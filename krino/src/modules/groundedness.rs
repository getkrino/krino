//! NLI-based groundedness / faithfulness checking.
//!
//! Verifies that every claim in an LLM output is entailed by the provided
//! context using Natural Language Inference. Operates at sentence level.

use crate::error::Result;
use crate::models::inference::{EmbeddingSimilarity, SequenceClassifier, SequenceClassifierInput};
use crate::pipeline::report::ModuleDetail;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// NLI label constants.
/// Matches MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli id2label mapping.
pub const LABEL_ENTAILMENT: usize = 0;
pub const LABEL_NEUTRAL: usize = 1;
pub const LABEL_CONTRADICTION: usize = 2;

/// Type alias for claim evidence map.
/// Maps claim index to vector of (`context_idx`, probs, `similarity_score`).
type EvidenceMap = HashMap<usize, Vec<(usize, Vec<f64>, Option<f32>)>>;

/// Configuration for groundedness checking.
//
// Four independent boolean knobs (strict-mode, adaptive top-K, matrix output,
// compound flagging). They control orthogonal behaviors and don't form a
// state machine, so the pedantic struct_excessive_bools lint doesn't apply.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundednessConfig {
    /// Confidence threshold for flagging contradictions [0.0, 1.0]
    pub contradiction_threshold: f64,

    /// Whether NEUTRAL verdicts count as unsupported
    /// (stricter mode for regulated industries)
    pub treat_neutral_as_unsupported: bool,

    /// Minimum claim length in characters to evaluate
    /// (skip very short fragments like "Yes." or "OK.")
    pub min_claim_length: usize,

    /// Number of top-K context sentences to evaluate per claim
    /// after embedding pre-filtering. Higher = more accurate, slower.
    /// Set to 0 to disable pre-filtering (evaluate ALL context sentences).
    pub top_k_context: usize,

    /// Minimum embedding similarity to consider a context sentence
    /// as a candidate for NLI evaluation. Sentences below this
    /// threshold are skipped even within the top-K.
    pub min_similarity_threshold: f32,

    /// When true, the per-claim top-K is scaled by claim length:
    /// `clamp(ceil(claim_chars / 40), 3, top_k_context)`. Short atomic
    /// claims get fewer candidates without changing verdict stability on
    /// the long tail. Has no effect when `top_k_context == 0`.
    pub adaptive_top_k: bool,

    /// Whether to include the full entailment matrix in the result.
    /// When true, the result contains NLI scores for every evaluated
    /// `(claim, context_sentence)` pair — useful for debugging and
    /// explainability. When false, only the per-claim verdict is stored.
    pub include_entailment_matrix: bool,

    /// Whether to flag compound sentences (containing "and", "while",
    /// semicolons, etc.) that may contain multiple atomic claims.
    /// Does not split them — just marks them in the verdict.
    pub flag_compound_claims: bool,

    /// Floor on per-pair entailment probability for a context sentence to
    /// be eligible as partial-evidence on a compound claim. Acts as a
    /// guardrail under the three-condition partial rule (see
    /// `partial_neutral_ceiling` and `partial_similarity_floor`); the rule
    /// itself does most of the gating. Set conservatively low (0.2 by
    /// default) so it only filters out pathological pairs where every
    /// probability is near-zero.
    pub partial_threshold: f64,

    /// Upper bound on per-pair neutral probability for a context sentence to
    /// count as partial evidence. The intuition: a high neutral means the
    /// model is *disinterestedly* uncertain (the sentence is off-topic);
    /// a moderate neutral with entailment > contradiction means the model
    /// is *interestedly* uncertain (the sentence partially supports the
    /// claim). Empirical data on RoBERTa-large MNLI INT8 (2026-05-19)
    /// showed true-compound mean n ≈ 0.59 vs unrelated-content n ≈ 0.71,
    /// so 0.65 splits the two cases.
    pub partial_neutral_ceiling: f64,

    /// Lower bound on embedding similarity for a context sentence to count
    /// as partial evidence. Filters out sentences the pre-filter only
    /// admitted because of top-K — they were the best available but not
    /// truly topically related. Has no effect when `top_k_context == 0`
    /// (pre-filtering disabled) since similarity scores are unavailable
    /// then; in that case the engine defers to the other two conditions.
    pub partial_similarity_floor: f32,
}

impl Default for GroundednessConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: 0.7,
            treat_neutral_as_unsupported: false,
            min_claim_length: 10,
            // Pre-filtering enabled by default for performance
            // Reduces N×M NLI calls to N×K where K=5
            // IMPORTANT: Requires a real embedding backend (not MockEmbedding)
            // With MockEmbedding, set top_k_context=0 to disable pre-filtering
            top_k_context: 5,
            min_similarity_threshold: 0.1,
            adaptive_top_k: false,
            include_entailment_matrix: false,
            flag_compound_claims: true,
            partial_threshold: 0.2,
            partial_neutral_ceiling: 0.65,
            partial_similarity_floor: 0.7,
        }
    }
}

/// Per-request overrides for [`GroundednessChecker::check_with_overrides`].
///
/// Each field is `Option<T>`: `Some(value)` overrides the corresponding
/// `GroundednessConfig` field for this request only, `None` keeps the
/// configured default. Intentionally narrow — only fields that are safe to
/// flip per-request without correctness side effects.
#[derive(Debug, Clone, Copy, Default)]
pub struct RequestOverrides {
    /// Override `include_entailment_matrix` for this request.
    pub include_matrix: Option<bool>,
    /// Override `top_k_context` for this request. `Some(0)` disables
    /// pre-filtering — every (claim, `context_sentence`) pair is sent to NLI.
    /// Useful for audit probes that need the full matrix.
    pub top_k_context: Option<usize>,
}

/// Computes the effective top-K for a single claim under the adaptive policy.
///
/// `K_effective = clamp(ceil(claim_chars / 40), 3, top_k_context)`
///
/// Short claims get fewer candidates (saves NLI calls) while longer compound
/// claims keep the full top-K (where multiple candidates can each cover
/// different sub-claims). Returns `top_k_context` unchanged when adaptive is off.
#[must_use]
pub fn effective_top_k(claim_chars: usize, top_k_context: usize, adaptive: bool) -> usize {
    if !adaptive || top_k_context == 0 {
        return top_k_context;
    }
    let scaled = claim_chars.div_ceil(40).max(3);
    scaled.min(top_k_context)
}

/// A sentence with its position in the source text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceSpan {
    pub text: String,
    pub index: usize,
    pub start: usize,
    pub end: usize,
}

/// Links a claim to its supporting or contradicting evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceLink {
    /// The context sentence text
    pub sentence: String,

    /// Index of this sentence in the context
    pub sentence_idx: usize,

    /// Character span in the original context (start, end)
    pub span: (usize, usize),

    /// Entailment probability for this pair
    pub entailment_prob: f64,

    /// Contradiction probability for this pair
    pub contradiction_prob: f64,

    /// Embedding similarity score (if pre-filtering was used)
    pub similarity_score: Option<f32>,
}

/// A single cell in the entailment matrix.
/// Represents the NLI relationship between one output claim and one context sentence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntailmentCell {
    /// Index of the output claim (row in the matrix)
    pub claim_idx: usize,

    /// Index of the context sentence (column in the matrix)
    pub context_idx: usize,

    /// The context sentence text
    pub context_sentence: String,

    /// NLI probabilities
    pub entailment_prob: f64,
    pub neutral_prob: f64,
    pub contradiction_prob: f64,

    /// Predicted label for this pair
    pub label: String,

    /// Embedding similarity that caused this pair to be evaluated
    /// (None if pre-filtering was disabled)
    pub similarity_score: Option<f32>,
}

/// Verdict for a single claim, with evidence tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimVerdict {
    /// The claim text
    pub claim: String,

    /// Index of this claim in the output
    pub claim_idx: usize,

    /// Character span in the original output (start, end)
    pub span: (usize, usize),

    /// Final NLI label: "entailment", "neutral", or "contradiction"
    pub label: String,

    /// Probability of entailment (from best evidence sentence)
    pub entailment_prob: f64,

    /// Probability of neutral (from best evidence sentence)
    pub neutral_prob: f64,

    /// Probability of contradiction (from best evidence sentence)
    pub contradiction_prob: f64,

    /// Whether this claim is considered supported
    pub supported: bool,

    /// The specific context sentence that best supports this claim.
    /// This is the sentence with the highest entailment score.
    pub best_evidence: Option<EvidenceLink>,

    /// The specific context sentence that most contradicts this claim
    /// (if any contradiction was detected above threshold).
    pub strongest_contradiction: Option<EvidenceLink>,

    /// Number of context sentences evaluated for this claim
    pub context_sentences_evaluated: usize,

    /// Whether this claim was flagged as potentially compound
    /// (may contain multiple atomic claims)
    pub is_compound: bool,

    /// Context sentences that jointly support this claim when the verdict
    /// is `"partial"`. Empty for all other verdicts. Each entry's
    /// `entailment_prob` was ≥ `partial_threshold`; mean of these is the
    /// claim's reported `entailment_prob`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supporting_evidence: Vec<EvidenceLink>,
}

/// Full result from groundedness checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundednessResult {
    /// Per-claim verdicts with evidence links
    pub verdicts: Vec<ClaimVerdict>,

    /// The full entailment matrix (only populated if `config.include_entailment_matrix`)
    /// Rows = output claims, Columns = context sentences
    /// Each cell contains the NLI scores for that `(claim, context_sentence)` pair
    pub entailment_matrix: Option<Vec<EntailmentCell>>,

    /// Context sentences that were extracted (for reference)
    pub context_sentences: Vec<SentenceSpan>,

    /// Output claims that were extracted (for reference)
    pub output_claims: Vec<SentenceSpan>,

    /// Aggregate faithfulness score [0.0, 1.0]
    /// Fraction of claims classified as entailed
    pub faithfulness_score: f64,

    /// Total claims evaluated
    pub total_claims: usize,

    /// Claims classified as entailed
    pub supported_claims: usize,

    /// Claims classified as contradicted
    pub contradicted_claims: usize,

    /// Claims classified as neutral
    pub neutral_claims: usize,

    /// Claims classified as `"partial"` — compound claims where ≥2
    /// distinct context sentences cleared `partial_threshold` and
    /// jointly support the claim. Counted as supported in the
    /// faithfulness score.
    #[serde(default)]
    pub partial_claims: usize,

    /// Number of NLI inference calls made
    pub nli_calls: usize,

    /// Number of NLI calls saved by embedding pre-filtering
    pub nli_calls_saved: usize,

    /// Number of NLI calls skipped by the substring fast-path (a claim that
    /// appears verbatim in a context sentence is entailed by definition).
    #[serde(default)]
    pub nli_calls_skipped_substring: usize,

    /// Total inference latency (ms)
    pub latency_ms: f64,

    /// Time spent splitting context + output into sentences/claims (ms).
    #[serde(default)]
    pub split_ms: f64,

    /// Time spent computing claim + context embeddings for pre-filtering (ms).
    /// Zero on the all-fast-path / no-context / pre-filter-disabled paths.
    #[serde(default)]
    pub embedding_ms: f64,

    /// Time spent in NLI batch inference (ms). Zero on the all-fast-path /
    /// no-context paths.
    #[serde(default)]
    pub nli_ms: f64,
}

/// Splits text into individual sentences/claims for NLI evaluation.
///
/// Uses rule-based sentence boundary detection. Not a model — deterministic
/// by construction.
///
/// Handles:
/// - Standard sentence endings (. ! ?)
/// - Abbreviations (Dr., Mr., U.S., etc.) — NOT treated as boundaries
/// - Decimal numbers (3.14) — NOT treated as boundaries
/// - Semicolons used as clause separators — treated as boundaries
///
/// Returns `Vec<(claim_text, start_offset, end_offset)>`
#[must_use]
pub fn split_into_claims(text: &str) -> Vec<(String, usize, usize)> {
    let mut claims = Vec::new();

    // Known abbreviations that should not trigger sentence splits
    let abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "i.e.", "e.g.", "U.S.",
        "U.K.",
    ];

    // Build char->byte index mapping once upfront (O(n) instead of O(n²))
    let char_to_byte: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let chars: Vec<char> = text.chars().collect();

    let mut current_start = 0;
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '.' || ch == '!' || ch == '?' || ch == ';' {
            // Check if this is an abbreviation (for '.')
            if ch == '.' {
                let preceding: String = chars[current_start..=i].iter().collect();
                let is_abbreviation = abbreviations
                    .iter()
                    .any(|abbr| preceding.trim().ends_with(abbr));

                // Check if it's a decimal number (digit.digit)
                let is_decimal = i > 0
                    && i + 1 < chars.len()
                    && chars[i - 1].is_ascii_digit()
                    && chars[i + 1].is_ascii_digit();

                if is_abbreviation || is_decimal {
                    i += 1;
                    continue;
                }
            }

            // Found a sentence boundary
            let end = i + 1;
            let claim_text: String = chars[current_start..end].iter().collect();
            let trimmed = claim_text.trim();

            if !trimmed.is_empty() {
                // O(1) byte offset lookup using pre-built index
                let byte_start = char_to_byte[current_start];
                let byte_end = if end >= chars.len() {
                    text.len()
                } else {
                    char_to_byte[end]
                };

                claims.push((trimmed.to_string(), byte_start, byte_end));
            }

            // Skip whitespace after boundary
            i += 1;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            current_start = i;
        } else {
            i += 1;
        }
    }

    // Capture any trailing text that didn't end with punctuation
    if current_start < chars.len() {
        let remaining: String = chars[current_start..].iter().collect();
        let trimmed = remaining.trim();
        if !trimmed.is_empty() {
            let byte_start = char_to_byte[current_start];
            claims.push((trimmed.to_string(), byte_start, text.len()));
        }
    }

    claims
}

/// Checks if a sentence likely contains multiple independently verifiable claims.
///
/// Flags sentences containing conjunctions, semicolons, relative clauses,
/// or multiple facts. Does NOT split them — just marks them
/// so the caller knows the granularity is sentence-level, not fact-level.
///
/// # Determinism
///
/// Pure string pattern matching. Always returns the same result for the same input.
///
/// # Examples
///
/// ```
/// use krino::modules::groundedness::is_compound_claim;
///
/// assert!(is_compound_claim("Revenue grew 15%, and operating margin expanded."));
/// assert!(is_compound_claim("The company, which was founded in 1987, reported earnings."));
/// assert!(!is_compound_claim("Revenue grew 15% year over year."));
/// ```
#[must_use]
pub fn is_compound_claim(text: &str) -> bool {
    let compound_indicators = [
        // Coordinating conjunctions joining independent clauses
        ", and ",
        ", but ",
        ", while ",
        ", whereas ",
        ", although ",
        // Relative clauses introducing additional facts
        ", which ",
        ", who ",
        ", where ",
        // Semicolons separating independent clauses
        "; ",
        // Appositives
        " — ",
        " – ",
    ];

    compound_indicators
        .iter()
        .any(|indicator| text.contains(indicator))
}

/// Normalizes a string for substring comparison in the fast-path:
/// lowercase, whitespace-collapsed, trailing punctuation stripped.
///
/// Preserves internal punctuation so "$4.2 billion" stays intact, but trims
/// trailing `. ! ? , ; :` and quotes so "Revenue grew." matches "Revenue grew".
///
/// # Determinism
///
/// Pure string transformation. Same input always produces the same output.
#[must_use]
pub fn normalize_for_substring(s: &str) -> String {
    let lowered = s.to_lowercase();
    let collapsed: String = lowered.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .trim_matches(|c: char| matches!(c, '.' | '!' | '?' | ',' | ';' | ':' | '"' | '\''))
        .to_string()
}

/// Returns the index of the first context sentence that contains `claim` as
/// a substring after normalization, or `None`.
///
/// A claim of length < `MIN_LEN` is never matched — single words and tiny
/// fragments are too likely to false-positive ("the" appears everywhere).
///
/// # Determinism
///
/// Iterates context sentences in order; same input always yields the same
/// first-match index.
fn find_substring_match(normalized_claim: &str, normalized_context: &[String]) -> Option<usize> {
    // 12 chars is the smallest claim length where a substring match is unlikely
    // to be coincidental. Below this, fragments like "the cat" or "in 2024"
    // appear in unrelated contexts and produce spurious entailment verdicts.
    // 12 chars admits short-but-distinctive claims like "$4.2 billion".
    const MIN_LEN: usize = 12;
    if normalized_claim.len() < MIN_LEN {
        return None;
    }
    normalized_context
        .iter()
        .position(|ctx| ctx.contains(normalized_claim))
}

/// Selects the top-K most similar context sentences for each claim
/// using embedding cosine similarity.
///
/// Returns a `Vec` of `(claim_idx, Vec<(context_idx, similarity_score)>)`
/// where each claim maps to its top-K context sentence candidates.
///
/// If `top_k` is 0, returns all context sentences for every claim (no filtering).
///
/// # Determinism
///
/// Given deterministic embeddings, this function is deterministic.
/// The sorting is stable and uses `partial_cmp` which is deterministic for
/// non-NaN values.
///
/// # Arguments
///
/// * `claim_embeddings` - Embedding vectors for each claim
/// * `context_embeddings` - Embedding vectors for each context sentence
/// * `top_k` - Number of top candidates to keep (0 = keep all)
/// * `min_similarity` - Minimum similarity score threshold
///
/// # Returns
///
/// For each claim, a list of `(context_idx, similarity)` pairs sorted by similarity descending.
///
/// # Performance
///
/// Uses matrix multiplication to compute all N×M similarities at once, which is
/// 10-50× faster than computing them one-by-one for typical groundedness workloads.
fn prefilter_by_embedding(
    claim_embeddings: &[Vec<f32>],
    context_embeddings: &[Vec<f32>],
    top_k: usize,
    min_similarity: f32,
) -> Vec<Vec<(usize, f32)>> {
    use crate::models::inference::cosine_similarity_matrix;

    if top_k == 0 {
        // No filtering — return all context indices for every claim
        return claim_embeddings
            .iter()
            .map(|_| {
                (0..context_embeddings.len())
                    .map(|idx| (idx, 1.0_f32))
                    .collect()
            })
            .collect();
    }

    // Compute full similarity matrix [N_claims, M_contexts] using matrix multiply
    // This is much faster than computing N×M similarities one-by-one
    let similarity_matrix = cosine_similarity_matrix(claim_embeddings, context_embeddings);

    // For each claim, find top-K most similar context sentences
    similarity_matrix
        .into_iter()
        .map(|similarities| {
            // Create (index, similarity) pairs
            let mut scored: Vec<(usize, f32)> = similarities
                .into_iter()
                .enumerate()
                .filter(|(_, sim)| *sim >= min_similarity)
                .collect();

            // Sort by similarity descending (stable sort for determinism)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-K
            scored.truncate(top_k);

            scored
        })
        .collect()
}

/// Chunks context into overlapping segments for models with short context windows.
///
/// Returns Vec of context chunk strings. If `max_tokens` is 0 or the context
/// fits within the limit, returns the original context as a single chunk.
fn chunk_context(
    context: &str,
    tokenizer: &tokenizers::Tokenizer,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<String> {
    if max_tokens == 0 {
        return vec![context.to_string()];
    }

    // Tokenize to check length
    let Ok(encoding) = tokenizer.encode(context, false) else {
        return vec![context.to_string()];
    };

    let token_count = encoding.get_ids().len();
    if token_count <= max_tokens {
        return vec![context.to_string()];
    }

    // Need to chunk — split by sentences first for cleaner boundaries
    let sentences = split_into_claims(context);
    let mut chunks = Vec::new();
    let mut current_chunk_sentences: Vec<String> = Vec::new();
    let mut current_token_count = 0;

    for (sentence, _, _) in &sentences {
        let sentence_tokens = match tokenizer.encode(sentence.as_str(), false) {
            Ok(enc) => enc.get_ids().len(),
            Err(_) => sentence.len() / 4, // rough estimate
        };

        if current_token_count + sentence_tokens > max_tokens && !current_chunk_sentences.is_empty()
        {
            // Emit current chunk
            chunks.push(current_chunk_sentences.join(" "));

            // Keep overlap: retain last N tokens worth of sentences
            let mut overlap_count = 0;
            let mut keep_from = current_chunk_sentences.len();
            for (idx, s) in current_chunk_sentences.iter().enumerate().rev() {
                let s_tokens = match tokenizer.encode(s.as_str(), false) {
                    Ok(enc) => enc.get_ids().len(),
                    Err(_) => s.len() / 4,
                };
                overlap_count += s_tokens;
                if overlap_count >= overlap_tokens {
                    keep_from = idx;
                    break;
                }
                keep_from = idx;
            }

            current_chunk_sentences = current_chunk_sentences[keep_from..].to_vec();
            current_token_count = overlap_count;
        }

        current_chunk_sentences.push(sentence.clone());
        current_token_count += sentence_tokens;
    }

    // Emit final chunk
    if !current_chunk_sentences.is_empty() {
        chunks.push(current_chunk_sentences.join(" "));
    }

    if chunks.is_empty() {
        chunks.push(context.to_string());
    }

    chunks
}

/// NLI-based groundedness checker.
///
/// NLI-based groundedness checker using `SummaC` sentence-to-sentence evaluation.
///
/// Instead of chunking context into overlapping windows, this approach:
/// 1. Splits both context and output into individual sentences
/// 2. Uses embedding similarity to pre-filter relevant context sentences per claim
/// 3. Runs NLI on each `(claim, relevant_context_sentence)` pair
/// 4. Builds an entailment matrix where rows=claims, columns=context sentences
/// 5. Takes max-entailment per row to determine each claim's verdict
///
/// This produces fine-grained evidence tracing: each verdict points to the
/// specific context sentence that supports or contradicts the claim.
pub struct GroundednessChecker {
    /// NLI backend (`roberta-large-mnli` or `DeBERTa`-MNLI via ONNX, or
    /// `ModernBERT-NLI` via Candle). All three emit probs in the engine's
    /// canonical `[entailment, neutral, contradiction]` order — the ONNX
    /// backend permutes raw logits at load time when needed.
    nli_backend: Arc<dyn SequenceClassifier>,

    /// Embedding backend for pre-filtering (sentence-transformers via Candle)
    embedding_backend: Arc<dyn EmbeddingSimilarity>,

    /// Configuration
    config: GroundednessConfig,
}

/// Thresholds for the multi-evidence partial-verdict rule. Bundled into a
/// struct so the helper signature doesn't keep growing as we tune the rule.
#[derive(Debug, Clone, Copy)]
struct PartialThresholds {
    entailment_floor: f64,
    neutral_ceiling: f64,
    similarity_floor: f32,
}

/// Gathers context sentences that pass the three-condition partial-evidence
/// rule into a deduplicated list, sorted by descending entailment.
///
/// A candidate `(ctx_idx, probs, similarity)` qualifies iff *all* hold:
///   - `entailment >= thresholds.entailment_floor` (pathological-pair guard)
///   - `entailment > contradiction` (model leans toward support)
///   - `neutral <= thresholds.neutral_ceiling` (uncertain in an *interested*
///     way — not just dismissing the pair as off-topic)
///   - `similarity >= thresholds.similarity_floor` if similarity is `Some`;
///     no constraint when similarity is `None` (pre-filtering disabled)
///
/// The dedup-by-`sentence_idx` step is what makes "two distinct sentences
/// supporting two facts" distinguishable from "two redundant entailments of
/// the same fact" — same sentence, same signal, count it once.
///
/// Empirical basis for the three-condition design: a curl probe against the
/// deployed RoBERTa-large MNLI INT8 on 2026-05-19 showed true compound
/// claims sit at e ≈ 0.27 / n ≈ 0.59 / sim ≈ 0.79 across 3 supporting
/// sentences, while unrelated content sat at e ≈ 0.19 / n ≈ 0.71 / sim ≈
/// 0.51. Entailment alone gives a 7-point gap; neutral and similarity each
/// give wider gaps. The conjunction of the three reliably separates the two
/// regimes.
///
/// # Determinism
///
/// Pure aggregation over inputs; stable sort. Same inputs always produce the
/// same output ordering.
fn collect_partial_evidence(
    evidence: &[(usize, Vec<f64>, Option<f32>)],
    context_sentences: &[(String, usize, usize)],
    thresholds: PartialThresholds,
) -> Vec<EvidenceLink> {
    use std::collections::HashSet;

    let qualifies = |probs: &[f64], similarity: &Option<f32>| -> bool {
        let e = probs.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0);
        let n = probs.get(LABEL_NEUTRAL).copied().unwrap_or(0.0);
        let c = probs.get(LABEL_CONTRADICTION).copied().unwrap_or(0.0);

        if e < thresholds.entailment_floor || e <= c || n > thresholds.neutral_ceiling {
            return false;
        }
        // Similarity check only enforced when pre-filtering was active.
        match similarity {
            Some(sim) => *sim >= thresholds.similarity_floor,
            None => true,
        }
    };

    // Sort candidates by entailment desc so the first occurrence of each
    // sentence_idx is its strongest entailment.
    let mut ranked: Vec<&(usize, Vec<f64>, Option<f32>)> = evidence
        .iter()
        .filter(|(_, probs, similarity)| qualifies(probs, similarity))
        .collect();
    ranked.sort_by(|a, b| {
        let ea = a.1.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0);
        let eb = b.1.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0);
        eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen: HashSet<usize> = HashSet::new();
    let mut out: Vec<EvidenceLink> = Vec::new();
    for (ctx_idx, probs, similarity) in ranked {
        if !seen.insert(*ctx_idx) {
            continue;
        }
        let (ctx_text, ctx_start, ctx_end) = &context_sentences[*ctx_idx];
        out.push(EvidenceLink {
            sentence: ctx_text.clone(),
            sentence_idx: *ctx_idx,
            span: (*ctx_start, *ctx_end),
            entailment_prob: probs.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0),
            contradiction_prob: probs.get(LABEL_CONTRADICTION).copied().unwrap_or(0.0),
            similarity_score: *similarity,
        });
    }
    out
}

impl GroundednessChecker {
    /// Creates a new groundedness checker.
    ///
    /// # Arguments
    ///
    /// * `nli_backend` - NLI model for entailment/contradiction detection
    /// * `embedding_backend` - Embedding model for pre-filtering (sentence-transformers)
    /// * `config` - Verification configuration
    pub fn new(
        nli_backend: Arc<dyn SequenceClassifier>,
        embedding_backend: Arc<dyn EmbeddingSimilarity>,
        config: GroundednessConfig,
    ) -> Self {
        Self {
            nli_backend,
            embedding_backend,
            config,
        }
    }

    /// Checks whether every claim in `output` is supported by `context`.
    ///
    /// Uses SummaC-style sentence-to-sentence NLI matrix with embedding pre-filtering.
    pub fn check(&self, context: &str, output: &str) -> Result<GroundednessResult> {
        self.check_inner(
            context,
            output,
            self.config.include_entailment_matrix,
            self.config.top_k_context,
        )
    }

    /// Like `check`, but lets the caller override a small set of per-request
    /// knobs. `None` on an override field means "use the configured value".
    ///
    /// Only `include_matrix` and `top_k_context` are overridable here; other
    /// config fields (thresholds, similarity floor, adaptive top-K, partial
    /// rule knobs) have correctness implications and are intentionally not
    /// per-request — flipping them is a wider design decision.
    pub fn check_with_overrides(
        &self,
        context: &str,
        output: &str,
        overrides: RequestOverrides,
    ) -> Result<GroundednessResult> {
        let include_matrix = overrides
            .include_matrix
            .unwrap_or(self.config.include_entailment_matrix);
        let top_k_context = overrides.top_k_context.unwrap_or(self.config.top_k_context);
        self.check_inner(context, output, include_matrix, top_k_context)
    }

    #[allow(clippy::too_many_lines)]
    fn check_inner(
        &self,
        context: &str,
        output: &str,
        include_matrix: bool,
        top_k_context: usize,
    ) -> Result<GroundednessResult> {
        let start = Instant::now();
        info!(
            "Running groundedness check: context={} chars, output={} chars",
            context.len(),
            output.len()
        );

        // Step 1: Split both into sentences
        let split_start = Instant::now();
        let context_sentences = split_into_claims(context);
        let output_claims = split_into_claims(output);
        let split_ms = split_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Split into {} context sentences, {} output claims",
            context_sentences.len(),
            output_claims.len()
        );

        // Filter claims by minimum length
        let evaluable_claims: Vec<_> = output_claims
            .iter()
            .enumerate()
            .filter(|(_, (text, _, _))| text.len() >= self.config.min_claim_length)
            .collect();

        if evaluable_claims.is_empty() {
            return Ok(Self::empty_result(
                &context_sentences,
                &output_claims,
                start,
                split_ms,
            ));
        }

        if context_sentences.is_empty() {
            return Ok(self.no_context_result(
                &evaluable_claims,
                &context_sentences,
                &output_claims,
                start,
                split_ms,
            ));
        }

        // Step 2: Substring fast-path. If a claim appears verbatim in any
        // context sentence (after normalization), entailment is definitional —
        // skip NLI entirely. We pre-normalize the context once and reuse it.
        let normalized_context: Vec<String> = context_sentences
            .iter()
            .map(|(text, _, _)| normalize_for_substring(text))
            .collect();

        let mut fast_path_verdicts: Vec<ClaimVerdict> = Vec::new();
        let mut nli_claims: Vec<&(usize, &(String, usize, usize))> =
            Vec::with_capacity(evaluable_claims.len());

        for entry in &evaluable_claims {
            let (original_idx, (claim_text, claim_start, claim_end)) = entry;
            let normalized_claim = normalize_for_substring(claim_text);

            if let Some(ctx_idx) = find_substring_match(&normalized_claim, &normalized_context) {
                let (ctx_text, ctx_start, ctx_end) = &context_sentences[ctx_idx];
                let evidence = EvidenceLink {
                    sentence: ctx_text.clone(),
                    sentence_idx: ctx_idx,
                    span: (*ctx_start, *ctx_end),
                    entailment_prob: 1.0,
                    contradiction_prob: 0.0,
                    similarity_score: None,
                };
                fast_path_verdicts.push(ClaimVerdict {
                    claim: (*claim_text).clone(),
                    claim_idx: *original_idx,
                    span: (*claim_start, *claim_end),
                    label: "entailment".to_string(),
                    entailment_prob: 1.0,
                    neutral_prob: 0.0,
                    contradiction_prob: 0.0,
                    supported: true,
                    best_evidence: Some(evidence),
                    strongest_contradiction: None,
                    context_sentences_evaluated: 1,
                    is_compound: self.config.flag_compound_claims && is_compound_claim(claim_text),
                    supporting_evidence: Vec::new(),
                });
            } else {
                nli_claims.push(entry);
            }
        }

        let nli_calls_skipped_substring = fast_path_verdicts.len();
        debug!(
            "Substring fast-path: {} claims short-circuited, {} sent to NLI",
            nli_calls_skipped_substring,
            nli_claims.len()
        );

        // If every claim was caught by the fast-path, we're done.
        if nli_claims.is_empty() {
            return Ok(self.assemble_result(
                fast_path_verdicts,
                None,
                &context_sentences,
                &output_claims,
                0,
                0,
                nli_calls_skipped_substring,
                start,
                split_ms,
                0.0,
                0.0,
            ));
        }

        // Step 3 & 4: Compute embeddings and pre-filter for the remaining claims.
        let mut embedding_ms: f64 = 0.0;
        let candidates = if top_k_context > 0 {
            // Pre-filtering enabled - compute embeddings and filter
            let claim_texts: Vec<&str> =
                nli_claims.iter().map(|(_, (t, _, _))| t.as_str()).collect();
            let context_texts: Vec<&str> = context_sentences
                .iter()
                .map(|(text, _, _)| text.as_str())
                .collect();

            let embed_start = Instant::now();
            let claim_embeddings = self.embedding_backend.embed(&claim_texts)?;
            let context_embeddings = self.embedding_backend.embed(&context_texts)?;
            embedding_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

            debug!(
                "Computed {} + {} embeddings",
                claim_embeddings.len(),
                context_embeddings.len()
            );

            let mut raw = prefilter_by_embedding(
                &claim_embeddings,
                &context_embeddings,
                top_k_context,
                self.config.min_similarity_threshold,
            );

            // Adaptive top-K: shrink each claim's candidate list by claim length.
            // Operates on the already-sorted top-K, so we just truncate.
            if self.config.adaptive_top_k {
                for (local_idx, candidate_list) in raw.iter_mut().enumerate() {
                    let (_, (claim_text, _, _)) = &nli_claims[local_idx];
                    let k = effective_top_k(claim_text.chars().count(), top_k_context, true);
                    candidate_list.truncate(k);
                }
            }

            raw
        } else {
            // Pre-filtering disabled - evaluate all context sentences for each claim
            debug!("Pre-filtering disabled, evaluating all context sentences");
            let all_context_indices: Vec<(usize, f32)> = (0..context_sentences.len())
                .map(|idx| (idx, 1.0_f32))
                .collect();

            vec![all_context_indices; nli_claims.len()]
        };

        let total_possible = nli_claims.len() * context_sentences.len();

        // Step 5: Build NLI inputs for all (claim, candidate_context) pairs
        // We collect all inputs upfront to enable batched inference
        let mut nli_inputs: Vec<SequenceClassifierInput> = Vec::new();
        let mut input_map: Vec<(usize, usize, Option<f32>)> = Vec::new();

        for (local_idx, candidate_list) in candidates.iter().enumerate() {
            let (original_idx, (claim_text, _, _)) = &nli_claims[local_idx];

            for &(ctx_idx, similarity) in candidate_list {
                let (ctx_text, _, _) = &context_sentences[ctx_idx];

                nli_inputs.push(SequenceClassifierInput {
                    text_a: ctx_text.clone(),      // premise = context sentence
                    text_b: (*claim_text).clone(), // hypothesis = claim
                });

                let sim = if top_k_context > 0 {
                    Some(similarity)
                } else {
                    None
                };
                input_map.push((*original_idx, ctx_idx, sim));
            }
        }

        let total_filtered = nli_inputs.len();
        let nli_calls_saved = total_possible.saturating_sub(total_filtered);

        debug!(
            "Pre-filter: {} NLI calls needed ({} saved from {} total possible)",
            total_filtered, nli_calls_saved, total_possible
        );

        // Step 6: Batch NLI inference
        let nli_start = Instant::now();
        let nli_results = self.nli_backend.classify(&nli_inputs)?;
        let nli_ms = nli_start.elapsed().as_secs_f64() * 1000.0;

        // Step 7: Assemble entailment matrix and per-claim verdicts
        let mut entailment_cells: Vec<EntailmentCell> = Vec::new();

        // Group NLI results by claim
        // Key: original claim index → Vec of (context_idx, probs, similarity)
        let mut claim_evidence: EvidenceMap = HashMap::new();

        for (result_idx, nli_result) in nli_results.iter().enumerate() {
            let (claim_idx, ctx_idx, similarity) = input_map[result_idx];

            let probs = &nli_result.probabilities;

            if include_matrix {
                let (ctx_text, _, _) = &context_sentences[ctx_idx];
                entailment_cells.push(EntailmentCell {
                    claim_idx,
                    context_idx: ctx_idx,
                    context_sentence: ctx_text.clone(),
                    entailment_prob: probs.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0),
                    neutral_prob: probs.get(LABEL_NEUTRAL).copied().unwrap_or(0.0),
                    contradiction_prob: probs.get(LABEL_CONTRADICTION).copied().unwrap_or(0.0),
                    label: nli_result.predicted_label.clone(),
                    similarity_score: similarity,
                });
            }

            claim_evidence
                .entry(claim_idx)
                .or_default()
                .push((ctx_idx, probs.clone(), similarity));
        }

        // Step 8: Build per-claim verdicts for the NLI-evaluated claims, then
        // merge with fast-path verdicts and sort by claim_idx to preserve
        // input ordering.
        let mut verdicts = fast_path_verdicts;
        verdicts.reserve(nli_claims.len());

        for entry in &nli_claims {
            let (original_idx, (claim_text, claim_start, claim_end)) = **entry;
            let evidence = claim_evidence
                .get(&original_idx)
                .cloned()
                .unwrap_or_default();

            let verdict = self.build_verdict(
                claim_text,
                original_idx,
                (*claim_start, *claim_end),
                &evidence,
                &context_sentences,
            );

            verdicts.push(verdict);
        }
        verdicts.sort_by_key(|v| v.claim_idx);

        let entailment_matrix = if include_matrix {
            Some(entailment_cells)
        } else {
            None
        };

        Ok(self.assemble_result(
            verdicts,
            entailment_matrix,
            &context_sentences,
            &output_claims,
            total_filtered,
            nli_calls_saved,
            nli_calls_skipped_substring,
            start,
            split_ms,
            embedding_ms,
            nli_ms,
        ))
    }

    /// Common tail of `check()`: turns verdicts + bookkeeping into a final
    /// `GroundednessResult`. Used by both the all-fast-path early return and
    /// the normal NLI path.
    //
    // `&self` is retained (rather than an associated fn) to match the shape
    // of build_verdict / classify_verdict, and so future config-dependent
    // logging or metric tagging won't require a call-site change.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn assemble_result(
        &self,
        verdicts: Vec<ClaimVerdict>,
        entailment_matrix: Option<Vec<EntailmentCell>>,
        context_sentences: &[(String, usize, usize)],
        output_claims: &[(String, usize, usize)],
        nli_calls: usize,
        nli_calls_saved: usize,
        nli_calls_skipped_substring: usize,
        start: Instant,
        split_ms: f64,
        embedding_ms: f64,
        nli_ms: f64,
    ) -> GroundednessResult {
        let total_claims = verdicts.len();
        let supported_claims = verdicts.iter().filter(|v| v.supported).count();
        let contradicted_claims = verdicts
            .iter()
            .filter(|v| v.label == "contradiction")
            .count();
        let neutral_claims = verdicts.iter().filter(|v| v.label == "neutral").count();
        let partial_claims = verdicts.iter().filter(|v| v.label == "partial").count();

        #[allow(clippy::cast_precision_loss)]
        let faithfulness_score = if total_claims > 0 {
            supported_claims as f64 / total_claims as f64
        } else {
            1.0
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Groundedness check complete: {supported_claims}/{total_claims} supported, \
             score={faithfulness_score:.3}, {nli_calls} NLI calls, \
             {nli_calls_skipped_substring} fast-path skips, {latency_ms:.2}ms"
        );

        GroundednessResult {
            verdicts,
            entailment_matrix,
            context_sentences: context_sentences
                .iter()
                .enumerate()
                .map(|(i, (text, start, end))| SentenceSpan {
                    text: text.clone(),
                    index: i,
                    start: *start,
                    end: *end,
                })
                .collect(),
            output_claims: output_claims
                .iter()
                .enumerate()
                .map(|(i, (text, start, end))| SentenceSpan {
                    text: text.clone(),
                    index: i,
                    start: *start,
                    end: *end,
                })
                .collect(),
            faithfulness_score,
            total_claims,
            supported_claims,
            contradicted_claims,
            neutral_claims,
            partial_claims,
            nli_calls,
            nli_calls_saved,
            nli_calls_skipped_substring,
            latency_ms,
            split_ms,
            embedding_ms,
            nli_ms,
        }
    }

    /// Builds a verdict for a single claim from its evidence pairs.
    fn build_verdict(
        &self,
        claim: &str,
        claim_idx: usize,
        span: (usize, usize),
        evidence: &[(usize, Vec<f64>, Option<f32>)],
        context_sentences: &[(String, usize, usize)],
    ) -> ClaimVerdict {
        let mut best_evidence: Option<EvidenceLink> = None;
        let mut best_probs = vec![0.0, 0.0, 0.0];
        // Pick `best_evidence` by max(entailment, contradiction) — the most
        // *informative* signal across evidence pairs. Neutral is excluded
        // because "the model is unsure" is never a useful verdict to surface.
        // If no pair has any informative signal (all-neutral), fall back to
        // the highest-similarity sentence so the user at least sees the most
        // topically-relevant context rather than the first one evaluated.
        let mut best_informative_score = -1.0_f64;
        let mut best_similarity_fallback = f32::NEG_INFINITY;

        let mut strongest_contradiction_prob = 0.0_f64;
        let mut strongest_contradiction: Option<EvidenceLink> = None;

        for (ctx_idx, probs, similarity) in evidence {
            let entailment = probs.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0);
            let contradiction = probs.get(LABEL_CONTRADICTION).copied().unwrap_or(0.0);
            let (ctx_text, ctx_start, ctx_end) = &context_sentences[*ctx_idx];

            let pair_informative = entailment.max(contradiction);
            let pair_better = if pair_informative > 0.0 || best_informative_score > 0.0 {
                // Primary mode: at least one pair has informative signal.
                pair_informative > best_informative_score
            } else {
                // All-neutral fallback: prefer the highest-similarity sentence.
                let sim = similarity.unwrap_or(f32::NEG_INFINITY);
                sim > best_similarity_fallback
            };

            if pair_better {
                best_informative_score = pair_informative;
                if let Some(sim) = similarity {
                    best_similarity_fallback = best_similarity_fallback.max(*sim);
                }
                best_probs.clone_from(probs);
                best_evidence = Some(EvidenceLink {
                    sentence: ctx_text.clone(),
                    sentence_idx: *ctx_idx,
                    span: (*ctx_start, *ctx_end),
                    entailment_prob: entailment,
                    contradiction_prob: contradiction,
                    similarity_score: *similarity,
                });
            }

            // Track strongest contradiction
            if contradiction > strongest_contradiction_prob
                && contradiction >= self.config.contradiction_threshold
            {
                strongest_contradiction_prob = contradiction;
                strongest_contradiction = Some(EvidenceLink {
                    sentence: ctx_text.clone(),
                    sentence_idx: *ctx_idx,
                    span: (*ctx_start, *ctx_end),
                    entailment_prob: entailment,
                    contradiction_prob: contradiction,
                    similarity_score: *similarity,
                });
            }
        }

        // Determine final verdict
        let (mut label, mut supported) = self.classify_verdict(&best_probs);
        let is_compound = self.config.flag_compound_claims && is_compound_claim(claim);

        // Multi-evidence aggregation for compound claims that landed neutral.
        // A compound claim covering multiple facts often gets neutral from any
        // single sentence — but if several distinct sentences each entail part
        // of it, the union supports the whole. We approximate "union entails"
        // by: ≥2 candidates over `partial_threshold` on distinct context
        // sentences, with no contradiction strong enough to override.
        //
        // This is a heuristic over NLI outputs, not true compositional
        // entailment: two sentences each weakly entailing the *same* half of
        // the claim will still trip the rule. The multi-evidence list is
        // surfaced so callers can judge.
        let mut supporting_evidence: Vec<EvidenceLink> = Vec::new();
        if is_compound && label == "neutral" && strongest_contradiction.is_none() {
            supporting_evidence = collect_partial_evidence(
                evidence,
                context_sentences,
                PartialThresholds {
                    entailment_floor: self.config.partial_threshold,
                    neutral_ceiling: self.config.partial_neutral_ceiling,
                    similarity_floor: self.config.partial_similarity_floor,
                },
            );

            if supporting_evidence.len() >= 2 {
                #[allow(clippy::cast_precision_loss)]
                let mean_entailment = supporting_evidence
                    .iter()
                    .map(|e| e.entailment_prob)
                    .sum::<f64>()
                    / supporting_evidence.len() as f64;

                // Rewrite the headline probs so downstream consumers (UI,
                // metrics, callers reading entailment_prob) see the aggregate
                // signal, not the single-best-evidence prob.
                best_probs[LABEL_ENTAILMENT] = mean_entailment;
                label = "partial".to_string();
                supported = true;
            } else {
                // < 2 distinct evidences → not partial, drop what we collected.
                supporting_evidence.clear();
            }
        }

        ClaimVerdict {
            claim: claim.to_string(),
            claim_idx,
            span,
            label,
            entailment_prob: best_probs.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0),
            neutral_prob: best_probs.get(LABEL_NEUTRAL).copied().unwrap_or(0.0),
            contradiction_prob: best_probs.get(LABEL_CONTRADICTION).copied().unwrap_or(0.0),
            supported,
            best_evidence,
            strongest_contradiction,
            context_sentences_evaluated: evidence.len(),
            is_compound,
            supporting_evidence,
        }
    }

    /// Determines the label and whether the claim is supported.
    fn classify_verdict(&self, probs: &[f64]) -> (String, bool) {
        let entailment = probs.get(LABEL_ENTAILMENT).copied().unwrap_or(0.0);
        let neutral = probs.get(LABEL_NEUTRAL).copied().unwrap_or(0.0);
        let contradiction = probs.get(LABEL_CONTRADICTION).copied().unwrap_or(0.0);

        if contradiction >= entailment
            && contradiction >= neutral
            && contradiction >= self.config.contradiction_threshold
        {
            ("contradiction".to_string(), false)
        } else if entailment >= neutral && entailment >= contradiction {
            ("entailment".to_string(), true)
        } else {
            let supported = !self.config.treat_neutral_as_unsupported;
            ("neutral".to_string(), supported)
        }
    }

    /// Helper for empty output (no claims to evaluate).
    fn empty_result(
        ctx: &[(String, usize, usize)],
        claims: &[(String, usize, usize)],
        start: Instant,
        split_ms: f64,
    ) -> GroundednessResult {
        GroundednessResult {
            verdicts: vec![],
            entailment_matrix: None,
            context_sentences: ctx
                .iter()
                .enumerate()
                .map(|(i, (t, s, e))| SentenceSpan {
                    text: t.clone(),
                    index: i,
                    start: *s,
                    end: *e,
                })
                .collect(),
            output_claims: claims
                .iter()
                .enumerate()
                .map(|(i, (t, s, e))| SentenceSpan {
                    text: t.clone(),
                    index: i,
                    start: *s,
                    end: *e,
                })
                .collect(),
            faithfulness_score: 1.0,
            total_claims: 0,
            supported_claims: 0,
            contradicted_claims: 0,
            neutral_claims: 0,
            partial_claims: 0,
            nli_calls: 0,
            nli_calls_saved: 0,
            nli_calls_skipped_substring: 0,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            split_ms,
            embedding_ms: 0.0,
            nli_ms: 0.0,
        }
    }

    /// Helper for when no context is provided (everything is neutral).
    fn no_context_result(
        &self,
        evaluable: &[(usize, &(String, usize, usize))],
        ctx: &[(String, usize, usize)],
        claims: &[(String, usize, usize)],
        start: Instant,
        split_ms: f64,
    ) -> GroundednessResult {
        let verdicts: Vec<ClaimVerdict> = evaluable
            .iter()
            .map(|(idx, (text, s, e))| ClaimVerdict {
                claim: text.clone(),
                claim_idx: *idx,
                span: (*s, *e),
                label: "neutral".to_string(),
                entailment_prob: 0.0,
                neutral_prob: 1.0,
                contradiction_prob: 0.0,
                supported: !self.config.treat_neutral_as_unsupported,
                best_evidence: None,
                strongest_contradiction: None,
                context_sentences_evaluated: 0,
                is_compound: self.config.flag_compound_claims && is_compound_claim(text),
                supporting_evidence: Vec::new(),
            })
            .collect();

        let total = verdicts.len();
        let supported = verdicts.iter().filter(|v| v.supported).count();

        GroundednessResult {
            verdicts,
            entailment_matrix: None,
            context_sentences: ctx
                .iter()
                .enumerate()
                .map(|(i, (t, s, e))| SentenceSpan {
                    text: t.clone(),
                    index: i,
                    start: *s,
                    end: *e,
                })
                .collect(),
            output_claims: claims
                .iter()
                .enumerate()
                .map(|(i, (t, s, e))| SentenceSpan {
                    text: t.clone(),
                    index: i,
                    start: *s,
                    end: *e,
                })
                .collect(),
            faithfulness_score: if total > 0 {
                #[allow(clippy::cast_precision_loss)]
                {
                    supported as f64 / total as f64
                }
            } else {
                1.0
            },
            total_claims: total,
            supported_claims: supported,
            contradicted_claims: 0,
            neutral_claims: total,
            partial_claims: 0,
            nli_calls: 0,
            nli_calls_saved: 0,
            nli_calls_skipped_substring: 0,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            split_ms,
            embedding_ms: 0.0,
            nli_ms: 0.0,
        }
    }

    /// Converts result to `ModuleDetail` variants for `KrinoReport` integration.
    #[must_use]
    pub fn to_module_details(result: &GroundednessResult) -> Vec<ModuleDetail> {
        result
            .verdicts
            .iter()
            .map(|v| ModuleDetail::NliClassification {
                claim: v.claim.clone(),
                label: v.label.clone(),
                confidence: match v.label.as_str() {
                    "entailment" | "partial" => v.entailment_prob,
                    "contradiction" => v.contradiction_prob,
                    "neutral" => v.neutral_prob,
                    _ => 0.0,
                },
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inference::SequenceClassifierOutput;

    /// Mock NLI backend for testing
    struct MockNliBackend {
        /// Fixed response for all inputs: (entailment, neutral, contradiction)
        fixed_probs: Vec<f64>,
    }

    impl SequenceClassifier for MockNliBackend {
        fn classify(
            &self,
            inputs: &[SequenceClassifierInput],
        ) -> Result<Vec<SequenceClassifierOutput>> {
            Ok(inputs
                .iter()
                .map(|_| {
                    let predicted_class = self
                        .fixed_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            // NaN-safe comparison
                        })
                        .map_or(0, |(idx, _)| idx);

                    SequenceClassifierOutput {
                        predicted_class,
                        predicted_label: ["entailment", "neutral", "contradiction"]
                            [predicted_class]
                            .to_string(),
                        probabilities: self.fixed_probs.clone(),
                        latency_ms: 1.0,
                    }
                })
                .collect())
        }

        fn device_info(&self) -> String {
            "MockNLI".to_string()
        }

        fn max_length(&self) -> usize {
            1024
        }

        fn label_map(&self) -> &[String] {
            // Return static reference workaround
            &[]
        }
    }

    /// Mock embedding backend for testing
    struct MockEmbedding;

    impl EmbeddingSimilarity for MockEmbedding {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            // Simple bag-of-characters embedding for testing
            // Texts with similar characters will have similar embeddings
            Ok(texts
                .iter()
                .map(|text| {
                    let mut vec = vec![0.0_f32; 26];
                    for ch in text.to_lowercase().chars() {
                        if ch.is_ascii_lowercase() {
                            vec[(ch as u8 - b'a') as usize] += 1.0;
                        }
                    }
                    // L2 normalize
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut vec {
                            *x /= norm;
                        }
                    }
                    vec
                })
                .collect())
        }

        fn embedding_dim(&self) -> usize {
            26
        }

        fn device_info(&self) -> String {
            "MockEmbedding".to_string()
        }
    }

    fn mock_tokenizer() -> tokenizers::Tokenizer {
        let wp = tokenizers::models::wordpiece::WordPiece::default();
        tokenizers::Tokenizer::new(wp)
    }

    // --- Claim splitting tests ---

    #[test]
    fn test_split_simple_sentences() {
        let claims = split_into_claims("The cat sat. The dog ran.");
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].0, "The cat sat.");
        assert_eq!(claims[1].0, "The dog ran.");
    }

    #[test]
    fn test_split_preserves_abbreviations() {
        let claims = split_into_claims("Dr. Smith went to Washington. He arrived Tuesday.");
        assert_eq!(claims.len(), 2);
        assert!(claims[0].0.contains("Dr. Smith"));
    }

    #[test]
    fn test_split_preserves_decimals() {
        let claims = split_into_claims("The rate was 3.14 percent. Revenue grew.");
        assert_eq!(claims.len(), 2);
        assert!(claims[0].0.contains("3.14"));
    }

    #[test]
    fn test_split_handles_semicolons() {
        let claims = split_into_claims("Revenue grew 15%; costs declined 3%.");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_split_trailing_text_no_punctuation() {
        let claims = split_into_claims("First sentence. Trailing text without period");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_split_empty_input() {
        let claims = split_into_claims("");
        assert!(claims.is_empty());
    }

    #[test]
    fn test_split_offset_tracking() {
        let text = "Hello world. Goodbye.";
        let claims = split_into_claims(text);
        assert_eq!(text[claims[0].1..claims[0].2].trim(), "Hello world.");
        assert_eq!(text[claims[1].1..claims[1].2].trim(), "Goodbye.");
    }

    // --- Groundedness checker tests ---

    #[test]
    fn test_all_entailed() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0, // Disable pre-filtering for deterministic tests
                ..Default::default()
            },
        );

        let result = checker
            .check(
                "France is in Europe. Paris is the capital.",
                "Paris is in France. France is European.",
            )
            .unwrap();

        assert_eq!(result.total_claims, 2);
        assert_eq!(result.supported_claims, 2);
        assert!((result.faithfulness_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contradiction_detected() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.05, 0.05, 0.9],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let result = checker
            .check("Revenue was $4.2 billion.", "Revenue declined sharply.")
            .unwrap();

        assert_eq!(result.contradicted_claims, 1);
        assert!(!result.verdicts[0].supported);
    }

    #[test]
    fn test_neutral_default_supported() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.1, 0.8, 0.1],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                treat_neutral_as_unsupported: false,
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let result = checker
            .check("Context.", "Some neutral claim here.")
            .unwrap();
        assert!(result.verdicts[0].supported); // neutral = supported by default
    }

    #[test]
    fn test_neutral_strict_mode_unsupported() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.1, 0.8, 0.1],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                treat_neutral_as_unsupported: true,
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let result = checker
            .check("Context.", "Some neutral claim here.")
            .unwrap();
        assert!(!result.verdicts[0].supported); // strict mode: neutral = unsupported
    }

    #[test]
    fn test_empty_output_vacuously_faithful() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let checker = GroundednessChecker::new(
            nli_backend,
            Arc::new(MockEmbedding),
            GroundednessConfig::default(),
        );

        let result = checker.check("Some context.", "").unwrap();
        assert_eq!(result.total_claims, 0);
        assert!((result.faithfulness_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_claim_length_filtering() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 20,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let result = checker
            .check(
                "Context here.",
                "Short. This is a much longer claim that exceeds the minimum.",
            )
            .unwrap();

        // "Short." is 6 chars < 20, should be skipped
        assert_eq!(result.total_claims, 1);
    }

    #[test]
    fn test_to_module_details() {
        let result = GroundednessResult {
            verdicts: vec![ClaimVerdict {
                claim: "Test claim.".to_string(),
                claim_idx: 0,
                span: (0, 11),
                label: "entailment".to_string(),
                entailment_prob: 0.9,
                neutral_prob: 0.05,
                contradiction_prob: 0.05,
                supported: true,
                best_evidence: None,
                strongest_contradiction: None,
                context_sentences_evaluated: 1,
                is_compound: false,
                supporting_evidence: Vec::new(),
            }],
            entailment_matrix: None,
            context_sentences: vec![],
            output_claims: vec![],
            faithfulness_score: 1.0,
            total_claims: 1,
            supported_claims: 1,
            contradicted_claims: 0,
            neutral_claims: 0,
            partial_claims: 0,
            nli_calls: 1,
            nli_calls_saved: 0,
            nli_calls_skipped_substring: 0,
            latency_ms: 10.0,
            split_ms: 0.0,
            embedding_ms: 0.0,
            nli_ms: 0.0,
        };

        let details = GroundednessChecker::to_module_details(&result);
        assert_eq!(details.len(), 1);
        match &details[0] {
            ModuleDetail::NliClassification {
                claim,
                label,
                confidence,
            } => {
                assert_eq!(claim, "Test claim.");
                assert_eq!(label, "entailment");
                assert!((confidence - 0.9).abs() < f64::EPSILON);
            }
            _ => panic!("Expected NliClassification"),
        }
    }

    // --- Determinism test ---

    #[test]
    fn test_determinism_fast_path_and_adaptive_top_k() {
        // Exercises the substring fast-path AND adaptive top-K together:
        // - First claim is a verbatim substring of context -> fast-path
        // - Second claim is paraphrased and longer -> falls through to NLI,
        //   with adaptive top-K shrinking its candidate list
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.85, 0.10, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 5,
                min_similarity_threshold: 0.0,
                adaptive_top_k: true,
                ..Default::default()
            },
        );

        let context = "Paris is the capital of France. The Eiffel Tower opened in 1889. \
             Mount Everest is the tallest mountain. The Pacific Ocean is the largest body of water.";
        let output = "Paris is the capital of France. \
             The summit of Everest was first reached in 1953.";

        let results: Vec<_> = (0..100)
            .map(|_| checker.check(context, output).unwrap())
            .collect();

        let first = &results[0];
        assert_eq!(first.total_claims, 2);
        assert!(
            first.nli_calls_skipped_substring >= 1,
            "fast-path branch must fire for this fixture"
        );
        assert!(first.nli_calls >= 1, "NLI branch must also fire");

        for result in &results[1..] {
            assert_eq!(result.total_claims, first.total_claims);
            assert_eq!(result.nli_calls, first.nli_calls, "nli_calls drifted");
            assert_eq!(
                result.nli_calls_skipped_substring, first.nli_calls_skipped_substring,
                "fast-path hit count drifted"
            );
            assert!(
                (result.faithfulness_score - first.faithfulness_score).abs() < f64::EPSILON,
                "faithfulness_score drifted"
            );

            for (v1, v2) in result.verdicts.iter().zip(first.verdicts.iter()) {
                assert_eq!(v1.claim_idx, v2.claim_idx, "verdict ordering drifted");
                assert_eq!(v1.claim, v2.claim);
                assert_eq!(v1.label, v2.label);
                assert!((v1.entailment_prob - v2.entailment_prob).abs() < f64::EPSILON);
                assert!((v1.neutral_prob - v2.neutral_prob).abs() < f64::EPSILON);
                assert!((v1.contradiction_prob - v2.contradiction_prob).abs() < f64::EPSILON);

                let (e1, e2) = (v1.best_evidence.as_ref(), v2.best_evidence.as_ref());
                match (e1, e2) {
                    (Some(a), Some(b)) => {
                        assert_eq!(a.sentence_idx, b.sentence_idx, "evidence drifted");
                    }
                    (None, None) => {}
                    _ => panic!("best_evidence presence differs across runs"),
                }
            }
        }
    }

    #[test]
    fn test_determinism() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.85, 0.10, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "The company reported revenue of $4.2 billion.";
        let output = "Revenue was $4.2 billion. Growth was strong.";

        // Run 100 iterations to verify determinism
        let results: Vec<_> = (0..100)
            .map(|_| checker.check(context, output).unwrap())
            .collect();

        // All results should be identical
        let first = &results[0];
        for result in &results[1..] {
            assert_eq!(
                result.total_claims, first.total_claims,
                "total_claims differs across runs"
            );
            assert!(
                (result.faithfulness_score - first.faithfulness_score).abs() < f64::EPSILON,
                "faithfulness_score differs across runs"
            );

            for (v1, v2) in result.verdicts.iter().zip(first.verdicts.iter()) {
                assert_eq!(v1.claim, v2.claim, "claim text differs");
                assert_eq!(v1.label, v2.label, "label differs");
                assert!(
                    (v1.entailment_prob - v2.entailment_prob).abs() < f64::EPSILON,
                    "entailment_prob differs"
                );
                assert!(
                    (v1.neutral_prob - v2.neutral_prob).abs() < f64::EPSILON,
                    "neutral_prob differs"
                );
                assert!(
                    (v1.contradiction_prob - v2.contradiction_prob).abs() < f64::EPSILON,
                    "contradiction_prob differs"
                );
            }
        }
    }

    // --- v2 specific tests ---

    // --- Substring fast-path tests ---

    #[test]
    fn test_normalize_lowercases_and_trims_trailing_punct() {
        assert_eq!(normalize_for_substring("Revenue Grew."), "revenue grew");
        assert_eq!(normalize_for_substring("Revenue grew!"), "revenue grew");
        assert_eq!(normalize_for_substring("Revenue grew?"), "revenue grew");
        assert_eq!(normalize_for_substring("\"Quoted claim.\""), "quoted claim");
    }

    #[test]
    fn test_normalize_collapses_internal_whitespace() {
        assert_eq!(
            normalize_for_substring("Revenue   grew\t  15%"),
            "revenue grew 15%"
        );
    }

    #[test]
    fn test_normalize_preserves_internal_punctuation() {
        // Decimals, currency, and percent signs must survive normalization
        // so "$4.2 billion" can still substring-match its context.
        assert_eq!(
            normalize_for_substring("Revenue was $4.2 billion."),
            "revenue was $4.2 billion"
        );
    }

    #[test]
    fn test_find_substring_match_hits_when_claim_in_context() {
        let context = vec![
            normalize_for_substring("Paris is the capital of France."),
            normalize_for_substring("Tokyo is a different city entirely."),
        ];
        let claim = normalize_for_substring("Paris is the capital of France.");
        assert_eq!(find_substring_match(&claim, &context), Some(0));
    }

    #[test]
    fn test_find_substring_match_misses_short_claim() {
        // Claims under MIN_LEN are never matched — too many false positives.
        let context = vec![normalize_for_substring("The cat sat on the mat.")];
        let claim = normalize_for_substring("The cat.");
        assert_eq!(find_substring_match(&claim, &context), None);
    }

    #[test]
    fn test_find_substring_match_misses_when_absent() {
        let context = vec![
            normalize_for_substring("Paris is the capital of France."),
            normalize_for_substring("The Eiffel Tower opened in 1889."),
        ];
        let claim = normalize_for_substring("Berlin is the capital of Germany.");
        assert_eq!(find_substring_match(&claim, &context), None);
    }

    #[test]
    fn test_find_substring_match_finds_first_when_duplicates() {
        // Deterministic: first-seen wins.
        let context = vec![
            normalize_for_substring("Revenue was $4.2 billion last quarter."),
            normalize_for_substring("Revenue was $4.2 billion."),
        ];
        let claim = normalize_for_substring("Revenue was $4.2 billion.");
        assert_eq!(find_substring_match(&claim, &context), Some(0));
    }

    #[test]
    fn test_fast_path_skips_nli_call() {
        // If every claim is a verbatim substring of context, no NLI call
        // should fire and the verdict should be entailment with prob 1.0.
        let nli_backend = Arc::new(MockNliBackend {
            // If the fast-path leaked, the mock would force contradiction,
            // making this test fail loudly.
            fixed_probs: vec![0.05, 0.05, 0.9],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "Paris is the capital of France. The Eiffel Tower opened in 1889.";
        let output = "Paris is the capital of France.";

        let result = checker.check(context, output).unwrap();

        assert_eq!(result.total_claims, 1);
        assert_eq!(result.nli_calls, 0, "fast-path should skip NLI entirely");
        assert_eq!(result.nli_calls_skipped_substring, 1);
        assert!(result.verdicts[0].supported);
        assert_eq!(result.verdicts[0].label, "entailment");
        assert!((result.verdicts[0].entailment_prob - 1.0).abs() < f64::EPSILON);
        let evidence = result.verdicts[0]
            .best_evidence
            .as_ref()
            .expect("fast-path verdict must carry evidence link");
        assert_eq!(evidence.sentence_idx, 0);
    }

    #[test]
    fn test_fast_path_does_not_match_paraphrase() {
        // A paraphrase is NOT a substring, so the fast-path must skip and
        // NLI runs normally. With contradiction-leaning mock, we expect
        // an NLI call to fire and the claim to be flagged.
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.05, 0.05, 0.9],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "Paris is the capital of France.";
        let output = "The capital of France is Paris.";

        let result = checker.check(context, output).unwrap();

        assert_eq!(result.nli_calls_skipped_substring, 0);
        assert!(result.nli_calls >= 1);
    }

    #[test]
    fn test_fast_path_mixed_with_nli_path() {
        // One claim hits the fast-path, one doesn't. Both must end up in
        // the verdict list with the right labels, ordering preserved.
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "Paris is the capital of France. Mount Everest is the tallest mountain.";
        let output = "Paris is the capital of France. Climbing is a popular activity.";

        let result = checker.check(context, output).unwrap();

        assert_eq!(result.total_claims, 2);
        assert_eq!(result.nli_calls_skipped_substring, 1);
        assert!(result.nli_calls >= 1);
        // Verdicts ordered by claim_idx (the input order in the output text).
        assert_eq!(result.verdicts[0].claim_idx, 0);
        assert_eq!(result.verdicts[1].claim_idx, 1);
        // First was the substring match; check it carries entailment_prob = 1.0.
        assert!((result.verdicts[0].entailment_prob - 1.0).abs() < f64::EPSILON);
    }

    // --- Adaptive top-K tests ---

    #[test]
    fn test_effective_top_k_off_returns_top_k() {
        assert_eq!(effective_top_k(5, 10, false), 10);
        assert_eq!(effective_top_k(500, 10, false), 10);
    }

    #[test]
    fn test_effective_top_k_zero_top_k() {
        assert_eq!(effective_top_k(50, 0, true), 0);
        assert_eq!(effective_top_k(500, 0, true), 0);
    }

    #[test]
    fn test_effective_top_k_short_claim_floors_at_3() {
        // 30 chars -> ceil(30/40) = 1 -> max(1, 3) = 3
        assert_eq!(effective_top_k(30, 10, true), 3);
        // 1 char  -> ceil(1/40)  = 1 -> max(1, 3) = 3
        assert_eq!(effective_top_k(1, 10, true), 3);
    }

    #[test]
    fn test_effective_top_k_medium_claim_scales_linear() {
        // 80 chars  -> ceil(80/40) = 2 -> max(2, 3) = 3
        assert_eq!(effective_top_k(80, 10, true), 3);
        // 120 chars -> ceil(120/40) = 3 -> max(3, 3) = 3
        assert_eq!(effective_top_k(120, 10, true), 3);
        // 160 chars -> ceil(160/40) = 4 -> max(4, 3) = 4
        assert_eq!(effective_top_k(160, 10, true), 4);
        // 200 chars -> ceil(200/40) = 5 -> max(5, 3) = 5
        assert_eq!(effective_top_k(200, 10, true), 5);
    }

    #[test]
    fn test_effective_top_k_long_claim_caps_at_top_k() {
        // 500 chars -> ceil(500/40) = 13 -> min(13, 10) = 10
        assert_eq!(effective_top_k(500, 10, true), 10);
        assert_eq!(effective_top_k(10_000, 10, true), 10);
    }

    #[test]
    fn test_is_compound_claim_simple() {
        assert!(!is_compound_claim("Revenue grew 15%."));
        assert!(!is_compound_claim("The company is doing well."));
    }

    #[test]
    fn test_is_compound_claim_with_conjunction() {
        assert!(is_compound_claim("Revenue grew 15%, and margins expanded."));
        assert!(is_compound_claim(
            "The CEO spoke, while investors listened."
        ));
    }

    #[test]
    fn test_is_compound_claim_with_relative_clause() {
        assert!(is_compound_claim(
            "The company, which was founded in 1987, reported earnings."
        ));
        assert!(is_compound_claim(
            "The CEO, who joined last year, announced changes."
        ));
    }

    #[test]
    fn test_is_compound_claim_with_semicolon() {
        assert!(is_compound_claim("Revenue grew; costs declined."));
    }

    fn default_partial_thresholds() -> PartialThresholds {
        PartialThresholds {
            entailment_floor: 0.2,
            neutral_ceiling: 0.65,
            similarity_floor: 0.7,
        }
    }

    #[test]
    fn test_collect_partial_evidence_picks_distinct_sentences_passing_rule() {
        let context = vec![
            ("Alpha sentence.".to_string(), 0, 15),
            ("Beta sentence.".to_string(), 16, 30),
            ("Gamma sentence.".to_string(), 31, 46),
        ];
        // ctx_idx → (entailment, neutral, contradiction), similarity
        let evidence = vec![
            // Alpha: passes (e > c, n < 0.65, sim >= 0.7)
            (0, vec![0.6, 0.3, 0.1], Some(0.8)),
            // Beta: passes (e=0.27 > c=0.13, n=0.6 < 0.65, sim=0.75 >= 0.7)
            (1, vec![0.27, 0.6, 0.13], Some(0.75)),
            // Gamma: fails neutral ceiling (n=0.7 > 0.65)
            (2, vec![0.2, 0.7, 0.1], Some(0.8)),
        ];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert_eq!(
            out.len(),
            2,
            "Alpha and Beta pass, Gamma fails neutral ceiling"
        );
        assert_eq!(out[0].sentence_idx, 0, "highest entailment first");
        assert_eq!(out[1].sentence_idx, 1);
    }

    #[test]
    fn test_collect_partial_evidence_dedups_repeated_sentences() {
        let context = vec![("Alpha sentence.".to_string(), 0, 15)];
        // Same ctx_idx appears twice — should be deduped to one entry.
        let evidence = vec![
            (0, vec![0.3, 0.6, 0.1], Some(0.8)),
            (0, vec![0.6, 0.3, 0.1], Some(0.8)),
        ];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert_eq!(out.len(), 1);
        assert!(
            (out[0].entailment_prob - 0.6).abs() < 1e-9,
            "keeps the strongest entailment for the deduped sentence"
        );
    }

    #[test]
    fn test_collect_partial_evidence_rejects_when_entailment_below_contradiction() {
        let context = vec![("Alpha.".to_string(), 0, 6)];
        // Contradiction beats entailment → must be rejected even if e clears floor.
        let evidence = vec![(0, vec![0.25, 0.45, 0.3], Some(0.8))];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert!(out.is_empty(), "e <= c must fail the rule");
    }

    #[test]
    fn test_collect_partial_evidence_rejects_when_neutral_above_ceiling() {
        let context = vec![("Alpha.".to_string(), 0, 6)];
        // High neutral = model is *disinterestedly* uncertain → off-topic noise.
        let evidence = vec![(0, vec![0.25, 0.7, 0.05], Some(0.8))];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert!(out.is_empty(), "n > neutral_ceiling must fail the rule");
    }

    #[test]
    fn test_collect_partial_evidence_rejects_when_similarity_below_floor() {
        let context = vec![("Alpha.".to_string(), 0, 6)];
        // Pair would pass on NLI metrics alone, but topical alignment is too weak.
        let evidence = vec![(0, vec![0.5, 0.4, 0.1], Some(0.4))];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert!(out.is_empty(), "sim < similarity_floor must fail the rule");
    }

    #[test]
    fn test_collect_partial_evidence_allows_missing_similarity() {
        let context = vec![("Alpha.".to_string(), 0, 6)];
        // None similarity → pre-filter disabled; rule defers to NLI conditions only.
        let evidence = vec![(0, vec![0.5, 0.4, 0.1], None)];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert_eq!(out.len(), 1, "None similarity is not a failure");
    }

    #[test]
    fn test_collect_partial_evidence_rejects_below_entailment_floor() {
        let context = vec![("Alpha.".to_string(), 0, 6)];
        // All probs near-zero shouldn't qualify — entailment floor is the guard.
        let evidence = vec![(0, vec![0.05, 0.5, 0.04], Some(0.9))];

        let out = collect_partial_evidence(&evidence, &context, default_partial_thresholds());

        assert!(out.is_empty(), "e < entailment_floor must fail the rule");
    }

    #[test]
    fn test_partial_verdict_promoted_for_compound_with_multi_evidence() {
        // Mock backend that returns (entailment=0.45, neutral=0.5, contradiction=0.05)
        // for every pair. Single-best verdict is "neutral" (0.5 > 0.45) but
        // each pair clears partial_threshold=0.4, so a compound claim with
        // ≥2 candidate context sentences should be promoted to "partial".
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.45, 0.5, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0, // evaluate all context sentences
                flag_compound_claims: true,
                ..Default::default()
            },
        );

        let context = "Rust ships in browsers. Rust ships in operating systems.";
        // Compound claim (em-dash trips is_compound_claim).
        let output = "Rust is used in browsers — and in operating systems.";

        let result = checker.check(context, output).unwrap();
        assert_eq!(result.verdicts.len(), 1);
        let v = &result.verdicts[0];

        assert!(v.is_compound, "claim should be flagged compound");
        assert_eq!(v.label, "partial");
        assert!(v.supported);
        assert_eq!(
            v.supporting_evidence.len(),
            2,
            "two distinct supporting sentences"
        );
        // Aggregate entailment is the mean of contributing pairs (each 0.45).
        assert!((v.entailment_prob - 0.45).abs() < 1e-6);
        assert_eq!(result.partial_claims, 1);
        assert_eq!(result.supported_claims, 1);
    }

    #[test]
    fn test_partial_verdict_not_applied_to_non_compound_claims() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.45, 0.5, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                flag_compound_claims: true,
                ..Default::default()
            },
        );

        let context = "Rust ships in browsers. Rust ships in operating systems.";
        // Simple (non-compound) claim — partial path must NOT fire.
        let output = "Rust is fast.";

        let result = checker.check(context, output).unwrap();
        let v = &result.verdicts[0];

        assert!(!v.is_compound);
        assert_eq!(v.label, "neutral");
        assert!(v.supporting_evidence.is_empty());
        assert_eq!(result.partial_claims, 0);
    }

    #[test]
    fn test_partial_verdict_not_applied_when_only_one_evidence_clears_threshold() {
        // Single context sentence → only one candidate possible, so even a
        // compound claim should fall back to neutral (no multi-evidence).
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.45, 0.5, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                flag_compound_claims: true,
                ..Default::default()
            },
        );

        let context = "Rust ships in browsers.";
        let output = "Rust is used in browsers — and in operating systems.";

        let result = checker.check(context, output).unwrap();
        let v = &result.verdicts[0];

        assert!(v.is_compound);
        assert_eq!(v.label, "neutral", "one evidence is not enough for partial");
        assert!(v.supporting_evidence.is_empty());
    }

    #[test]
    fn test_partial_verdict_not_applied_when_contradiction_dominates() {
        // Even if multiple sentences clear partial_threshold on entailment,
        // a contradiction over contradiction_threshold should win the verdict.
        let nli_backend = Arc::new(MockNliBackend {
            // Contradiction dominates — but entailment also clears 0.4.
            fixed_probs: vec![0.4, 0.0, 0.6],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                flag_compound_claims: true,
                partial_threshold: 0.4,
                contradiction_threshold: 0.5,
                ..Default::default()
            },
        );

        let context = "Rust ships in browsers. Rust ships in operating systems.";
        let output = "Rust is used in browsers — and in operating systems.";

        let result = checker.check(context, output).unwrap();
        let v = &result.verdicts[0];

        assert_eq!(v.label, "contradiction");
        assert!(v.supporting_evidence.is_empty());
    }

    #[test]
    fn test_evidence_tracing() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "The company reported revenue of $4.2 billion. Profit margin was 18%.";
        let output = "Revenue was $4.2 billion.";

        let result = checker.check(context, output).unwrap();

        assert_eq!(result.verdicts.len(), 1);
        assert!(result.verdicts[0].best_evidence.is_some());

        let evidence = result.verdicts[0].best_evidence.as_ref().unwrap();
        assert!(evidence.sentence.contains("$4.2 billion"));
        assert_eq!(evidence.sentence_idx, 0);
    }

    #[test]
    fn test_entailment_matrix_populated() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                include_entailment_matrix: true,
                ..Default::default()
            },
        );

        let context = "Revenue was $4.2 billion.";
        let output = "Revenue was high.";

        let result = checker.check(context, output).unwrap();

        assert!(result.entailment_matrix.is_some());
        let matrix = result.entailment_matrix.unwrap();
        assert!(!matrix.is_empty());
        assert_eq!(matrix[0].claim_idx, 0);
        assert_eq!(matrix[0].context_idx, 0);
    }

    #[test]
    fn test_check_with_overrides_can_enable_matrix() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                include_entailment_matrix: false,
                ..Default::default()
            },
        );

        let context = "Revenue was $4.2 billion.";
        let output = "Revenue was high.";

        let result = checker
            .check_with_overrides(
                context,
                output,
                RequestOverrides {
                    include_matrix: Some(true),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(
            result.entailment_matrix.is_some(),
            "override should enable matrix even when config has it off"
        );
    }

    #[test]
    fn test_check_with_overrides_can_disable_matrix() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                include_entailment_matrix: true,
                ..Default::default()
            },
        );

        let context = "Revenue was $4.2 billion.";
        let output = "Revenue was high.";

        let result = checker
            .check_with_overrides(
                context,
                output,
                RequestOverrides {
                    include_matrix: Some(false),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(
            result.entailment_matrix.is_none(),
            "override should disable matrix even when config has it on"
        );
    }

    #[test]
    fn test_check_with_overrides_top_k_zero_disables_prefilter() {
        // Config has top_k_context = 2 (pre-filter cuts to 2 sentences), but
        // an override of Some(0) should evaluate ALL context sentences. With
        // 3 context sentences and matrix on, we should see all 3 cells.
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 2,
                include_entailment_matrix: false,
                ..Default::default()
            },
        );

        let context = "Alpha sentence. Beta sentence. Gamma sentence.";
        let output = "Some claim.";

        let result = checker
            .check_with_overrides(
                context,
                output,
                RequestOverrides {
                    include_matrix: Some(true),
                    top_k_context: Some(0),
                },
            )
            .unwrap();

        let matrix = result
            .entailment_matrix
            .expect("matrix override should be active");
        // All 3 context sentences × 1 claim = 3 cells when pre-filter is off.
        assert_eq!(matrix.len(), 3, "top_k=0 must evaluate every context pair");
        // similarity_score is None when pre-filter is off (engine signal).
        assert!(matrix.iter().all(|c| c.similarity_score.is_none()));
    }

    #[test]
    fn test_check_with_overrides_top_k_some_uses_override_value() {
        // Config disables pre-filter (top_k_context = 0), but override sets
        // Some(2) — we should see exactly 2 cells in the matrix per claim.
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                adaptive_top_k: false,
                include_entailment_matrix: false,
                min_similarity_threshold: 0.0,
                ..Default::default()
            },
        );

        let context = "Alpha sentence. Beta sentence. Gamma sentence.";
        let output = "Some claim.";

        let result = checker
            .check_with_overrides(
                context,
                output,
                RequestOverrides {
                    include_matrix: Some(true),
                    top_k_context: Some(2),
                },
            )
            .unwrap();

        let matrix = result
            .entailment_matrix
            .expect("matrix override should be active");
        assert_eq!(matrix.len(), 2, "top_k=2 must keep only 2 candidates");
        assert!(matrix.iter().all(|c| c.similarity_score.is_some()));
    }

    #[test]
    fn test_entailment_matrix_not_populated_when_disabled() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                include_entailment_matrix: false,
                ..Default::default()
            },
        );

        let context = "Revenue was $4.2 billion.";
        let output = "Revenue was high.";

        let result = checker.check(context, output).unwrap();

        assert!(result.entailment_matrix.is_none());
    }

    #[test]
    fn test_prefiltering_reduces_nli_calls() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);

        // With pre-filtering (top-K = 2)
        let checker_filtered = GroundednessChecker::new(
            nli_backend.clone(),
            embedding_backend.clone(),
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 2,
                ..Default::default()
            },
        );

        // Without pre-filtering
        let checker_unfiltered = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.";
        let output = "Claim here.";

        let result_filtered = checker_filtered.check(context, output).unwrap();
        let result_unfiltered = checker_unfiltered.check(context, output).unwrap();

        // With top-K=2, should only make 2 NLI calls (1 claim × 2 context sentences)
        assert_eq!(result_filtered.nli_calls, 2);

        // Without filtering, should make 5 NLI calls (1 claim × 5 context sentences)
        assert_eq!(result_unfiltered.nli_calls, 5);

        // Should have saved 3 calls
        assert_eq!(result_filtered.nli_calls_saved, 3);
    }

    #[test]
    fn test_compound_claim_flagging() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                flag_compound_claims: true,
                ..Default::default()
            },
        );

        let context = "Context here.";
        let output = "Revenue grew, and margins expanded."; // Compound claim

        let result = checker.check(context, output).unwrap();

        assert_eq!(result.verdicts.len(), 1);
        assert!(result.verdicts[0].is_compound);
    }

    #[test]
    fn test_context_sentences_and_output_claims_tracked() {
        let nli_backend = Arc::new(MockNliBackend {
            fixed_probs: vec![0.9, 0.05, 0.05],
        });
        let embedding_backend = Arc::new(MockEmbedding);
        let checker = GroundednessChecker::new(
            nli_backend,
            embedding_backend,
            GroundednessConfig {
                min_claim_length: 3,
                top_k_context: 0,
                ..Default::default()
            },
        );

        let context = "Sentence one. Sentence two.";
        let output = "Claim one. Claim two.";

        let result = checker.check(context, output).unwrap();

        assert_eq!(result.context_sentences.len(), 2);
        assert_eq!(result.output_claims.len(), 2);
        assert_eq!(result.context_sentences[0].index, 0);
        assert_eq!(result.context_sentences[1].index, 1);
    }
}
