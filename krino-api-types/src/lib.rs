//! Shared wire types for the Krino HTTP API.
//!
//! Both the server (`krino-api`) and any Rust client use these types so the
//! request/response format cannot drift between the two.

use serde::{Deserialize, Serialize};

/// A single chunk of context supplied by the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    /// Caller-assigned identifier, echoed back in evidence links.
    #[serde(default)]
    pub id: Option<String>,
    pub text: String,
}

/// Per-request configuration overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestConfig {
    /// `"claim"` (default) or `"token"`.
    #[serde(default)]
    pub granularity: Option<String>,
    /// Minimum faithfulness score to pass. Default: 0.7.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Count NEUTRAL verdicts as unsupported. Default: false.
    #[serde(default)]
    pub strict: Option<bool>,
    /// Include the full entailment matrix. Default: false.
    #[serde(default)]
    pub include_matrix: Option<bool>,
    /// Per-request override for embedding pre-filter top-K. `Some(0)` disables
    /// pre-filtering and evaluates every (claim, context_sentence) pair —
    /// useful for audit probes that need the full matrix. `None` keeps the
    /// server's configured default. Other engine knobs (similarity floor,
    /// adaptive top-K, thresholds) stay startup-time only; changing them
    /// per-request has correctness implications that warrant a wider design pass.
    #[serde(default)]
    pub top_k_context: Option<usize>,
}

/// POST /api/v1/evaluate — request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluateRequest {
    /// Source documents to verify against. Required, non-empty.
    pub context: Vec<ContextChunk>,
    /// LLM output to verify.
    pub output: String,
    #[serde(default)]
    pub config: Option<RequestConfig>,
}

/// A grounding evidence link from the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceResponse {
    /// Chunk ID from the input, if the caller supplied one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<String>,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entailment_prob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contradiction_prob: Option<f64>,
    /// Cosine similarity score from the embedding pre-filter (if used).
    /// Surfaces *why* this sentence was chosen as candidate evidence —
    /// a high similarity with a neutral NLI verdict suggests the
    /// pre-filter ranked correctly but the NLI model couldn't decide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f64>,
}

/// Per-claim verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResponse {
    pub text: String,
    /// `"entailment"` | `"contradiction"` | `"neutral"` | `"partial"`.
    ///
    /// `"partial"` is set on a compound claim where ≥2 distinct context
    /// sentences each cleared the engine's `partial_threshold`. The headline
    /// `score` is the mean of those entailments, `supported` is `true`, and
    /// `supporting_evidence` lists the contributing sentences in
    /// descending-entailment order.
    pub verdict: String,
    /// Entailment probability [0.0, 1.0].
    pub score: f64,
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceResponse>,
    pub is_compound: bool,
    /// Context sentences that jointly support a `"partial"` verdict. Empty
    /// (skipped on the wire) for all other verdicts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supporting_evidence: Vec<EvidenceResponse>,
}

/// One cell in the per-request entailment matrix.
///
/// Mirrors `krino::modules::groundedness::EntailmentCell` on the wire. Only
/// populated when the request sets `config.include_matrix = true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntailmentMatrixCell {
    pub claim_idx: usize,
    pub context_idx: usize,
    pub context_sentence: String,
    pub entailment_prob: f64,
    pub neutral_prob: f64,
    pub contradiction_prob: f64,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f64>,
}

/// Per-token span (granularity = "token", not yet implemented).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanResponse {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
}

/// A single faithfulness issue surfaced by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueResponse {
    pub text: String,
    /// `"contradiction"` | `"unsupported"` | `"unfaithful_span"`
    pub issue_type: String,
    /// `"high"` | `"medium"` | `"low"`
    pub severity: String,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceResponse>,
}

/// Evaluation metadata included in every response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaResponse {
    pub granularity: String,
    pub model: String,
    pub latency_ms: f64,
    /// Number of NLI forward passes. Aliased to `inference_calls` on the
    /// wire for backward compatibility with pre-0.10 clients.
    #[serde(rename = "inference_calls", alias = "nli_calls")]
    pub nli_calls: usize,
    /// Fraction of claims where the engine produced a decisive verdict
    /// (entailment or contradiction), in `[0.0, 1.0]`. A low value means
    /// the model returned mostly neutral, so the faithfulness score should
    /// be interpreted with caution. Optional for pre-0.10 wire compat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_confidence: Option<f64>,
    /// Time spent splitting input into sentences/claims (ms). Optional
    /// for pre-0.10 wire compat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_ms: Option<f64>,
    /// Time spent computing embeddings for pre-filtering (ms). Zero / absent
    /// when pre-filtering is disabled or every claim hits the fast-path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_ms: Option<f64>,
    /// Time spent in NLI batch inference (ms). Zero / absent on the
    /// all-fast-path / no-context paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nli_ms: Option<f64>,
    pub engine_version: String,
}

/// POST /api/v1/evaluate — response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluateResponse {
    /// Overall faithfulness score [0.0, 1.0]. Serialized as `score` on the
    /// wire for backward compatibility with pre-0.10 clients; deserializers
    /// also accept `faithfulness_score`.
    #[serde(rename = "score", alias = "faithfulness_score")]
    pub faithfulness_score: f64,
    /// Whether the output passes the configured threshold.
    pub pass: bool,
    /// Issues ordered by severity.
    pub issues: Vec<IssueResponse>,
    /// Per-claim verdicts (granularity = "claim").
    pub claims: Vec<ClaimResponse>,
    /// Per-token spans (granularity = "token").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spans: Option<Vec<SpanResponse>>,
    /// Full entailment matrix (only present when `RequestConfig.include_matrix = true`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entailment_matrix: Option<Vec<EntailmentMatrixCell>>,
    pub meta: MetaResponse,
}
