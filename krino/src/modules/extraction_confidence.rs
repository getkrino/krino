//! Confidence scoring for LLM-extracted structured fields.
//!
//! When an LLM extracts `{field: value}` pairs from a source document, this
//! module scores how well each extracted value is actually grounded in the
//! source — instead of trusting the LLM's own self-reported confidence,
//! which is known to be poorly calibrated.
//!
//! Each field is turned into a synthetic claim sentence and run through the
//! same [`GroundednessChecker`] pipeline already used for summary
//! faithfulness. This module is pure orchestration: no new model, no new
//! inference path, no new thresholds to calibrate from scratch.

use crate::error::Result;
use crate::modules::groundedness::{EvidenceLink, GroundednessChecker, RequestOverrides};
use serde::{Deserialize, Serialize};

/// Configuration for extraction confidence scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfidenceConfig {
    /// Minimum confidence for a field to be considered `Grounded` rather
    /// than `PartiallyGrounded`. Default: 0.7, mirrors
    /// `GroundednessConfig::contradiction_threshold`'s role as the one
    /// caller-visible decision boundary.
    pub min_confidence_threshold: f64,
}

impl Default for ExtractionConfidenceConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
        }
    }
}

/// The type of an extracted field's value. Drives how the synthetic claim
/// sentence is rendered; does not change the scoring mechanism itself in
/// v1 (string and number both flow through the same NLI/substring path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Number,
    Boolean,
}

/// A single field an LLM claims to have extracted from a source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedField {
    /// Field name, e.g. `"invoice_total"`.
    pub name: String,
    /// Extracted value, rendered as text for the synthetic claim.
    pub value: String,
    pub value_type: FieldType,
}

/// How a field's confidence was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    /// The value (or its containing claim sentence) appeared verbatim in
    /// the source — the groundedness engine's substring fast-path fired.
    ExactSubstring,
    /// No verbatim match; confidence came from NLI entailment probability.
    NliEntailment,
    /// No context sentence supported the field at all.
    NoMatch,
}

/// Verdict for a single extracted field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldVerdict {
    /// Confidence at or above `min_confidence_threshold`.
    Grounded,
    /// Some support found, but below `min_confidence_threshold`.
    PartiallyGrounded,
    /// Support was found but the engine's verdict was a contradiction —
    /// the source appears to state something else for this field.
    Ungrounded,
    /// No matching evidence anywhere in the source. The likely-hallucinated
    /// case: the LLM extracted a value that isn't in the document at all.
    NotFound,
}

/// Confidence result for a single extracted field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfidence {
    pub name: String,
    pub value: String,
    /// Composite confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    pub verdict: FieldVerdict,
    pub match_kind: MatchKind,
    /// The source sentence that best supports (or contradicts) this field,
    /// if any was found.
    pub evidence: Option<EvidenceLink>,
}

/// Full result from extraction confidence scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfidenceResult {
    pub fields: Vec<FieldConfidence>,
    /// Mean confidence across all fields. `1.0` when `fields` is empty.
    pub overall_confidence: f64,
    /// Fraction of fields with a decisive verdict (anything except
    /// `PartiallyGrounded` from a neutral NLI read) — mirrors
    /// `GroundednessResult`'s `engine_confidence` concept: low values mean
    /// the headline score should be read with caution.
    pub engine_confidence: f64,
}

/// Builds the synthetic claim sentence for a field.
///
/// Deterministic string formatting — same input always produces the same
/// sentence. Kept simple on purpose: `"{name} is {value}"` reads naturally
/// for most field names (`"invoice_total is 1,204.50"`, `"vendor is Acme
/// Corp"`) and is what the underlying NLI model was trained to score well
/// against source prose.
fn synthetic_claim(field: &ExtractedField) -> String {
    let readable_name = field.name.replace(['_', '-'], " ");
    match field.value_type {
        FieldType::Boolean => {
            if field.value.eq_ignore_ascii_case("true") {
                format!("{readable_name} is true.")
            } else {
                format!("{readable_name} is false.")
            }
        }
        FieldType::String | FieldType::Number => {
            format!("{readable_name} is {}.", field.value)
        }
    }
}

/// Scores confidence for a set of extracted fields against their source
/// document by composing the existing [`GroundednessChecker`] pipeline.
pub struct ExtractionConfidenceChecker {
    groundedness: GroundednessChecker,
    config: ExtractionConfidenceConfig,
}

impl ExtractionConfidenceChecker {
    #[must_use]
    pub fn new(groundedness: GroundednessChecker, config: ExtractionConfidenceConfig) -> Self {
        Self {
            groundedness,
            config,
        }
    }

    /// Scores each field in `fields` against `source`.
    ///
    /// One `GroundednessChecker::check_with_overrides` call per field: the
    /// field's synthetic claim is the "output", `source` is the context.
    /// This mirrors exactly how a single-sentence groundedness check would
    /// score a claim, since a synthetic claim is by construction one
    /// sentence (no compound-claim aggregation needed for a single fact).
    pub fn check(
        &self,
        source: &str,
        fields: &[ExtractedField],
    ) -> Result<ExtractionConfidenceResult> {
        let mut scored_fields = Vec::with_capacity(fields.len());

        for field in fields {
            let claim_text = synthetic_claim(field);
            let result = self.groundedness.check_with_overrides(
                source,
                &claim_text,
                RequestOverrides::default(),
            )?;

            scored_fields.push(self.score_field(field, &result));
        }

        #[allow(clippy::cast_precision_loss)]
        let overall_confidence = if scored_fields.is_empty() {
            1.0
        } else {
            let sum: f64 = scored_fields.iter().map(|f| f.confidence).sum();
            sum / scored_fields.len() as f64
        };

        #[allow(clippy::cast_precision_loss)]
        let engine_confidence = if scored_fields.is_empty() {
            1.0
        } else {
            let decisive = scored_fields
                .iter()
                .filter(|f| f.match_kind != MatchKind::NoMatch)
                .count();
            decisive as f64 / scored_fields.len() as f64
        };

        Ok(ExtractionConfidenceResult {
            fields: scored_fields,
            overall_confidence,
            engine_confidence,
        })
    }

    /// Converts a single field's synthetic-claim groundedness result into a
    /// `FieldConfidence`. The synthetic claim always produces exactly zero
    /// or one verdict (it's one sentence), so this takes the first verdict
    /// if present.
    fn score_field(
        &self,
        field: &ExtractedField,
        result: &crate::modules::groundedness::GroundednessResult,
    ) -> FieldConfidence {
        let Some(verdict) = result.verdicts.first() else {
            // Below min_claim_length or otherwise unevaluated — treat as
            // not found rather than silently defaulting to a score.
            return FieldConfidence {
                name: field.name.clone(),
                value: field.value.clone(),
                confidence: 0.0,
                verdict: FieldVerdict::NotFound,
                match_kind: MatchKind::NoMatch,
                evidence: None,
            };
        };

        let evidence = verdict
            .best_evidence
            .clone()
            .or_else(|| verdict.strongest_contradiction.clone());

        let match_kind = match evidence.as_ref() {
            // Substring fast-path always reports entailment_prob == 1.0
            // with no similarity score attached (see groundedness.rs).
            Some(e)
                if verdict.label == "entailment"
                    && (e.entailment_prob - 1.0).abs() < f64::EPSILON
                    && e.similarity_score.is_none() =>
            {
                MatchKind::ExactSubstring
            }
            Some(_) => MatchKind::NliEntailment,
            None => MatchKind::NoMatch,
        };

        let (confidence, field_verdict) = match verdict.label.as_str() {
            "contradiction" => (0.0, FieldVerdict::Ungrounded),
            "entailment" | "partial" => {
                let c = verdict.entailment_prob;
                let v = if c >= self.config.min_confidence_threshold {
                    FieldVerdict::Grounded
                } else {
                    FieldVerdict::PartiallyGrounded
                };
                (c, v)
            }
            // "neutral" with no context at all (no_context_result path)
            // and "neutral" from an inconclusive NLI read both land here.
            _ => {
                if evidence.is_none() {
                    (0.0, FieldVerdict::NotFound)
                } else {
                    (verdict.entailment_prob, FieldVerdict::PartiallyGrounded)
                }
            }
        };

        FieldConfidence {
            name: field.name.clone(),
            value: field.value.clone(),
            confidence,
            verdict: field_verdict,
            match_kind,
            evidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::groundedness::GroundednessConfig;
    use std::sync::Arc;

    // Reuses the test scaffolding pattern from groundedness.rs: a fixed-probs
    // mock NLI backend and a mock embedding backend, so these tests exercise
    // real pipeline wiring (synthetic claim -> check_with_overrides -> verdict
    // mapping) without needing real model weights.
    use crate::models::inference::{
        EmbeddingSimilarity, SequenceClassifier, SequenceClassifierInput, SequenceClassifierOutput,
    };

    struct MockNliBackend {
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
            512
        }

        fn label_map(&self) -> &[String] {
            &[]
        }
    }

    struct MockEmbeddingBackend;

    impl EmbeddingSimilarity for MockEmbeddingBackend {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            // Distinct-ish deterministic vectors so cosine similarity isn't
            // degenerate; exact values don't matter since top_k_context=0
            // in these tests bypasses pre-filtering anyway.
            #[allow(clippy::cast_precision_loss)]
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 1.0, 0.0])
                .collect())
        }

        fn embedding_dim(&self) -> usize {
            3
        }

        fn device_info(&self) -> String {
            "MockEmbedding".to_string()
        }
    }

    fn checker_with_probs(probs: Vec<f64>) -> ExtractionConfidenceChecker {
        let nli = Arc::new(MockNliBackend { fixed_probs: probs });
        let embedding = Arc::new(MockEmbeddingBackend);
        let config = GroundednessConfig {
            top_k_context: 0, // disable pre-filtering; mock embeddings are meaningless
            min_claim_length: 5,
            ..Default::default()
        };
        let groundedness = GroundednessChecker::new(nli, embedding, config);
        ExtractionConfidenceChecker::new(groundedness, ExtractionConfidenceConfig::default())
    }

    fn field(name: &str, value: &str, ty: FieldType) -> ExtractedField {
        ExtractedField {
            name: name.to_string(),
            value: value.to_string(),
            value_type: ty,
        }
    }

    #[test]
    fn synthetic_claim_formats_readable_field_name() {
        let f = field("invoice_total", "1,204.50", FieldType::Number);
        assert_eq!(synthetic_claim(&f), "invoice total is 1,204.50.");
    }

    #[test]
    fn synthetic_claim_formats_boolean_true() {
        let f = field("is_paid", "true", FieldType::Boolean);
        assert_eq!(synthetic_claim(&f), "is paid is true.");
    }

    #[test]
    fn exact_substring_yields_full_confidence() {
        let checker = checker_with_probs(vec![0.9, 0.05, 0.05]);
        let source = "The vendor is Acme Corp and the invoice total is 1,204.50 dollars.";
        let fields = vec![field("invoice total", "1,204.50", FieldType::Number)];

        let result = checker.check(source, &fields).unwrap();
        assert_eq!(result.fields.len(), 1);
        let f = &result.fields[0];
        assert_eq!(f.confidence, 1.0);
        assert_eq!(f.verdict, FieldVerdict::Grounded);
        assert_eq!(f.match_kind, MatchKind::ExactSubstring);
    }

    #[test]
    fn nli_entailment_below_threshold_is_partially_grounded() {
        // entailment=0.5 clears "entailment is max" but is below the 0.7
        // min_confidence_threshold.
        let checker = checker_with_probs(vec![0.5, 0.3, 0.2]);
        let source = "Acme Corp shipped the order on the fifteenth of March.";
        let fields = vec![field("vendor", "Acme Corporation", FieldType::String)];

        let result = checker.check(source, &fields).unwrap();
        let f = &result.fields[0];
        assert_eq!(f.match_kind, MatchKind::NliEntailment);
        assert_eq!(f.verdict, FieldVerdict::PartiallyGrounded);
        assert!((f.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn contradiction_yields_ungrounded_and_zero_confidence() {
        let checker = checker_with_probs(vec![0.1, 0.1, 0.8]);
        let source = "The invoice total is 900.00 dollars, paid in full.";
        let fields = vec![field("invoice total", "1,204.50", FieldType::Number)];

        let result = checker.check(source, &fields).unwrap();
        let f = &result.fields[0];
        assert_eq!(f.verdict, FieldVerdict::Ungrounded);
        assert_eq!(f.confidence, 0.0);
        assert!(f.evidence.is_some());
    }

    #[test]
    fn hallucinated_field_with_no_context_is_not_found() {
        let checker = checker_with_probs(vec![0.9, 0.05, 0.05]);
        let fields = vec![field("vendor", "Acme Corp", FieldType::String)];

        // Empty source -> no_context_result path in groundedness.rs.
        let result = checker.check("", &fields).unwrap();
        let f = &result.fields[0];
        assert_eq!(f.verdict, FieldVerdict::NotFound);
        assert_eq!(f.confidence, 0.0);
        assert_eq!(f.match_kind, MatchKind::NoMatch);
    }

    #[test]
    fn overall_confidence_is_mean_of_field_confidences() {
        // First field: exact substring (confidence 1.0).
        // Second field: contradiction (confidence 0.0).
        let checker = checker_with_probs(vec![0.9, 0.05, 0.05]);
        let source = "The vendor is Acme Corp.";
        let fields = vec![
            field("vendor", "Acme Corp", FieldType::String),
            field("vendor", "Beta LLC", FieldType::String),
        ];

        // Note: both fields use the same mock backend response (fixed
        // probs), so the second field's substring miss still resolves via
        // NLI to "entailment" here rather than contradiction — this test
        // only asserts the mean aggregation, not per-field semantics
        // (covered above).
        let result = checker.check(source, &fields).unwrap();
        assert_eq!(result.fields.len(), 2);
        let expected_mean = f64::midpoint(result.fields[0].confidence, result.fields[1].confidence);
        assert!((result.overall_confidence - expected_mean).abs() < 1e-9);
    }

    #[test]
    fn empty_fields_returns_full_confidence() {
        let checker = checker_with_probs(vec![0.9, 0.05, 0.05]);
        let result = checker.check("some source text", &[]).unwrap();
        assert_eq!(result.overall_confidence, 1.0);
        assert_eq!(result.engine_confidence, 1.0);
        assert!(result.fields.is_empty());
    }
}
