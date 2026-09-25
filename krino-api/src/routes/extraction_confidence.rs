use axum::{Json, extract::State};
use std::sync::Arc;

use krino::modules::extraction_confidence::{ExtractedField, FieldType as EngineFieldType};
use krino_api_types::{
    EvidenceResponse, ExtractionConfidenceRequest, ExtractionConfidenceResponse,
    FieldConfidenceResponse, MetaResponse,
};

use crate::error::ApiError;
use crate::metrics::{Timer, record_evaluation};
use crate::state::AppState;

/// POST /api/v1/extraction-confidence
///
/// Scores confidence for a set of LLM-extracted structured fields against a
/// source document. Each field is composed into a synthetic claim and run
/// through the same groundedness pipeline used by `/evaluate` — the
/// resulting entailment probability (or a substring exact-match) becomes
/// the field's confidence score, instead of trusting the extracting LLM's
/// own self-reported confidence.
pub async fn extraction_confidence(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExtractionConfidenceRequest>,
) -> Result<Json<ExtractionConfidenceResponse>, ApiError> {
    let timer = Timer::new();

    // ── Validate ──
    if req.source.is_empty() {
        return Err(ApiError::bad_request(
            "source is required. Provide the document the fields were extracted from.",
        ));
    }

    if req.fields.is_empty() {
        return Err(ApiError::bad_request(
            "fields is required and must contain at least one extracted field.",
        ));
    }

    let total_source_chars: usize = req.source.iter().map(|c| c.text.len()).sum();
    if total_source_chars > state.config.faithfulness.max_context_chars {
        return Err(ApiError::bad_request(format!(
            "source exceeds maximum of {} characters ({} provided)",
            state.config.faithfulness.max_context_chars, total_source_chars,
        )));
    }

    let mut fields = Vec::with_capacity(req.fields.len());
    for f in &req.fields {
        if f.name.trim().is_empty() {
            return Err(ApiError::bad_request("field name must not be empty"));
        }
        let value_type = match f.value_type.as_str() {
            "string" => EngineFieldType::String,
            "number" => EngineFieldType::Number,
            "boolean" => EngineFieldType::Boolean,
            other => {
                return Err(ApiError::bad_request(format!(
                    "invalid value_type '{other}' for field '{}'. Use 'string', 'number', or 'boolean'.",
                    f.name
                )));
            }
        };
        fields.push(ExtractedField {
            name: f.name.clone(),
            value: f.value.clone(),
            value_type,
        });
    }

    // ── Build source string ──
    let source_text = req
        .source
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    // ── Dispatch to worker pool ──
    let result = state
        .worker_pool
        .check_extraction_confidence(source_text, fields)
        .await?;

    let latency = timer.elapsed();

    let field_responses: Vec<FieldConfidenceResponse> = result
        .fields
        .iter()
        .map(|f| {
            let verdict = match f.verdict {
                krino::modules::extraction_confidence::FieldVerdict::Grounded => "grounded",
                krino::modules::extraction_confidence::FieldVerdict::PartiallyGrounded => {
                    "partially_grounded"
                }
                krino::modules::extraction_confidence::FieldVerdict::Ungrounded => "ungrounded",
                krino::modules::extraction_confidence::FieldVerdict::NotFound => "not_found",
            };
            let match_kind = match f.match_kind {
                krino::modules::extraction_confidence::MatchKind::ExactSubstring => {
                    "exact_substring"
                }
                krino::modules::extraction_confidence::MatchKind::NliEntailment => "nli_entailment",
                krino::modules::extraction_confidence::MatchKind::NoMatch => "no_match",
            };

            FieldConfidenceResponse {
                name: f.name.clone(),
                value: f.value.clone(),
                confidence: f.confidence,
                verdict: verdict.to_string(),
                match_kind: match_kind.to_string(),
                evidence: f.evidence.as_ref().map(|e| EvidenceResponse {
                    chunk_id: None,
                    text: e.sentence.clone(),
                    entailment_prob: Some(e.entailment_prob),
                    contradiction_prob: Some(e.contradiction_prob),
                    similarity_score: e.similarity_score.map(f64::from),
                }),
            }
        })
        .collect();

    record_evaluation("extraction_confidence", latency, field_responses.len());

    Ok(Json(ExtractionConfidenceResponse {
        fields: field_responses,
        overall_confidence: result.overall_confidence,
        engine_confidence: result.engine_confidence,
        meta: MetaResponse {
            granularity: "field".to_string(),
            model: "krino-faithfulness-v1".to_string(),
            latency_ms: latency.as_secs_f64() * 1000.0,
            nli_calls: 0,
            engine_confidence: Some(result.engine_confidence),
            split_ms: None,
            embedding_ms: None,
            nli_ms: None,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }))
}
