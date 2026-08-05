use axum::{Json, extract::State};
use std::sync::Arc;

use krino_api_types::{
    ClaimResponse, ContextChunk, EntailmentMatrixCell, EvaluateRequest, EvaluateResponse,
    EvidenceResponse, IssueResponse, MetaResponse,
};

use crate::error::ApiError;
use crate::metrics::{Timer, record_evaluation, record_nli_inference};
use crate::state::AppState;

/// POST /api/v1/evaluate
///
/// Verifies that LLM output is faithful to the provided context.
/// Context is required. Granularity (claim or token) is configurable.
pub async fn evaluate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, ApiError> {
    let timer = Timer::new();

    // ── Validate ──
    if req.context.is_empty() {
        return Err(ApiError::bad_request(
            "context is required. Provide the source documents, retrieved chunks, \
             or reference text that the output should be faithful to.",
        ));
    }

    if req.output.trim().is_empty() {
        return Err(ApiError::bad_request("output must not be empty"));
    }

    let total_context_chars: usize = req.context.iter().map(|c| c.text.len()).sum();

    if total_context_chars > state.config.faithfulness.max_context_chars {
        return Err(ApiError::bad_request(format!(
            "context exceeds maximum of {} characters ({} provided)",
            state.config.faithfulness.max_context_chars, total_context_chars,
        )));
    }

    if req.output.len() > state.config.faithfulness.max_output_chars {
        return Err(ApiError::bad_request(format!(
            "output exceeds maximum of {} characters ({} provided)",
            state.config.faithfulness.max_output_chars,
            req.output.len(),
        )));
    }

    // ── Parse granularity ──
    let granularity = req
        .config
        .as_ref()
        .and_then(|c| c.granularity.as_deref())
        .unwrap_or("claim");

    match granularity {
        "claim" => {}
        "token" => {
            return Err(ApiError::bad_request(
                "Token-level granularity not yet implemented. Use 'claim' granularity.",
            ));
        }
        other => {
            return Err(ApiError::bad_request(format!(
                "invalid granularity '{other}'. Use 'claim' or 'token'."
            )));
        }
    }

    let threshold = req
        .config
        .as_ref()
        .and_then(|c| c.threshold)
        .unwrap_or(state.config.faithfulness.default_threshold);

    let include_matrix = req.config.as_ref().and_then(|c| c.include_matrix);
    let top_k_context = req.config.as_ref().and_then(|c| c.top_k_context);
    let overrides = krino::modules::groundedness::RequestOverrides {
        include_matrix,
        top_k_context,
    };

    // ── Build context string ──
    let context_text = req
        .context
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    // ── Dispatch to worker pool ──
    let chunks = req.context.clone();
    let result = state
        .worker_pool
        .evaluate(context_text, req.output.clone(), overrides)
        .await?;

    let latency = timer.elapsed();

    // ── Map to response ──
    let issues: Vec<IssueResponse> = result
        .verdicts
        .iter()
        .filter(|v| !v.supported)
        .map(|v| {
            let evidence = v.best_evidence.as_ref().map(|e| {
                let chunk_id = find_chunk_id(&chunks, &e.sentence);
                EvidenceResponse {
                    chunk_id,
                    text: e.sentence.clone(),
                    entailment_prob: Some(e.entailment_prob),
                    contradiction_prob: Some(e.contradiction_prob),
                    similarity_score: e.similarity_score.map(f64::from),
                }
            });

            let (issue_type, severity, confidence) = if v.label == "contradiction" {
                ("contradiction", "high", v.contradiction_prob)
            } else {
                ("unsupported", "medium", 1.0 - v.entailment_prob)
            };

            IssueResponse {
                text: v.claim.clone(),
                issue_type: issue_type.to_string(),
                severity: severity.to_string(),
                confidence,
                evidence,
            }
        })
        .collect();

    let claims: Vec<ClaimResponse> = result
        .verdicts
        .iter()
        .map(|v| {
            let evidence = v.best_evidence.as_ref().map(|e| {
                let chunk_id = find_chunk_id(&chunks, &e.sentence);
                EvidenceResponse {
                    chunk_id,
                    text: e.sentence.clone(),
                    entailment_prob: Some(e.entailment_prob),
                    contradiction_prob: Some(e.contradiction_prob),
                    similarity_score: e.similarity_score.map(f64::from),
                }
            });

            let supporting_evidence: Vec<EvidenceResponse> = v
                .supporting_evidence
                .iter()
                .map(|e| EvidenceResponse {
                    chunk_id: find_chunk_id(&chunks, &e.sentence),
                    text: e.sentence.clone(),
                    entailment_prob: Some(e.entailment_prob),
                    contradiction_prob: Some(e.contradiction_prob),
                    similarity_score: e.similarity_score.map(f64::from),
                })
                .collect();

            ClaimResponse {
                text: v.claim.clone(),
                verdict: v.label.clone(),
                score: v.entailment_prob,
                supported: v.supported,
                evidence,
                is_compound: v.is_compound,
                supporting_evidence,
            }
        })
        .collect();

    record_evaluation("faithfulness", latency, result.total_claims);
    record_nli_inference(result.latency_ms);

    let entailment_matrix = result.entailment_matrix.as_ref().map(|cells| {
        cells
            .iter()
            .map(|c| EntailmentMatrixCell {
                claim_idx: c.claim_idx,
                context_idx: c.context_idx,
                context_sentence: c.context_sentence.clone(),
                entailment_prob: c.entailment_prob,
                neutral_prob: c.neutral_prob,
                contradiction_prob: c.contradiction_prob,
                label: c.label.clone(),
                similarity_score: c.similarity_score.map(f64::from),
            })
            .collect()
    });

    Ok(Json(EvaluateResponse {
        faithfulness_score: result.faithfulness_score,
        pass: result.faithfulness_score >= threshold,
        issues,
        claims,
        spans: None,
        entailment_matrix,
        meta: MetaResponse {
            granularity: "claim".to_string(),
            model: "krino-faithfulness-v1".to_string(),
            latency_ms: latency.as_secs_f64() * 1000.0,
            nli_calls: result.nli_calls,
            // REST API keeps the engine's native score — no UI re-weighting.
            engine_confidence: None,
            split_ms: Some(result.split_ms),
            embedding_ms: Some(result.embedding_ms),
            nli_ms: Some(result.nli_ms),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }))
}

/// Find which input chunk contains the evidence sentence.
pub fn find_chunk_id(chunks: &[ContextChunk], evidence_text: &str) -> Option<String> {
    chunks
        .iter()
        .find(|c| c.text.contains(evidence_text))
        .and_then(|c| c.id.clone())
}
