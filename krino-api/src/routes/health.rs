use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use std::sync::Arc;

use crate::state::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

#[derive(Serialize)]
pub struct ReadyResponse {
    pub status: &'static str,
    pub model_loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// GET /health — Liveness probe. Always returns 200 if the process is up.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// GET /health/ready — Readiness probe. Submits a trivial evaluation to the
/// worker pool; returns 503 if it fails (e.g. ONNX session crashed, queue
/// permanently saturated). ALB target groups should health-check this path,
/// not `/health`, to avoid sending traffic to instances whose model has
/// failed to load.
pub async fn ready(State(state): State<Arc<AppState>>) -> Response {
    // Trivial paired input: a single context sentence and a verbatim claim.
    // Engine hits the substring fast-path and returns near-instantly without
    // a full NLI inference, but still exercises the worker queue end-to-end.
    let ctx = "Krino is a faithfulness checker.".to_string();
    let out = "Krino is a faithfulness checker.".to_string();

    match state
        .worker_pool
        .evaluate(
            ctx,
            out,
            krino::modules::groundedness::RequestOverrides::default(),
        )
        .await
    {
        Ok(_) => (
            StatusCode::OK,
            Json(ReadyResponse {
                status: "ready",
                model_loaded: true,
                error: None,
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyResponse {
                status: "not_ready",
                model_loaded: false,
                error: Some(format!("{err:?}")),
            }),
        )
            .into_response(),
    }
}
