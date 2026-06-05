use axum::{Router, extract::State};
use std::sync::Arc;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::auth::auth_middleware;
use crate::routes;
use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    let state = Arc::new(state);

    // Public routes (no auth)
    let public = Router::new()
        .route("/health", axum::routing::get(routes::health::health))
        .route("/health/ready", axum::routing::get(routes::health::ready))
        .route("/metrics", axum::routing::get(metrics_handler));

    // Authenticated API routes (x-api-key header)
    let api = Router::new()
        .route(
            "/v1/evaluate",
            axum::routing::post(routes::evaluate::evaluate),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::<Arc<AppState>>::new()
        .merge(public)
        .nest("/api", api)
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> Result<String, axum::http::StatusCode> {
    Ok(state.metrics.render())
}
