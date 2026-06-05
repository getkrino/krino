use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Unauthorized,
    RateLimited { retry_after_secs: u64 },
    ServiceUnavailable(String),
    Internal(String),
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    code: &'static str,
    message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self::RateLimited { retry_after_secs }
    }

    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self::ServiceUnavailable(msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "Invalid or missing API key".to_string(),
            ),
            ApiError::RateLimited { retry_after_secs } => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                format!("Rate limit exceeded. Retry after {retry_after_secs}s"),
            ),
            ApiError::ServiceUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable", msg)
            }
            ApiError::Internal(msg) => {
                // Log the actual error but don't expose it
                tracing::error!(error = %msg, "Internal server error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "An internal error occurred".to_string(),
                )
            }
        };

        (
            status,
            Json(ErrorBody {
                error: ErrorDetail { code, message },
            }),
        )
            .into_response()
    }
}

// Convert from Krino engine errors to API errors
impl From<krino::error::KrinoError> for ApiError {
    fn from(err: krino::error::KrinoError) -> Self {
        ApiError::internal(err.to_string())
    }
}
