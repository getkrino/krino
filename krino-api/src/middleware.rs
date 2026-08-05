use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};
use uuid::Uuid;

/// Middleware to add request ID if not present
pub async fn request_id_middleware(mut req: Request, next: Next) -> Response {
    // Generate request ID if not present and extract it
    let request_id = if !req.headers().contains_key("x-request-id") {
        let id = Uuid::new_v4().to_string();
        req.headers_mut()
            .insert("x-request-id", HeaderValue::from_str(&id).unwrap());
        id
    } else {
        req.headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string()
    };

    let mut response = next.run(req).await;

    // Propagate request ID to response headers
    if let Ok(header_value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", header_value);
    }

    response
}
