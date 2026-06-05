use axum::http::StatusCode;

/// Basic integration test to verify the API structure compiles
/// Full end-to-end tests require the ONNX model to be present

#[test]
fn test_error_serialization() {
    use krino_api::error::ApiError;

    let err = ApiError::bad_request("test error");
    let response = axum::response::IntoResponse::into_response(err);
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn test_config_defaults() {
    use krino_api::config::{FaithfulnessApiConfig, ServerConfig};

    let server = ServerConfig::default();
    assert_eq!(server.port, 8080);

    let faithfulness = FaithfulnessApiConfig::default();
    assert_eq!(faithfulness.default_top_k, 10);
    assert_eq!(faithfulness.default_threshold, 0.7);
    assert_eq!(faithfulness.available_granularities, vec!["claim"]);
}
