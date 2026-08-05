//! Error types for Krino evaluation engine.
//!
//! This module defines all error types used throughout Krino. All errors are
//! designed to be actionable and provide clear context for debugging.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for Krino operations.
pub type Result<T> = std::result::Result<T, KrinoError>;

/// Main error type for Krino operations.
///
/// All Krino operations that can fail return this error type. Each variant
/// provides specific context about what went wrong and how to fix it.
#[derive(Error, Debug)]
pub enum KrinoError {
    /// Model-related errors (loading, inference, version mismatches)
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Configuration errors (invalid settings, missing required fields)
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// Evaluation errors (module failures, invalid inputs)
    #[error("Evaluation error: {0}")]
    Evaluation(#[from] EvaluationError),

    /// I/O errors (file not found, permission denied)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors (invalid JSON, parsing failures)
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Errors related to model loading and inference.
#[derive(Error, Debug)]
pub enum ModelError {
    /// Model file not found at expected path
    #[error("Model not found: {path}")]
    NotFound { path: PathBuf },

    /// Model version mismatch (expected vs actual)
    #[error("Model version mismatch: expected {expected}, found {actual}")]
    VersionMismatch { expected: String, actual: String },

    /// Model weights failed SHA256 verification
    #[error("Model integrity check failed: expected hash {expected}, computed {actual}")]
    IntegrityCheckFailed { expected: String, actual: String },

    /// Model failed to load (corrupt file, unsupported format)
    #[error("Failed to load model from {path}: {reason}")]
    LoadFailed { path: PathBuf, reason: String },

    /// Inference operation failed
    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },

    /// Model not initialized (attempted inference before loading)
    #[error("Model not initialized: {model_id}")]
    NotInitialized { model_id: String },

    /// Unsupported model format or architecture
    #[error("Unsupported model format: {format}")]
    UnsupportedFormat { format: String },
}

/// Errors related to configuration.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Required field missing from configuration
    #[error("Missing required configuration field: {field}")]
    MissingField { field: String },

    /// Invalid value for configuration field
    #[error("Invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },

    /// Configuration file not found
    #[error("Configuration file not found: {path}")]
    NotFound { path: PathBuf },

    /// Failed to parse configuration file
    #[error("Failed to parse configuration from {path}: {reason}")]
    ParseFailed { path: PathBuf, reason: String },
}

/// Errors related to evaluation operations.
#[derive(Error, Debug)]
pub enum EvaluationError {
    /// Required input missing (e.g., context required but not provided)
    #[error("Missing required input: {input}. {suggestion}")]
    MissingInput { input: String, suggestion: String },

    /// Invalid input (e.g., empty string, malformed data)
    #[error("Invalid input for {field}: {reason}")]
    InvalidInput { field: String, reason: String },

    /// Module failed during evaluation
    #[error("Module {module} failed: {reason}")]
    ModuleFailed { module: String, reason: String },

    /// Evaluation timeout exceeded
    #[error("Evaluation timeout exceeded: {module} took {duration_ms}ms (limit: {limit_ms}ms)")]
    Timeout {
        module: String,
        duration_ms: u64,
        limit_ms: u64,
    },

    /// Non-determinism detected (same input produced different outputs)
    #[error("Non-determinism detected in {module}: {details}")]
    NonDeterministic { module: String, details: String },
}

impl ModelError {
    /// Creates a new `NotFound` error with the given path.
    pub fn not_found(path: impl Into<PathBuf>) -> Self {
        Self::NotFound { path: path.into() }
    }

    /// Creates a new `LoadFailed` error with the given path and reason.
    pub fn load_failed(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::LoadFailed {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Creates a new `InferenceFailed` error with the given reason.
    pub fn inference_failed(reason: impl Into<String>) -> Self {
        Self::InferenceFailed {
            reason: reason.into(),
        }
    }
}

impl ConfigError {
    /// Creates a new `MissingField` error.
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
        }
    }

    /// Creates a new `InvalidValue` error.
    pub fn invalid_value(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidValue {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl EvaluationError {
    /// Creates a new `MissingInput` error with a helpful suggestion.
    pub fn missing_input(input: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self::MissingInput {
            input: input.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Creates a new `InvalidInput` error.
    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Creates a new `ModuleFailed` error.
    pub fn module_failed(module: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ModuleFailed {
            module: module.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_error_display() {
        let err = ModelError::not_found("/path/to/model.bin");
        assert_eq!(err.to_string(), "Model not found: /path/to/model.bin");
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::missing_field("model_path");
        assert_eq!(
            err.to_string(),
            "Missing required configuration field: model_path"
        );
    }

    #[test]
    fn test_evaluation_error_display() {
        let err = EvaluationError::missing_input(
            "context",
            "Context is required for hallucination detection. Provide grounding documents.",
        );
        assert!(err.to_string().contains("Missing required input: context"));
    }

    #[test]
    fn test_error_conversion() {
        let model_err = ModelError::not_found("/test/path");
        let krino_err: KrinoError = model_err.into();
        assert!(matches!(krino_err, KrinoError::Model(_)));
    }
}
