//! JSON Schema validation for structured LLM outputs.
//!
//! Validates that LLM-generated JSON conforms to a JSON Schema specification.
//! Purely rule-based and deterministic - no models involved.
//!
//! Supports JSON Schema Draft 7 (default in jsonschema crate).

use crate::error::Result;
use crate::pipeline::report::ModuleDetail;
use jsonschema::{Draft, Validator};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};

/// Configuration for schema validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaConfig {
    /// JSON Schema specification (as a JSON value)
    pub schema: serde_json::Value,

    /// Whether to fail on additional properties not in schema (strict mode)
    /// This is separate from `additionalProperties: false` in the schema itself
    pub strict_mode: bool,

    /// Maximum allowed nesting depth for JSON (prevents deeply nested outputs)
    pub max_depth: Option<usize>,
}

impl Default for SchemaConfig {
    fn default() -> Self {
        Self {
            schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": true
            }),
            strict_mode: false,
            max_depth: Some(10),
        }
    }
}

/// A single validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// JSON path to the error (e.g., "/user/age")
    pub path: String,

    /// Error message
    pub message: String,

    /// Error kind (e.g., "type", "required", "pattern", "enum")
    pub kind: String,
}

/// Result from schema validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationResult {
    /// Whether the output is valid according to the schema
    pub valid: bool,

    /// List of validation errors (empty if valid)
    pub errors: Vec<ValidationError>,

    /// Total number of errors found
    pub error_count: usize,

    /// The parsed JSON value (if parsing succeeded)
    pub parsed_json: Option<serde_json::Value>,

    /// Whether JSON parsing succeeded
    pub json_parse_success: bool,

    /// JSON parsing error (if parsing failed)
    pub json_parse_error: Option<String>,

    /// Nesting depth of the JSON structure
    pub nesting_depth: usize,

    /// Total validation latency (ms)
    pub latency_ms: f64,
}

/// JSON Schema validator for LLM outputs.
///
/// # Determinism
///
/// This validator is fully deterministic - same input always produces
/// identical validation results. No models or random processes involved.
pub struct SchemaValidator {
    /// Compiled JSON Schema
    schema: Validator,

    /// Configuration
    config: SchemaConfig,
}

impl SchemaValidator {
    /// Creates a new schema validator with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the schema is invalid or cannot be compiled.
    pub fn new(config: SchemaConfig) -> Result<Self> {
        info!("Compiling JSON Schema");

        // Compile the schema (validates schema is well-formed)
        let schema = Validator::options()
            .with_draft(Draft::Draft7)
            .build(&config.schema)
            .map_err(|e| {
                crate::error::EvaluationError::module_failed(
                    "schema",
                    format!("Invalid JSON Schema: {e}"),
                )
            })?;

        debug!("Schema compiled successfully");

        Ok(Self { schema, config })
    }

    /// Validates a JSON string against the schema.
    ///
    /// # Determinism
    ///
    /// This function is deterministic - same inputs always produce identical outputs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let schema = serde_json::json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "name": { "type": "string" },
    ///         "age": { "type": "number", "minimum": 0 }
    ///     },
    ///     "required": ["name"]
    /// });
    ///
    /// let validator = SchemaValidator::new(SchemaConfig {
    ///     schema,
    ///     ..Default::default()
    /// })?;
    ///
    /// let result = validator.validate(r#"{"name": "Alice", "age": 30}"#)?;
    /// assert!(result.valid);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error only if internal processing fails (not for validation failures).
    /// Validation failures are reported in the result's `errors` field.
    pub fn validate(&self, json_str: &str) -> Result<SchemaValidationResult> {
        let start = Instant::now();

        info!(
            "Validating JSON output ({} chars) against schema",
            json_str.len()
        );

        // Step 1: Parse JSON
        let parsed_json = match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(json) => json,
            Err(e) => {
                // JSON parsing failed - return immediately
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(SchemaValidationResult {
                    valid: false,
                    errors: vec![ValidationError {
                        path: "/".to_string(),
                        message: format!("JSON parsing failed: {e}"),
                        kind: "parse_error".to_string(),
                    }],
                    error_count: 1,
                    parsed_json: None,
                    json_parse_success: false,
                    json_parse_error: Some(e.to_string()),
                    nesting_depth: 0,
                    latency_ms,
                });
            }
        };

        debug!("JSON parsed successfully");

        // Step 2: Check nesting depth
        let nesting_depth = compute_nesting_depth(&parsed_json);
        if let Some(max_depth) = self.config.max_depth
            && nesting_depth > max_depth
        {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(SchemaValidationResult {
                valid: false,
                errors: vec![ValidationError {
                    path: "/".to_string(),
                    message: format!("Nesting depth {nesting_depth} exceeds maximum {max_depth}"),
                    kind: "max_depth_exceeded".to_string(),
                }],
                error_count: 1,
                parsed_json: Some(parsed_json),
                json_parse_success: true,
                json_parse_error: None,
                nesting_depth,
                latency_ms,
            });
        }

        // Step 3: Validate against schema
        let validation_result = self.schema.validate(&parsed_json);

        let errors: Vec<ValidationError> = if validation_result.is_ok() {
            Vec::new()
        } else {
            self.schema
                .iter_errors(&parsed_json)
                .map(|e| {
                    let path = e.instance_path().to_string();
                    let message = e.to_string();
                    let kind = format!("{:?}", e.kind()); // Extract error kind (e.g., Type, Required)

                    ValidationError {
                        path: if path.is_empty() {
                            "/".to_string()
                        } else {
                            path
                        },
                        message,
                        kind,
                    }
                })
                .collect()
        };

        let error_count = errors.len();
        let valid = error_count == 0;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Schema validation complete: valid={}, errors={}, {:.2}ms",
            valid, error_count, latency_ms
        );

        Ok(SchemaValidationResult {
            valid,
            errors,
            error_count,
            parsed_json: Some(parsed_json),
            json_parse_success: true,
            json_parse_error: None,
            nesting_depth,
            latency_ms,
        })
    }

    /// Converts result to `ModuleDetail` variants for `KrinoReport` integration.
    #[must_use]
    pub fn to_module_details(result: &SchemaValidationResult) -> Vec<ModuleDetail> {
        result
            .errors
            .iter()
            .map(|e| ModuleDetail::SchemaValidation {
                path: e.path.clone(),
                error: e.message.clone(),
            })
            .collect()
    }
}

/// Computes the maximum nesting depth of a JSON value.
///
/// - Primitives (null, bool, number, string): depth 0
/// - Arrays/Objects: 1 + max depth of children
fn compute_nesting_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => 0,
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                1
            } else {
                1 + arr.iter().map(compute_nesting_depth).max().unwrap_or(0)
            }
        }
        serde_json::Value::Object(obj) => {
            if obj.is_empty() {
                1
            } else {
                1 + obj.values().map(compute_nesting_depth).max().unwrap_or(0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Nesting depth tests ---

    #[test]
    fn test_nesting_depth_primitives() {
        assert_eq!(compute_nesting_depth(&serde_json::json!(null)), 0);
        assert_eq!(compute_nesting_depth(&serde_json::json!(true)), 0);
        assert_eq!(compute_nesting_depth(&serde_json::json!(42)), 0);
        assert_eq!(compute_nesting_depth(&serde_json::json!("hello")), 0);
    }

    #[test]
    fn test_nesting_depth_empty_containers() {
        assert_eq!(compute_nesting_depth(&serde_json::json!([])), 1);
        assert_eq!(compute_nesting_depth(&serde_json::json!({})), 1);
    }

    #[test]
    fn test_nesting_depth_flat_array() {
        assert_eq!(compute_nesting_depth(&serde_json::json!([1, 2, 3])), 1);
    }

    #[test]
    fn test_nesting_depth_nested_array() {
        assert_eq!(
            compute_nesting_depth(&serde_json::json!([[1, 2], [3, 4]])),
            2
        );
        assert_eq!(compute_nesting_depth(&serde_json::json!([[[1]], [[2]]])), 3);
    }

    #[test]
    fn test_nesting_depth_flat_object() {
        assert_eq!(
            compute_nesting_depth(&serde_json::json!({"a": 1, "b": 2})),
            1
        );
    }

    #[test]
    fn test_nesting_depth_nested_object() {
        assert_eq!(
            compute_nesting_depth(&serde_json::json!({"a": {"b": 1}})),
            2
        );
        assert_eq!(
            compute_nesting_depth(&serde_json::json!({"a": {"b": {"c": 1}}})),
            3
        );
    }

    #[test]
    fn test_nesting_depth_mixed() {
        assert_eq!(
            compute_nesting_depth(&serde_json::json!({
                "users": [
                    {"name": "Alice", "tags": ["admin", "user"]},
                    {"name": "Bob", "tags": ["user"]}
                ]
            })),
            4 // object (1) -> array (2) -> object (3) -> array (4)
        );
    }

    // --- Schema validation tests ---

    #[test]
    fn test_valid_simple_object() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let result = validator
            .validate(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        assert!(result.valid);
        assert!(result.json_parse_success);
        assert_eq!(result.error_count, 0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_missing_required_field() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let result = validator.validate(r#"{"name": "Alice"}"#).unwrap();

        assert!(!result.valid);
        assert_eq!(result.error_count, 1);
        assert!(result.errors[0].message.contains("age"));
    }

    #[test]
    fn test_wrong_type() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "age": {"type": "number"}
            }
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let result = validator.validate(r#"{"age": "thirty"}"#).unwrap();

        assert!(!result.valid);
        assert!(result.errors[0].message.contains("number"));
    }

    #[test]
    fn test_pattern_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                }
            }
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let valid_result = validator
            .validate(r#"{"email": "alice@example.com"}"#)
            .unwrap();
        assert!(valid_result.valid);

        let invalid_result = validator.validate(r#"{"email": "not-an-email"}"#).unwrap();
        assert!(!invalid_result.valid);
    }

    #[test]
    fn test_enum_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"]
                }
            }
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let valid_result = validator.validate(r#"{"status": "active"}"#).unwrap();
        assert!(valid_result.valid);

        let invalid_result = validator.validate(r#"{"status": "unknown"}"#).unwrap();
        assert!(!invalid_result.valid);
    }

    #[test]
    fn test_number_constraints() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "age": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 120
                }
            }
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let valid_result = validator.validate(r#"{"age": 30}"#).unwrap();
        assert!(valid_result.valid);

        let invalid_result = validator.validate(r#"{"age": -5}"#).unwrap();
        assert!(!invalid_result.valid);

        let invalid_result2 = validator.validate(r#"{"age": 150}"#).unwrap();
        assert!(!invalid_result2.valid);
    }

    #[test]
    fn test_array_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5
                }
            }
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let valid_result = validator.validate(r#"{"tags": ["rust", "json"]}"#).unwrap();
        assert!(valid_result.valid);

        let invalid_result = validator.validate(r#"{"tags": []}"#).unwrap();
        assert!(!invalid_result.valid); // minItems violation

        let invalid_result2 = validator.validate(r#"{"tags": [1, 2, 3]}"#).unwrap();
        assert!(!invalid_result2.valid); // wrong item type
    }

    #[test]
    fn test_nested_object_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["user"]
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let valid_result = validator
            .validate(r#"{"user": {"name": "Alice", "email": "alice@example.com"}}"#)
            .unwrap();
        assert!(valid_result.valid);

        let invalid_result = validator.validate(r#"{"user": {}}"#).unwrap();
        assert!(!invalid_result.valid); // missing required name
    }

    #[test]
    fn test_json_parse_error() {
        let schema = serde_json::json!({"type": "object"});

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let result = validator.validate(r#"{"invalid json"#).unwrap();

        assert!(!result.valid);
        assert!(!result.json_parse_success);
        assert!(result.json_parse_error.is_some());
        assert_eq!(result.error_count, 1);
        assert_eq!(result.errors[0].kind, "parse_error");
    }

    #[test]
    fn test_max_depth_exceeded() {
        let schema = serde_json::json!({"type": "object"});

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            max_depth: Some(2),
            ..Default::default()
        })
        .unwrap();

        // Depth 3: object -> object -> object
        let result = validator.validate(r#"{"a": {"b": {"c": 1}}}"#).unwrap();

        assert!(!result.valid);
        assert_eq!(result.nesting_depth, 3);
        assert_eq!(result.error_count, 1);
        assert_eq!(result.errors[0].kind, "max_depth_exceeded");
    }

    #[test]
    fn test_additional_properties_allowed() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": true
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let result = validator
            .validate(r#"{"name": "Alice", "extra": "field"}"#)
            .unwrap();

        assert!(result.valid);
    }

    #[test]
    fn test_additional_properties_forbidden() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": false
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let result = validator
            .validate(r#"{"name": "Alice", "extra": "field"}"#)
            .unwrap();

        assert!(!result.valid);
    }

    #[test]
    fn test_to_module_details() {
        let result = SchemaValidationResult {
            valid: false,
            errors: vec![
                ValidationError {
                    path: "/name".to_string(),
                    message: "Missing required field".to_string(),
                    kind: "required".to_string(),
                },
                ValidationError {
                    path: "/age".to_string(),
                    message: "Expected number, got string".to_string(),
                    kind: "type".to_string(),
                },
            ],
            error_count: 2,
            parsed_json: None,
            json_parse_success: true,
            json_parse_error: None,
            nesting_depth: 1,
            latency_ms: 1.5,
        };

        let details = SchemaValidator::to_module_details(&result);
        assert_eq!(details.len(), 2);

        match &details[0] {
            ModuleDetail::SchemaValidation { path, error } => {
                assert_eq!(path, "/name");
                assert_eq!(error, "Missing required field");
            }
            _ => panic!("Expected SchemaValidation"),
        }

        match &details[1] {
            ModuleDetail::SchemaValidation { path, error } => {
                assert_eq!(path, "/age");
                assert_eq!(error, "Expected number, got string");
            }
            _ => panic!("Expected SchemaValidation"),
        }
    }

    // --- Determinism test ---

    #[test]
    fn test_determinism() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name"]
        });

        let validator = SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap();

        let json_str = r#"{"name": "Alice", "age": 30, "tags": ["admin", "user"]}"#;

        // Run 100 iterations to verify determinism
        let results: Vec<_> = (0..100)
            .map(|_| validator.validate(json_str).unwrap())
            .collect();

        // All results should be identical
        let first = &results[0];
        for result in &results[1..] {
            assert_eq!(result.valid, first.valid, "valid differs across runs");
            assert_eq!(result.error_count, first.error_count, "error_count differs");
            assert_eq!(
                result.json_parse_success, first.json_parse_success,
                "json_parse_success differs"
            );
            assert_eq!(
                result.nesting_depth, first.nesting_depth,
                "nesting_depth differs"
            );

            // Compare errors
            assert_eq!(
                result.errors.len(),
                first.errors.len(),
                "errors.len differs"
            );
            for (e1, e2) in result.errors.iter().zip(first.errors.iter()) {
                assert_eq!(e1.path, e2.path, "error path differs");
                assert_eq!(e1.message, e2.message, "error message differs");
                assert_eq!(e1.kind, e2.kind, "error kind differs");
            }
        }
    }
}
