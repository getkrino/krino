# Schema Validation Library Examples

This guide shows how to use Krino schema validation in your Rust applications.

## Quick Start

```rust
use krino::modules::schema::{SchemaConfig, SchemaValidator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define schema
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name"]
    });

    // Create validator
    let validator = SchemaValidator::new(SchemaConfig {
        schema,
        strict_mode: false,
        max_depth: Some(10),
    })?;

    // Validate JSON
    let result = validator.validate(r#"{"name": "Alice", "age": 30}"#)?;

    if result.valid {
        println!("✅ Valid JSON");
    } else {
        println!("❌ Validation errors:");
        for error in &result.errors {
            println!("  {}: {}", error.path, error.message);
        }
    }

    Ok(())
}
```

## Loading Schema from File

```rust
use std::fs;
use krino::modules::schema::{SchemaConfig, SchemaValidator};

fn load_validator(schema_path: &str) -> Result<SchemaValidator, Box<dyn std::error::Error>> {
    let schema_str = fs::read_to_string(schema_path)?;
    let schema: serde_json::Value = serde_json::from_str(&schema_str)?;

    Ok(SchemaValidator::new(SchemaConfig {
        schema,
        strict_mode: false,
        max_depth: Some(10),
    })?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let validator = load_validator("examples/schema_validation/user_schema.json")?;

    let json_input = fs::read_to_string("user_data.json")?;
    let result = validator.validate(&json_input)?;

    println!("Valid: {}", result.valid);
    println!("Errors: {}", result.error_count);

    Ok(())
}
```

## Validating LLM Function Calls

```rust
use krino::modules::schema::{SchemaConfig, SchemaValidator};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct FunctionCall {
    function: String,
    parameters: serde_json::Value,
}

fn validate_llm_function_call(
    validator: &SchemaValidator,
    llm_output: &str,
) -> Result<FunctionCall, Box<dyn std::error::Error>> {
    // First validate structure
    let result = validator.validate(llm_output)?;

    if !result.valid {
        return Err(format!(
            "Invalid function call structure: {} errors",
            result.error_count
        )
        .into());
    }

    // Parse into struct
    let function_call: FunctionCall = serde_json::from_str(llm_output)?;

    Ok(function_call)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "function": {
                "type": "string",
                "enum": ["search", "create", "update"]
            },
            "parameters": {"type": "object"}
        },
        "required": ["function", "parameters"]
    });

    let validator = SchemaValidator::new(SchemaConfig {
        schema,
        ..Default::default()
    })?;

    let llm_output = r#"{
        "function": "search",
        "parameters": {"query": "rust crates"}
    }"#;

    match validate_llm_function_call(&validator, llm_output) {
        Ok(call) => println!("✅ Valid function call: {}", call.function),
        Err(e) => println!("❌ Invalid: {}", e),
    }

    Ok(())
}
```

## Batch Validation

```rust
use krino::modules::schema::{SchemaConfig, SchemaValidator};

fn validate_batch(
    validator: &SchemaValidator,
    json_inputs: &[&str],
) -> Vec<(bool, usize)> {
    json_inputs
        .iter()
        .map(|input| {
            validator
                .validate(input)
                .map(|r| (r.valid, r.error_count))
                .unwrap_or((false, 1))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "value": {"type": "number"}
        },
        "required": ["id"]
    });

    let validator = SchemaValidator::new(SchemaConfig {
        schema,
        ..Default::default()
    })?;

    let inputs = vec![
        r#"{"id": "1", "value": 100}"#,
        r#"{"id": "2", "value": 200}"#,
        r#"{"value": 300}"#, // Missing required 'id'
    ];

    let results = validate_batch(&validator, &inputs);

    for (i, (valid, errors)) in results.iter().enumerate() {
        println!("Input {}: valid={}, errors={}", i + 1, valid, errors);
    }

    Ok(())
}
```

## Integration with Krino Pipeline

```rust
use krino::modules::schema::{SchemaConfig, SchemaValidator};
use krino::modules::hallucination::{HallucinationDetector, HallucinationConfig};
use std::sync::Arc;

struct LlmOutputValidator {
    schema_validator: SchemaValidator,
    hallucination_detector: HallucinationDetector,
}

impl LlmOutputValidator {
    fn new(
        schema: serde_json::Value,
        detector: HallucinationDetector,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            schema_validator: SchemaValidator::new(SchemaConfig {
                schema,
                ..Default::default()
            })?,
            hallucination_detector: detector,
        })
    }

    fn validate_all(&self, llm_output: &str) -> Result<ValidationReport, Box<dyn std::error::Error>> {
        // Step 1: Validate JSON structure
        let schema_result = self.schema_validator.validate(llm_output)?;

        if !schema_result.valid {
            return Ok(ValidationReport {
                schema_valid: false,
                schema_errors: schema_result.error_count,
                hallucination_score: None,
                overall_pass: false,
            });
        }

        // Step 2: Check for hallucinations
        let hallucination_result = self.hallucination_detector.detect(llm_output)?;

        Ok(ValidationReport {
            schema_valid: true,
            schema_errors: 0,
            hallucination_score: Some(hallucination_result.aggregate_score),
            overall_pass: hallucination_result.aggregate_score < 0.3,
        })
    }
}

#[derive(Debug)]
struct ValidationReport {
    schema_valid: bool,
    schema_errors: usize,
    hallucination_score: Option<f64>,
    overall_pass: bool,
}
```

## Error Handling Patterns

### Pattern 1: Early Return on Invalid Schema

```rust
fn process_llm_output(
    validator: &SchemaValidator,
    llm_output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = validator.validate(llm_output)?;

    if !result.valid {
        eprintln!("Validation failed with {} errors:", result.error_count);
        for error in &result.errors {
            eprintln!("  - {}: {}", error.path, error.message);
        }
        return Err("Invalid JSON structure".into());
    }

    // Continue processing valid JSON
    println!("Processing valid JSON...");
    Ok(())
}
```

### Pattern 2: Collect All Errors

```rust
use krino::modules::schema::ValidationError;

fn validate_and_report(
    validator: &SchemaValidator,
    llm_output: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let result = validator.validate(llm_output)?;

    let error_messages: Vec<String> = result
        .errors
        .iter()
        .map(|e| format!("{}: {}", e.path, e.message))
        .collect();

    if error_messages.is_empty() {
        Ok(vec!["✅ Validation passed".to_string()])
    } else {
        Ok(error_messages)
    }
}
```

### Pattern 3: Typed Error Responses

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum ValidationError {
    #[error("Schema validation failed: {0} errors")]
    SchemaInvalid(usize),

    #[error("JSON parsing failed: {0}")]
    ParseError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

fn validate_strict(
    validator: &SchemaValidator,
    llm_output: &str,
) -> Result<serde_json::Value, ValidationError> {
    let result = validator
        .validate(llm_output)
        .map_err(|e| ValidationError::Internal(e.to_string()))?;

    if !result.json_parse_success {
        return Err(ValidationError::ParseError(
            result.json_parse_error.unwrap_or_default(),
        ));
    }

    if !result.valid {
        return Err(ValidationError::SchemaInvalid(result.error_count));
    }

    Ok(result.parsed_json.unwrap())
}
```

## Performance Optimization

### Reuse Validators

```rust
use std::collections::HashMap;
use krino::modules::schema::SchemaValidator;

struct ValidatorCache {
    validators: HashMap<String, SchemaValidator>,
}

impl ValidatorCache {
    fn new() -> Self {
        Self {
            validators: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        schema_name: &str,
        schema: serde_json::Value,
    ) -> Result<&SchemaValidator, Box<dyn std::error::Error>> {
        if !self.validators.contains_key(schema_name) {
            let validator = SchemaValidator::new(SchemaConfig {
                schema,
                ..Default::default()
            })?;
            self.validators.insert(schema_name.to_string(), validator);
        }

        Ok(self.validators.get(schema_name).unwrap())
    }
}
```

### Parallel Validation

```rust
use rayon::prelude::*;
use krino::modules::schema::SchemaValidator;

fn validate_parallel(
    validator: &SchemaValidator,
    inputs: Vec<String>,
) -> Vec<bool> {
    inputs
        .par_iter()
        .map(|input| {
            validator
                .validate(input)
                .map(|r| r.valid)
                .unwrap_or(false)
        })
        .collect()
}
```

## Testing with Schema Validation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use krino::modules::schema::{SchemaConfig, SchemaValidator};

    fn create_test_validator() -> SchemaValidator {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["id"]
        });

        SchemaValidator::new(SchemaConfig {
            schema,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn test_valid_json() {
        let validator = create_test_validator();
        let result = validator.validate(r#"{"id": "test", "count": 5}"#).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_missing_required() {
        let validator = create_test_validator();
        let result = validator.validate(r#"{"count": 5}"#).unwrap();
        assert!(!result.valid);
        assert_eq!(result.error_count, 1);
    }

    #[test]
    fn test_wrong_type() {
        let validator = create_test_validator();
        let result = validator.validate(r#"{"id": "test", "count": "five"}"#).unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_determinism() {
        let validator = create_test_validator();
        let input = r#"{"id": "test", "count": 5}"#;

        let results: Vec<_> = (0..100)
            .map(|_| validator.validate(input).unwrap())
            .collect();

        // All results should be identical
        for result in &results[1..] {
            assert_eq!(result.valid, results[0].valid);
            assert_eq!(result.error_count, results[0].error_count);
        }
    }
}
```

## Custom Schema Configurations

### Strict Mode

```rust
let validator = SchemaValidator::new(SchemaConfig {
    schema,
    strict_mode: true, // Extra strictness
    max_depth: Some(5),
})?;
```

### Unlimited Nesting

```rust
let validator = SchemaValidator::new(SchemaConfig {
    schema,
    strict_mode: false,
    max_depth: None, // No depth limit
})?;
```

### Shallow Nesting Only

```rust
let validator = SchemaValidator::new(SchemaConfig {
    schema,
    strict_mode: false,
    max_depth: Some(3), // Max 3 levels deep
})?;
```

## Integration Examples

### Web API Validation

```rust
use axum::{Json, response::IntoResponse, http::StatusCode};
use krino::modules::schema::SchemaValidator;

async fn validate_request(
    validator: &SchemaValidator,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let json_str = serde_json::to_string(&payload).unwrap();

    match validator.validate(&json_str) {
        Ok(result) if result.valid => {
            (StatusCode::OK, Json(payload))
        }
        Ok(result) => {
            let errors: Vec<_> = result
                .errors
                .iter()
                .map(|e| format!("{}: {}", e.path, e.message))
                .collect();

            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "errors": errors }))
            )
        }
        Err(e) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() }))
            )
        }
    }
}
```

## See Also

- [CLI Examples](CLI_EXAMPLES.md) - Command-line usage
- [Main README](README.md) - Overview and concepts
- [Krino Documentation](../../README.md) - Full Krino guide
