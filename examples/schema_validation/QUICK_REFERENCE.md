# Schema Validation Quick Reference

## CLI Commands

```bash
# Validate JSON string
krino validate-schema --schema SCHEMA.json --json '{"key": "value"}'

# Validate JSON file
krino validate-schema --schema SCHEMA.json --file INPUT.json

# JSON output format
krino validate-schema --schema SCHEMA.json --file INPUT.json --output json

# Custom max depth
krino validate-schema --schema SCHEMA.json --file INPUT.json --max-depth 5
```

## Library Usage

```rust
use krino::modules::schema::{SchemaConfig, SchemaValidator};

// Create validator
let validator = SchemaValidator::new(SchemaConfig {
    schema: serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }),
    strict_mode: false,
    max_depth: Some(10),
})?;

// Validate
let result = validator.validate(r#"{"name": "Alice"}"#)?;

// Check result
if result.valid {
    println!("✅ Valid");
} else {
    for error in &result.errors {
        println!("❌ {}: {}", error.path, error.message);
    }
}
```

## Common Schema Patterns

### Required Fields
```json
{
  "type": "object",
  "properties": {
    "id": {"type": "string"}
  },
  "required": ["id"]
}
```

### Type Validation
```json
{
  "name": {"type": "string"},
  "age": {"type": "number"},
  "active": {"type": "boolean"},
  "tags": {"type": "array"},
  "config": {"type": "object"}
}
```

### String Constraints
```json
{
  "email": {
    "type": "string",
    "pattern": "^[^@]+@[^@]+\\.[^@]+$"
  },
  "status": {
    "type": "string",
    "enum": ["active", "inactive"]
  }
}
```

### Number Constraints
```json
{
  "age": {
    "type": "number",
    "minimum": 0,
    "maximum": 120
  },
  "price": {
    "type": "number",
    "minimum": 0,
    "exclusiveMinimum": true
  }
}
```

### Array Validation
```json
{
  "tags": {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10,
    "uniqueItems": true
  }
}
```

### Nested Objects
```json
{
  "user": {
    "type": "object",
    "properties": {
      "name": {"type": "string"}
    },
    "required": ["name"]
  }
}
```

### Additional Properties
```json
{
  "type": "object",
  "properties": {
    "id": {"type": "string"}
  },
  "additionalProperties": false  // No extra fields
}
```

## Result Fields

```rust
pub struct SchemaValidationResult {
    pub valid: bool,              // Overall pass/fail
    pub errors: Vec<ValidationError>,  // List of errors
    pub error_count: usize,       // Total errors
    pub parsed_json: Option<Value>,    // Parsed JSON (if successful)
    pub json_parse_success: bool, // Whether parsing succeeded
    pub json_parse_error: Option<String>,  // Parse error message
    pub nesting_depth: usize,     // Actual nesting depth
    pub latency_ms: f64,          // Validation time
}
```

## Error Fields

```rust
pub struct ValidationError {
    pub path: String,    // JSON path (e.g., "/user/age")
    pub message: String, // Error description
    pub kind: String,    // Error type (Required, Type, etc.)
}
```

## Configuration Options

```rust
pub struct SchemaConfig {
    pub schema: serde_json::Value,  // JSON Schema spec
    pub strict_mode: bool,           // Extra strictness
    pub max_depth: Option<usize>,    // Max nesting (None = unlimited)
}
```

## Common Error Types

| Kind | Description | Example |
|------|-------------|---------|
| `Required` | Missing required field | `"age" is a required property` |
| `Type` | Wrong data type | `"30" is not of type "number"` |
| `Pattern` | Regex pattern mismatch | `"abc" does not match pattern` |
| `Enum` | Value not in enum | `"unknown" is not one of ["active", "inactive"]` |
| `Minimum` | Below minimum | `"-5" is less than minimum 0` |
| `Maximum` | Above maximum | `"200" is greater than maximum 120` |
| `MinItems` | Too few array items | `[] has less than minimum 1 items` |
| `MaxItems` | Too many array items | Array exceeds maximum |
| `parse_error` | Invalid JSON syntax | `EOF while parsing` |
| `max_depth_exceeded` | Too deeply nested | `Nesting depth 11 exceeds maximum 10` |

## Typical Workflow

1. **Define Schema** - Create JSON Schema file
2. **Load Validator** - Create `SchemaValidator` with schema
3. **Validate Output** - Call `validate()` with LLM output
4. **Check Result** - If `result.valid == false`, inspect `result.errors`
5. **Process or Reject** - Continue if valid, else handle errors

## Performance

- **Latency:** <1ms typical
- **Throughput:** >10,000 validations/sec
- **Determinism:** 100% (same input = same output)
- **No models:** Pure rule-based validation

## Example Files

```
examples/schema_validation/
├── README.md                    # Full documentation
├── CLI_EXAMPLES.md             # CLI usage examples
├── LIBRARY_EXAMPLES.md         # Rust library examples
├── QUICK_REFERENCE.md          # This file
├── user_schema.json            # User profile schema
├── function_call_schema.json   # LLM function call schema
├── product_schema.json         # E-commerce product schema
├── config_schema.json          # Nested config schema
├── strict_schema.json          # Strict mode example
└── test_data/
    ├── user_valid.json
    ├── user_invalid_*.json
    ├── product_valid.json
    └── function_call_valid.json
```

## Testing Your Schema

```bash
# Valid case (should pass)
krino validate-schema --schema user_schema.json --json '{"name": "Alice", "age": 30}'

# Invalid case (should fail)
krino validate-schema --schema user_schema.json --json '{"name": "Alice"}'
```

## Resources

- [JSON Schema Official Docs](https://json-schema.org/)
- [Schema Validator Online](https://www.jsonschemavalidator.net/)
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/)
- [Krino Documentation](../../README.md)
