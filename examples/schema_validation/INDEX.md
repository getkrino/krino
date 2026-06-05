# Schema Validation Examples - Index

Complete documentation for Krino JSON Schema validation.

## 📚 Documentation

### Getting Started
- **[README.md](README.md)** - Overview, concepts, and best practices
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Cheat sheet for common patterns

### Usage Guides
- **[CLI_EXAMPLES.md](CLI_EXAMPLES.md)** - Command-line usage with real examples
- **[LIBRARY_EXAMPLES.md](LIBRARY_EXAMPLES.md)** - Rust library integration examples

## 📋 Example Schemas

### Basic Schemas
- **[user_schema.json](user_schema.json)** - User profile validation
  - Required fields, email/phone patterns, nested address
  - Use case: User registration, profile updates

- **[strict_schema.json](strict_schema.json)** - API response validation
  - No additional properties allowed
  - Use case: Strict API contract enforcement

### Advanced Schemas
- **[function_call_schema.json](function_call_schema.json)** - LLM function calling
  - Enum validation for function names
  - Use case: Tool use, function calling agents

- **[product_schema.json](product_schema.json)** - E-commerce products
  - Complex constraints, arrays, nested inventory
  - Use case: Product catalog generation

- **[config_schema.json](config_schema.json)** - Application configuration
  - Deeply nested objects, multiple validation rules
  - Use case: Config file generation, system settings

## 🧪 Test Data

Located in `test_data/`:

### Valid Examples
- `user_valid.json` - Complete valid user profile
- `product_valid.json` - Complete valid product listing
- `function_call_valid.json` - Valid function call

### Invalid Examples (for testing)
- `user_invalid_missing_required.json` - Missing required 'age' field
- `user_invalid_wrong_type.json` - Wrong type for 'age' (string instead of number)

## 🚀 Quick Start

### CLI
```bash
# Validate a valid example
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --file examples/schema_validation/test_data/user_valid.json

# Test with invalid data
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --file examples/schema_validation/test_data/user_invalid_missing_required.json
```

### Library
```rust
use krino::modules::schema::{SchemaConfig, SchemaValidator};

let schema = std::fs::read_to_string("examples/schema_validation/user_schema.json")?;
let schema_json: serde_json::Value = serde_json::from_str(&schema)?;

let validator = SchemaValidator::new(SchemaConfig {
    schema: schema_json,
    ..Default::default()
})?;

let result = validator.validate(llm_output)?;
println!("Valid: {}", result.valid);
```

## 📖 Learning Path

1. **Start Here**: [README.md](README.md) - Understand concepts
2. **Try Examples**: Use CLI with provided test data
3. **Learn Patterns**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. **Integrate**: [LIBRARY_EXAMPLES.md](LIBRARY_EXAMPLES.md)
5. **Advanced**: [CLI_EXAMPLES.md](CLI_EXAMPLES.md) - Pipelines, CI/CD

## 🎯 Use Case Index

### LLM Function Calling
- Schema: `function_call_schema.json`
- Example: `test_data/function_call_valid.json`
- Guide: [README.md#common-use-cases](README.md#common-use-cases)

### User Data Validation
- Schema: `user_schema.json`
- Examples: `test_data/user_*.json`
- Guide: [CLI_EXAMPLES.md#testing-invalid-json](CLI_EXAMPLES.md#testing-invalid-json)

### E-Commerce
- Schema: `product_schema.json`
- Example: `test_data/product_valid.json`
- Guide: [LIBRARY_EXAMPLES.md#validating-llm-function-calls](LIBRARY_EXAMPLES.md#validating-llm-function-calls)

### Configuration Generation
- Schema: `config_schema.json`
- Guide: [README.md#nested-objects](README.md#nested-objects)

### Strict API Validation
- Schema: `strict_schema.json`
- Guide: [README.md#strict-mode-example](README.md#strict-mode-example)

## 🔍 Common Tasks

| Task | File | Section |
|------|------|---------|
| Validate from command line | CLI_EXAMPLES.md | Basic Usage |
| Integrate in Rust app | LIBRARY_EXAMPLES.md | Quick Start |
| Test edge cases | CLI_EXAMPLES.md | Testing Invalid JSON |
| Batch validation | LIBRARY_EXAMPLES.md | Batch Validation |
| CI/CD integration | CLI_EXAMPLES.md | CI/CD Integration |
| Error handling | LIBRARY_EXAMPLES.md | Error Handling Patterns |
| Performance testing | CLI_EXAMPLES.md | Performance Testing |

## 📊 Validation Types Covered

- ✅ Type validation (string, number, boolean, array, object)
- ✅ Required fields
- ✅ Pattern matching (regex)
- ✅ Enum values
- ✅ Number constraints (min, max, exclusive)
- ✅ String constraints (minLength, maxLength)
- ✅ Array constraints (items, minItems, maxItems, uniqueItems)
- ✅ Nested object validation
- ✅ Additional properties control
- ✅ Nesting depth limits

## 🧩 File Organization

```
examples/schema_validation/
│
├── 📘 Documentation
│   ├── INDEX.md (this file)
│   ├── README.md
│   ├── QUICK_REFERENCE.md
│   ├── CLI_EXAMPLES.md
│   └── LIBRARY_EXAMPLES.md
│
├── 📋 Schemas
│   ├── user_schema.json
│   ├── function_call_schema.json
│   ├── product_schema.json
│   ├── config_schema.json
│   └── strict_schema.json
│
└── 🧪 Test Data
    └── test_data/
        ├── user_valid.json
        ├── user_invalid_missing_required.json
        ├── user_invalid_wrong_type.json
        ├── product_valid.json
        └── function_call_valid.json
```

## ⚡ Performance Characteristics

- **Latency**: <1ms typical
- **Throughput**: >10,000 validations/sec
- **Determinism**: 100% (verified with 100-iteration tests)
- **Memory**: Minimal allocations
- **Startup**: Instant (no model loading)

## 🔗 Related Documentation

- [Krino Main Documentation](../../README.md)
- [Hallucination Detection](../hallucination/)
- [Groundedness Checking](../groundedness/)
- [JSON Schema Specification](https://json-schema.org/)

## 💡 Tips

1. **Start simple** - Begin with basic schemas, add constraints incrementally
2. **Test both valid and invalid** - Verify schema catches errors
3. **Use examples** - Schema examples field helps document expected structure
4. **Version schemas** - Keep schemas in version control
5. **Validate early** - Check structure before processing
6. **Reuse validators** - Create once, validate many times
7. **Check error paths** - Use error.path to identify exactly what failed

## 🐛 Troubleshooting

See [README.md#troubleshooting](README.md#troubleshooting) for:
- JSON parsing errors
- Missing required fields
- Type mismatches
- Pattern validation failures
- Common pitfalls

## 📝 License

Apache 2.0 - Same as Krino
