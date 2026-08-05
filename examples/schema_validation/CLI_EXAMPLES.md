# Schema Validation CLI Examples

This guide shows practical examples of using the Krino CLI to validate JSON against schemas.

## Basic Usage

### Validate JSON String

```bash
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --json '{"name": "Alice", "age": 30}'
```

**Output:**
```
📋 Loading schema from: examples/schema_validation/user_schema.json
✅ Schema loaded successfully

🧪 Validating JSON (27 chars)...

📊 Results:
  Valid: ✅ Yes
  JSON Parse Success: true
  Nesting Depth: 1
  Error Count: 0
  Latency: 0.12ms

✅ JSON is valid according to schema!
```

### Validate JSON File

```bash
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --file examples/schema_validation/test_data/user_valid.json
```

### Get JSON Output

```bash
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --file examples/schema_validation/test_data/user_valid.json \
  --output json
```

**Output:**
```json
{
  "valid": true,
  "errors": [],
  "error_count": 0,
  "parsed_json": {
    "name": "Alice Johnson",
    "age": 30,
    "email": "alice.johnson@example.com"
  },
  "json_parse_success": true,
  "json_parse_error": null,
  "nesting_depth": 2,
  "latency_ms": 0.15
}
```

## Testing Invalid JSON

### Missing Required Field

```bash
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --file examples/schema_validation/test_data/user_invalid_missing_required.json
```

**Output:**
```
📋 Loading schema from: examples/schema_validation/user_schema.json
✅ Schema loaded successfully

🧪 Validating JSON (48 chars)...

📊 Results:
  Valid: ❌ No
  JSON Parse Success: true
  Nesting Depth: 1
  Error Count: 1
  Latency: 0.09ms

❌ Found 1 validation error(s):

  1. Path: /
     Error: "age" is a required property
     Kind: Required { property: "age" }
```

### Wrong Data Type

```bash
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --file examples/schema_validation/test_data/user_invalid_wrong_type.json
```

**Output:**
```
📊 Results:
  Valid: ❌ No
  JSON Parse Success: true
  Nesting Depth: 1
  Error Count: 1
  Latency: 0.11ms

❌ Found 1 validation error(s):

  1. Path: /age
     Error: "thirty-five" is not of type "number"
     Kind: Type { kind: Single(Number) }
```

### Malformed JSON

```bash
krino validate-schema \
  --schema examples/schema_validation/user_schema.json \
  --json '{"name": "Alice", "age": 30'  # Missing closing brace
```

**Output:**
```
❌ JSON Parsing Failed:
   EOF while parsing an object at line 1 column 29
```

## Advanced Examples

### Custom Max Nesting Depth

Prevent deeply nested malicious payloads:

```bash
krino validate-schema \
  --schema examples/schema_validation/config_schema.json \
  --file config.json \
  --max-depth 5
```

### Unlimited Nesting Depth

```bash
krino validate-schema \
  --schema schema.json \
  --file data.json \
  --max-depth 0  # 0 = unlimited
```

## Real-World Workflows

### 1. Validate LLM Function Call Output

```bash
# Prompt LLM to generate function call
llm_output=$(curl -X POST https://api.example.com/llm \
  -d '{"prompt": "Search for wireless headphones under $200"}')

# Validate the output
echo "$llm_output" | krino validate-schema \
  --schema examples/schema_validation/function_call_schema.json \
  --json "$llm_output"
```

### 2. Validate Product Listings

```bash
# Generate product listing with LLM
krino validate-schema \
  --schema examples/schema_validation/product_schema.json \
  --file generated_product.json
```

### 3. Pipeline Integration

```bash
#!/bin/bash
# validate_and_process.sh

SCHEMA="examples/schema_validation/user_schema.json"
INPUT="llm_output.json"

# Validate JSON structure
if krino validate-schema --schema "$SCHEMA" --file "$INPUT" --output json | jq -e '.valid' > /dev/null; then
  echo "✅ Validation passed, processing..."
  # Run other Krino evaluations
  krino eval-hallucination --file "$INPUT" --model-path ./models/modernbert
else
  echo "❌ Validation failed, aborting"
  exit 1
fi
```

### 4. Batch Validation

```bash
#!/bin/bash
# Validate multiple files

for file in outputs/*.json; do
  echo "Validating $file..."
  krino validate-schema \
    --schema user_schema.json \
    --file "$file" \
    --output json > "results/$(basename $file)"
done
```

### 5. CI/CD Integration

```yaml
# .github/workflows/validate-llm-outputs.yml
name: Validate LLM Outputs

on: [push]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Krino
        run: cargo install krino --features cli

      - name: Validate All Outputs
        run: |
          for schema in schemas/*.json; do
            for output in outputs/*.json; do
              krino validate-schema \
                --schema "$schema" \
                --file "$output"
            done
          done
```

## Error Handling

### Check Exit Code

```bash
krino validate-schema --schema schema.json --file data.json

if [ $? -eq 0 ]; then
  echo "Validation successful"
else
  echo "Validation failed"
fi
```

### Capture JSON Output

```bash
result=$(krino validate-schema \
  --schema schema.json \
  --file data.json \
  --output json)

# Check if valid
if echo "$result" | jq -e '.valid' > /dev/null; then
  echo "Valid!"
else
  # Extract error details
  echo "$result" | jq '.errors'
fi
```

## Performance Testing

### Single Validation

```bash
time krino validate-schema \
  --schema examples/schema_validation/product_schema.json \
  --file examples/schema_validation/test_data/product_valid.json
```

### Batch Performance

```bash
#!/bin/bash
# Test 1000 validations

start=$(date +%s%N)
for i in {1..1000}; do
  krino validate-schema \
    --schema user_schema.json \
    --json '{"name": "User", "age": 30}' \
    > /dev/null
done
end=$(date +%s%N)

elapsed=$(( ($end - $start) / 1000000 ))
echo "1000 validations in ${elapsed}ms"
echo "Average: $(( $elapsed / 1000 ))ms per validation"
```

## Troubleshooting

### Schema Not Found

```bash
$ krino validate-schema --schema missing.json --json '{}'
Error: No such file or directory (os error 2)
```

**Solution:** Check the schema file path is correct.

### Both --json and --file Specified

```bash
$ krino validate-schema --schema schema.json --json '{}' --file data.json
Error: Cannot specify both --json and --file
```

**Solution:** Use only one input source.

### Neither --json nor --file Specified

```bash
$ krino validate-schema --schema schema.json
Error: Must specify either --json or --file
```

**Solution:** Provide input via --json or --file.

## Tips & Tricks

### 1. Pretty Print JSON Output

```bash
krino validate-schema \
  --schema schema.json \
  --file data.json \
  --output json | jq '.'
```

### 2. Extract Only Errors

```bash
krino validate-schema \
  --schema schema.json \
  --file data.json \
  --output json | jq '.errors'
```

### 3. Count Errors

```bash
krino validate-schema \
  --schema schema.json \
  --file data.json \
  --output json | jq '.error_count'
```

### 4. Validate Multiple Files Against Same Schema

```bash
for file in *.json; do
  krino validate-schema --schema schema.json --file "$file"
done
```

### 5. Create Validation Report

```bash
#!/bin/bash
echo "# Validation Report" > report.md
echo "Generated: $(date)" >> report.md
echo "" >> report.md

for file in outputs/*.json; do
  result=$(krino validate-schema --schema schema.json --file "$file" --output json)
  valid=$(echo "$result" | jq -r '.valid')
  errors=$(echo "$result" | jq -r '.error_count')

  echo "## $(basename $file)" >> report.md
  echo "- Valid: $valid" >> report.md
  echo "- Errors: $errors" >> report.md
  echo "" >> report.md
done
```

## See Also

- [Main README](README.md) - Overview and concepts
- [JSON Schema Docs](https://json-schema.org/) - Schema specification
- [Krino Documentation](../../README.md) - Full Krino guide
