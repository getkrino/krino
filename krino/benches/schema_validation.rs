use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use krino::modules::schema::{SchemaConfig, SchemaValidator};

fn create_simple_validator() -> SchemaValidator {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
            }
        },
        "required": ["name", "age"]
    });

    SchemaValidator::new(SchemaConfig {
        schema,
        strict_mode: false,
        max_depth: Some(10),
    })
    .unwrap()
}

fn create_complex_validator() -> SchemaValidator {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "age": {"type": "number", "minimum": 0, "maximum": 120}
                        },
                        "required": ["name"]
                    },
                    "settings": {
                        "type": "object",
                        "properties": {
                            "theme": {"type": "string", "enum": ["light", "dark"]},
                            "notifications": {"type": "boolean"}
                        }
                    }
                },
                "required": ["profile"]
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10
            }
        },
        "required": ["user"]
    });

    SchemaValidator::new(SchemaConfig {
        schema,
        strict_mode: false,
        max_depth: Some(10),
    })
    .unwrap()
}

fn bench_simple_valid(c: &mut Criterion) {
    let validator = create_simple_validator();
    let json = r#"{"name": "Alice", "age": 30, "email": "alice@example.com"}"#;

    let mut group = c.benchmark_group("schema_validation_simple");
    group.throughput(Throughput::Bytes(json.len() as u64));

    group.bench_function("valid", |b| {
        b.iter(|| {
            let result = validator.validate(black_box(json)).unwrap();
            assert!(result.valid);
        });
    });

    group.finish();
}

fn bench_simple_invalid(c: &mut Criterion) {
    let validator = create_simple_validator();
    let json = r#"{"name": "Bob"}"#; // Missing required 'age'

    let mut group = c.benchmark_group("schema_validation_simple");
    group.throughput(Throughput::Bytes(json.len() as u64));

    group.bench_function("invalid_missing_field", |b| {
        b.iter(|| {
            let result = validator.validate(black_box(json)).unwrap();
            assert!(!result.valid);
        });
    });

    group.finish();
}

fn bench_simple_wrong_type(c: &mut Criterion) {
    let validator = create_simple_validator();
    let json = r#"{"name": "Charlie", "age": "thirty"}"#; // Wrong type

    let mut group = c.benchmark_group("schema_validation_simple");
    group.throughput(Throughput::Bytes(json.len() as u64));

    group.bench_function("invalid_wrong_type", |b| {
        b.iter(|| {
            let result = validator.validate(black_box(json)).unwrap();
            assert!(!result.valid);
        });
    });

    group.finish();
}

fn bench_complex_valid(c: &mut Criterion) {
    let validator = create_complex_validator();
    let json = r#"{
        "user": {
            "profile": {
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "age": 30
            },
            "settings": {
                "theme": "dark",
                "notifications": true
            }
        },
        "tags": ["admin", "developer", "reviewer"]
    }"#;

    let mut group = c.benchmark_group("schema_validation_complex");
    group.throughput(Throughput::Bytes(json.len() as u64));

    group.bench_function("valid", |b| {
        b.iter(|| {
            let result = validator.validate(black_box(json)).unwrap();
            assert!(result.valid);
        });
    });

    group.finish();
}

fn bench_complex_nested_invalid(c: &mut Criterion) {
    let validator = create_complex_validator();
    let json = r#"{
        "user": {
            "profile": {},
            "settings": {"theme": "dark"}
        },
        "tags": ["admin"]
    }"#; // Missing required 'name' in nested profile

    let mut group = c.benchmark_group("schema_validation_complex");
    group.throughput(Throughput::Bytes(json.len() as u64));

    group.bench_function("invalid_nested", |b| {
        b.iter(|| {
            let result = validator.validate(black_box(json)).unwrap();
            assert!(!result.valid);
        });
    });

    group.finish();
}

fn bench_parse_error(c: &mut Criterion) {
    let validator = create_simple_validator();
    let json = r#"{"name": "Alice", "age": 30"#; // Malformed JSON

    let mut group = c.benchmark_group("schema_validation_errors");

    group.bench_function("json_parse_error", |b| {
        b.iter(|| {
            let result = validator.validate(black_box(json)).unwrap();
            assert!(!result.valid);
            assert!(!result.json_parse_success);
        });
    });

    group.finish();
}

fn bench_varying_sizes(c: &mut Criterion) {
    let validator = create_simple_validator();

    let mut group = c.benchmark_group("schema_validation_sizes");

    for size in [10, 50, 100, 500, 1000].iter() {
        // Generate JSON with varying number of fields
        let mut json_obj = serde_json::json!({
            "name": "Test User",
            "age": 30
        });

        if let Some(obj) = json_obj.as_object_mut() {
            for i in 0..*size {
                obj.insert(format!("field{i}"), serde_json::json!(format!("value{i}")));
            }
        }

        let json = serde_json::to_string(&json_obj).unwrap();

        group.throughput(Throughput::Bytes(json.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = validator.validate(black_box(&json)).unwrap();
                assert!(result.valid);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_valid,
    bench_simple_invalid,
    bench_simple_wrong_type,
    bench_complex_valid,
    bench_complex_nested_invalid,
    bench_parse_error,
    bench_varying_sizes
);

criterion_main!(benches);
