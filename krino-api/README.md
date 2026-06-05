# Krino API

HTTP API service for the Krino evaluation engine. Exposes groundedness checking as a REST API with authentication, rate limiting, and metrics.

## Quick Start

### Local Development

```bash
# Build the workspace
cargo build --release --package krino-api

# Run with default config
./target/release/krino-api

# Or with custom config
KRINO_SERVER__PORT=9000 ./target/release/krino-api
```

### Docker

```bash
# Build image
docker build -t krino-api .

# Run container
docker run -p 8080:8080 krino-api

# Or use docker-compose
docker-compose up
```

## API Reference

### Authentication

All API endpoints (except `/health` and `/metrics`) require authentication via API key.

**Headers:**
```
x-api-key: sk-krino-your-api-key
```

Or:
```
Authorization: Bearer sk-krino-your-api-key
```

### Endpoints

#### `GET /health`

Liveness probe. Returns 200 if the service is running.

**Response:**
```json
{
  "status": "ok",
  "version": "0.8.2"
}
```

#### `GET /health/ready`

Readiness probe. Returns 200 if models are loaded and service is ready.

**Response:**
```json
{
  "status": "ready",
  "model_loaded": true
}
```

#### `GET /metrics`

Prometheus metrics endpoint.

#### `POST /api/v1/evaluate/groundedness`

Evaluate groundedness/faithfulness of an LLM output against context.

**Request:**
```json
{
  "context": [
    {
      "id": "chunk-1",
      "text": "The Eiffel Tower was completed in 1889..."
    },
    {
      "id": "chunk-2",
      "text": "It stands 330 meters tall..."
    }
  ],
  "output": "The Eiffel Tower was completed in 1889 and is 330 meters tall.",
  "config": {
    "threshold": 0.7,
    "evidence_granularity": "sentence"
  }
}
```

**Response:**
```json
{
  "faithfulness_score": 0.95,
  "pass": true,
  "claims": [
    {
      "text": "The Eiffel Tower was completed in 1889",
      "verdict": "entailed",
      "score": 0.98,
      "evidence": {
        "chunk_id": "chunk-1",
        "text": "The Eiffel Tower was completed in 1889...",
        "entailment_prob": 0.98,
        "contradiction_prob": 0.01
      },
      "is_compound": false
    }
  ],
  "meta": {
    "model": "krino-groundedness-v1",
    "latency_ms": 125.4,
    "nli_calls": 3,
    "claims_evaluated": 2,
    "engine_version": "0.8.2"
  }
}
```

**Error Response:**
```json
{
  "error": {
    "code": "bad_request",
    "message": "context must not be empty"
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad request (invalid input)
- `401` - Unauthorized (missing or invalid API key)
- `429` - Rate limit exceeded
- `500` - Internal server error

## Configuration

Configuration is loaded from `krino-api.toml` and can be overridden with environment variables using the `KRINO_` prefix.

**Example:**
```bash
# Override port
export KRINO_SERVER__PORT=9000

# Override log level
export KRINO_LOGGING__LEVEL=debug

# Override API key
export KRINO_AUTH__API_KEYS__0__KEY=sk-krino-new-key
```

### Configuration Options

```toml
[server]
port = 8080
blocking_threads = 4

[models]
nli_model_path = "models/deberta-nli-onnx"

[groundedness]
default_top_k = 10
default_threshold = 0.7
max_context_chars = 500000
max_output_chars = 50000

[auth]
[[auth.api_keys]]
customer_id = "demo"
key = "sk-krino-demo-key"

[rate_limit]
requests_per_minute = 60
burst = 10

[logging]
level = "info"
format = "json"  # or "pretty"
ort_level = "warn"
```

## Rate Limiting

The API uses a token bucket algorithm for rate limiting:
- **Default:** 60 requests/minute
- **Burst:** 10 requests
- **Per-customer:** Rate limits are applied per API key

When rate limited, you'll receive a `429` response with a `Retry-After` header.

## Metrics

Prometheus metrics are available at `/metrics`:

**Key Metrics:**
- `http_requests_total` - Total HTTP requests (by method, path, status)
- `http_request_duration_seconds` - Request latency histogram
- `evaluations_total` - Total evaluations (by type)
- `evaluation_duration_seconds` - Evaluation latency histogram
- `evaluation_claims` - Number of claims per evaluation

## Deployment

### AWS EC2

```bash
# Launch c7a.xlarge (AVX-512 optimized)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type c7a.xlarge \
    --key-name krino-prod \
    --security-group-ids sg-xxx \
    --subnet-id subnet-xxx

# On the instance
docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/krino-api:latest
docker run -d \
    --name krino-api \
    -p 8080:8080 \
    -e KRINO_AUTH__API_KEYS__0__KEY=sk-prod-xxx \
    --restart unless-stopped \
    krino-api:latest
```

### AWS ECS/Fargate

See deployment guide in `/scripts/` for ECS task definitions and ALB configuration.

## Development

### Project Structure

```
krino-api/
├── src/
│   ├── main.rs          # Entry point
│   ├── server.rs        # Axum router
│   ├── state.rs         # App state (models, config)
│   ├── config.rs        # Configuration types
│   ├── error.rs         # Error handling
│   ├── auth.rs          # API key authentication
│   ├── rate_limit.rs    # Token bucket rate limiter
│   ├── metrics.rs       # Prometheus metrics
│   ├── middleware.rs    # Request middleware
│   └── routes/
│       ├── health.rs    # Health endpoints
│       └── groundedness.rs  # Groundedness evaluation
└── Cargo.toml
```

### Testing

```bash
# Run unit tests
cargo test --package krino-api

# Integration test with curl
curl -X POST http://localhost:8080/api/v1/evaluate/groundedness \
  -H "x-api-key: sk-krino-demo-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "context": [{"text": "Paris is the capital of France."}],
    "output": "Paris is the capital of France."
  }'
```

## Performance

**Target Metrics:**
- Single groundedness evaluation: <200ms
- Throughput: ~130 evaluations/minute (single c7a.xlarge)
- Memory: ~500MB (with loaded models)

**Optimization:**
- ONNX INT8 quantization
- BM25 pre-filtering for context selection
- AVX-512 SIMD instructions (on c7a instances)
- Tokio `spawn_blocking` for CPU-bound inference

## Security

- API keys are hashed with SHA256
- Constant-time comparison to prevent timing attacks
- Rate limiting per customer
- Input size limits (500K context, 50K output)
- CORS can be tightened in production

## Troubleshooting

**Model not found:**
```
Error: Models not found at path: models/deberta-nli-onnx
```
Download models or update `nli_model_path` in config.

**Out of memory:**
Reduce `blocking_threads` or increase instance RAM.

**Slow inference:**
Ensure AVX-512 is available (`/health` logs CPU capabilities).

## License

UNLICENSED - Proprietary software
