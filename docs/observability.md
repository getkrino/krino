# Observability

How to know what Krino is doing in production.

## Logs

Set the log level and format in `[logging]` (see
[configuration.md](configuration.md#logging)):

```toml
[logging]
level     = "info"
format    = "json"
ort_level = "warn"
```

- **`json`** emits one JSON object per line, suitable for piping into
  CloudWatch, Loki, or any structured log store. This is the default.
- **`pretty`** emits ANSI-colored human-readable lines. Use this in dev
  only.

A typical evaluation produces one or two `info`-level lines per
request — the request itself and the result summary. Use `debug` to
also see per-stage timings, per-claim verdicts, and ONNX session
diagnostics. Use `trace` if you're chasing a specific bug.

### What every request logs

At `info`, every successful evaluation produces a line like:

```json
{
  "ts": "2026-06-05T18:14:22Z",
  "level": "INFO",
  "target": "krino_api::routes::evaluate",
  "msg": "evaluation complete",
  "claims": 7,
  "supported": 5,
  "score": 0.714,
  "latency_ms": 943,
  "engine_version": "0.1.0"
}
```

The exact field set varies by event; everything is tagged with a
`target` that names the Rust module emitting the log so you can
filter precisely.

### Startup diagnostics

At startup, the server logs (at `info`):

- CPU capabilities (AVX2, AVX-512, available cores).
- Worker pool configuration (`n_workers`, `threads_per_worker`,
  `queue_depth`).
- Model paths and successful load events.

These are useful for verifying that your deployed `krino-api.toml`
took effect. If your container is reporting `n_workers=8` but you
configured `n_workers=1`, the override didn't land — check your env
vars or the mounted config file.

## Metrics

Krino exposes Prometheus-format metrics at `GET /metrics` (no auth
required).

A minimal Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: krino
    scrape_interval: 15s
    static_configs:
      - targets: ['krino-api:8080']
```

### Metric names

| Metric | Type | Labels | Description |
|---|---|---|---|
| `http_requests_total` | counter | `method`, `path`, `status` | Total HTTP requests handled. |
| `http_request_duration_seconds` | histogram | `method`, `path` | Wall time per HTTP request. |
| `evaluations_total` | counter | `type` | Total successful evaluations. `type` is always `"faithfulness"` today. |
| `evaluation_duration_seconds` | histogram | `type` | Server-side wall time for the engine's `check_inner`. |
| `evaluation_claims` | histogram | `type` | Number of claims produced by the splitter per request. |
| `nli_inference_latency_seconds` | histogram | none | Sum of NLI batch-inference time per evaluation. |

The default Prometheus histogram buckets are appropriate for the
expected latency range (10ms to 10s).

### What to alert on

Suggested starting alerts:

- `rate(http_requests_total{status=~"5.."}[5m]) > 0.01` — sustained
  5xx rate above 1%.
- `histogram_quantile(0.95, rate(evaluation_duration_seconds_bucket[5m])) > 3.0`
  — 95th-percentile evaluation latency above 3 seconds.
- `rate(http_requests_total{status="503"}[5m]) > 0` — any sustained
  back-pressure (workers saturated).

## Per-request timing fields

Every `/api/v1/evaluate` response carries the engine's own
self-measured breakdown in `meta`:

```json
{
  "meta": {
    "latency_ms": 943,
    "split_ms": 1.2,
    "embedding_ms": 762,
    "nli_ms": 178,
    "inference_calls": 18,
    "engine_confidence": 0.86
  }
}
```

| Field | Meaning |
|---|---|
| `latency_ms` | Total server-side wall time inside the engine. |
| `split_ms` | Time spent splitting context and output into sentences. Should be sub-millisecond on typical inputs. |
| `embedding_ms` | Time spent computing claim and context embeddings for the pre-filter. Scales primarily with context size. |
| `nli_ms` | Time spent in NLI batch inference. Scales with number of `(claim, candidate)` pairs (i.e., `claims × top_k`). |
| `inference_calls` | Number of NLI forward passes. |
| `engine_confidence` | Fraction of claims with a decisive verdict (entailment / contradiction / partial — anything except neutral). A low value means the engine wasn't sure, and the `score` should be read with caution. |

`embedding_ms + nli_ms + split_ms` should approximately equal
`latency_ms`. The remainder is verdict assembly, response
serialization, and Rayon-pool overhead — usually under 5ms.

## Tracing what was decided and why

For *what* the engine decided, set `config.include_matrix: true` on
the request. The response then carries the full
`(claim, context_sentence)` matrix with per-pair entailment, neutral,
and contradiction probabilities, plus pre-filter similarity scores.

This is exactly what to look at when a verdict surprises you. See
[api-reference.md](api-reference.md) for the matrix structure.
