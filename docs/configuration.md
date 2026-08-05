# Configuration reference

Every field in `krino-api.toml`, with what to set it to and why. The
example file at the repository root (`krino-api.toml.example`) carries
short inline comments; this page goes into the rationale.

Every field can be overridden by an environment variable using the
double-underscore notation for nesting:

```bash
KRINO_SERVER__PORT=9000 krino-api
KRINO_FAITHFULNESS__DEFAULT_TOP_K=5 krino-api
```

## `[server]`

```toml
[server]
port = 8080
```

| Field | Default | Description |
|---|---|---|
| `port` | `8080` | HTTP port the API listens on. |

## `[models]`

```toml
[models]
nli_model_path       = "models/roberta-large-mnli-static-int8"
embedding_model_path = "models/all-MiniLM-L6-v2-onnx-quantized"
```

| Field | Default | Description |
|---|---|---|
| `nli_model_path` | (required) | Path to the NLI ONNX model directory. Must contain `model.onnx`, `tokenizer.json`, and `config.json`. |
| `embedding_model_path` | (required) | Path to the embedding ONNX model directory. Used for the candidate pre-filter. |

Defaults match what `scripts/download_models.sh` produces. If you keep
models elsewhere, update both paths.

## `[faithfulness]`

```toml
[faithfulness]
default_top_k             = 10
default_threshold         = 0.7
contradiction_threshold   = 0.5
min_similarity_threshold  = 0.25
adaptive_top_k            = false
max_context_chars         = 500000
max_output_chars          = 50000
available_granularities   = ["claim"]
```

### `default_top_k`

Number of context-sentence candidates evaluated per claim by NLI after
the embedding pre-filter. Higher = more NLI calls = more compute, but
also more chance of finding the right evidence.

10 is the calibrated default for RoBERTa-large-MNLI INT8 with
`all-MiniLM-L6-v2` as the pre-filter. Cutting to 5 saves about 50% of
NLI cost, but for compound claims that span multiple facts, the
*supporting* sentences can sit at ranks 6–13 by embedding similarity —
because the embedding model ranks lexical-topical overlap rather than
semantic relevance. Cutting top-K too aggressively silently regresses
the partial-verdict path.

If your traffic is mostly short, non-compound claims, 5 is fine.

### `default_threshold`

The minimum overall `score` for the API to mark a response as
`pass: true`. A request can override this per-call via
`config.threshold`.

### `contradiction_threshold`

Minimum NLI contradiction probability for the engine to declare a
verdict of `contradiction` (vs. neutral).

0.5 is calibrated for RoBERTa-large-MNLI INT8. FP32 models produce
much sharper contradiction probabilities (often 0.95+) and tolerate
higher thresholds; INT8 quantization softens the distribution to
roughly 0.55–0.6 on real contradictions, so 0.7 (a previous default)
silently demoted real contradictions to neutral.

If you swap to a different NLI backend, recalibrate this with a few
known-contradiction probes through `/api/v1/evaluate` with
`include_matrix: true` to see actual contradiction probabilities.

### `min_similarity_threshold`

The embedding pre-filter discards any context sentence with cosine
similarity below this floor *before* sending the survivors to NLI.

Raising this cuts NLI calls but risks dropping borderline-relevant
context. 0.25 is the production-tuned default.

### `adaptive_top_k`

When true, per-claim top-K scales with claim length:
`clamp(ceil(claim_chars / 40), 3, default_top_k)`.

Off by default. The intent was to save NLI calls on short claims, but
in practice short claims sometimes had a single decisive evidence
sentence buried at rank 4 or 5, and the adaptive cap (K=3) dropped it.
Leave off unless you've measured no regression on your inputs.

### `max_context_chars`, `max_output_chars`

Hard limits enforced before the engine sees the input. Requests above
either limit return 400.

500,000 is generous for typical RAG inputs (~80 pages of text).

### `available_granularities`

Reserved. Always `["claim"]`. `"token"` granularity is wired into the
API but not implemented by the engine yet.

## `[auth]`

```toml
[[auth.api_keys]]
customer_id = "default"
key = "sk-krino-replace-me"
```

Multiple `[[auth.api_keys]]` blocks may be defined. Each must have
`customer_id` (an opaque identifier for your records) and `key` (the
actual secret string).

Keys are SHA-256-hashed at startup and held in memory; the plaintext
is never logged. Comparing requested keys against the table uses
constant-time comparison via the `subtle` crate.

**Rotate keys by editing the config and restarting.** There is no
admin endpoint to add or revoke keys at runtime.

## `[rate_limit]`

```toml
[rate_limit]
requests_per_minute = 60
burst               = 10
```

Per-API-key rate limiting using a token bucket. `requests_per_minute`
is the steady-state rate; `burst` is the bucket capacity.

Requests over the limit return 429.

## `[logging]`

```toml
[logging]
level     = "info"
format    = "json"
ort_level = "warn"
```

| Field | Values | Description |
|---|---|---|
| `level` | `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"` | Tracing level for the application. |
| `format` | `"json"` or `"pretty"` | JSON for production (machine-parseable), `pretty` for dev (human-readable). |
| `ort_level` | same as `level` | Separate level for ONNX Runtime's own logging. `warn` suppresses harmless BFC arena chatter at startup. |

## `[workers]`

```toml
[workers]
n_workers          = 1
threads_per_worker = 7
queue_depth        = 1
batch_size         = 16
```

### `n_workers` and `threads_per_worker`

Each worker owns its own ONNX session and runs requests in a private
Rayon thread pool. The product `n_workers × threads_per_worker` should
not exceed your available vCPUs minus one (reserved for tokio/OS
overhead).

The default is **1 worker × (vCPUs − 1) threads** — i.e., a single
request gets every available core. This maximizes single-request
latency, which is what most demo and integration workloads care
about. Concurrent requests queue.

For **concurrent throughput** (many users, each tolerant of higher
per-request latency), invert the ratio. On 8 vCPUs:

| Goal | n_workers | threads_per_worker |
|---|---|---|
| Lowest single-request latency (default) | 1 | 7 |
| Two concurrent users, balanced | 2 | 3 |
| High concurrency, latency tolerant | 4 | 1 |

### `queue_depth`

Bounded channel depth for waiting requests. With `n_workers=1` and
`queue_depth=1`, the 3rd concurrent request returns 503 immediately
rather than piling up.

Raise this for spiky workloads where you'd rather queue than reject.

### `batch_size`

NLI pairs grouped into a single ORT forward pass. Larger batches
amortize the per-pass overhead but cost more peak memory.

16 is the calibrated default at typical sequence lengths. Bump to 32
if you have memory headroom and want slightly tighter throughput;
drop to 8 if you see OOM.
