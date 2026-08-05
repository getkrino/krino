# API reference

Krino exposes a small HTTP API. This page documents every endpoint
field. The wire types themselves live in the
[`krino-api-types`](https://docs.rs/krino-api-types) crate; this
document is a narrative overlay.

## Authentication

All `/api/v1/*` routes require an API key. Either:

- `x-api-key: <key>` header, or
- `Authorization: Bearer <key>` header.

Public routes (`/health`, `/health/ready`, `/metrics`) do not require
auth.

## `POST /api/v1/evaluate`

Verify that an LLM output is faithful to the provided context.

### Request body

```json
{
  "context": [
    {"id": "doc1", "text": "Source text..."},
    {"id": "doc2", "text": "More source text..."}
  ],
  "output": "What the model produced.",
  "config": {
    "granularity": "claim",
    "threshold": 0.7,
    "strict": false,
    "include_matrix": false,
    "top_k_context": null
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `context` | `Array<ContextChunk>` | yes | Source documents to verify against. Must contain at least one entry. |
| `output` | `string` | yes | The LLM output being evaluated. |
| `config` | `RequestConfig` | no | Per-request overrides. See below. |

**`ContextChunk`**

| Field | Type | Description |
|---|---|---|
| `id` | `string?` | Caller-assigned identifier. Echoed back in evidence links so callers can map evidence to chunks. |
| `text` | `string` | The chunk's text content. |

**`RequestConfig`** (all fields optional)

| Field | Type | Default | Description |
|---|---|---|---|
| `granularity` | `"claim"` or `"token"` | `"claim"` | Verdict granularity. `"token"` is reserved; only `"claim"` is implemented today. |
| `threshold` | `number` (0.0–1.0) | from `krino-api.toml` | Minimum faithfulness score for the response's `pass` field to be true. |
| `strict` | `bool` | `false` | If true, neutral verdicts count as unsupported. Reserved for future use; not currently consulted by the engine. |
| `include_matrix` | `bool` | `false` | Include the full entailment matrix in the response. Useful for debugging; significantly enlarges the response. |
| `top_k_context` | `usize` | from `krino-api.toml` | Override the embedding pre-filter top-K. `0` disables pre-filtering and evaluates every `(claim, context_sentence)` pair — useful for audit probes. Other engine knobs (similarity floor, thresholds) remain startup-time only. |

### Response body

```json
{
  "score": 0.92,
  "pass": true,
  "issues": [],
  "claims": [ /* one per output sentence */ ],
  "spans": null,
  "entailment_matrix": null,
  "meta": { /* ... */ }
}
```

**Top-level fields**

| Field | Type | Description |
|---|---|---|
| `score` | `number` (0.0–1.0) | Overall faithfulness score: fraction of claims with a supportive verdict. |
| `pass` | `bool` | Whether `score >= threshold`. |
| `issues` | `Array<IssueResponse>` | Per-issue summary ordered by severity. One issue per non-supported claim. |
| `claims` | `Array<ClaimResponse>` | One entry per claim extracted from the output. |
| `spans` | `Array<SpanResponse>?` | Reserved for `granularity: "token"`. Always `null` today. |
| `entailment_matrix` | `Array<EntailmentMatrixCell>?` | Present only when `include_matrix: true`. One cell per `(claim, evaluated_context_sentence)` pair. |
| `meta` | `MetaResponse` | Timing, model, and engine metadata. |

**`ClaimResponse`**

| Field | Type | Description |
|---|---|---|
| `text` | `string` | The claim as extracted. |
| `verdict` | `string` | One of `"entailment"`, `"contradiction"`, `"neutral"`, or `"partial"`. See [Verdicts](#verdicts) below. |
| `score` | `number` (0.0–1.0) | Entailment probability for the best evidence sentence. For `"partial"`, this is the mean across contributing sentences. |
| `supported` | `bool` | True for `"entailment"` and `"partial"`. |
| `evidence` | `EvidenceResponse?` | The single context sentence that best supports the claim. |
| `is_compound` | `bool` | True if the engine flagged this claim as containing multiple atomic assertions. |
| `supporting_evidence` | `Array<EvidenceResponse>` | Non-empty only for `"partial"` verdicts: the set of context sentences that jointly support the claim, ranked by entailment. |

**`EvidenceResponse`**

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `string?` | The `id` of the context chunk this evidence came from, if the caller supplied one. |
| `text` | `string` | The context sentence text. |
| `entailment_prob` | `number?` | NLI entailment probability for this evidence. |
| `contradiction_prob` | `number?` | NLI contradiction probability for this evidence. |
| `similarity_score` | `number?` | Cosine similarity from the embedding pre-filter. Absent when pre-filtering is disabled (`top_k_context: 0`). |

**`MetaResponse`**

| Field | Type | Description |
|---|---|---|
| `granularity` | `string` | Always `"claim"` for now. |
| `model` | `string` | Identifier of the NLI model backing the engine. |
| `latency_ms` | `number` | Server-side wall time. |
| `inference_calls` | `number` | Number of NLI forward passes. (Wire field is `inference_calls`; deserializers also accept `nli_calls`.) |
| `engine_confidence` | `number?` | Fraction of claims with a decisive verdict. Low values mean the score itself is shaky; consult individual claims. |
| `split_ms` | `number?` | Time spent splitting input into sentences. |
| `embedding_ms` | `number?` | Time spent computing embeddings for the pre-filter. |
| `nli_ms` | `number?` | Time spent in NLI batch inference. |
| `engine_version` | `string` | Krino engine version. |

## Verdicts

Per claim:

- **`entailment`** — A single context sentence supports the claim with
  high probability. Default supported.
- **`contradiction`** — A context sentence asserts the opposite of the
  claim with probability over the configured `contradiction_threshold`.
  Not supported.
- **`neutral`** — The model can find no sentence that decisively
  supports or contradicts the claim. Not supported by default; counts
  as supported if the engine was configured with
  `treat_neutral_as_unsupported = false` (the default).
- **`partial`** — Reserved for *compound* claims (multi-fact). The
  engine couldn't find a single sentence covering everything, but two
  or more sentences each cover part of the claim with sufficient
  entailment, low neutral, and high similarity. Counted as supported.
  The contributing sentences are in `supporting_evidence`.

See [architecture.md](architecture.md) for the algorithm that produces
each verdict.

## `GET /health`

Liveness probe. Always returns 200 with `{"status": "ok"}` if the
process is running.

## `GET /health/ready`

Readiness probe. Submits a trivial fast-path evaluation to the worker
pool. Returns 200 if the engine is loaded and responsive, 503
otherwise. Use this for load balancer health checks.

## `GET /metrics`

Prometheus metrics endpoint. See [observability.md](observability.md)
for the metric names and labels.

## Error responses

All error responses are JSON:

```json
{"error": {"type": "...", "message": "..."}}
```

| HTTP | `type` | Cause |
|---|---|---|
| 400 | `bad_request` | Malformed request body, missing required fields, invalid granularity. |
| 401 | `unauthorized` | Missing or invalid API key. |
| 429 | `too_many_requests` | Rate limit exceeded. |
| 503 | `service_unavailable` | Worker pool saturated; retry. |
| 500 | `internal` | Engine error. |
