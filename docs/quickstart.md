# Quickstart

Get a Krino server running locally and send your first evaluation in
about five minutes.

## Prerequisites

- **Docker**, or alternatively **Rust 1.93+** if you want to build from
  source.
- **About 1 GB free disk** for the model weights.
- **An x86_64 host**. ARM has not been tested; the INT8 models assume
  AVX-512 VNNI for acceptable throughput.

## 1. Get the models

The engine ships without weights — you fetch them on first run.

```bash
git clone https://github.com/getkrino/krino
cd krino
./scripts/download_models.sh
```

This pulls the source models from Hugging Face and runs the export and
quantization scripts under `scripts/` to produce the two ONNX models
Krino needs:

- `models/all-MiniLM-L6-v2-onnx-quantized/` — embedding model used for
  the candidate pre-filter.
- `models/roberta-large-mnli-static-int8/` — INT8-quantized NLI model
  that does the actual entailment classification.

The first run takes a few minutes; later runs are skipped.

## 2. Configure

Copy the example configuration and edit the API key:

```bash
cp krino-api.toml.example krino-api.toml
# Edit krino-api.toml — at minimum, replace `sk-krino-replace-me`.
```

The defaults match the model paths from step 1, so you don't need to
change anything else for local development.

## 3. Run the server

**With Docker:**

```bash
docker compose up --build
```

**From source:**

```bash
cargo run --release -p krino-api --bin krino-api
```

Either way the server is now listening on port 8080.

## 4. Verify

In a second terminal:

```bash
# Liveness probe (no auth)
curl -sS http://localhost:8080/health

# Readiness probe (exercises the worker pool)
curl -sS http://localhost:8080/health/ready
```

Both should return JSON with `"status": "ok"` or `"ready"`.

## 5. Your first evaluation

```bash
curl -sS -X POST http://localhost:8080/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: sk-krino-replace-me' \
  -d '{
    "context": [
      {"id": "src1", "text": "Rust was first released in May 2015."}
    ],
    "output": "Rust shipped its first stable release in 2015."
  }' | jq
```

Expected response (abbreviated):

```json
{
  "score": 1.0,
  "pass": true,
  "issues": [],
  "claims": [
    {
      "text": "Rust shipped its first stable release in 2015.",
      "verdict": "entailment",
      "score": 0.97,
      "supported": true,
      "evidence": {
        "chunk_id": "src1",
        "text": "Rust was first released in May 2015.",
        "entailment_prob": 0.97
      },
      "is_compound": false
    }
  ],
  "meta": {
    "model": "krino-faithfulness-v1",
    "latency_ms": 142,
    "engine_version": "0.1.0"
  }
}
```

The claim's `verdict` is `"entailment"`, meaning the context supports
it. If you change the output to something contradicted (`"Rust was
first released in 2018."`) the verdict becomes `"contradiction"` and
the overall `score` drops.

## What's next

- [CLI usage](cli.md) — the standalone `krino` binary, for one-off
  checks without running the server.
- [API reference](api-reference.md) — every request/response field, all
  request-time overrides (matrix output, per-request top-K).
- [Configuration reference](configuration.md) — every field in
  `krino-api.toml`, with calibration notes.
- [Deployment guide](deployment.md) — Docker, AWS, systemd, worker
  tuning.
- [Architecture](architecture.md) — how the engine actually decides.
- [Observability](observability.md) — logs, `/metrics`, what each
  timing field means.
