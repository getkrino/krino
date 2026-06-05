```
██╗  ██╗██████╗ ██╗███╗   ██╗ ██████╗
██║ ██╔╝██╔══██╗██║████╗  ██║██╔═══██╗
█████╔╝ ██████╔╝██║██╔██╗ ██║██║   ██║
██╔═██╗ ██╔══██╗██║██║╚██╗██║██║   ██║
██║  ██╗██║  ██║██║██║ ╚████║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
        groundedness for LLM outputs
```

Krino is a self-hostable faithfulness engine for LLM outputs. Given a
context (retrieved documents, source text, reference material) and an
output (what your model produced), Krino tells you which claims in the
output are supported, contradicted, or unsupported by the context — and
points at the specific sentences that justify each verdict.

The engine runs locally via ONNX, returns evidence-linked verdicts in
under a second on typical inputs, and exposes everything as a simple
HTTP API.

> **Status:** pre-1.0. The HTTP API and engine outputs are stable, but
> configuration field names and crate APIs may shift before 1.0.

## Quick start

```bash
# 1. Run the server with the example config (Docker)
docker run -p 8080:8080 ghcr.io/smithjustinm/krino:latest

# 2. Verify
curl -sS http://localhost:8080/health

# 3. Send your first evaluation
curl -sS -X POST http://localhost:8080/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{
    "context": [{"text": "Rust was first released in 2015."}],
    "output": "Rust shipped its first stable release in 2015."
  }'
```

See [`docs/quickstart.md`](docs/quickstart.md) for the full walkthrough,
and [`docs/deployment.md`](docs/deployment.md) for production deployment
guidance (systemd, AWS, worker tuning, observability).

## What Krino does

- **Per-claim verdicts.** Splits LLM output into claims and classifies
  each as `entailment`, `contradiction`, `neutral`, or `partial`
  against the supplied context.
- **Evidence tracing.** Every verdict links to the exact context
  sentence that justifies it, with NLI probabilities exposed.
- **Compound claim handling.** Multi-fact claims that no single
  sentence covers are detected and aggregated across multiple
  supporting sentences.
- **Configurable strictness.** Per-request overrides for matrix
  output and pre-filter top-K, plus server-side configuration for
  contradiction threshold, similarity floor, and more.

## What Krino does not (yet) do

- Token-level granularity. The API accepts `granularity: "token"`
  but the engine only implements claim-level today.
- GPU inference. CPU-only, AVX-512 VNNI recommended.
- Streaming responses. The engine returns a single JSON body per
  request.

## System requirements

- **CPU**: x86_64. AVX-512 VNNI strongly recommended (used by the
  INT8 quantized NLI model). ARM has not been tested.
- **Memory**: 4 GB minimum. 8 GB recommended for production.
- **Models**: ~600 MB on disk (RoBERTa-large MNLI INT8 + MiniLM-L6 embedding).

## Crates

| Crate              | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| `krino`            | Engine library: NLI, embedding pre-filter, verdict logic. |
| `krino-api`        | HTTP server binary built on the engine.                   |
| `krino-api-types`  | Shared wire types between server and clients.             |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
