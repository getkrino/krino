```
██╗  ██╗██████╗ ██╗███╗   ██╗ ██████╗
██║ ██╔╝██╔══██╗██║████╗  ██║██╔═══██╗
█████╔╝ ██████╔╝██║██╔██╗ ██║██║   ██║
██╔═██╗ ██╔══██╗██║██║╚██╗██║██║   ██║
██║  ██╗██║  ██║██║██║ ╚████║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
        groundedness for LLM outputs
```

[![CI](https://github.com/getkrino/krino/actions/workflows/ci.yml/badge.svg)](https://github.com/getkrino/krino/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Krino is a self-hostable faithfulness engine for LLM outputs. Given a
context (retrieved documents, source text, reference material) and an
output (what your model produced), Krino tells you which claims in
the output are supported, contradicted, or unsupported by the
context — and points at the specific sentences that justify each
verdict.

The engine runs locally via ONNX. Verdicts are
**deterministic by construction** (same inputs, same outputs) and
**explainable** (every verdict carries evidence). No LLM-as-judge.

> **Status:** pre-1.0. The HTTP API and engine outputs are usable but
> not yet stable — field names and crate APIs may change before the
> 1.0 release.

## Quick start

```bash
git clone https://github.com/getkrino/krino
cd krino
./scripts/download_models.sh         # ~600 MB, takes a few minutes
cp krino-api.toml.example krino-api.toml
docker compose up --build
```

In another terminal:

```bash
curl -sS -X POST http://localhost:8080/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: sk-krino-replace-me' \
  -d '{
    "context": [{"id": "src1", "text": "Rust was first released in 2015."}],
    "output": "Rust shipped its first stable release in 2015."
  }' | jq
```

The full walkthrough is in [docs/quickstart.md](docs/quickstart.md).

## CLI

Krino also ships a standalone `krino` CLI for one-off checks without
running the server:

```bash
curl -sSf https://raw.githubusercontent.com/getkrino/krino/main/install.sh | sh
krino validate-schema --json '{"a":1}' --schema schema.json
```

`install.sh` detects your platform (Linux/macOS, x64/arm64) and
downloads the matching release binary — no Rust toolchain or Docker
required. Windows: grab `krino-windows-x64.exe` from
[Releases](https://github.com/getkrino/krino/releases). See
[CLI usage](docs/cli.md) for every subcommand, or build from source
with `make install` (`cargo install --path krino --features cli`).

## What Krino does

- **Per-claim verdicts.** Splits LLM output into claims and
  classifies each as `entailment`, `contradiction`, `neutral`, or
  `partial` against the supplied context.
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
- GPU inference. CPU-only ONNX, AVX-512 VNNI recommended.
- Streaming responses. Each `/evaluate` returns a single JSON body.
- The CLI's `eval-hallucination` subcommand. It requires a Candle
  ModernBERT token-classification model, and no script or published
  weights produce one yet — see [CLI usage](docs/cli.md#eval-hallucination-not-yet-usable).

## System requirements

- **CPU**: x86_64. AVX-512 VNNI strongly recommended (used by the
  INT8 quantized NLI model). ARM has not been tested.
- **Memory**: 4 GB minimum. 8 GB recommended for production.
- **Models**: ~600 MB on disk (RoBERTa-large MNLI INT8 + MiniLM-L6
  embedding).

## Crates

| Crate             | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `krino`           | Engine library (NLI, embedding pre-filter, verdict logic) plus the standalone `krino` CLI binary (`--features cli`). |
| `krino-api`       | HTTP server binary built on the engine.                    |
| `krino-api-types` | Shared wire types between server and clients.              |

## Documentation

- [Quickstart](docs/quickstart.md) — 5-minute walkthrough.
- [CLI usage](docs/cli.md) — the `krino` binary: install, subcommands, known gaps.
- [API reference](docs/api-reference.md) — every request and response field.
- [Configuration reference](docs/configuration.md) — every field in `krino-api.toml`, with calibration notes.
- [Deployment guide](docs/deployment.md) — Docker, AWS, systemd, worker tuning.
- [Architecture](docs/architecture.md) — how the engine actually decides.
- [Observability](docs/observability.md) — logs, `/metrics`, timing fields.
- [Release procedure](docs/release.md) — how new versions ship.
- [Contributing](CONTRIBUTING.md) — dev setup, code style, PR process.
- [Security policy](SECURITY.md) — vulnerability reporting.
- [Code of conduct](CODE_OF_CONDUCT.md).

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
