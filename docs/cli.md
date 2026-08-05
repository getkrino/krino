# CLI usage

`krino` is a standalone binary for one-off checks — config validation,
JSON Schema validation — without running the `krino-api` server. It's
built from the same crate as the engine library, gated behind the
`cli` feature.

## Install

```bash
curl -sSf https://raw.githubusercontent.com/getkrino/krino/main/install.sh | sh
```

This downloads the `krino-linux-x64` binary from the latest GitHub
Release and installs it to `~/.local/bin` (or `/usr/local/bin` if
writable and `~/.local/bin` isn't on `PATH`). Only Linux x86_64 is
published today — see [Platform support](#platform-support).

**From source**, if you need another platform or want a dev build:

```bash
git clone https://github.com/getkrino/krino
cd krino
make install    # cargo install --path krino --features cli
```

## Subcommands

### `version`

Prints the engine version. No config or models required.

```bash
krino version
```

### `show-config`

Prints the default `KrinoConfig` as JSON — a starting point for
writing your own config file.

```bash
krino show-config
```

### `validate-config`

Validates a `krino.json`-style config file (model paths, performance
limits) without starting the engine.

```bash
krino validate-config --config krino.json
```

### `validate-schema`

Validates JSON against a JSON Schema.

```bash
krino validate-schema --json '{"a": 1}' --schema schema.json
krino validate-schema --file output.json --schema schema.json --output json
```

### `eval-hallucination` (not yet usable)

Intended to run token-level hallucination detection against a Candle
`ModernBERT` token-classification model:

```bash
krino eval-hallucination --context ctx.txt --answer answer.txt \
  --model-path ./models/some-modernbert-token-classifier
```

**This has no working model source today.** It requires a directory
with `tokenizer.json` + `model.safetensors` in `ModernBERT`
token-classification format, loaded via
[`CandleBackend::from_pretrained`](../krino/src/models/backends/candle.rs).
The only model-provisioning script in this repo,
[`scripts/download_models.sh`](../scripts/download_models.sh),
produces **ONNX** sequence-classification models for `krino-api`
instead — a different format and a different trait
(`SequenceClassifier`, one label per input pair) than what this
subcommand needs (`TokenClassifier`, one label per token). There is no
script, published weights, or documented recipe that produces a
compatible model, and no compatible off-the-shelf model is known to
exist publicly.

Until one of the following happens, don't rely on this subcommand:

- A `ModernBERT` token-classification checkpoint (fine-tuned for
  hallucination/entailment span detection) is trained or found and
  published under the Krino org on Hugging Face, with an export script
  added alongside `download_models.sh`, or
- An ONNX-based `TokenClassifier` implementation is added to
  `krino/src/models/backends/onnx.rs` and a token-classification ONNX
  model is sourced for it.

Track this in the repo issues before depending on it in a pipeline.

## Platform support

Release binaries are built for `x86_64-unknown-linux-gnu` only (see
[`release.yml`](../.github/workflows/release.yml)). macOS and ARM
users should build from source with `make install`; `install.sh` will
refuse to run rather than install the wrong binary.
