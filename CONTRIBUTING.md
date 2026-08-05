# Contributing to Krino

Thanks for your interest. This document covers the dev environment,
how to propose changes, and the conventions the codebase follows.

## Dev environment

### Prerequisites

- **Rust 1.93+** (the workspace specifies `rust-version = "1.93"` and
  uses 2024 edition).
- For ONNX model regeneration: **Python 3.10+** and
  [`uv`](https://github.com/astral-sh/uv) (the model export scripts
  under `scripts/` use `uv run`).
- For Docker-based smoke testing: **Docker** and **Docker Compose**.

### First-time setup

```bash
git clone https://github.com/getkrino/krino
cd krino
cargo build
```

Some integration tests require model weights. To download (or
regenerate) them:

```bash
./scripts/download_models.sh
```

### Quick commands

The `Makefile` collects common workflows:

```bash
make check         # cargo check --workspace
make test          # cargo test --workspace (skips ignored integration tests)
make fmt           # cargo fmt --all
make clippy        # cargo clippy --workspace --all-targets -D warnings
make ci            # everything CI runs
make bench         # cargo bench
```

### Recommended pre-commit hook

Hooks aren't committed to the repo (kept personal). The Makefile can
install one for you:

```bash
make pre-commit
```

This writes `.git/hooks/pre-commit` that runs `cargo fmt --check`,
`cargo clippy -D warnings`, and `cargo test --workspace` on every
commit. Skip in a pinch with `git commit --no-verify` — CI runs the
same checks anyway.

## Making changes

### Branches and pull requests

- Branch off `main`.
- Keep PRs focused. Smaller is easier to review.
- Tests for new behavior. Bug fixes get a regression test.
- Give the PR a Conventional Commit title (see below) — `CHANGELOG.md`
  is generated from it, not hand-edited.
- Run `make ci` locally; CI runs the same.

### Commit messages and PR titles

We squash-merge, and the squash commit's subject becomes the PR
title — so **your PR title must be a Conventional Commit**, since
that's what drives releases (see
[docs/release.md](docs/release.md)). CI lints this on every PR.

```
feat: add per-request top_k override
fix: handle 0-byte context chunks
docs: clarify partial verdict semantics
chore: bump tokenizers to 0.21
```

Individual commits within a PR can be messier (fixups, WIP) since
they get squashed away — but keep the PR title accurate to what's
actually shipping. Only `feat`, `fix`, and `perf` trigger a release;
`chore`, `docs`, `refactor`, `test`, `ci`, and `build` are allowed but
won't bump the version.

The changelog is auto-generated from these titles by `git-cliff` at
release time — it isn't hand-edited.

### Code style

Standard `rustfmt` and `clippy` are the baseline. Beyond that:

- **No `unwrap()` on `Result` outside of tests.** Use `?`, `.context()`
  from `anyhow`, or explicit error variants.
- **No `unwrap()` on `Option`** unless an invariant in the surrounding
  code obviously guarantees `Some`. When you do, leave a comment
  explaining the invariant.
- **Document what `pub` items do, not what they are.** A docstring
  describing `pub fn check_with_overrides` should explain *when* a
  caller would want it, not just restate the signature.
- **Don't write `#[derive(Debug, Clone)]` reflexively.** Add `Debug`
  when you'd want to log the value; add `Clone` when something
  actually clones it.
- **Tests next to code** where reasonable: `#[cfg(test)] mod tests`
  blocks at the bottom of the module they test, integration tests in
  `tests/`.

### What gets reviewed

For most PRs:

- **Correctness.** Does the change do what the description says?
- **Determinism.** Krino's contract is "same inputs, same outputs."
  Anything that introduces randomness (HashMap iteration order, etc.)
  is rejected without a justification. See the existing test
  `tests/groundedness_onnx_integration.rs::test_determinism` for the
  pattern.
- **API surface.** Public types in `krino-api-types` are the wire
  format. Breaking changes are version-bumps; consider whether the
  change can be additive (new optional field) instead.

## Reporting bugs

Open a GitHub issue with:

1. What you ran (command line or the request body for API issues).
2. What you expected.
3. What happened instead.
4. Engine version (`meta.engine_version` from any response, or the
   `Cargo.toml` `version`).
5. If reproducible from a curl: the smallest possible curl that shows
   the bug.

For correctness bugs ("the engine gave the wrong verdict"), the most
useful additional thing is the matrix output: send the same request
with `config.include_matrix: true` and include the result.

## Reporting security issues

Don't open a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree your contributions will be licensed under
both Apache 2.0 and MIT (the dual-license under which Krino is
released).
