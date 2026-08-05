# Security policy

## Supported versions

Krino is pre-1.0. Only the latest minor release on `main` receives
security updates. Older tags are point-in-time snapshots.

| Version | Supported |
|---|---|
| 0.1.x (latest) | ✓ |
| < 0.1 | ✗ |

When 1.0 ships, this table will list the supported release line.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **thinkingincode@proton.me** with:

- A description of the vulnerability and its impact.
- A reproducer — the smallest possible request, configuration, or
  input that demonstrates the issue.
- The affected version (`meta.engine_version` from a response, or the
  workspace `Cargo.toml` `version`).
- Optionally, a suggested fix.

You can expect:

- **Acknowledgement within 72 hours**, including a confidential issue
  tracker for follow-up.
- **A fix or mitigation plan within 7 days** for high-severity issues.
- **A coordinated public disclosure** after a fix is released, with
  credit to the reporter unless they prefer anonymity.

## Scope

Issues considered in scope for security disclosure:

- Authentication bypass in `/api/v1/*` routes.
- Information disclosure via crafted requests (e.g. error messages
  that leak internal paths, partial-evaluation results that leak
  unrelated context).
- Resource exhaustion attacks that aren't already bounded by the
  documented `max_context_chars`, `max_output_chars`, `queue_depth`,
  and rate-limit configuration.
- Cryptographic weaknesses in API key handling.
- Supply-chain issues in published artifacts (the Docker image, the
  release binaries, or crates.io publications).

Issues considered **out of scope**:

- Behavior of the engine on adversarial inputs that are within
  configured limits and don't cause resource exhaustion (e.g. "the
  engine returns 'neutral' on a sentence we think it should
  understand"). These are correctness issues, not security issues —
  open a normal GitHub issue with the matrix output.
- Vulnerabilities in third-party model weights downloaded by
  `scripts/download_models.sh`. Those are upstream concerns; report
  them to the model maintainers.
- Configuration mistakes where the deployer hasn't followed the
  guidance in [docs/deployment.md](docs/deployment.md) — for example,
  shipping the example API key (`sk-krino-replace-me`).

## Pre-release notes

Krino doesn't currently sign release binaries or publish SBOMs. This
will be revisited as the project matures.
