# Release procedure

Krino uses manual releases. The `release.yml` workflow is triggered
from the Actions tab; there is no automatic version bumping or
auto-publishing on push to `main`. This is deliberate for the
pre-1.0 period — once API and behavior stabilize, we'll revisit.

## Steps

### 1. Open a release branch

```bash
git checkout main
git pull
git checkout -b release/0.X.Y
```

### 2. Bump the version

Edit the workspace `Cargo.toml`:

```toml
[workspace.package]
version = "0.X.Y"
```

Every member crate inherits this via `version.workspace = true`, so
the bump propagates automatically.

Run `cargo check --workspace` to update `Cargo.lock`.

### 3. Update the changelog

Move `[Unreleased]` content into a new versioned section:

```markdown
## [Unreleased]

## [0.X.Y] - 2026-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Each entry should be a short user-facing description. PR numbers help
("(#42)") but aren't required.

### 4. Open a PR, get it merged

Standard review. CI runs the same checks it always does.

### 5. Trigger the release workflow

1. Go to **Actions → Release → Run workflow**.
2. Set the `version` input to `0.X.Y` (must match `Cargo.toml`).
3. Click **Run workflow**.

The workflow will:

- Verify the input version matches the workspace `Cargo.toml`. (If
  not, it fails immediately — bump `Cargo.toml` first.)
- Build the `krino` CLI binary (`-p krino --features cli --bin krino`)
  and the `krino-api` HTTP server binary in release mode.
- Tag the commit `vX.Y.Z` and push the tag.
- Create a GitHub Release with the two binaries attached.

### 6. (Optional) Publish to crates.io

The release workflow does **not** publish to crates.io today. If you
want to:

```bash
# In dependency order
cargo publish -p krino-api-types
cargo publish -p krino
cargo publish -p krino-api
```

You'll need `cargo login <token>` first with a token that has publish
permissions on each crate.

## Versioning policy

Pre-1.0:

- **Patch (`0.X.Y → 0.X.(Y+1)`)** — bug fixes, internal refactors,
  no public-API or wire-format changes.
- **Minor (`0.X.* → 0.(X+1).0`)** — public-API or wire-format changes,
  including additions. May be breaking.
- **Major (`0.X.* → 1.0.0`)** — stability commitment, full semver
  rules apply going forward.

Post-1.0, standard semver applies — breaking changes only on major.

## What goes in a release

| Artifact | Where | Notes |
|---|---|---|
| `krino-linux-x64` | GitHub Releases | CLI binary built from `-p krino --features cli`. |
| `krino-api-linux-x64` | GitHub Releases | HTTP server binary. |
| Tagged source | `v0.X.Y` git tag | The point-in-time source. |
| crates.io | (manual, optional) | The three workspace crates. |
| Docker image | (not yet) | Future work. |

## Rolling back

If a release ships with a critical regression:

1. Don't delete the tag or release — they're a public record.
2. Open a fix PR against `main` with the regression test.
3. Once merged, cut a new patch release (`0.X.(Y+1)`).
4. Update the release notes of the broken version with a "Known
   issue: use 0.X.(Y+1) instead" note.

For security issues, see [SECURITY.md](../SECURITY.md).
