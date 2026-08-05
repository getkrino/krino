# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-05

### Added

- Initial open-source release.
- Engine library (`krino` crate): deterministic NLI-based faithfulness
  checker. Per-claim verdicts (`entailment` / `contradiction` /
  `neutral` / `partial`), evidence tracing, compound-claim detection
  with multi-evidence aggregation, embedding pre-filter for large
  context windows.
- HTTP server (`krino-api` crate): `POST /api/v1/evaluate`, health
  and metrics endpoints, configurable worker pool.
- Shared wire types (`krino-api-types` crate).
- Docker quickstart and example deployment configurations.
