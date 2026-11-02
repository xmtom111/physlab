# Changelog

All notable changes to physlab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-02

### Added

- Repository scaffold for the physlab package (`src` layout, configs, tests, examples, docs).
- Core layer: quaternion and pose math, shared types, error hierarchy, YAML configuration
  with `PHYSLAB_` environment overrides, structlog setup, JSONL event log, and a typed registry.
- Command line entry points `physlab --version` and `physlab config show`.
- Vendored PyOpenGL 3.1.0 wheel with the stray egg-info file removed so `uv run` can sync.
