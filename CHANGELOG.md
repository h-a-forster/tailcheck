# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: package layout, typed API stubs, CLI entry point, CI, design document
  (`docs/DESIGN.md`) with module specifications, test oracles, Monte Carlo plan and milestones.
- `scripts/r_fixtures/`: a Node runner for R reference fixtures through webR (R compiled to
  WebAssembly), so the R packages `esreg` and `esback` can serve as black-box oracles
  without a local R installation.
