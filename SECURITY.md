# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Report privately through the repository's **Security** tab ("Report a vulnerability").
Do not open a public issue.

Include the tailcheck version, your OS, what goes wrong, and how to reproduce it. Expect an
acknowledgement within a week.

## Threat model

tailcheck is a local statistics library and command-line tool.

- **No credentials.** tailcheck needs no API keys. The data downloaders fetch public files
  over HTTPS from `data.binance.vision` and `fred.stlouisfed.org` and nothing else.
- **Downloaded data is untrusted input.** Parsers validate column counts, types, dates and
  finiteness (non-finite or non-positive prices are rejected), and reject files that do not
  match the expected layout. Binance archives are verified against their published SHA-256
  checksums; FRED publishes none, so its CSV is parsed and validated before it is cached.
  Downloads are written to the cache atomically. Downloaded archives are read in memory;
  nothing from them is executed.
- **Cache and run directories** hold market data and intermediate results only. They are
  local, gitignored, and safe to delete.
- **The R fixture runner** (`scripts/r_fixtures/`) is a developer tool. It installs the `webr`
  npm package and R packages from the webR binary repository into a gitignored
  `node_modules/`. It is not part of the Python package and is never run by the tests or CI.
