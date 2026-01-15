# Changelog

## 2026-01-15
### Added
- Multi-round audit flow with configurable `MAX_AUDIT_ROUNDS`.
- Term adjustment reporting via `TermAdjusted` sheet in review report.
- Log output compatibility with tqdm progress bars via `USE_TQDM_WRITE`.

### Changed
- Audit workflow now preserves bilingual draft formatting and appends source text for Chinese-only entries before audit.
- Glossary extraction strips English tokens and collapses duplicate numeric suffixes.
- Prompting tightened to enforce Simplified Chinese and avoid tag leakage.

### Fixed
- Duplicate Chinese prefixes and repeated numeric suffixes in short strings.
- Residual injected tag artifacts (e.g., `|原文:` fragments, stray markers).
