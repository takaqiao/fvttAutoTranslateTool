# Changelog

## 2026-01-17
- 测试模式新增完整流程模拟与规范化输出。
- HTML 比对更稳定（忽略 `title`，属性归一）。
- 含 HTML 文本在修复后直接返回，避免误改。
- 修复测试模式下的非字符串崩溃问题。

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
