# FVTT Python Scripts Index

Last updated: 2026-03-18
Scope: `c:\Users\Taka\Desktop\fvtt` (including subfolders)

This document summarizes all Python scripts currently found in this folder tree.

## Quick Map

| Category | Scripts |
| --- | --- |
| Translation / AI | `pf2e_translator.py`, `sf2e_translator.py`, `crucible_translator.py`, `docx_batch_translator.py`, `pdf_vision_translator.py`, `术语工具/glossaryextraction.py` |
| Glossary / Term governance | `term_governor.py`, `table_to_glossary_json.py`, `更新merge/merge_fvtt_journals.py` |
| Cleanup / Repair | `clean_control_chars.py`, `repair_uuid_tokens.py`, `replace_name_newlines.py`, `strip_dscryb_names.py` |
| Rename / File operations | `rename_files_translate.py`, `rename_from_csv.py` |
| Extraction / Reporting | `extract_entries_titles.py`, `json_to_md.py`, `list_files_to_txt.py`, `merge_actor_txts.py` |
| Media conversion | `convert_mp4_to_webm.py`, `extend+optimizemp4/convert_mp4_to_webm.py` |

## Shared Dependencies

- Python standard library is enough for many scripts.
- OpenAI-based scripts require `openai` and API key setup.
- Progress bars in some scripts require `tqdm`.
- DOCX translation requires `python-docx`.
- PDF translation/extraction scripts require `PyMuPDF` (`fitz`).
- MP4/WebM conversion scripts require external `ffmpeg` and `ffprobe` binaries.

## Script-by-Script Notes

### `pf2e_translator.py`
- Purpose: Batch-translate FVTT/PF2E JSON content with glossary support, cache, retries, fallback model, and adaptive glossary learning.
- Inputs: `simple_config.json`, target JSON file(s), `glossary.json`, optional extra/adaptive glossary JSON.
- Outputs: Updated target JSON(s), `simple_run.log`, `simple_cache.json`, optional adaptive glossary output.
- Run: `python pf2e_translator.py`
- Notes: Main production translator in this repo; currently includes long-text timeout scaling and special handling to avoid endless retries on index-like page names.

### `sf2e_translator.py`
- Purpose: SF2E-oriented translation pipeline cloned from `pf2e_translator.py`, with independent config and safer default concurrency.
- Inputs: `sf2e_simple_config.json`, SF2E JSON target(s), glossary JSON files.
- Outputs: Updated target JSON(s), `sf2e_simple_run.log`, `sf2e_simple_cache.json`, optional adaptive glossary output.
- Run: `python 翻译工具/sf2e_translator.py`
- Notes: Uses SF2E prompt profile and defaults tuned for stable long runs.

### `crucible_translator.py`
- Purpose: Crucible-oriented translation pipeline based on the SF2E/PF2E translators, with independent config/log/cache.
- Inputs: `crucible_simple_config.json`, Crucible JSON target(s), glossary JSON files.
- Outputs: Updated target JSON(s), `crucible_simple_run.log`, `crucible_simple_cache.json`, optional adaptive glossary output.
- Run: `python 翻译工具/crucible_translator.py`
- Notes: Supports adaptive glossary modes `file` / `entry` / `entry_batch` where `entry_batch` extracts terms every N entries.

### `docx_batch_translator.py`
- Purpose: Translate DOCX content in batches using OpenAI, with glossary matching and cache.
- Inputs: `docx_translate_config.json`, DOCX source files, `glossary.json`.
- Outputs: Translated DOCX and side outputs (markdown/json/cache/log as configured).
- Run: `python docx_batch_translator.py`
- Notes: Good for document workflows outside FVTT JSON.

### `pdf_vision_translator.py`
- Purpose: Translate PDF content using page rendering and vision-based prompts.
- Inputs: `pdf_vision_config.json`, source PDF(s), `glossary.json`.
- Outputs: Translation artifacts (markdown/json/text), `pdf_vision_cache.json`, `pdf_vision_run.log`.
- Run: `python pdf_vision_translator.py`
- Notes: API-cost heavy; tune page ranges and cache for large PDFs.

### `冒险提取术语表/glossaryextraction.py`
- Purpose: Extract bilingual glossary candidates from PDFs and aggregate conflicts/candidates.
- Inputs: `glossary_extract_config.json`, PDF directory tree.
- Outputs: `glossary_from_pdf.json`, `glossary_candidates_from_pdf.json`, `glossary_conflicts_from_pdf.json`, cache/log.
- Run: `python 冒险提取术语表/glossaryextraction.py`
- Notes: Useful for building domain glossary before running translation scripts.

### `term_governor.py`
- Purpose: Manage glossary extraction/apply/restore workflows for JSON text terms.
- Inputs: target JSON and glossary/terms files via CLI arguments.
- Outputs: terms JSON/CSV, optional backup file, modified target JSON (depending on command).
- Run: subcommand style script with `argparse` (extract/apply/restore/ai-suggest flows are implemented).
- Notes: Higher-control tooling for term lifecycle and recoverability.

### `table_to_glossary_json.py`
- Purpose: Convert CSV/TSV/XLSX term tables into glossary JSON for translators.
- Inputs: tabular file with English and Chinese columns.
- Outputs: dictionary glossary JSON (default) or `terms`-style JSON.
- Run: `python 术语工具/table_to_glossary_json.py --input <table.xlsx> --output glossary_sf2e_from_table.json`
- Notes: Supports duplicate conflict strategies (`first`/`last`/`list`) and optional merge with existing glossary.

### `更新merge/merge_fvtt_journals.py`
- Purpose: Merge and harmonize FVTT journal bilingual content with HTML-aware text processing.
- Inputs: old/new JSON via CLI flags (`--old`, `--new`) and merge tuning options.
- Outputs: merged JSON (`--out`) and report JSON (`--report`, default includes `merge_report.json`).
- Run: `python 更新merge/merge_fvtt_journals.py --old <old.json> --new <new.json> --out <out.json>`
- Notes: Advanced merge script; uses rich normalization/harmonization functions for mixed EN/ZH+HTML content.

### `clean_control_chars.py`
- Purpose: Detect/remove hidden control characters from text files.
- Inputs: path(s), glob pattern, optional in-place flag.
- Outputs: cleaned files when `-i/--in-place` is used.
- Run: `python clean_control_chars.py [paths] [-g "**/*.json"] [-i]`
- Notes: Safe to dry-run without `-i`.

### `repair_uuid_tokens.py`
- Purpose: Repair broken UUID/Compendium token patterns by comparing source and target JSON.
- Inputs: `--source`, `--target`.
- Outputs: fixed JSON (`--output`) and report (`--report`).
- Run: `python repair_uuid_tokens.py --source <src.json> --target <target.json> --output <fixed.json> --report <report.json>`
- Notes: Use `--dry-run` first on important data.

### `replace_name_newlines.py`
- Purpose: Normalize `name` fields by removing embedded newline issues.
- Inputs: configured JSON path in script constants.
- Outputs: rewritten JSON file.
- Run: `python replace_name_newlines.py`
- Notes: In-place behavior; check file constants before execution.

### `strip_dscryb_names.py`
- Purpose: Strip selected dScryb fields (including page names/Image page content) to reduce payload.
- Inputs: input JSON path.
- Outputs: stripped JSON (`-o/--output`).
- Run: `python strip_dscryb_names.py <input.json> -o <output.json>`
- Notes: Destructive by design for removed fields; keep source backup.

### `rename_files_translate.py`
- Purpose: Use OpenAI to translate filenames and rename files to bilingual names.
- Inputs: files in current directory and OpenAI API key.
- Outputs: renamed files (in place).
- Run: `python rename_files_translate.py`
- Notes: No dry-run mode in script; test in a copy first.

### `rename_from_csv.py`
- Purpose: Rename/move files based on mapping rows from CSV.
- Inputs: `--csv`, `--base`, optional `--encoding`.
- Outputs: renamed files in target paths.
- Run: `python rename_from_csv.py --csv 索引.csv --base . [--dry-run] [--overwrite]`
- Notes: Supports dry-run and overwrite controls; recommended for large rename batches.

### `extract_entries_titles.py`
- Purpose: Extract title/label style metadata from configured FVTT pack JSON files.
- Inputs: hardcoded input pack list in script.
- Outputs: `labels.json`, `titles.json`.
- Run: `python extract_entries_titles.py`
- Notes: Edit script input list when adding/removing source packs.

### `json_to_md.py`
- Purpose: Convert entry-like JSON content into Markdown documentation.
- Inputs: JSON input path.
- Outputs: Markdown output path (`-o/--output`).
- Run: `python json_to_md.py <input.json> -o <output.md>`
- Notes: Read-only on input source.

### `list_files_to_txt.py`
- Purpose: Recursively list files under the current directory.
- Inputs: current working directory.
- Outputs: `files_list.txt`.
- Run: `python list_files_to_txt.py`
- Notes: Lightweight inventory helper.

### `merge_actor_txts.py`
- Purpose: Merge actor-related TXT fragments into consolidated files.
- Inputs: input dir and naming conventions for actor text fragments.
- Outputs: merged TXT files (output dir configurable).
- Run: `python merge_actor_txts.py -d <input_dir> -o <output_dir> [--overwrite]`
- Notes: Has robust CLI flags for separator, headers, and strictness.

### `convert_mp4_to_webm.py`
- Purpose: Convert MP4/WebM using ffmpeg/ffprobe, with GPU-aware options and batching.
- Inputs: media files in scan scope, ffmpeg/ffprobe availability.
- Outputs: converted `.webm` files.
- Run: `python convert_mp4_to_webm.py`
- Notes: Compute-heavy; check constants before full-batch runs.

### `extend+optimizemp4/convert_mp4_to_webm.py`
- Purpose: Same media conversion tool as root script.
- Inputs: same as root converter.
- Outputs: same as root converter.
- Run: `python extend+optimizemp4/convert_mp4_to_webm.py`
- Notes: File hash check confirms it is byte-identical to `convert_mp4_to_webm.py`; keep one canonical copy to reduce maintenance drift.

## Risk Classification

### High risk (modifies files in place / move / delete-like effects)
- `pf2e_translator.py`
- `crucible_translator.py`
- `rename_files_translate.py`
- `rename_from_csv.py`
- `repair_uuid_tokens.py`
- `replace_name_newlines.py`
- `strip_dscryb_names.py`
- `更新merge/merge_fvtt_journals.py`

### Medium risk (output generation, may overwrite output files)
- `docx_batch_translator.py`
- `pdf_vision_translator.py`
- `冒险提取术语表/glossaryextraction.py`
- `term_governor.py`
- `merge_actor_txts.py`
- `convert_mp4_to_webm.py`
- `extend+optimizemp4/convert_mp4_to_webm.py`

### Low risk (mostly read/report)
- `clean_control_chars.py` (without `-i`)
- `extract_entries_titles.py`
- `json_to_md.py`
- `list_files_to_txt.py`

## Recommended Operating Checklist

- Set and verify `OPENAI_API_KEY` before AI scripts.
- Keep source backups before running high-risk scripts.
- Use dry-run flags when available (`rename_from_csv.py`, `repair_uuid_tokens.py`, `clean_control_chars.py`).
- Keep only one copy of duplicate scripts unless you intentionally branch behavior.
- For long translation jobs, keep cache enabled and monitor log files.
