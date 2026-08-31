import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


ACTION_SECTION_RE = re.compile(r'<section class="action"[^>]*>.*?</section>', re.S)
ENCOUNTER_SECTION_RE = re.compile(r'<section class="encounter"[^>]*>.*?</section>', re.S)
UUID_RE = re.compile(r'@UUID\[([^\]]+)\]\{([^{}]+)\}')
SRC_RE = re.compile(r'src="([^"]+)"')

EMPTY_LINES_RE = re.compile(r"\n{3,}")


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def save_json_minified(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def _is_duplicate_empty_action(block: str, prior_text: str) -> bool:
    lowered = block.lower()
    if "<ul" in lowered or "<p" in lowered:
        return False
    if contains_cjk(block):
        return False

    uuid_matches = UUID_RE.findall(block)
    if not uuid_matches:
        return False

    labels = [label for _, label in uuid_matches]
    if not all(contains_latin(label) and not contains_cjk(label) for label in labels):
        return False

    # Safe guard: only remove the shell if the same UUID reference already appears earlier.
    return all(f"@UUID[{uuid}]" in prior_text for uuid, _ in uuid_matches)


def _is_duplicate_empty_encounter(block: str, prior_text: str) -> bool:
    lowered = block.lower()
    if "<p" in lowered or "<ul" in lowered or "<table" in lowered:
        return False
    if contains_cjk(block):
        return False
    if '<div class="header">' not in lowered:
        return False

    srcs = SRC_RE.findall(block)
    if not srcs:
        return False

    non_generic = [
        src
        for src in srcs
        if "information-icon" not in src
        and "roll-icon" not in src
        and "danger-icon" not in src
        and "treasure-icon" not in src
    ]
    compare_pool = non_generic if non_generic else srcs

    # If all visual assets already appeared earlier, this encounter shell is likely duplicated tail.
    return all(src in prior_text for src in compare_pool)


def clean_journal_html(text: str) -> tuple[str, int, int]:
    if not isinstance(text, str) or not contains_cjk(text):
        return text, 0, 0

    remove_spans: list[tuple[int, int]] = []
    removed_actions = 0
    removed_encounters = 0

    for match in ACTION_SECTION_RE.finditer(text):
        block = match.group(0)
        if _is_duplicate_empty_action(block, text[: match.start()]):
            remove_spans.append((match.start(), match.end()))
            removed_actions += 1

    for match in ENCOUNTER_SECTION_RE.finditer(text):
        block = match.group(0)
        if _is_duplicate_empty_encounter(block, text[: match.start()]):
            remove_spans.append((match.start(), match.end()))
            removed_encounters += 1

    if not remove_spans:
        return text, 0, 0

    remove_spans.sort()
    merged: list[list[int]] = []
    for start, end in remove_spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    chunks: list[str] = []
    cursor = 0
    for start, end in merged:
        chunks.append(text[cursor:start])
        cursor = end
    chunks.append(text[cursor:])

    cleaned = "".join(chunks)
    cleaned = EMPTY_LINES_RE.sub("\n\n", cleaned)
    return cleaned, removed_actions, removed_encounters


def process_adventure_json(data: Any) -> tuple[Any, dict[str, Any]]:
    changed_pages = 0
    removed_actions = 0
    removed_encounters = 0
    page_changes: list[dict[str, Any]] = []

    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return data, {
            "changed_pages": 0,
            "removed_action_shells": 0,
            "removed_encounter_shells": 0,
            "page_changes": [],
        }

    for adventure_name, adventure in entries.items():
        if not isinstance(adventure, dict):
            continue

        journals = adventure.get("journals")
        if not isinstance(journals, dict):
            continue

        for journal_name, journal in journals.items():
            if not isinstance(journal, dict):
                continue

            pages = journal.get("pages")
            if not isinstance(pages, dict):
                continue

            for page_name, page in pages.items():
                if not isinstance(page, dict):
                    continue

                text_node = page.get("text")
                content: str | None = None
                mode = ""

                if isinstance(text_node, str):
                    content = text_node
                    mode = "str"
                elif isinstance(text_node, dict) and isinstance(text_node.get("content"), str):
                    content = text_node["content"]
                    mode = "dict-content"

                if content is None:
                    continue

                cleaned, ra, re_ = clean_journal_html(content)
                if cleaned == content:
                    continue

                if mode == "str":
                    page["text"] = cleaned
                else:
                    text_node["content"] = cleaned

                changed_pages += 1
                removed_actions += ra
                removed_encounters += re_
                page_changes.append(
                    {
                        "adventure": adventure_name,
                        "journal": journal_name,
                        "page": page_name,
                        "removed_action_shells": ra,
                        "removed_encounter_shells": re_,
                        "before_len": len(content),
                        "after_len": len(cleaned),
                    }
                )

    return data, {
        "changed_pages": changed_pages,
        "removed_action_shells": removed_actions,
        "removed_encounter_shells": removed_encounters,
        "page_changes": page_changes,
    }


def resolve_files(target: Path, pattern: str, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        if recursive:
            return sorted(path for path in target.rglob(pattern) if path.is_file())
        return sorted(path for path in target.glob(pattern) if path.is_file())
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicated empty action/encounter shells from translated FVTT adventure journals"
    )
    parser.add_argument("--target", required=True, help="Target JSON file or directory")
    parser.add_argument("--glob", default="*.json", help="File pattern for directory mode (default: *.json)")
    parser.add_argument("--recursive", action="store_true", help="Recurse subdirectories in directory mode")
    parser.add_argument("--in-place", action="store_true", help="Overwrite source files")
    parser.add_argument("--output", default="", help="Output path for single-file mode when not using --in-place")
    parser.add_argument("--backup-dir", default="", help="Backup directory used with --in-place")
    parser.add_argument("--report", default="strip_duplicate_render_shells_report.json", help="Report JSON path")
    parser.add_argument("--sample-limit", type=int, default=80, help="Max page changes per file in report")
    args = parser.parse_args()

    target = Path(args.target)
    files = resolve_files(target, args.glob, args.recursive)
    if not files:
        raise FileNotFoundError(f"No JSON files found from target: {target}")

    if target.is_dir() and not args.in_place:
        print("[INFO] Directory mode without --in-place runs as dry-run")

    backup_root: Path | None = None
    if args.in_place:
        if args.backup_dir:
            backup_root = Path(args.backup_dir)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_root = Path(f"_backup_strip_duplicate_render_shells_{stamp}")
        backup_root.mkdir(parents=True, exist_ok=True)

    report_files: list[dict[str, Any]] = []
    total_changed_files = 0
    total_changed_pages = 0
    total_removed_actions = 0
    total_removed_encounters = 0

    for file_path in files:
        try:
            data = load_json(file_path)
        except Exception as exc:
            report_files.append({"file": str(file_path), "error": str(exc)})
            continue

        new_data, summary = process_adventure_json(data)
        changed_pages = int(summary["changed_pages"])

        if changed_pages > 0:
            total_changed_files += 1
            total_changed_pages += changed_pages
            total_removed_actions += int(summary["removed_action_shells"])
            total_removed_encounters += int(summary["removed_encounter_shells"])

            if args.in_place:
                assert backup_root is not None
                backup_path = backup_root / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                save_json_minified(file_path, new_data)
            elif target.is_file():
                output_path = (
                    Path(args.output)
                    if args.output
                    else file_path.with_name(file_path.stem + ".dedup-shells" + file_path.suffix)
                )
                save_json_minified(output_path, new_data)

        report_files.append(
            {
                "file": str(file_path),
                "changed_pages": changed_pages,
                "removed_action_shells": int(summary["removed_action_shells"]),
                "removed_encounter_shells": int(summary["removed_encounter_shells"]),
                "samples": summary["page_changes"][: max(0, args.sample_limit)],
            }
        )

    report_payload = {
        "meta": {
            "target": str(target),
            "in_place": args.in_place,
            "files_scanned": len(files),
            "files_changed": total_changed_files,
            "pages_changed": total_changed_pages,
            "removed_action_shells": total_removed_actions,
            "removed_encounter_shells": total_removed_encounters,
            "backup_dir": str(backup_root) if backup_root else "",
        },
        "files": report_files,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "Scanned:"
        f" {len(files)} | Changed files: {total_changed_files}"
        f" | Changed pages: {total_changed_pages}"
        f" | Removed action shells: {total_removed_actions}"
        f" | Removed encounter shells: {total_removed_encounters}"
    )
    print(f"Report: {report_path}")
    if backup_root:
        print(f"Backup: {backup_root}")


if __name__ == "__main__":
    main()
