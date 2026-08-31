import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


HTML_TAG_RE = re.compile(r"<[^>]+>")
FVTT_REF_NODE_RE = re.compile(r"^(?:@(?:UUID|Compendium|Localize)\[[^\]]+\](?:\{[^}]*\})?)$")
HTML_TAG_ONLY_LINE_RE = re.compile(r"^\s*(?:<[^>]+>\s*)+$")
HTML_PARSE_TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9:-]+)([^>]*)>")

SELF_CLOSING_TAGS = {
    "br", "hr", "img", "input", "meta", "link", "source", "track", "area",
    "base", "col", "embed", "param", "wbr",
}

BALANCE_CHECK_TAGS = {
    "p", "section", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "td", "th", "blockquote", "em", "strong",
    "span", "div", "figure", "figcaption", "article", "aside", "header", "footer",
    "a",
}


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def looks_like_html(text: str) -> bool:
    return bool(HTML_TAG_RE.search(text or ""))


def is_html_tag_only_line(text: str) -> bool:
    return bool(HTML_TAG_ONLY_LINE_RE.fullmatch(text or ""))


def is_html_balanced(html: str) -> bool:
    if not html or not looks_like_html(html):
        return True

    stack: list[str] = []
    for m in HTML_PARSE_TAG_RE.finditer(html):
        closing = bool(m.group(1))
        tag = m.group(2).lower()
        full = m.group(0)
        self_closing = tag in SELF_CLOSING_TAGS or full.endswith("/>")

        if tag not in BALANCE_CHECK_TAGS or self_closing:
            continue

        if closing:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
        else:
            stack.append(tag)

    return not stack


def split_bilingual_html_parts(text: str) -> tuple[str | None, str | None]:
    if not text or not looks_like_html(text):
        return None, None
    if not (contains_cjk(text) and contains_latin(text)):
        return None, None

    lines = text.splitlines()
    if len(lines) < 2:
        return None, None

    for i in range(1, len(lines)):
        zh_part = "\n".join(lines[:i]).strip()
        en_shell = "\n".join(lines[i:]).strip()
        if not zh_part or not en_shell:
            continue
        if not contains_cjk(zh_part):
            continue
        if not contains_latin(en_shell):
            continue
        if contains_cjk(en_shell):
            continue
        if not looks_like_html(en_shell):
            continue
        if not is_html_balanced(en_shell):
            continue
        return zh_part, en_shell

    return None, None


def extract_cjk_text_nodes(html: str) -> list[str]:
    parts = re.split(r"(<[^>]+>)", html)
    nodes: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            continue
        stripped = part.strip()
        if not stripped:
            continue
        if contains_cjk(stripped):
            nodes.append(stripped)
    return nodes


def rebuild_with_english_shell(text: str) -> str | None:
    zh_part, en_shell = split_bilingual_html_parts(text)
    if not zh_part or not en_shell:
        return None

    zh_nodes = extract_cjk_text_nodes(zh_part)
    if not zh_nodes:
        return None

    parts = re.split(r"(<[^>]+>)", en_shell)
    idx = 0
    replaced = 0

    for i, part in enumerate(parts):
        if i % 2 == 1:
            continue

        node = part
        stripped = node.strip()
        if not stripped:
            continue
        if contains_cjk(stripped):
            continue
        if not contains_latin(stripped):
            continue
        if FVTT_REF_NODE_RE.fullmatch(stripped):
            continue

        if idx < len(zh_nodes):
            leading = len(node) - len(node.lstrip())
            trailing = len(node) - len(node.rstrip())
            parts[i] = (" " * leading) + zh_nodes[idx] + (" " * trailing)
            idx += 1
            replaced += 1
        else:
            parts[i] = ""

    if replaced == 0:
        return None

    rebuilt = cleanup_empty_html_fragments("".join(parts))
    if not is_html_balanced(rebuilt):
        return None
    return rebuilt


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cleanup_empty_html_fragments(html: str) -> str:
    prev = None
    cur = html

    while cur != prev:
        prev = cur
        cur = re.sub(
            r"<(h[1-6]|p|em|strong|li|figcaption)(\s[^>]*)?>\s*</\1>",
            "",
            cur,
            flags=re.IGNORECASE,
        )
        cur = re.sub(
            r"<(article|aside|section|header|div|span)(\s[^>]*)?>\s*</\1>",
            "",
            cur,
            flags=re.IGNORECASE,
        )
        cur = re.sub(r"<ul(\s[^>]*)?>\s*</ul>", "", cur, flags=re.IGNORECASE)
        cur = re.sub(r"<ol(\s[^>]*)?>\s*</ol>", "", cur, flags=re.IGNORECASE)

    return cur


def strip_english_text_nodes_preserving_html(text: str) -> tuple[str, int]:
    if not text or not looks_like_html(text):
        return text, 0

    if not (contains_cjk(text) and contains_latin(text)):
        return text, 0

    # Safety-first strategy: only strip when we can clearly identify a
    # trailing English shell in bilingual HTML (ZH block followed by EN block).
    zh_part, en_shell = split_bilingual_html_parts(text)
    if zh_part and en_shell:
        cleaned = cleanup_empty_html_fragments(zh_part)
        return cleaned, 1

    # No clear bilingual shell detected; keep original to avoid deleting
    # inline proper nouns/brand words (e.g. Ember, Foundry VTT).
    return text, 0


def strip_english_lines_after_chinese(text: str) -> tuple[str, int]:
    if not text or "\n" not in text:
        return text, 0

    if not (contains_cjk(text) and contains_latin(text)):
        return text, 0

    lines = text.splitlines()
    keep: list[str] = []
    removed = 0
    cjk_seen = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            keep.append(line)
            continue

        has_cjk = contains_cjk(stripped)
        has_latin = contains_latin(stripped)

        if has_cjk:
            cjk_seen = True
            keep.append(line)
            continue

        if cjk_seen and has_latin and not has_cjk:
            # Preserve structural HTML lines such as </h2> and <section ...>.
            if is_html_tag_only_line(stripped):
                keep.append(line)
                continue
            if FVTT_REF_NODE_RE.fullmatch(stripped):
                keep.append(line)
                continue
            removed += 1
            continue

        keep.append(line)

    if removed == 0:
        return text, 0

    result = "\n".join(keep)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result, removed


def format_path(parts: list[Any]) -> str:
    out = "root"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def should_skip_path(path_str: str) -> bool:
    skip_suffixes = {
        ".command",  # macros/scripts
    }
    return any(path_str.endswith(s) for s in skip_suffixes)


def walk_and_strip(node: Any, parts: list[Any] | None = None, changes: list[dict[str, Any]] | None = None) -> tuple[Any, int, int]:
    if parts is None:
        parts = []
    if changes is None:
        changes = []

    changed_nodes = 0
    removed_lines = 0

    if isinstance(node, dict):
        for k, v in list(node.items()):
            new_v, c_nodes, c_lines = walk_and_strip(v, parts + [k], changes)
            node[k] = new_v
            changed_nodes += c_nodes
            removed_lines += c_lines
        return node, changed_nodes, removed_lines

    if isinstance(node, list):
        for i, v in enumerate(list(node)):
            new_v, c_nodes, c_lines = walk_and_strip(v, parts + [i], changes)
            node[i] = new_v
            changed_nodes += c_nodes
            removed_lines += c_lines
        return node, changed_nodes, removed_lines

    if isinstance(node, str):
        path_str = format_path(parts)
        if should_skip_path(path_str):
            return node, 0, 0

        original = node
        total_removed_here = 0

        current = original
        if looks_like_html(current):
            current, removed_nodes = strip_english_text_nodes_preserving_html(current)
            total_removed_here += removed_nodes
        else:
            current, removed_lines_plain = strip_english_lines_after_chinese(current)
            total_removed_here += removed_lines_plain

        if current != original:
            changes.append(
                {
                    "path": path_str,
                    "removed_segments": total_removed_here,
                    "before_preview": re.sub(r"\s+", " ", original)[:120],
                    "after_preview": re.sub(r"\s+", " ", current)[:120],
                }
            )
            return current, 1, total_removed_here

    return node, 0, 0


def resolve_files(target: Path, pattern: str, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        if recursive:
            return sorted(p for p in target.rglob(pattern) if p.is_file())
        return sorted(p for p in target.glob(pattern) if p.is_file())
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗双语 JSON：移除中文段后的英文段，避免双语导致样式布局问题")
    parser.add_argument("--target", required=True, help="目标 JSON 文件或目录")
    parser.add_argument("--glob", default="*.json", help="目录模式下文件匹配（默认 *.json）")
    parser.add_argument("--recursive", action="store_true", help="目录模式下递归子目录")
    parser.add_argument("--in-place", action="store_true", help="直接覆盖原文件")
    parser.add_argument("--output", default="", help="单文件模式下输出路径（不 in-place 时可用）")
    parser.add_argument("--backup-dir", default="", help="in-place 时备份目录（默认自动生成）")
    parser.add_argument("--report", default="strip_bilingual_english_report.json", help="报告文件路径")
    parser.add_argument("--sample-limit", type=int, default=80, help="报告中每文件保留的样本上限")
    args = parser.parse_args()

    target = Path(args.target)
    files = resolve_files(target, args.glob, args.recursive)
    if not files:
        raise FileNotFoundError(f"No JSON files found from target: {target}")

    if target.is_dir() and not args.in_place:
        print("⚠️ 目录模式默认只做 dry-run。若要写回请加 --in-place")

    backup_root: Path | None = None
    if args.in_place:
        if args.backup_dir:
            backup_root = Path(args.backup_dir)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_root = Path(f"_backup_strip_bilingual_english_{stamp}")
        backup_root.mkdir(parents=True, exist_ok=True)

    total_changed_files = 0
    total_changed_nodes = 0
    total_removed_segments = 0
    file_reports: list[dict[str, Any]] = []

    for file_path in files:
        try:
            data = load_json(file_path)
        except Exception as exc:
            file_reports.append(
                {
                    "file": str(file_path),
                    "error": str(exc),
                }
            )
            continue

        changes: list[dict[str, Any]] = []
        new_data, changed_nodes, removed_segments = walk_and_strip(data, changes=changes)

        if changed_nodes > 0:
            total_changed_files += 1
            total_changed_nodes += changed_nodes
            total_removed_segments += removed_segments

            if args.in_place:
                assert backup_root is not None
                if file_path.is_absolute():
                    drive = file_path.drive.replace(":", "")
                    rel_parts = file_path.parts[1:]
                    backup_rel = Path(drive, *rel_parts) if drive else Path(*rel_parts)
                    backup_path = backup_root / backup_rel
                else:
                    backup_path = backup_root / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                save_json(file_path, new_data)
            elif target.is_file():
                output_path = Path(args.output) if args.output else file_path.with_name(file_path.stem + ".zh-only" + file_path.suffix)
                save_json(output_path, new_data)

        file_reports.append(
            {
                "file": str(file_path),
                "changed_nodes": changed_nodes,
                "removed_segments": removed_segments,
                "samples": changes[: max(0, args.sample_limit)],
            }
        )

    report_payload = {
        "meta": {
            "target": str(target),
            "in_place": args.in_place,
            "files_scanned": len(files),
            "files_changed": total_changed_files,
            "nodes_changed": total_changed_nodes,
            "segments_removed": total_removed_segments,
            "backup_dir": str(backup_root) if backup_root else "",
        },
        "files": file_reports,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scanned: {len(files)} | Changed files: {total_changed_files} | Changed nodes: {total_changed_nodes} | Removed segments: {total_removed_segments}")
    print(f"Report: {report_path}")
    if backup_root:
        print(f"Backup: {backup_root}")


if __name__ == "__main__":
    main()
