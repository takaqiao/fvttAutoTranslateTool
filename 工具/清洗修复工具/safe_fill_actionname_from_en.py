import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


LATIN_RE = re.compile(r"[A-Za-z]")
ACTIONNAME_STR_RE = re.compile(r'"actionname"\s*:\s*"((?:\\.|[^"\\])*)"')
ACTIONNAME_LIST_RE = re.compile(r'"actionname"\s*:\s*(\[(?:.|\n|\r)*?\])')


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def decode_json_string(raw: str) -> str:
    return json.loads(f'"{raw}"')


def extract_braced_block(text: str, start_brace_idx: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_brace_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_brace_idx : i + 1]
    return None


def extract_actionname_from_block(block: str):
    m_list = ACTIONNAME_LIST_RE.search(block)
    if m_list:
        raw_list = m_list.group(1)
        try:
            value = json.loads(raw_list)
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                return tuple(value)
        except Exception:
            pass

    m_str = ACTIONNAME_STR_RE.search(block)
    if m_str:
        try:
            return decode_json_string(m_str.group(1))
        except Exception:
            return None

    return None


def extract_candidate_actionnames(en_text: str, item_key: str):
    target = f'"{item_key}"'
    idx = 0
    candidates = set()

    while True:
        pos = en_text.find(target, idx)
        if pos == -1:
            break

        colon = en_text.find(":", pos + len(target))
        if colon == -1:
            break

        brace = en_text.find("{", colon + 1)
        if brace == -1:
            break

        block = extract_braced_block(en_text, brace)
        if block:
            action = extract_actionname_from_block(block)
            if action is not None:
                candidates.add(action)

        idx = pos + len(target)

    return candidates


def merge_actionname(cn_value, en_value):
    if isinstance(cn_value, str) and isinstance(en_value, str):
        if has_latin(cn_value):
            return cn_value, False, "cn_has_latin"
        if not en_value.strip() or not has_latin(en_value):
            return cn_value, False, "en_empty_or_nonlatin"
        merged = f"{cn_value.strip()} {en_value.strip()}".strip()
        return merged, merged != cn_value, "ok"

    if isinstance(cn_value, list) and isinstance(en_value, tuple):
        out = list(cn_value)
        changed = False
        n = min(len(cn_value), len(en_value))
        for i in range(n):
            if isinstance(cn_value[i], str) and isinstance(en_value[i], str):
                if not has_latin(cn_value[i]) and has_latin(en_value[i]) and en_value[i].strip():
                    merged = f"{cn_value[i].strip()} {en_value[i].strip()}".strip()
                    if merged != cn_value[i]:
                        out[i] = merged
                        changed = True
        return out, changed, "ok"

    return cn_value, False, "type_mismatch"


def walk_and_fill(node, path, en_text, cache, report):
    changed_count = 0

    if isinstance(node, dict):
        item_key = path[-1] if path else None

        if item_key and "actionname" in node:
            if item_key not in cache:
                cache[item_key] = extract_candidate_actionnames(en_text, item_key)

            candidates = cache[item_key]
            if len(candidates) == 1:
                en_value = next(iter(candidates))
                new_value, changed, reason = merge_actionname(node["actionname"], en_value)
                if changed:
                    node["actionname"] = new_value
                    changed_count += 1
                    report["changed"].append({
                        "path": ".".join(path + ["actionname"]),
                        "item": item_key,
                    })
                else:
                    report["skipped"].append({
                        "path": ".".join(path + ["actionname"]),
                        "item": item_key,
                        "reason": reason,
                    })
            elif len(candidates) == 0:
                report["skipped"].append({
                    "path": ".".join(path + ["actionname"]),
                    "item": item_key,
                    "reason": "no_en_candidate",
                })
            else:
                report["skipped"].append({
                    "path": ".".join(path + ["actionname"]),
                    "item": item_key,
                    "reason": "ambiguous_en_candidate",
                    "candidate_count": len(candidates),
                })

        for k, v in node.items():
            changed_count += walk_and_fill(v, path + [k], en_text, cache, report)

    elif isinstance(node, list):
        for i, v in enumerate(node):
            changed_count += walk_and_fill(v, path + [str(i)], en_text, cache, report)

    return changed_count


def process_file(base_dir: Path, filename: str, apply: bool):
    cn_path = base_dir / filename
    en_path = base_dir / filename.replace(".json", "-en.json")
    out = {
        "file": filename,
        "changed_count": 0,
        "changed": [],
        "skipped": [],
        "errors": [],
    }

    if not cn_path.exists() or not en_path.exists():
        out["errors"].append("missing_cn_or_en_file")
        return out

    try:
        cn_obj = json.loads(cn_path.read_text(encoding="utf-8"))
    except Exception as e:
        out["errors"].append(f"cn_json_error: {e}")
        return out

    en_text = en_path.read_text(encoding="utf-8")
    cache = {}
    changed = walk_and_fill(cn_obj, [], en_text, cache, out)
    out["changed_count"] = changed

    if apply and changed > 0:
        backup_root = base_dir / f"_backup_actionname_safe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cn_path, backup_root / filename)
        cn_path.write_text(json.dumps(cn_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        out["backup"] = str(backup_root / filename)

    return out


def main():
    parser = argparse.ArgumentParser(description="Safely fill actionname from malformed -en JSON using per-item block extraction.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--files", nargs="*", default=[
        "ember.crucible-adventure.json",
        "ember.crucible-adversary.json",
        "ember.crucible-character.json",
    ])
    parser.add_argument("--report", default="safe_actionname_fill_report.json")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    report = {
        "base_dir": str(base_dir),
        "apply": args.apply,
        "files": [],
        "summary": {
            "changed": 0,
            "skipped": 0,
            "errors": 0,
        },
    }

    for fn in args.files:
        res = process_file(base_dir, fn, args.apply)
        report["files"].append(res)
        report["summary"]["changed"] += res.get("changed_count", 0)
        report["summary"]["skipped"] += len(res.get("skipped", []))
        report["summary"]["errors"] += len(res.get("errors", []))

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"changed={report['summary']['changed']} skipped={report['summary']['skipped']} errors={report['summary']['errors']}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()