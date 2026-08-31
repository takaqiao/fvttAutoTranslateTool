import argparse
import json
import re
from pathlib import Path
from typing import Any

IGNORE_KEYS = {"Name", "name", "prototypeToken", "folder", "folders", "notes"}

# Metadata-like keys that are usually configuration, not translation content.
SKIP_KEYS = {
    "label", "path", "converter", "mapping", "uuid", "id", "_id", "type", "img", "src",
}

# Path fragments that are usually config blocks rather than player-facing localized strings.
SKIP_PATH_PARTS = {
    "mapping", "flags", "ownership", "permission", "sort", "folder", "folders", "prototypeToken",
}

# Words that are usually technical/formatting and should not be treated as missed translation.
ALLOWED_WORDS = {
    "ac", "hp", "dc", "xp", "cr", "ft", "mi", "lb", "kg",
    "str", "dex", "con", "int", "wis", "cha",
    "id", "uuid", "img", "jpg", "jpeg", "png", "svg", "webp", "gif",
    "mp3", "ogg", "wav", "webm", "http", "https", "html", "css", "js", "json",
    "ref", "uuid", "actor", "item", "token", "self", "target",
    "crucible", "foundry", "active", "effect", "group", "settings", "configuration",
    "player", "character", "compendium", "complete",
}

EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]{1,}")
EN_PHRASE_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*)+")
CJK_RE = re.compile(r"[\u3400-\u9FFF]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def should_skip_path(path: tuple[str, ...]) -> bool:
    if any(part in IGNORE_KEYS for part in path):
        return True

    if any(part in SKIP_PATH_PARTS for part in path):
        return True

    if path and path[-1] in SKIP_KEYS:
        return True

    return False


def strip_format_noise(text: str) -> str:
    out = text
    out = re.sub(r"<[^>]+>", " ", out)  # HTML tags
    out = re.sub(r"@\w+\[[^\]]*\](?:\{[^}]*\})?", " ", out)  # Foundry inline refs
    out = re.sub(r"\{\{[^}]+\}\}", " ", out)  # Handlebars
    out = re.sub(r"\[[^\]]+\]\([^\)]+\)", " ", out)  # Markdown links
    out = re.sub(r"\b\d*d\d+(?:[+-]\d+)?\b", " ", out, flags=re.IGNORECASE)  # Dice
    out = re.sub(r"\b\d+(?:\.\d+)?%?\b", " ", out)  # Plain numbers
    out = re.sub(r"[_`~*#>|\\/=+^]", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def normalize_for_compare(text: str) -> str:
    out = strip_format_noise(text)
    out = re.sub(r"[^A-Za-z0-9\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip().lower()
    return out


def collect_english_words(text: str) -> list[str]:
    words = []
    for m in EN_WORD_RE.finditer(text):
        w = m.group(0)
        lw = w.lower()
        if lw in ALLOWED_WORDS:
            continue
        if len(lw) <= 2:
            continue
        words.append(w)
    return words


def english_phrases(text: str) -> list[str]:
    phrases = []
    for m in EN_PHRASE_RE.finditer(text):
        phrase = m.group(0).strip()
        if len(phrase) >= 6:
            phrases.append(phrase)
    return phrases


def classify_suspicion(cn_text: str, en_text: str | None) -> tuple[bool, str, list[str]]:
    cn_clean = strip_format_noise(cn_text)
    if not re.search(r"[A-Za-z]", cn_clean):
        return False, "", []

    cn_words = collect_english_words(cn_clean)
    if not cn_words:
        return False, "", []

    # Bilingual working source: if the field has CJK content alongside English,
    # treat the English half as the intentional bilingual EN and only flag when
    # there is NO CJK at all (genuinely untranslated).
    if has_cjk(cn_text):
        return False, "", []

    cn_norm = normalize_for_compare(cn_text)
    en_norm = normalize_for_compare(en_text or "")

    # Strong signal: CN field is almost same as EN (after removing formatting noise).
    if en_norm and cn_norm and cn_norm == en_norm:
        return True, "exact-en-match", cn_words

    # Strong signal: CN still contains an English phrase copied from EN.
    if en_text:
        en_lower = (en_text or "").lower()
        for phrase in english_phrases(cn_clean):
            if phrase.lower() in en_lower:
                return True, "en-phrase-carried", cn_words

        overlap = {w.lower() for w in cn_words if w.lower() in en_lower}
        if len(overlap) >= 2:
            return True, "multi-en-word-overlap", cn_words

    # Medium signal: mixed CN + significant English words left.
    if has_cjk(cn_clean) and len(cn_words) >= 3:
        return True, "mixed-cn-en", cn_words

    return False, "", []


def walk(cn_node: Any, en_node: Any, path: tuple[str, ...], findings: list[dict[str, Any]]) -> None:
    if isinstance(cn_node, dict):
        for key, value in cn_node.items():
            en_value = en_node.get(key) if isinstance(en_node, dict) else None
            walk(value, en_value, path + (key,), findings)
        return

    if isinstance(cn_node, list):
        for i, value in enumerate(cn_node):
            en_value = en_node[i] if isinstance(en_node, list) and i < len(en_node) else None
            walk(value, en_value, path + (str(i),), findings)
        return

    if not isinstance(cn_node, str):
        return

    if should_skip_path(path):
        return

    en_text = en_node if isinstance(en_node, str) else None
    suspicious, reason, words = classify_suspicion(cn_node, en_text)
    if not suspicious:
        return

    findings.append(
        {
            "path": ".".join(path),
            "reason": reason,
            "english_words": sorted({w for w in words}, key=lambda x: x.lower())[:20],
            "cn_preview": cn_node[:180].replace("\n", "\\n"),
            "en_preview": (en_text or "")[:180].replace("\n", "\\n"),
        }
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scan_file(cn_file: Path, en_file: Path | None) -> list[dict[str, Any]]:
    cn_data = load_json(cn_file)
    en_data = load_json(en_file) if en_file and en_file.exists() else None

    findings: list[dict[str, Any]] = []
    walk(cn_data, en_data, (), findings)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find suspicious leftover English in CN JSON values, excluding keys: "
            "Name/name/folder/folders/notes/prototypeToken."
        )
    )
    parser.add_argument("--cn-root", default="compendium/cn", help="CN folder to scan")
    parser.add_argument("--en-root", default="compendium/en", help="EN reference folder")
    parser.add_argument("--glob", default="*.json", help="File glob under --cn-root")
    parser.add_argument("--include-lang", action="store_true", help="Also scan lang/cn.json against lang/en.json")
    parser.add_argument("--output", help="Optional output JSON path for full findings")
    parser.add_argument("--max-per-file", type=int, default=50, help="Max findings printed per file")
    args = parser.parse_args()

    cn_root = Path(args.cn_root)
    en_root = Path(args.en_root)

    all_findings: dict[str, list[dict[str, Any]]] = {}

    for cn_file in sorted(cn_root.glob(args.glob)):
        if not cn_file.is_file():
            continue
        en_file = en_root / cn_file.name
        findings = scan_file(cn_file, en_file if en_file.exists() else None)
        if findings:
            all_findings[str(cn_file)] = findings

    if args.include_lang:
        cn_lang = Path("lang/cn.json")
        en_lang = Path("lang/en.json")
        if cn_lang.exists():
            findings = scan_file(cn_lang, en_lang if en_lang.exists() else None)
            if findings:
                all_findings[str(cn_lang)] = findings

    total = sum(len(v) for v in all_findings.values())
    print(f"Scanned files with findings: {len(all_findings)}")
    print(f"Total suspicious entries: {total}")

    for file_path, findings in all_findings.items():
        print(f"\n== {file_path} ({len(findings)}) ==")
        for item in findings[: args.max_per_file]:
            print(f"- {item['path']}")
            print(f"  reason: {item['reason']}")
            print(f"  english_words: {', '.join(item['english_words'])}")
            print(f"  cn: {item['cn_preview']}")
            if item["en_preview"]:
                print(f"  en: {item['en_preview']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(all_findings, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\nSaved detailed report: {out}")


if __name__ == "__main__":
    main()
