import argparse
import json
import re
from pathlib import Path

DELETE = object()


def should_keep_bilingual(path: tuple[str, ...], key: str) -> bool:
    if key in {"name", "prototypeToken"}:
        return True
    return ("notes" in path) or ("folders" in path)


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def ensure_bilingual(cn_text: str, en_text: str) -> str:
    cn_text = cn_text.strip()
    en_text = en_text.strip()
    if not en_text:
        return cn_text
    if not cn_text:
        return en_text
    if en_text in cn_text:
        return cn_text
    return f"{cn_text} {en_text}"


def remove_english(cn_text: str, en_text: str) -> str:
    cn = cn_text.strip()
    en = en_text.strip()
    if not en:
        return cn

    if cn == en:
        return ""

    replacements = [
        f" {en}",
        f"\n{en}",
        f"/{en}",
        f" / {en}",
        f"|{en}",
        f" | {en}",
        f"（{en}）",
        f"({en})",
        f"【{en}】",
        f"[{en}]",
        en,
    ]

    out = cn
    for token in replacements:
        out = out.replace(token, "")

    return normalize_spaces(out)


def process_node(cn_node, en_node, path: tuple[str, ...]):
    changed = False

    if isinstance(cn_node, dict):
        result = {}
        for key, value in cn_node.items():
            en_value = en_node.get(key) if isinstance(en_node, dict) else None
            next_node, node_changed = process_node(value, en_value, path + (key,))
            changed = changed or node_changed
            if next_node is not DELETE:
                result[key] = next_node
            else:
                changed = True
        return result, changed

    if isinstance(cn_node, list):
        result = []
        for index, value in enumerate(cn_node):
            en_value = en_node[index] if isinstance(en_node, list) and index < len(en_node) else None
            next_node, node_changed = process_node(value, en_value, path + (str(index),))
            changed = changed or node_changed
            if next_node is not DELETE:
                result.append(next_node)
            else:
                changed = True
        return result, changed

    if not isinstance(cn_node, str):
        return cn_node, False

    key = path[-1] if path else ""
    if not isinstance(en_node, str):
        return cn_node, False

    if should_keep_bilingual(path, key):
        updated = ensure_bilingual(cn_node, en_node)
        return updated, (updated != cn_node)

    updated = remove_english(cn_node, en_node)
    if updated == "":
        return DELETE, True
    return updated, (updated != cn_node)


def main():
    parser = argparse.ArgumentParser(description="Normalize Ember CN translation with EN reference.")
    parser.add_argument("--cn", required=True, help="Path to translated json file")
    parser.add_argument("--en", required=True, help="Path to English reference json file")
    args = parser.parse_args()

    cn_path = Path(args.cn)
    en_path = Path(args.en)

    with cn_path.open("r", encoding="utf-8") as f:
        cn_data = json.load(f)

    with en_path.open("r", encoding="utf-8") as f:
        en_data = json.load(f)

    updated = dict(cn_data)
    changed = False
    for section in ("entries", "folders"):
        cn_section = cn_data.get(section) if isinstance(cn_data, dict) else None
        en_section = en_data.get(section) if isinstance(en_data, dict) else None
        if isinstance(cn_section, dict) and isinstance(en_section, dict):
            updated_section, section_changed = process_node(cn_section, en_section, (section,))
            updated[section] = updated_section
            changed = changed or section_changed

    if not changed:
        print("No changes needed.")
        return

    with cn_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Updated:", cn_path)


if __name__ == "__main__":
    main()
