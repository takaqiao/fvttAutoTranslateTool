import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def normalize_name(name: str) -> str:
    return " ".join(name.splitlines()).strip()


def extract_text(pages: Any) -> Optional[str]:
    if not isinstance(pages, dict):
        return None
    # Prefer a "Text" page if present
    text_page = pages.get("Text")
    if isinstance(text_page, dict) and isinstance(text_page.get("text"), str):
        return text_page["text"]
    # Otherwise pick the first page with a text field
    for page in pages.values():
        if isinstance(page, dict) and isinstance(page.get("text"), str):
            return page["text"]
    return None


def json_to_md(data: Dict[str, Any]) -> str:
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return ""

    blocks = []
    for entry_id, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        text = extract_text(entry.get("pages"))
        if not isinstance(text, str) or not text.strip():
            continue
        title = normalize_name(name)
        blocks.append(f"# {title}\n**ID**：{entry_id}\n**描述**：{text}\n\n---")
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dScryb JSON to Markdown with name and text.")
    parser.add_argument("input", nargs="?", default="world.dscryb.stripped.json", help="Input JSON file path")
    parser.add_argument("-o", "--output", default="world.dscryb.md", help="Output Markdown file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    md = json_to_md(data)
    output_path.write_text(md, encoding="utf-8")
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
