import argparse
import json
from pathlib import Path


def keep_chinese_only(value: str) -> str:
    if not isinstance(value, str):
        return value
    return value


def strip_page_names(data: dict) -> int:
    removed = 0
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return removed

    for entry in entries.values():
        if not isinstance(entry, dict):
            continue

        if "name" in entry:
            entry["name"] = keep_chinese_only(entry["name"])

        pages = entry.get("pages")
        if not isinstance(pages, dict):
            continue
        if "Image" in pages:
            pages.pop("Image", None)
            removed += 1

        for page in pages.values():
            if not isinstance(page, dict):
                continue
            if "name" in page:
                page.pop("name", None)
                removed += 1
            if "text" in page:
                page["text"] = keep_chinese_only(page["text"])
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove page.name fields from dScryb JSON to reduce size.")
    parser.add_argument("input", nargs="?", default="world.dscryb.json", help="Input JSON file path")
    parser.add_argument("-o", "--output", default="world.dscryb.stripped.json", help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    removed = strip_page_names(data)

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Removed {removed} page name fields.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
