"""normalize_gmguide.py - term pass over the GM guide, without touching its English source.

The GM guide carries its own copy of the source text: `.translated.pages.json` keeps
`vision_extracted_text` / `text_reference` next to `translated`, and the `.md` keeps the
same inside `<details>` blocks.  Those are the record of what was translated FROM and
must stay byte-identical, so the generic `normalize_terms.py` (which walks every string
leaf) cannot be pointed at these files: a rule like `Cynemi -> 西涅米` would rewrite the
English source too.

So this applies the same kind of rules, to the translated text only.

Usage:
  python normalize_gmguide.py --terms _gmguide_terms.json --pages <pages.json>
                              [--md <file.md>] [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

DETAILS = re.compile(r"<details>.*?</details>", re.S)


def load_terms(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def apply_terms(text, terms, counter):
    for wrong, right in terms.items():
        if wrong in text:
            counter[wrong] += text.count(wrong)
            text = text.replace(wrong, right)
    return text


def do_pages(path, terms, counter, write, stamp):
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for page in data:
        before = page.get("translated", "")
        after = apply_terms(before, terms, counter)
        if after != before:
            page["translated"] = after
            changed += 1
    if changed and write:
        backup = path.parent / "_backup" / f"gmguide_{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / path.name)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    return changed


def do_markdown(path, terms, counter, write, stamp):
    text = path.read_text(encoding="utf-8")
    # Protect every <details> block (the extracted English) before replacing.
    blocks = []

    def stash(m):
        blocks.append(m.group(0))
        return f"\x00BLOCK{len(blocks) - 1}\x00"

    masked = DETAILS.sub(stash, text)
    fixed = apply_terms(masked, terms, counter)
    for i, block in enumerate(blocks):
        fixed = fixed.replace(f"\x00BLOCK{i}\x00", block)
    if fixed == text:
        return 0
    if write:
        backup = path.parent / "_backup" / f"gmguide_{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / path.name)
        path.write_text(fixed, encoding="utf-8", newline="\n")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terms", required=True, type=Path)
    parser.add_argument("--pages", required=True, type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    terms = load_terms(args.terms)
    counter = Counter()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"{len(terms)} rules")

    n_pages = do_pages(args.pages, terms, counter, args.write, stamp)
    print(f"[{'write' if args.write else 'dry  '}] pages.json: {n_pages} pages changed")
    if args.md and args.md.exists():
        n_md = do_markdown(args.md, terms, counter, args.write, stamp)
        print(f"[{'write' if args.write else 'dry  '}] markdown : {'changed' if n_md else 'unchanged'}")

    print(f"\ntotal replacements: {sum(counter.values())}")
    for key, n in counter.most_common():
        print(f"    {n:>4}  {key} -> {terms[key]}")
    unused = [k for k in terms if k not in counter]
    if unused:
        print(f"\nrules that matched nothing ({len(unused)}): {unused}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"counts": dict(counter), "unused": unused},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
