"""strip_label_translated_suffix.py - remove an appended English block whose enricher
labels were translated along with the Chinese half.

`strip_english_suffix.py` proves an appended block is the source text by requiring the
leaf to END with the English baseline, byte for byte (modulo `<hr />` and whitespace).
That test cannot fire when an earlier tool ran `normalize_enricher_labels`-style work over
the WHOLE leaf: the appended English keeps its English sentences but its `{label}` tails
came out Chinese, so the tail no longer equals the baseline:

    …译文…</p>\n<p>The creature is @UUID[Compendium.pf2e.conditionitems.Item.fesd…]{恶心}.</p>
                                    baseline says ……………………………………………………………{Sickened 1}

Every one of those leaves renders the same rules text twice, and the duplicate carries a
live second copy of each enricher.

The evidence used here is the same one, with `{...}` bodies masked on both sides: outside
the labels the tail must still be the baseline exactly. A label is prose, so masking it
costs nothing - what proves the block is a duplicate is the English *around* the labels.

Guards, all of which must hold before a byte is dropped:

  * the leaf is not a name-class leaf (`中文 English` legitimately ends in English)
  * the head that survives still contains Chinese
  * the head is HTML-balanced on its own
  * the tail actually starts at a tag or line boundary, never mid-sentence

Usage:
  python strip_label_translated_suffix.py --cn-dir <dir> --en-dir <dir> [--write]
                                          [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
LABEL = re.compile(r"\{[^{}]*\}")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
VOID = {"br", "hr", "img", "input", "meta", "link", "col", "source", "area", "base",
        "wbr", "embed"}


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.ok = [], True

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.ok = False


def balanced(text: str) -> bool:
    parser = Balance()
    try:
        parser.feed(text)
    except Exception:
        return False
    return parser.ok and not parser.stack


def mask(text: str) -> str:
    """Everything the comparison is allowed to ignore: label bodies, `<hr />` vs `<hr>`,
    and whitespace runs.  Nothing else."""
    text = LABEL.sub("{}", text)
    text = re.sub(r"<hr\s*/?>", "<hr>", text)
    return re.sub(r"\s+", " ", text).strip()


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def split_points(text: str):
    """Offsets a duplicated block may legitimately start at: a tag opening or a newline.
    Anything else would cut a sentence in half."""
    for match in re.finditer(r"(?:^|(?<=>)|(?<=\n))\s*(?=<)", text):
        yield match.end()


def find_tail(cn: str, en: str):
    target = mask(en)
    if not target:
        return None
    for start in split_points(cn):
        if start == 0:
            continue
        if mask(cn[start:]) == target:
            return start
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--en-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    stats = Counter()
    changes = []

    for cn_file in sorted(args.cn_dir.glob("*.json")):
        en_file = args.en_dir / cn_file.name
        if not en_file.exists():
            stats["no-baseline-file"] += 1
            continue
        cn_data = json.loads(cn_file.read_text(encoding="utf-8"))
        en_flat = {".".join(p): v
                   for p, v in walk(json.loads(en_file.read_text(encoding="utf-8"))
                                    .get("entries", {}))}
        entries = cn_data.get("entries", {})
        touched = False

        for path, value in list(walk(entries)):
            if path[-1] in NAME_KEYS:
                stats["skipped-name"] += 1
                continue
            if not CJK.search(value):
                stats["no-chinese"] += 1
                continue
            baseline = en_flat.get(".".join(path))
            if not baseline:
                stats["no-baseline-key"] += 1
                continue
            start = find_tail(value, baseline)
            if start is None:
                stats["no-suffix"] += 1
                continue
            head = value[:start].rstrip()
            if not CJK.search(head):
                stats["head-has-no-chinese"] += 1
                continue
            if not balanced(head):
                stats["head-unbalanced"] += 1
                continue
            stats["stripped"] += 1
            changes.append({"file": cn_file.name, "path": ".".join(path),
                            "kept": head, "removed": value[start:]})
            node = entries
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = head
            touched = True

        if touched and args.write:
            backup = cn_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            shutil.copy2(cn_file, backup)
            cn_file.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    print(f"\nstripped {stats['stripped']} leaves"
          f"{'' if args.write else ' (dry run; pass --write)'}")
    for key, count in stats.most_common():
        print(f"    {key:24s} {count}")
    for change in changes[:3]:
        print(f"\n  {change['file']}:{change['path']}")
        print(f"      kept   : {change['kept'][:140]}")
        print(f"      removed: {change['removed'][:140]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(changes, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
