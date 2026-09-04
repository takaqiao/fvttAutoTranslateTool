"""strip_prose_parentheticals.py - take the English gloss back out of prose.

The project standard splits the two styles cleanly: a NAME leaf is `中文 English`, and
everything else is pure Chinese. A translator who carries the name convention into running
text produces a third thing that no check was looking for:

    你从野蛮人获得的训练技能为特技（Acrobatics）而非运动（Athletics）
    你获得「旋风突进（Whirling Advance）」动作

Each gloss is short, so the >=60-character English-run test never fires; each leaf is
overwhelmingly Chinese, so the coverage ratio never fires either. The text simply reads
twice as long as it should.

The same passes leave behind the gloss's empty shell when the thing inside it was an
enricher that a later tool removed:

    你不会获得 Quick Tempered 自由动作（）

Two rules, both narrow:

  * `（<latin>）` immediately after a Chinese character is dropped - but only when the
    content is a WORD. `（AC 10）`, `（DC 39）`, `（1d6）` are values the standard keeps in
    Latin, and a bare capitalised abbreviation (`GM`, `NPC`, `PC`, `XP`, `HP`) is a term
    the standard keeps in Latin too.
  * an empty `（）` / `()` is dropped along with a space in front of it.

Both leave the surrounding punctuation alone, and neither ever touches a name-class leaf.

Usage:
  python strip_prose_parentheticals.py --cn-dir <dir> [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# A gloss: parentheses right after a Chinese character holding Latin words.
GLOSS = re.compile(r"(?<=[一-鿿])[（(]\s*([A-Za-z][A-Za-z0-9 '’\-\.]{1,40}?)\s*[)）]")
EMPTY = re.compile(r"[ 　]?[（(]\s*[)）]")
# Kept in Latin by the standard, so a parenthetical carrying one is not a gloss.
KEEP = {"AC", "DC", "HP", "XP", "GM", "NPC", "PC", "TN", "PF", "SF"}
HAS_DIGIT = re.compile(r"\d")


def is_gloss(body: str) -> bool:
    """A gloss names a thing; a value states a number."""
    if HAS_DIGIT.search(body):
        return False
    return not all(word.upper() in KEEP for word in body.split())


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def clean(text: str, stats: Counter):
    def drop(match):
        if is_gloss(match.group(1)):
            stats[f"gloss:{match.group(1)}"] += 1
            return ""
        stats["kept-value"] += 1
        return match.group(0)

    text = GLOSS.sub(drop, text)
    text, n = EMPTY.subn("", text)
    stats["empty-parens"] += n
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    stats = Counter()
    changes = []
    for cn_file in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_file.read_text(encoding="utf-8"))
        entries = data.get("entries", {})
        touched = 0
        for path, value in list(walk(entries)):
            if path[-1] in NAME_KEYS:
                continue
            new = clean(value, stats)
            if new == value:
                continue
            node = entries
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = new
            changes.append({"file": cn_file.stem, "path": ".".join(path),
                            "before": value, "after": new})
            touched += 1
        if touched:
            print(f"[{'write' if args.write else 'dry  '}] {cn_file.stem:52s} leaves={touched}")
            if args.write:
                shutil.copy2(cn_file,
                             cn_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
                cn_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")

    glosses = {k[6:]: v for k, v in stats.items() if k.startswith("gloss:")}
    print(f"\nglosses removed {sum(glosses.values())} in {len(changes)} leaves"
          f"{'' if args.write else ' (dry run; pass --write)'}"
          f"   empty parens {stats['empty-parens']}   values kept {stats['kept-value']}")
    for term, count in sorted(glosses.items(), key=lambda kv: -kv[1])[:20]:
        print(f"    x{count:<3d} （{term}）")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(changes, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
