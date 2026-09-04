"""normalize_grade_suffix.py - one English grade word, one Chinese grade word.

PF2e grades an item by appending `(Lesser)`, `(Moderate)`, `(Greater)`, `(Major)` or
`(True)` to its name. A bilingual name leaf carries the English half verbatim, so the leaf
states its own grade - which makes this the one naming question that never needs a vote:

    阿林达姆之鞭（强效） Arindham's Whip (Greater)
    精金龙息槌（中阶）   Adamantine Dragonbreath Maul (Greater)
    炼金天使（高级）     Alchemical Angel (Greater)

Three renderings of one grade, and the middle one is worse than inconsistent: `中阶` reads
as the MODERATE tier, so the item's name tells the player the wrong tier. Nothing else in
the pipeline can see this - each leaf is internally valid, the English keys differ, and
`scan_name_variants.py` strips the suffix off both sides before it compares them.

The rewrite touches only the parenthetical at the very end of the Chinese half, and only
when the English half ends with a grade word. Prose is never touched, and a Chinese name
that ends in parentheses for another reason (`效果：揭示秘示域（军械）`) is left alone
because its English half carries no grade.

`--grades` overrides the table. The default is this corpus's majority rendering, which is
NOT the official compendium's: `pf2e_compendium_chn` uses a PREFIX (`高等烧蚀护甲甲片`),
and converting a suffix corpus to a prefix corpus rewrites every name, every pack label
that quotes one and every enricher label that echoes one. Vocabulary is worth unifying;
shape is not worth that.

Usage:
  python normalize_grade_suffix.py --cn-dir <dir> [--grades g.json] [--write] [--report r.json]
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
GRADES = {"Minor": "初级", "Lesser": "次级", "Moderate": "中级",
          "Greater": "高级", "Major": "超级", "True": "真级"}
# The English half is the last ASCII run of the leaf; the grade is its final parenthetical.
EN_GRADE = re.compile(r"\((Minor|Lesser|Moderate|Greater|Major|True)\)\s*$", re.IGNORECASE)
BILINGUAL = re.compile(r"^(.*[一-鿿][^\x00-\x7F]*)(\s+)([\x20-\x7E]+)$")
ZH_PAREN = re.compile(r"[（(]\s*([^（）()]{1,6})\s*[)）]\s*$")


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--grades", type=Path, help='{"Greater": "高级", ...}')
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    table = dict(GRADES)
    if args.grades and args.grades.exists():
        table.update({k: v for k, v in
                      json.loads(args.grades.read_text(encoding="utf-8")).items()
                      if not k.startswith("_")})

    stats, changes = Counter(), []
    for cn_file in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_file.read_text(encoding="utf-8"))
        entries = data.get("entries", {})
        touched = 0
        for path, value in list(walk(entries)):
            if path[-1] not in NAME_KEYS:
                continue
            split = BILINGUAL.match(value.strip())
            if not split:
                continue
            chinese, gap, english = split.groups()
            grade = EN_GRADE.search(english)
            if not grade:
                stats["no-grade"] += 1
                continue
            want = table.get(grade.group(1).title())
            if not want:
                continue
            found = ZH_PAREN.search(chinese)
            if not found:
                stats["chinese-has-no-suffix"] += 1
                changes.append({"file": cn_file.stem, "path": ".".join(path),
                                "value": value, "verdict": "no-chinese-suffix",
                                "grade": grade.group(1)})
                continue
            if found.group(1) == want:
                stats["already-right"] += 1
                continue
            new_zh = chinese[: found.start()] + f"（{want}）"
            new = f"{new_zh}{gap}{english}"
            node = entries
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = new
            stats[f"{grade.group(1)}: {found.group(1)} -> {want}"] += 1
            changes.append({"file": cn_file.stem, "path": ".".join(path),
                            "before": value, "after": new, "verdict": "rewritten"})
            touched += 1
        if touched:
            print(f"[{'write' if args.write else 'dry  '}] {cn_file.stem:52s} names={touched}")
            if args.write:
                shutil.copy2(cn_file,
                             cn_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
                cn_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")

    rewritten = sum(v for k, v in stats.items() if "->" in k)
    print(f"\n{rewritten} grade suffixes rewritten"
          f"{'' if args.write else ' (dry run; pass --write)'}"
          f"   already right: {stats['already-right']}"
          f"   Chinese half has no suffix: {stats['chinese-has-no-suffix']}")
    for key, count in sorted(stats.items()):
        if "->" in key:
            print(f"    {key:34s} x{count}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(changes, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
