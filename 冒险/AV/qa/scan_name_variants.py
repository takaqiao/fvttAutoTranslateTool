"""scan_name_variants.py - one entity, one rendering, even when the English keys differ.

`scan_name_consistency.py` groups bilingual name leaves by their English half, so it only
sees a conflict when two leaves carry the SAME English string. That misses the case the
standard calls out (§4②): the English keys differ by a decoration, the grouping never
happens, and the player still sees two names for one thing.

In a PF2e module the decorations are systematic:

    Ledge Creeper                 -> 攀壁藤        (the item, in `pf2e-items`)
    Effect: Ledge Creeper         -> 攀岩常春藤    (its effect, in `pf2e-misc`)

    Concealing Puffball (Lesser)  -> 遮蔽尘菌（次级）
    Effect: Concealing Puffball   -> 遮蔽马勃菌

Both halves are internally consistent, both files pass every existing check, and one
document links to the other - so the reader meets both names in one sentence.

This strips the decorations from BOTH halves before grouping:

  * an English document-class prefix (`Effect:`, `Spell Effect:`, `Aura:`, `Stance:`,
    `Feature:`) and its Chinese counterpart (`效果：`, `法术效果：`, `灵光：`, `架势：`)
  * a grade suffix in either language, in either bracket style - upstream writes
    `(Lesser)` / `(Greater)`, translations write `（次级）` / `（高阶）` / `（强效）`,
    and the two vocabularies do not line up one-to-one

What remains is the entity. A group with more than one Chinese rendering is reported with
the files each came from, so the adjudication has evidence attached. Nothing is rewritten:
the winner is a judgement call (which pack is the anchor, which wording reads better), and
that belongs in `_terms.json` with a reason, not in a heuristic.

Usage:
  python scan_name_variants.py --cn-dir <dir> [--exemptions f.json] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
BILINGUAL = re.compile(r"^(.*?[一-鿿][^\x00-\x7F]*)\s+([\x20-\x7E]+)$")

EN_PREFIX = re.compile(r"^(?:Ephemeral Effect|Spell Effect|Effect|Aura|Stance|Feature|Focus Spell)\s*:\s*",
                       re.IGNORECASE)
ZH_PREFIX = re.compile(r"^(?:瞬时效果|法术效果|效果|灵光|灵气|光环|架势|姿态|特性|聚能法术)\s*[:：]\s*")

# Upstream's grade words and every rendering of them this corpus uses. They do not line
# up one-to-one (`Greater` appears as 高级 and 高阶 and 强效), which is exactly why the
# two sides have to be stripped separately rather than compared.
EN_GRADE = r"Lesser|Moderate|Greater|Major|True|Minor"
ZH_GRADE = r"次级|次等|中级|中等|中阶|高级|高等|高阶|超级|强效|真级|真实|真|初级|低阶|主要|弱级|上等"
EN_GRADE_RE = re.compile(rf"\s*[（(](?:{EN_GRADE})[)）]\s*$", re.IGNORECASE)
ZH_GRADE_RE = re.compile(rf"\s*[（(](?:{ZH_GRADE})[)）]\s*$")


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def strip_en(text: str) -> str:
    return EN_GRADE_RE.sub("", EN_PREFIX.sub("", text)).strip()


def strip_zh(text: str) -> str:
    return ZH_GRADE_RE.sub("", ZH_PREFIX.sub("", text)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--exemptions", type=Path,
                        help='{"entities": [{"en": "...", "why": "..."}]} - entities whose '
                             "two renderings are both correct")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    exempt = set()
    if args.exemptions and args.exemptions.exists():
        exempt = {row["en"] for row in
                  json.loads(args.exemptions.read_text(encoding="utf-8")).get("entities", [])}

    groups = defaultdict(lambda: defaultdict(list))
    leaves = 0
    for cn_file in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_file.read_text(encoding="utf-8")).get("entries", {})
        for path, value in walk(data):
            if path[-1] not in NAME_KEYS or not CJK.search(value):
                continue
            match = BILINGUAL.match(value.strip())
            if not match:
                continue
            leaves += 1
            english = strip_en(match.group(2).strip())
            chinese = strip_zh(match.group(1).strip())
            if english and chinese:
                groups[english][chinese].append(f"{cn_file.stem}:{'.'.join(path[:-1])}")

    conflicts = {en: ren for en, ren in groups.items()
                 if len(ren) > 1 and en not in exempt}
    print(f"bilingual name leaves: {leaves}   entities: {len(groups)}")
    print(f"entities with more than one Chinese rendering: {len(conflicts)}"
          + (f"  (exempt: {len(exempt)})" if exempt else ""))
    for english, renderings in sorted(conflicts.items()):
        parts = " | ".join(f"{zh}×{len(where)}" for zh, where in
                           sorted(renderings.items(), key=lambda kv: -len(kv[1])))
        print(f"  {english[:42]:42s} {parts}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            [{"en": en, "renderings": {zh: where for zh, where in ren.items()}}
             for en, ren in sorted(conflicts.items())],
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
