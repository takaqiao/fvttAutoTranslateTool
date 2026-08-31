"""Normalize PF2e spell tradition / preparation / Lore term inconsistencies
to pf2_cn canonical forms.

Canonical pf2_cn terms (from system/pf2_cn/zh_Hans/zh_Hans.json):
  PF2E.TraitArcane         -> 奥术
  PF2E.TraitDivine         -> 神术
  PF2E.TraitOccult         -> 异能
  PF2E.TraitPrimal         -> 原能
  PF2E.TraitComposition    -> 组曲
  PF2E.PreparationTypeInnate      -> 内在
  PF2E.PreparationTypePrepared    -> 准备
  PF2E.PreparationTypeSpontaneous -> 自发
  PF2E.PreparationTypeFocus       -> 聚能
  PF2E.Focus.Spells               -> 聚能法术
  PF2E.LoreSkillFormat            -> {name}学识

Strategy: exact-match replacement of full bilingual spell-header strings
(e.g., "天授原始法术 Innate Primal Spells" -> "内在原能法术 Innate Primal Spells").
This avoids substring confusion (e.g., 信念 used non-tradition contexts as "faith").

Usage:
  python normalize_tradition_terms.py [target_dir]
  default target: FotRP/需要翻译/
"""
import json
import re
import sys
import os
from pathlib import Path

# Master replacement table: old bilingual string -> new bilingual string
# Built from observed FotRP variants → pf2_cn canonical
REPLACEMENTS = {
    # Prepared <Tradition>
    "准备原始法术 Prepared Primal Spells": "准备原能法术 Prepared Primal Spells",
    "准备奥能法术 Prepared Arcane Spells": "准备奥术法术 Prepared Arcane Spells",
    "准备奥法法术 Prepared Arcane Spells": "准备奥术法术 Prepared Arcane Spells",
    "准备神秘法术 Prepared Occult Spells": "准备异能法术 Prepared Occult Spells",
    "准备秘学法术 Prepared Occult Spells": "准备异能法术 Prepared Occult Spells",
    "准备神圣法术 Prepared Divine Spells": "准备神术法术 Prepared Divine Spells",
    "准备信念法术 Prepared Divine Spells": "准备神术法术 Prepared Divine Spells",

    # Innate <Tradition> (天授/天生 -> 内在; 奥能/奥法/秘学/神秘/神圣/信念/原始 -> canonical)
    "天授奥能法术 Innate Arcane Spells": "内在奥术法术 Innate Arcane Spells",
    "天授奥法法术 Innate Arcane Spells": "内在奥术法术 Innate Arcane Spells",
    "天授奥术法术 Innate Arcane Spells": "内在奥术法术 Innate Arcane Spells",
    "天授信念法术 Innate Divine Spells": "内在神术法术 Innate Divine Spells",
    "天授神圣法术 Innate Divine Spells": "内在神术法术 Innate Divine Spells",
    "天授神术法术 Innate Divine Spells": "内在神术法术 Innate Divine Spells",
    "天授神秘法术 Innate Occult Spells": "内在异能法术 Innate Occult Spells",
    "天授秘学法术 Innate Occult Spells": "内在异能法术 Innate Occult Spells",
    "天授异能法术 Innate Occult Spells": "内在异能法术 Innate Occult Spells",
    "天授原始法术 Innate Primal Spells": "内在原能法术 Innate Primal Spells",
    "天授原能法术 Innate Primal Spells": "内在原能法术 Innate Primal Spells",
    "天生奥能法术 Innate Arcane Spells": "内在奥术法术 Innate Arcane Spells",
    "天生奥法法术 Innate Arcane Spells": "内在奥术法术 Innate Arcane Spells",
    "天生秘学法术 Innate Occult Spells": "内在异能法术 Innate Occult Spells",
    "天生神秘法术 Innate Occult Spells": "内在异能法术 Innate Occult Spells",
    "天生神圣法术 Innate Divine Spells": "内在神术法术 Innate Divine Spells",
    "天生信念法术 Innate Divine Spells": "内在神术法术 Innate Divine Spells",
    "天生原能法术 Innate Primal Spells": "内在原能法术 Innate Primal Spells",
    "天生原始法术 Innate Primal Spells": "内在原能法术 Innate Primal Spells",

    # Reversed-English-order: "Tradition Innate Spells" / "Tradition Prepared Spells"
    "原始天授法术 Primal Innate Spells": "原能内在法术 Primal Innate Spells",
    "原能天授法术 Primal Innate Spells": "原能内在法术 Primal Innate Spells",
    "奥术天生法术 Arcane Innate Spells": "奥术内在法术 Arcane Innate Spells",
    "奥术天授法术 Arcane Innate Spells": "奥术内在法术 Arcane Innate Spells",
    "神秘天授法术 Occult Innate Spells": "异能内在法术 Occult Innate Spells",
    "神秘准备法术 Occult Prepared Spells": "异能准备法术 Occult Prepared Spells",

    # Spontaneous <Tradition>
    "自发奥能法术 Spontaneous Arcane Spells": "自发奥术法术 Spontaneous Arcane Spells",
    "自发奥法法术 Spontaneous Arcane Spells": "自发奥术法术 Spontaneous Arcane Spells",
    "自发神秘法术 Spontaneous Occult Spells": "自发异能法术 Spontaneous Occult Spells",
    "自发秘学法术 Spontaneous Occult Spells": "自发异能法术 Spontaneous Occult Spells",
    "自发神圣法术 Spontaneous Divine Spells": "自发神术法术 Spontaneous Divine Spells",
    "自发信念法术 Spontaneous Divine Spells": "自发神术法术 Spontaneous Divine Spells",
    "自发原始法术 Spontaneous Primal Spells": "自发原能法术 Spontaneous Primal Spells",

    # Composition
    "吟游诗人即兴法术 Bard Composition Spells": "诗人组曲法术 Bard Composition Spells",
    "诗人即兴法术 Bard Composition Spells": "诗人组曲法术 Bard Composition Spells",
    "诗人组曲法术 Bard Composition Spells": "诗人组曲法术 Bard Composition Spells",  # already correct (no-op)

    # Focus
    "焦点法术 Focus Spells": "聚能法术 Focus Spells",
    "神圣专注法术 Divine Focus Spells": "神术聚能法术 Divine Focus Spells",
    "信念聚能法术 Divine Focus Spells": "神术聚能法术 Divine Focus Spells",

    # Order
    "德鲁伊教团法术 Druid Order Spells": "德鲁伊教派法术 Druid Order Spells",

    # Constant (no canonical issue, kept for reference)
    # "常驻法术 Constant Spells": "常驻法术 Constant Spells",  # leave as-is
}


def walk_replace(obj, stats):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, str) and v in REPLACEMENTS:
                if obj[k] != REPLACEMENTS[v]:
                    obj[k] = REPLACEMENTS[v]
                    stats["replaced"] += 1
            else:
                walk_replace(v, stats)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            v = obj[i]
            if isinstance(v, str) and v in REPLACEMENTS:
                if obj[i] != REPLACEMENTS[v]:
                    obj[i] = REPLACEMENTS[v]
                    stats["replaced"] += 1
            else:
                walk_replace(v, stats)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "FotRP/需要翻译"
    if not os.path.isdir(target):
        print(f"Target dir not found: {target}")
        sys.exit(1)
    total_replaced = 0
    files_changed = 0
    for root, dirs, files in os.walk(target):
        # Skip backup / NEW / cache dirs
        if any(s in root for s in ["NEW", "_backup", "_pdf", "_zh_synthetic", "_qa_reports"]):
            continue
        for f in files:
            if not f.endswith(".json"):
                continue
            fp = os.path.join(root, f)
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception as e:
                print(f"  skip (parse): {f}: {e}")
                continue
            stats = {"replaced": 0}
            walk_replace(data, stats)
            if stats["replaced"] > 0:
                with open(fp, "w", encoding="utf-8") as wf:
                    json.dump(data, wf, ensure_ascii=False, indent=2)
                print(f"  {f}: {stats['replaced']} replacements")
                total_replaced += stats["replaced"]
                files_changed += 1
    print(f"\nTotal: {total_replaced} replacements across {files_changed} files")


if __name__ == "__main__":
    main()
