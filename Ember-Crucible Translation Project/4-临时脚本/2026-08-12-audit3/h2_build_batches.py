# -*- coding: utf-8 -*-
"""H2-B: build apply_translations batches for the FR-crosscheck term fixes.

Each rule is (english_regex_on_the_leaf, old_cn_substring, new_cn_substring),
applied ONLY to leaves whose paired ENGLISH matches — the mechanical form of
PROJECT.md's "先查英文再判中文".  Restricted to `name`/`adjective` leaves plus
explicitly listed prose paths, so no rule can rewrite unrelated narrative.

Usage:
  python h2_build_batches.py --outdir <batchdir> [--report <json>]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {"crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
         "ember": os.path.join(ROOT, "1-Ember汉化插件")}
SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# (label, english regex, cn old, cn new, leaf-name filter)
# leaf filter: 'name' -> last path segment is name/adjective; 'any' -> any leaf
RULES = [
    ("Heater Shield",   r"(?i)\bheater shield\b", "鸢形盾", "熨斗盾", "any"),
    ("Buckler",         r"(?i)\bbuckler\b",   "圆盾",   "小圆盾", "any"),
    ("Stiletto",        r"\bStiletto\b",      "细剑",   "锥刺匕首", "any"),
    ("Tenacity",        r"\bTenacity\b",      "坚韧",   "顽强",   "name"),
    # Nimbleness is currently split 敏捷 (5) / 灵巧 (1); BOTH are already spoken
    # for by `Dexterity` (敏捷 257 : 灵巧 24 under the English gate), so the
    # affix has to move off both rather than pick one of them.
    ("Nimbleness",      r"\bNimbleness\b",    "敏捷",   "轻捷",   "name"),
    ("Nimbleness",      r"\bNimbleness\b",    "灵巧",   "轻捷",   "name"),
    ("Resilient",       r"\bResilient\b",     "坚韧",   "复原体质", "name"),
    ("Justiciar",       r"\bJusticiar",       "审判官", "执法官", "any"),
    ("Trader",          r"^Trader$",          "商人",   "贸易商", "name"),
    ("Plaguebearer",    r"\bPlaguebearer\b",  "疫病先驱", "播疫者", "name"),
    ("Pestilent Tongue", r"\bPestilent Tongue\b", "疫病鞭笞", "疫病之舌", "name"),
    ("Frost Visitor",   r"\bFrost Visitor\b", "霜访者", "寒霜访客", "name"),
    ("Earth Sprite",    r"\bEarth Sprite\b",  "土精灵", "大地精灵", "name"),

    # ---- same English, two Chinese renderings; FR has exactly one each -------
    # Direction chosen by the evidence ladder: entry `name` field first, then
    # whole-library majority. Counts are in findings/H2_fr_crosscheck.json.
    ("split:Armor Crusher",  r"^Armor Crusher$",  "护甲粉碎者", "碎甲者", "name"),
    ("split:Bite",           r"^Bite$",           "噬咬",   "啃咬",   "name"),
    ("split:Venomous Bite",  r"^Venomous Bite$",  "剧毒噬咬", "剧毒啃咬", "name"),
    ("split:Claws",          r"^Claws$",          "利爪",   "爪",     "name"),
    ("split:Evasive Shot",   r"^Evasive Shot$",   "游走射击", "闪避射击", "name"),
    ("split:Headbutt",       r"^Headbutt$",       "头槌攻击", "头槌",   "name"),
    ("split:Heavy Strike",   r"^Heavy Strike$",   "重击",   "强力打击", "name"),
    ("split:Heavy Weapon Training", r"^Heavy Weapon Training$", "重武器训练", "重型武器训练", "name"),
    ("split:Luminary",       r"^Luminary$",       "辉耀者", "辉耀",   "name"),
    ("split:Restless Dead",  r"^Restless Dead$",  "不安亡魂", "躁动亡者", "name"),
    ("split:Shield Bash",    r"^Shield Bash$",    "盾击",   "盾牌猛击", "name"),
    ("split:Sunlight Weakness", r"^Sunlight Weakness$", "日光弱点", "阳光弱点", "name"),
]

NAME_LEAVES = {"name", "adjective", "tokenName"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    batches = defaultdict(dict)
    report = defaultdict(list)

    for repo_tag, repo in REPOS.items():
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(f for f in os.listdir(en_dir)
                         if f.endswith(".json") and not f.startswith("_")):
            cn_path = os.path.join(cn_dir, fn)
            if not os.path.exists(cn_path):
                continue
            with open(os.path.join(en_dir, fn), encoding="utf-8-sig") as f:
                en = json.load(f)
            with open(cn_path, encoding="utf-8-sig") as f:
                cn = json.load(f)
            rows = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
            for path, e, c in rows:
                if not c:
                    continue
                last = path.split(".")[-1]
                bp = path[len("entries."):]
                for label, rx, old, new, scope in RULES:
                    if scope == "name" and last not in NAME_LEAVES:
                        continue
                    if not re.search(rx, e):
                        continue
                    if old not in c:
                        continue
                    newc = c.replace(old, new)
                    if newc == c:
                        continue
                    batches[(repo_tag, fn)][bp] = newc
                    report[label].append({"repo": repo_tag, "pack": fn,
                                          "batch_path": bp, "en": e[:140],
                                          "old": c, "new": newc})

    for (repo_tag, fn), d in sorted(batches.items()):
        out = os.path.join(args.outdir, f"H2__{repo_tag}__{fn}")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        print(f"{len(d):4} edits -> {os.path.basename(out)}")

    for label, rows in report.items():
        print(f"  {label:18} {len(rows)} leaves")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
