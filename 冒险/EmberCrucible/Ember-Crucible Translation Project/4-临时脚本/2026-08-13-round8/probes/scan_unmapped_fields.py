# -*- coding: utf-8 -*-
"""判据：抽取器没抽的字段（LevelDB 全字段枚举 vs mappings.mjs 覆盖集）

所有既有静态检查都以「英文基准里有这一条」为起点，基准里没有的字段
不在任何检查的定义域内。本判据反过来做：先无条件枚举 LevelDB packs 里
每个文档的每个字符串叶子，再减去 mappings.mjs 声明的覆盖集，
剩下的凡是**人类可读文本**（不是 id / 枚举 / 数字 / 文件路径 / class 名 /
UUID / 颜色 / 本地化键）就是候选盲区。

数据由同目录的 census_pack_fields.mjs 采集（需要 node + classic-level）。
本脚本只做过滤与排序，不写任何 compendium/ 或 lang/。

用法:
  python scan_unmapped_fields.py --repo crucible --out out.json
  python scan_unmapped_fields.py --repo ember    --out out.json [--all]
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROUND = os.path.join(ROOT, r"4-临时脚本\2026-08-13-round8")
CENSUS = os.path.join(ROUND, "findings")
PKG_DIR = {
    "crucible": r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible",
    "ember": r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember",
}
EN_DIR = {
    "crucible": os.path.join(ROOT, r"2-Crucible汉化插件\compendium\en"),
    "ember": os.path.join(ROOT, r"1-Ember汉化插件\compendium\en"),
}

# ---------------------------------------------------------------- filters
FOUNDRY_ID = re.compile(r"^[A-Za-z0-9]{16}$")
HEXCOLOR = re.compile(r"^#[0-9a-fA-F]{3,8}$")
NUMERICISH = re.compile(r"^[-+0-9.,%/ dD*x×()\[\]@]+$")
FILEPATH = re.compile(r"^[\w./\\ +'()\-,&!]+\.(webp|png|jpg|jpeg|svg|ogg|mp3|wav|webm|mp4|json|db|otf|ttf|woff2?|m4a)$", re.I)
URLISH = re.compile(r"^(https?:|data:|modules/|systems/|icons/|worlds/|assets/|ui/|sounds/|cards/)")
UUIDISH = re.compile(r"^(Compendium|Actor|Item|JournalEntry|Scene|Macro|RollTable|Cards|Folder)\.")
LOCKEY = re.compile(r"^[A-Z][A-Z0-9]*(\.[A-Za-z0-9_]+)+$")   # CRUCIBLE.Foo.Bar
SLUG = re.compile(r"^[a-z0-9*]+([-_.:][a-z0-9*]+)*$")         # enum id / 引用 slug: two-handed, languages:standard:common
CSSISH = re.compile(r"^[a-z0-9]+([- ][a-z0-9]+)*$")
DICEEXPR = re.compile(r"^[0-9dD+\-*/ ().@a-z_]+$")
HASLETTER = re.compile(r"[A-Za-z]")
CJK = re.compile(r"[\u4e00-\u9fff]")

# 字段名本身就说明它不是正文的路径尾（id / 枚举 / 资源 / 技术键）
TECH_TAIL = re.compile(
    r"(^|\.)(_id|_key|id|key|uuid|src|img|icon|texture|path|type|subtype|folder|sort|"
    r"color|colour|tint|font|fontFamily|scale|mode|status|group|category|slug|"
    r"origin|parent|module|system|version|author|_stats|coreVersion|systemVersion|"
    r"createdTime|modifiedTime|lastModifiedBy|compendiumSource|duplicateSource|"
    r"exportSource|ownership|permission|flags|macro|command|scope|hotbar|"
    r"statuses|priority|formula|denomination|faces|results\[\]\.type)"
    r"(\[\])?$"
)
# `value` 只在这些技术容器下才是技术字段。曾经把 `value` 无条件打进 TECH_TAIL，
# 结果把 dnd5e 的 `system.description.value`（150 万字符的正文）整类漏掉 ——
# 灵敏度回测就是这么翻出来的，别再收紧成通配。
TECH_VALUE = re.compile(r"(^|\.)(changes|attributes|resources|abilities|skills|currency|"
                        r"movement|senses|traits|bonuses|scale|uses|spells|ac|hp|init|"
                        r"details\.level|details\.cr|xp|prof|attunement)\b.*\.value(\[\])?$")
# 上面 TECH_TAIL 里 flags 只挡尾节点，flags.* 子树整体也不要。
# `system.sounds` / `system.source` / advancement 的 configuration|value 子树整片是
# 音频 id、书目 id 与 `languages:standard:common` 这类引用 slug，不是正文。
TECH_PREFIX = re.compile(
    r"(^|\.)(flags|_stats|ownership|permission|filePathFields|"
    r"system\.sounds|system\.source)(\.|\[\]|$)"
    r"|system\.advancement(\[\]|\.<id>)\.(configuration|value)(\.|\[\]|$)")

# 收紧轮加的：字段名本身就是 id / 枚举 / 引用槽的，值再像人话也不是正文。
# 每一条都对应一族实测假阳性，改的时候连样例一起看：
#   identifier=healingElixir  rarity=uncommon  expiry=turnEnd  eventId=oozeControlIntro
#   locationId=shentMoonTemple  locations[]=theHallows  biomes[]=ordainFlats
#   favorites[]=strike  tags[]=iconicSpell  doorSound=woodBasic  style=solidLines
#   coefficients={"constant":"Infinity"}  scene.active.level=veiledChainHQ
ENUM_TAIL = re.compile(
    r"(^|\.)(identifier|rarity|expiry|period|eventId|locationId|locations|biomes|"
    r"favorites|tags|doorSound|style|coefficients|sourceItem|lastAction|"
    r"soundscape|arrangements|units|recovery|activation\.type|level|"
    r"script|grants|chosen|pool|restriction\.list)(\[\])?$")


CAMEL = re.compile(r"^[a-z][A-Za-z0-9]*$")           # healingElixir / turnEnd / solidLines
JSONISH = re.compile(r"^\s*[\{\[].*[\}\]]\s*$", re.S)
JSCODE = re.compile(r"(=>|\bawait\b|\bconst\b|\blet\b|\bfunction\b|\breturn\b)\s")
ONLY_ENRICHER = re.compile(r"^\s*(@(UUID|Embed|Check|Damage|Lookup)\[[^\]]*\](\{[^}]*\})?|\[\[[^\]]*\]\])+\s*$")


def human_readable(v: str) -> bool:
    """值看起来是给玩家看的文本吗。"""
    s = v.strip()
    if len(s) < 2:
        return False
    if not HASLETTER.search(s) and not CJK.search(s):
        return False
    if FOUNDRY_ID.match(s) or HEXCOLOR.match(s) or NUMERICISH.match(s):
        return False
    if FILEPATH.match(s) or URLISH.match(s) or UUIDISH.match(s) or LOCKEY.match(s):
        return False
    if "/" in s and " " not in s:
        return False
    if JSONISH.match(s) or JSCODE.search(s) or ONLY_ENRICHER.match(s):
        return False
    if CJK.search(s):
        return True
    if SLUG.match(s) or CSSISH.match(s) or CAMEL.match(s):
        return False          # 纯小写 slug / camelCase 枚举 id / class 名
    if DICEEXPR.match(s) and not s[0].isupper():
        return False
    # 到这里：含大写字母、空格、标点或 HTML —— 当作人类可读
    return True


def load_en_values(repo):
    """英文基准里出现过的全部字符串叶子（用来判断这条是不是真的没进基准）。"""
    vals = set()

    def leaves(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from leaves(v)
        elif isinstance(o, list):
            for v in o:
                yield from leaves(v)
        elif isinstance(o, str):
            yield o

    d = EN_DIR[repo]
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            for s in leaves(json.load(f)):
                vals.add(s)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, choices=["crucible", "ember"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--recensus", action="store_true", help="强制重跑 node 普查")
    ap.add_argument("--all", action="store_true", help="连被过滤掉的路径也一起输出")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--census", help="改用指定的普查文件（回测注入用，判据逻辑不变）")
    a = ap.parse_args()

    census_path = a.census or os.path.join(CENSUS, f"census_{a.repo}.json")
    if a.recensus or not os.path.exists(census_path):
        subprocess.run(
            ["node", os.path.join(ROUND, "probes", "census_pack_fields.mjs"),
             "--package", PKG_DIR[a.repo], "--out", census_path],
            cwd=r"C:\Users\Taka\Desktop\fvtt", check=True)

    with open(census_path, encoding="utf-8") as f:
        census = json.load(f)
    en_vals = load_en_values(a.repo)

    cand, rejected = [], []
    for r in census["rows"]:
        if r["mapped"]:
            continue
        p = r["relpath"]
        reason = None
        if TECH_PREFIX.search(p):
            reason = "tech-subtree"
        elif TECH_VALUE.search(p):
            reason = "tech-value-container"
        elif TECH_TAIL.search(p):
            reason = "tech-field-name"
        elif ENUM_TAIL.search(p):
            reason = "enum-or-reference-slot"
        else:
            hr = [s for s in r["samples"] if human_readable(s["v"])]
            # 半数以上样例不是人话 -> 整条按枚举处理（`alignment` 这种
            # 「Unaligned / Neutral Good」混着枚举与自由文本的字段靠这条挡掉大半）
            if len(hr) * 2 < len(r["samples"]):
                reason = "no-human-readable-sample"
        if reason:
            rejected.append({**r, "reject": reason})
            continue
        # 这条的样例值有多少已经在英文基准里出现过（同名字段可能被别处抽走）
        in_en = sum(1 for s in r["samples"] if s["v"] in en_vals)
        if r["chars"] < a.min_chars:
            rejected.append({**r, "reject": "below-min-chars"})
            continue
        cand.append({**r, "samples_in_en_baseline": in_en, "samples_total": len(r["samples"])})

    cand.sort(key=lambda x: (-x["chars"], -x["n"]))
    out = {
        "repo": a.repo,
        "package": census["package"],
        "version": census["version"],
        "docs_walked": census["docsWalked"],
        "field_paths_total": census["fieldPaths"],
        "mapped_paths": sum(1 for r in census["rows"] if r["mapped"]),
        "unmapped_paths": sum(1 for r in census["rows"] if not r["mapped"]),
        "candidates": len(cand),
        "rows": cand,
    }
    if a.all:
        out["rejected"] = rejected
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{a.repo}: docs={out['docs_walked']} paths={out['field_paths_total']} "
          f"mapped={out['mapped_paths']} unmapped={out['unmapped_paths']} "
          f"candidates={out['candidates']}")
    for r in cand[:40]:
        st = f"{r['documentType']}{'.' + r['subtype'] if r['subtype'] else ''}"
        print(f"  {r['chars']:>8}ch n={r['n']:<5} uniq={r['uniq']:<5} "
              f"inEN={r['samples_in_en_baseline']}/{r['samples_total']}  {st}  {r['relpath']}")


if __name__ == "__main__":
    main()
