# -*- coding: utf-8 -*-
"""U3 第二遍：按逐条裁决产出批次（整叶完整新文本，只替换指定的那一处）。

每条编辑 = (repo, pack, leafpath, old_fragment, new_fragment, expect_count)
读现值 → 断言片段出现次数 → 精确替换 → 写批次。任何断言不过就报错退出。
"""
import json, os, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
OUT = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"
EM = "1-Ember汉化插件"
ADV, CRU = "ember.adventure.json", "ember.crucible-adventure.json"
CB = "Actor.hDGMA97hjsiEszDf"

EDITS = [
    # --- 197：音译标签 → 明暗野兽（同英文标签全库 9:2:1，孪生包同页即 明暗野兽） ---
    (EM, CRU, "entries.Ember Early Access.journals.Local Color.pages.Matters of Perspective.text",
     f"@UUID[{CB}]{{基亚罗斯库兰野兽}}", f"@UUID[{CB}]{{明暗野兽}}", 1),
    # --- 202 / 203：与孪生包 ember.adventure 同路径写法对齐 ---
    (EM, CRU, "entries.Ember Early Access.journals.Redwalk Ramble.pages.Lower Plaza.text",
     f"@UUID[{CB}]{{明暗兽}}", f"@UUID[{CB}]{{明暗野兽}}", 1),
    (EM, CRU, "entries.Ember Early Access.journals.Redwalk Ramble.pages.Promontory.text",
     f"@UUID[{CB}]{{明暗兽}}", f"@UUID[{CB}]{{明暗野兽}}", 1),
    # --- 判据自身的错：Chiaroscuran 是形容词（英文有小写 chiaroscuran beasts），音译是机翻 ---
    (EM, ADV, "entries.Ember Early Access.actors.Chiaroscuran Beast.name",
     "基亚罗斯库兰野兽", "明暗野兽", 1),
    (EM, ADV, "entries.Ember Early Access.actors.Chiaroscuran Beast.tokenName",
     "基亚洛斯库里安野兽", "明暗野兽", 1),
    (EM, CRU, "entries.Ember Early Access.actors.Chiaroscuran Beast.name",
     "基亚罗斯库兰野兽", "明暗野兽", 1),
    (EM, CRU, "entries.Ember Early Access.actors.Chiaroscuran Beast.tokenName",
     "基亚洛斯库里安野兽", "明暗野兽", 1),
    # --- Raster Thorn 是人（六尺半的壮汉、帮派头目 "himself"/"he sleeps"），28 处标签皆作 拉斯特·索恩 ---
    (EM, ADV, "entries.Ember Early Access.actors.Raster Thorn.name",
     "栅格荆棘", "拉斯特·索恩", 1),
    (EM, ADV, "entries.Ember Early Access.actors.Raster Thorn.tokenName",
     "栅格棘刺", "拉斯特·索恩", 1),
    (EM, CRU, "entries.Ember Early Access.actors.Raster Thorn.name",
     "栅格荆棘", "拉斯特·索恩", 1),
    (EM, CRU, "entries.Ember Early Access.actors.Raster Thorn.tokenName",
     "栅格棘刺", "拉斯特·索恩", 1),
]


def leaf_get(doc, dotted):
    node = doc
    for seg in dotted.split("."):
        if isinstance(node, list):
            node = node[int(seg)]
        else:
            node = node[seg]
    return node


def main():
    packs = {}
    out = defaultdict(dict)
    bad = 0
    for repo, pack, path, old, new, n in EDITS:
        key = (repo, pack)
        if key not in packs:
            packs[key] = json.load(open(os.path.join(P, repo, "compendium", "cn", pack), encoding="utf-8"))
        bp = path[len("entries."):]
        cur = out[key].get(bp)
        if cur is None:
            cur = leaf_get(packs[key], path)
        c = cur.count(old)
        if c != n:
            print(f"!! 片段计数不符 {pack} :: {bp}  期望 {n} 实得 {c}  片段={old!r}")
            bad += 1
            continue
        out[key][bp] = cur.replace(old, new)
    if bad:
        sys.exit(f"有 {bad} 条断言失败，未写批次")
    os.makedirs(OUT, exist_ok=True)
    for (repo, pack), d in out.items():
        tag = "ember" if repo == EM else "crucible"
        fn = os.path.join(OUT, f"U3__{tag}__{pack}")
        json.dump(d, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"写出 {fn}  {len(d)} 条")


main()
