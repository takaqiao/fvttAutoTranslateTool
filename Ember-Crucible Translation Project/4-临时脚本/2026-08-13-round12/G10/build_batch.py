# -*- coding: utf-8 -*-
"""Build G10 batch for J02 Arctus Plateau Gazetteer (twin packs)."""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"
OUT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\batches"
PACKS = ["ember.adventure.json", "ember.crucible-adventure.json"]
J = "Ember Early Access.journals.Arctus Plateau Gazetteer."

# (batch_path, [(old, new, expected_count), ...])
EDITS = [
    (J + "pages.Bloodwoods.text", [
        ("打破了浓密的白杨林", "打破了浓密的黑杨林", 1),
    ]),
    (J + "pages.Brevin.overview", [
        ("它们在这些偏小的建筑中舒适地生活着", "他们在这些偏小的建筑中舒适地生活着", 1),
    ]),
    (J + "pages.Nain.text", [
        ('<h2 class="divider">地标 - 通路区</h2><p>通路区是这座聚落的中心城区',
         '<h2 class="divider">地标 - 通衢区</h2><p>通衢区是这座聚落的中心城区', 1),
        ('<h2 class="divider">重要人物 - 通路区</h2>',
         '<h2 class="divider">重要人物 - 通衢区</h2>', 1),
    ]),
    (J + "pages.Redrak Fields.text", [
        ("还有翻土鸟，它们常在农场里惹人厌烦", "还有耕鸟，它们常在农场里惹人厌烦", 1),
        ("流淌的塔尔卡斯河所滋养", "流淌的塔尔卡河所滋养", 1),
        ("{奔草兽}", "{草地奔行者}", 1),
    ]),
    (J + "pages.Storsa's Strand.overview", [
        ("以及每夜举行、为敬奉兰提尔的逝去而进行的奇异仪式而闻名",
         "以及每夜举行、为敬奉兰提尔西沉而进行的奇异仪式而闻名", 1),
    ]),
    (J + "pages.Rustvar Valleys.text", [
        ("也带来了若干现实层面的威胁", "也带来了若干关乎存亡的威胁", 1),
    ]),
    (J + "pages.Lake Jinro Lunar Shrine.text", [
        ("而获得一项临时的宇宙恩惠骰。", "而获得一项临时的宇宙恩赐。", 1),
    ]),
]


def get(d, path):
    cur = d
    for k in path.split("."):
        cur = cur[k]
    return cur


ok = True
for pack in PACKS:
    cn = json.load(open(os.path.join(REPO, "compendium", "cn", pack), encoding="utf-8"))["entries"]
    batch = {}
    for bp, reps in EDITS:
        old_leaf = get(cn, bp)
        new_leaf = old_leaf
        for a, b, n in reps:
            cnt = new_leaf.count(a)
            if cnt != n:
                print(f"!! {pack} {bp}: expected {n} of {a!r}, found {cnt}")
                ok = False
            new_leaf = new_leaf.replace(a, b)
        if new_leaf == old_leaf:
            print(f"!! {pack} {bp}: no change")
            ok = False
        # id-preservation self-check
        if new_leaf.count('id="') < old_leaf.count('id="'):
            print(f'!! {pack} {bp}: id="" count dropped')
            ok = False
        batch[bp] = new_leaf
    p = os.path.join(OUT, f"G10.1.{pack}")
    json.dump(batch, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {p}  keys={len(batch)}")
print("OK" if ok else "PROBLEMS")
