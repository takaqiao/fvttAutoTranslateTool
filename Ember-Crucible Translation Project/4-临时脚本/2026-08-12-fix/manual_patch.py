# -*- coding: utf-8 -*-
"""人工核对过、但自动判据证不出来的几叶（标签写法与文档名不一致、或目标文档
不在 ember 的 packs 里，文档名证据链断在那里）。

每条 spec 都由我逐句读过英文与中文后写下：给出「同叶内哪个标签应当挂哪个目标」。
脚本只做机械换位，并做与自动流程完全相同的两条自检：
  ① 可见文字逐字不变  ② @UUID[...] 多重集逐字不变
"""
import json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uuid_fix2 import load_pairs, links, key_of, UUID_RX

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"
BASE = os.path.dirname(os.path.abspath(__file__))
MAIN, TWIN = "ember.crucible-adventure.json", "ember.adventure.json"

# (pack, 路径尾, [(标签, 第几次出现, 目标文档名或 id)])
SPECS = [
    # EN: principal architects were the {Aedir Pantheon->Solaru} of {Shard Gods}
    (both := "both", "History.pages.Age of the Tower.contentGamemaster",
     [("碎片诸神", 0, "Shard Gods"), ("艾迪尔万神殿", 0, "Solaru")]),
    # EN: the only Kirargy Knight to ever bond with a {Cruel Dragon} in the
    #     history of the {Tayan Kingdoms->Tayan}
    (both, "History.pages.Night of Swords.text",
     [("塔扬诸王国", 0, "Tayan"), ("残酷龙", 0, "Cruel Dragons")]),
    # EN: High Priest of the Temple of {Sockets} in {Ordain}
    (MAIN, "actors.Conaris Haid.biography.private",
     [("奥尔丹", 0, "Ordain"), ("插孔神殿", 0, "Sockets")]),
    # EN: dominance over {inkaro pearl->Inkaro Pearl, White} trade and destabilize
    #     {House Cevher}, at the behest of {Zerranyss->v6lXjfETDkrt0J7E}
    (TWIN, "Glitter in the Dark.pages.The Story So Far.text",
     [("泽拉尼丝", 0, "v6lXjfETDkrt0J7E"), ("因卡罗珍珠", 0, "Inkaro Pearl, White"),
      ("杰夫赫尔家族", 0, "House Cevher")]),
    # EN: an {Arcturian} {Caravanner} on a journey back to {Ordain}, or even a
    #     {Tayan} {Star Dreamer} from far-off {Caren'ac}
    (both, "Players' Guide.pages.Creation Overview.text",
     [("奥尔丹", 0, "4gRSL7Tq1pgccdIW"), ("阿克图里亚人", 1, "Arcturian"),
      ("商队旅人", 0, "Caravanner"), ("卡伦阿克", 0, "Caren'ac"),
      ("塔扬", 0, "Tayan"), ("星辰逐梦者", 0, "Star Dreamer")]),
    # EN: tracks completed {Events->Quests and Events} as the party moves across
    #     the {Region Map->Region Exploration}
    (both, "Players' Guide.pages.The Codex.text",
     [("区域地图", 0, "Region Exploration"), ("事件", 0, "Quests and Events")]),
    # EN: a {Waerd} temple in the {Lowland Bastions->Lowlands} …
    #     traced back to the {Oaken} of {Oakengarde}
    (both, "Character Classes.pages.Monk.text",
     [("低地堡垒", 0, "Lowlands"), ("瓦尔德", 0, "Waerd"),
      ("奥肯加德", 0, "Oakengarde"), ("奥肯", 0, "Oaken")]),
]

ids = json.load(open(os.path.join(BASE, "reports", "ember_ids.json"), encoding="utf-8"))


def en_name(k):
    r = ids.get(k.split("#")[0])
    return r.get("name") if r else None


def patch(pack, tail, spec, out, log):
    for r in load_pairs(REPO, pack):
        if not r["cn"] or not r["path"].endswith(tail):
            continue
        cl = [(t, l, s, e) for t, l, s, e in links(r["cn"]) if l]
        ck = [key_of(t) for t, _, _, _ in cl]
        idx, seen = [], Counter()
        for lab, occ, _ in spec:
            hits = [j for j in range(len(cl)) if cl[j][1] == lab]
            if occ >= len(hits):
                raise SystemExit(f"{pack} {tail}: 标签「{lab}」第 {occ} 次出现找不到")
            idx.append(hits[occ])
        want = []
        for (lab, occ, tgt), j in zip(spec, idx):
            cands = [x for x in idx if ck[x] == tgt or en_name(ck[x]) == tgt]
            if len(cands) != 1:
                raise SystemExit(f"{pack} {tail}: 「{lab}」的目标 {tgt} 在指定位置里"
                                 f"匹配到 {len(cands)} 个")
            want.append(cands[0])
        new = [t for t, _, _, _ in cl]
        for j, src in zip(idx, want):
            new[j] = cl[src][0]
        s, parts, prev = r["cn"], [], 0
        for (t, l, st, en_), nt in zip(cl, new):
            parts.append(s[prev:st]); parts.append(f"@UUID[{nt}]{{{l}}}"); prev = en_
        parts.append(s[prev:])
        fixed = "".join(parts)
        vis = lambda x: UUID_RX.sub(lambda m: (m.group(3) or ""), x)
        assert vis(fixed) == vis(s), tail
        assert Counter(x[0] for x in links(fixed)) == Counter(x[0] for x in links(s)), tail
        assert fixed != s, f"{pack} {tail}: 没有实际改动，spec 可能已过期"
        out[r["batch_path"]] = fixed
        log.append({"pack": pack, "path": r["path"],
                    "moved": [{"pos": j, "label": cl[j][1],
                               "from": en_name(ck[j]) or cl[j][0],
                               "to": en_name(key_of(new[j])) or new[j]}
                              for j in idx if new[j] != cl[j][0]]})
        return
    raise SystemExit(f"{pack}: 找不到 {tail}")


if __name__ == "__main__":
    batches = {MAIN: {}, TWIN: {}}
    log = []
    for pack, tail, spec in SPECS:
        for p in ([MAIN, TWIN] if pack == "both" else [pack]):
            patch(p, tail, spec, batches[p], log)
    for p, b in batches.items():
        fp = os.path.join(BASE, "batches", f"uuid-manual-{p}")
        json.dump(b, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{fp}  entries={len(b)}")
    json.dump(log, open(os.path.join(BASE, "reports", "manual_log.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    for e in log:
        print(e["pack"][6:20], e["path"].split(".journals.")[-1][:60])
        for m in e["moved"]:
            print(f"    [{m['pos']:>2}] {m['label']:<16} {m['from']} -> {m['to']}")
