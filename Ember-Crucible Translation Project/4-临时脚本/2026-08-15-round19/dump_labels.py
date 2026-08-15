# -*- coding: utf-8 -*-
"""把全库可配对的「英文标签 / 中文标签」倾倒出来，供人工挑判据。

配对**按 (动词, 目标) 分组后组内取序**，不是全叶取序 ——
实测全叶取序有 1388/30650 对（4.5%）会配歪：中文「定语在前」经常把增强器整个搬位。

用法：python dump_labels.py [<英文正则>] [--rebuild]
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
sys.path.insert(0, P + "/3-常用脚本/qa")
import assert_resolutions as A          # noqa: E402

AT = re.compile(r"@([A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?")
PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*"([^"]*)"')
TEXT_PARAMS = {"label", "readaloud"}
HERE = os.path.dirname(os.path.abspath(__file__))


def key(m):
    return (m.group(1).lower(), m.group(2).split(" ", 1)[0])


def slots(m):
    out = []
    if m.group(3) is not None:
        out.append(("label", m.group(3)))
    for pm in PARAM.finditer(m.group(2)):
        if pm.group(1).lower() in TEXT_PARAMS:
            out.append((f"param:{pm.group(1).lower()}", pm.group(2)))
    return out


def pair(ev, cv):
    """按 (动词, 目标) 分组、组内取序配对。返回 (配对列表, 无法配对数, 两侧总数)。"""
    eg, cg = defaultdict(list), defaultdict(list)
    for m in AT.finditer(ev):
        eg[key(m)].append(m)
    for m in AT.finditer(cv):
        cg[key(m)].append(m)
    pairs, unpaired = [], 0
    for k in set(eg) | set(cg):
        a, b = eg.get(k, []), cg.get(k, [])
        n = min(len(a), len(b))
        pairs += list(zip(a[:n], b[:n]))
        unpaired += abs(len(a) - len(b))
    return pairs, unpaired, sum(len(v) for v in eg.values())


def collect():
    repos = {"ember": os.path.join(P, "1-Ember汉化插件"),
             "crucible": os.path.join(P, "2-Crucible汉化插件")}
    ctx = A.Ctx(repos, {})
    rows, unp, tot = [], 0, 0
    for repo in repos:
        for pack, path, ev, cv in ctx.pairs[repo]:
            if "@" not in ev and "@" not in cv:
                continue
            ps, u, t = pair(ev, cv)
            unp += u
            tot += t
            for a, b in ps:
                es, cs = dict(slots(a)), dict(slots(b))
                for k in set(es) | set(cs):
                    rows.append({"repo": repo, "pack": pack, "path": path,
                                 "tgt": key(a)[1][:70], "slot": k,
                                 "en": es.get(k), "cn": cs.get(k)})
    print(f"增强器 EN 侧 {tot} 个；无法按 (动词,目标) 配对 {unp} 个")
    return rows


def main():
    cache = os.path.join(HERE, "labels.json")
    if os.path.exists(cache) and "--rebuild" not in sys.argv:
        rows = json.load(open(cache, encoding="utf-8"))
    else:
        rows = collect()
        json.dump(rows, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    print("槽位分布:", Counter(r["slot"] for r in rows))
    print("两侧都有值:", sum(1 for r in rows if r["en"] is not None and r["cn"] is not None))
    print("只有 CN:", sum(1 for r in rows if r["en"] is None))
    print("只有 EN:", sum(1 for r in rows if r["cn"] is None))
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if args:
        rx = re.compile(args[0], re.IGNORECASE)
        hits = [r for r in rows if r["en"] and rx.search(r["en"])]
        print(f"\n英文槽匹配 /{args[0]}/ 的 {len(hits)} 处：")
        seen = Counter((r["en"][:90], r["cn"]) for r in hits)
        for (en, cn), n in seen.most_common(200):
            print(f"  {n:>4}  EN={en!r}  CN={cn!r}")


main()
