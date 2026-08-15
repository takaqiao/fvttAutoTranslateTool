#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_setat_clobber.py — 「防御性 shim 过度生效 / 静默改写」类判据在**构建管线**上的落地

p22 那条支线只扫了插件 JS 的持久化 sink；shipped JSON 是由 3-常用脚本/qa/*.py
写出来的，那一层没人查过。本探针查 apply_translations.py::set_at 的两个
happy-path 静默改写：

  S1  set_at 造容器时：
        if p not in node or not isinstance(node[p], (dict, list)):
            node[p] = [] if isinstance(nxt_shape, list) else {}
      —— CN 树上该键**已经是一个字符串译文**时，直接被换成 {}，旧译文无声消失，
      而计数器仍然 applied += 1。触发前提 = 同一路径 EN 是容器 / CN 是标量
      （或反过来）。这里数这种形状错配在现网 shipped 数据里有多少处。

  S2  set_at 补列表时：
        while len(node) <= idx: node.append({})
      —— 用 {} 填洞。若 EN 同位置是字符串数组，CN 就会出现 {} 元素。
      这里数 CN 侧数组/对象里的空 {}。

判据的假阳性：
  * S1 只是**前提**存在，不等于已经被踩过；报的时候要说清是「潜在」还是「已发生」。
  * S2 的空 {} 也可能是提取器正常产物（例如 crucibleActions.extract 的
    effects 占位 {}）——所以按「EN 同位置是不是字符串」二次过滤。

只读，不写库。
"""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]


def kind(v):
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        return "list"
    if isinstance(v, str):
        return "str"
    return type(v).__name__


def walk(en, cn, path, out_mismatch, out_emptydict):
    ke, kc = kind(en), kind(cn)
    if cn is None:
        return
    if ke != kc:
        out_mismatch.append({"path": ".".join(path), "en": ke, "cn": kc,
                             "cn_preview": (cn if isinstance(cn, str) else json.dumps(cn, ensure_ascii=False))[:80]})
        return
    if ke == "dict":
        for k, v in cn.items():
            if k in en:
                walk(en[k], v, path + [k], out_mismatch, out_emptydict)
    elif ke == "list":
        for i, v in enumerate(cn):
            if isinstance(v, dict) and not v:
                ep = en[i] if i < len(en) else None
                out_emptydict.append({"path": ".".join(path + [str(i)]),
                                      "en_kind": kind(ep)})
                continue
            if i < len(en):
                walk(en[i], v, path + [str(i)], out_mismatch, out_emptydict)


tot_packs = tot_leaves = 0
mism_all, empty_all = [], []
for repo in REPOS:
    cnd = ROOT / repo / "compendium" / "cn"
    end = ROOT / repo / "compendium" / "en"
    if not cnd.exists():
        continue
    for cnp in sorted(cnd.glob("*.json")):
        enp = end / cnp.name
        if not enp.exists():
            print(f"  (no EN baseline) {repo}/{cnp.name}")
            continue
        tot_packs += 1
        en = json.loads(enp.read_text(encoding="utf-8"))
        cn = json.loads(cnp.read_text(encoding="utf-8"))
        m, e = [], []
        walk(en, cn, [], m, e)
        for x in m:
            x["pack"] = f"{repo}/{cnp.name}"
        for x in e:
            x["pack"] = f"{repo}/{cnp.name}"
        mism_all += m
        empty_all += e

print(f"packs compared: {tot_packs}")
print(f"S1  EN/CN 同路径形状错配 : {len(mism_all)}")
for x in mism_all[:40]:
    print(f'   {x["pack"]}  {x["path"]}   EN={x["en"]}  CN={x["cn"]}   {x["cn_preview"]}')
if len(mism_all) > 40:
    print(f"   ... and {len(mism_all)-40} more")

print(f"\nS2  CN 侧数组里的空 {{}} : {len(empty_all)}")
from collections import Counter
print("   按 EN 同位置类型分布:", Counter(x["en_kind"] for x in empty_all))
for x in empty_all[:20]:
    print(f'   {x["pack"]}  {x["path"]}   EN_kind={x["en_kind"]}')

out = Path(__file__).with_name("setat_clobber.json")
out.write_text(json.dumps({"mismatch": mism_all, "empty_dict": empty_all},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n-> {out}")
