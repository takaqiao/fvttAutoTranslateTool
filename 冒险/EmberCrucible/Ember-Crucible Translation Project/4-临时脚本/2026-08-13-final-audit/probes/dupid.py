# -*- coding: utf-8 -*-
"""专查「注入锚点造成同页 id 重复」：中文侧 id 重复而英文侧不重复的叶。

顺带查：id 值里含空白（HTML 里 id 不允许空白，#锚点跳转必然失效）。
并核对该 id 是否真的被 @UUID[...#anchor] 引用过 —— 被引用的重复 id 才是真伤害。
"""
import json, re, sys, collections
from pathlib import Path

ID_ATTR = re.compile(r'\bid\s*=\s*"([^"]*)"')
ANCHOR_REF = re.compile(r'@UUID\[([^\]]*?)#([^\]\s]+)\]')


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


for repo in sys.argv[1:]:
    repo = Path(repo)
    print("#" * 100)
    print(repo.name)
    all_refs = collections.Counter()
    rows = []
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in cn.items():
            for m in ANCHOR_REF.finditer(s):
                all_refs[m.group(2)] += 1
        for p, s in cn.items():
            if "id=" not in s:
                continue
            cids = ID_ATTR.findall(s)
            eids = ID_ATTR.findall(en.get(p, ""))
            cdup = {v: n for v, n in collections.Counter(cids).items() if n > 1}
            edup = {v: n for v, n in collections.Counter(eids).items() if n > 1}
            newdup = {v: n for v, n in cdup.items() if edup.get(v, 0) < n}
            ws = [v for v in set(cids) if re.search(r"\s", v)]
            ws_en = [v for v in set(eids) if re.search(r"\s", v)]
            if newdup or (ws and set(ws) - set(ws_en)):
                rows.append((f.name, p, newdup, sorted(set(ws) - set(ws_en)), s))
    for pack, p, newdup, ws, s in rows:
        print("-" * 96)
        print(pack, "|", p)
        if newdup:
            print("  中文侧新增重复 id:", {k: (v, f"被 #{k} 引用 {all_refs.get(k,0)} 次") for k, v in newdup.items()})
        if ws:
            print("  id 含空白:", ws)
        for v in list(newdup) + ws:
            for m in re.finditer(r'<[^<>]*id\s*=\s*"' + re.escape(v) + r'"[^<>]*>([^<]{0,60})', s):
                print("     -> ", m.group(0)[:160])
    print(f"共 {len(rows)} 叶")
