# -*- coding: utf-8 -*-
"""Reverse probe: CN combat-轮 present while EN block has NO time-unit 'round' at all.
That would be a turn->轮 swap (the mirror of the reported defect)."""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_broad import collect, TIME_RX, NEG, CN_LUN_NEG  # noqa

def main():
    rows = collect()
    out = []
    for r in rows:
        cn = r["cn"] or ""
        if "轮" not in cn: continue
        lun_all = len(re.findall("轮", cn))
        lun_neg = sum(m.group(0).count("轮") for m in CN_LUN_NEG.finditer(cn))
        lun = lun_all - lun_neg
        if lun <= 0: continue
        en = r["en"]
        hits = [m.group(0) for m in TIME_RX.finditer(en) if not NEG.search(m.group(0))]
        if hits: continue           # EN has round(s): not a mirror case
        if re.search(r"\bround", en, re.I): continue  # some other 'round' form
        ctx = []
        for m in re.finditer("轮", cn):
            ctx.append(cn[max(0,m.start()-18):m.start()+12].replace("\n"," "))
        out.append({"repo": r["repo"], "pack": r["pack"], "path": r["path"],
                    "cn_lun_net": lun, "en_has_turn": bool(re.search(r"\bturns?\b", en, re.I)),
                    "ctx": ctx[:6]})
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_rev.json")
    json.dump({"n": len(out), "rows": out}, open(dst,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print("mirror candidates:", len(out), "->", dst)
    for o in out:
        print(f"  轮={o['cn_lun_net']} enTurn={o['en_has_turn']} {o['pack'][:28]:<28} {o['path'][:90]}")
        for c in o["ctx"]: print("      ", c)

main()
