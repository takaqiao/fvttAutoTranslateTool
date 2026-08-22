# -*- coding: utf-8 -*-
"""部件全集 v2：把 Color/Mask 帧反推回 partId 一并纳入（P1/P2 前置自证同 enum_parts.py）。"""
import io, json, os, re, sys
EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
ATLAS = os.path.join(EMBER, "assets/tokens/maker")
OUT = os.path.dirname(os.path.abspath(__file__))
TRUTH = {"Character0.json": 1631, "Character1.json": 3013, "Monster0.json": 783, "Party0.json": 222}
def die(m): sys.stderr.write(m+"\n"); sys.exit(2)
frames=[]
for f,n in TRUTH.items():
    d=json.load(io.open(os.path.join(ATLAS,f),encoding="utf-8")); ks=list(d["frames"])
    if len(ks)!=n: die("P1 FAIL %s %d != %d"%(f,len(ks),n))
    frames+=ks
if len(set(frames))!=5649: die("P1 FAIL 合计")
print("P1 OK 5649 frames")
for s in (frames[0], frames[1631], frames[5648]):
    if len(s.split("/"))!=3: die("P2 FAIL %r"%s)
print("P2 OK 三段式抽样", frames[0], "|", frames[5648])

allf=set(frames)
part_ids=set()
for f in allf:
    if f.endswith("Color"): part_ids.add(f[:-5])
    elif f.endswith("Mask") and f[:-4] in allf: pass
    else: part_ids.add(f)
# 去掉纯 Mask
part_ids={p for p in part_ids if p}
seg2layers={}
for p in part_ids:
    q=p.split("/")
    if len(q)!=3: continue
    seg2layers.setdefault(q[2],set()).add(q[1])
RE=re.compile(r'(?<!^)([A-Z1-9])')
disp=lambda s: RE.sub(r' \1', s)
d2l={}
for seg,ls in seg2layers.items(): d2l.setdefault(disp(seg),set()).update(ls)
print("part ids(全)=%d  唯一末段=%d  唯一显示名=%d"%(len(part_ids),len(seg2layers),len(d2l)))
from collections import Counter
c=Counter()
for p in part_ids:
    q=p.split("/")
    if len(q)==3: c[q[1]]+=1
print("按图层：")
for k,v in c.most_common(): print("  %-14s %d"%(k,v))
json.dump({"seg2layers":{k:sorted(v) for k,v in sorted(seg2layers.items())},
           "display2layers":{k:sorted(v) for k,v in sorted(d2l.items())},
           "by_layer":dict(c)},
          io.open(os.path.join(OUT,"parts_universe2.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("已写 parts_universe2.json")
