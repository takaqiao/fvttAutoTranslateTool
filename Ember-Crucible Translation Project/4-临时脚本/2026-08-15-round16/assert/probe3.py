# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
P=r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-15-round16\assert\probe_gates.py"
exec(open(P,encoding="utf-8").read().split("ALL = []")[0])
ROOT=r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ALL=[]
for name,rel in REPOS.items():
    for row in pairs(os.path.join(ROOT,rel)): ALL.append((name,)+row)

def find(sub, only=None):
    return [(r,p,path,ev,cv) for r,p,path,ev,cv in ALL if sub in path and (only is None or only==p)]

targets=["History.pages.Age of Rediscovery.text","Character Classes.pages.Barbarian.text",
 "Deities.pages.Thoma.text","Deities.pages.Kinalathi.contentGamemaster",
 "History.pages.Age of Rediscovery.contentGamemaster","actors.Sigil.archetype.description"]
for t in targets:
    rows=find(t)
    if not rows: print("!! 找不到",t); continue
    r,p,path,ev,cv=rows[0]
    print("\n#####",t,f"({len(rows)} 叶)")
    for m in re.finditer(r"[Ss]hard [Gg]ods?\b", ev):
        print("   EN:", ev[max(0,m.start()-110):m.end()+70].replace("\n"," "))
    for m in re.finditer(r"碎片[之诸]?神|神族|众神|诸神", cv):
        print("   CN:", cv[max(0,m.start()-45):m.end()+45].replace("\n"," "))
        break
    print("   CN 含碎片之神:", "碎片之神" in cv, "| 含碎片诸神:", "碎片诸神" in cv, "| 含碎片:", "碎片" in cv)

print("\n\n### 波因特 全库")
print(len([1 for r,p,path,ev,cv in ALL if "波因特" in cv]))
print("### 岬 全库")
for r,p,path,ev,cv in ALL:
    if "岬" in cv:
        i=cv.find("岬"); print("  ",r,p,path[:70],"|",cv[max(0,i-30):i+20])
print("\n### \bPoint\b 闸命中")
rp=re.compile(r"\bPoints?\b")
print(len([1 for r,p,path,ev,cv in ALL if rp.search(ev)]))

print("\n### 代币 4 叶路径")
for r,p,path,ev,cv in ALL:
    if "代币" in cv: print("  ",r,"|",p,"|",path)
print("\n### Cyclonic 违规叶路径")
rc=re.compile(r"\bCyclonic\b",re.I)
for r,p,path,ev,cv in ALL:
    if rc.search(ev) and "气旋" not in cv: print("  ",r,"|",p,"|",path)
print("\n### Cora HTML class 叶")
for r,p,path,ev,cv in ALL:
    if re.search(r"\bCora\b",ev,re.I) and "科拉" not in cv: print("  ",r,"|",p,"|",path)
