# -*- coding: utf-8 -*-
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SRC = {"ember": r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember",
       "crucible": r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"}
CN = {"ember": "1-Ember汉化插件", "crucible": "2-Crucible汉化插件"}
def walk(n,p=""):
    if isinstance(n,dict):
        for k,v in n.items(): yield from walk(v,f"{p}.{k}" if p else k)
    elif isinstance(n,list):
        for i,v in enumerate(n): yield from walk(v,f"{p}[{i}]")
    elif isinstance(n,str): yield p,n
L={}
for k in SRC:
    en=dict(walk(json.load(open(os.path.join(SRC[k],"lang","en.json"),encoding="utf-8-sig"))))
    cn=dict(walk(json.load(open(os.path.join(ROOT,CN[k],"lang","cn.json"),encoding="utf-8-sig"))))
    L[k]=(en,cn)

PH=re.compile(r"\{[^}]*\}")
def gate(pat, req, label, strip_ph=True):
    r=re.compile(pat,re.I); hits=0; bad=[]
    for repo,(en,cn) in L.items():
        for key,ev in en.items():
            probe = PH.sub(" ", ev) if strip_ph else ev
            if not r.search(probe): continue
            hits+=1
            cv=cn.get(key)
            if cv is None: bad.append((repo,key,"缺键",ev[:60])); continue
            if req not in cv: bad.append((repo,key,f"缺「{req}」",cv[:80]))
    print(f"\n[{label}] 命中 {hits} 键，违规 {len(bad)}")
    for b in bad: print("   ",b[0],b[1],"|",b[2],"|",b[3])
    return hits,bad

gate(r"\bRegion Maps?\b","地区地图","RegionMap")
gate(r"\bArea Maps?\b","区域地图","AreaMap")
gate(r"\bRanks?\b","阶位","Rank")
gate(r"\bTiers?\b","阶","Tier")
gate(r"\blevels?\b","级","level")
gate(r"\bRounds?\b","轮","Round")
gate(r"\bTurns?\b","回合","Turn")
gate(r"\bTokens?\b","指示物","Token")
gate(r"\bhex(es)?\b","六边格","Hex")
gate(r"\bCorla\b","科尔拉","Corla")
gate(r"\bShard Gods?\b","碎片","ShardGod")
print("\n--- Aura 相关 lang 键 ---")
for repo,(en,cn) in L.items():
    for key,ev in en.items():
        if re.search(r"aura|auric", ev, re.I) or "Aura" in key:
            print(f"  {repo} {key} | {ev[:70]} | {cn.get(key)}")
