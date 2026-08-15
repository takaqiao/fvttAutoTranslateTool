# -*- coding: utf-8 -*-
"""Resolve target id -> EN name (via ember_ids.json) -> CN name leaves."""
import json,os,sys,collections
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
IDS=json.load(open(os.path.join(ROOT,"4-临时脚本","2026-08-12-fix","reports","ember_ids.json"),encoding="utf-8"))
REPOS=["1-Ember汉化插件","2-Crucible汉化插件"]
def walk(en,cn,path,out):
    if isinstance(en,dict):
        for k,v in en.items():
            if k in ("_id","path","_variants","_when"): continue
            walk(v,cn.get(k) if isinstance(cn,dict) else None,path+[str(k)],out)
    elif isinstance(en,list):
        for i,v in enumerate(en): walk(v,cn[i] if isinstance(cn,list) and i<len(cn) else None,path+[str(i)],out)
    elif isinstance(en,str): out.append((".".join(path),en,cn if isinstance(cn,str) else None))
ROWS=[]
for repo in REPOS:
    ed=os.path.join(ROOT,repo,"compendium","en"); cd=os.path.join(ROOT,repo,"compendium","cn")
    if not os.path.isdir(ed): continue
    for fn in sorted(os.listdir(ed)):
        if not fn.endswith(".json") or fn.startswith("_"): continue
        en=json.load(open(os.path.join(ed,fn),encoding="utf-8"))
        cp=os.path.join(cd,fn); cn=json.load(open(cp,encoding="utf-8")) if os.path.isfile(cp) else {}
        r=[];walk(en,cn,[],r); ROWS+= [(repo[:1],fn)+x for x in r]
for tid in sys.argv[1:]:
    meta=IDS.get(tid)
    print(f'== {tid}  EN name = {meta["name"]!r}  type={meta.get("type")}' if meta else f'== {tid}  (not in id index)')
    if not meta: continue
    c=collections.Counter()
    for repo,fn,p,e,cv in ROWS:
        if p.split(".")[-1] not in ("name","label","tokenName"): continue
        if e!=meta["name"]: continue
        c[cv]+=1
    for v,n in c.most_common(): print(f'     CN name x{n}: {v!r}')
