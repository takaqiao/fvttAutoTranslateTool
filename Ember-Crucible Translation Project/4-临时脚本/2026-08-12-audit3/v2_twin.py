# -*- coding: utf-8 -*-
"""After simulating the U2 batches, report residual old labels still present in the repo."""
import json,os,re,sys,collections,copy
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPO=os.path.join(ROOT,"1-Ember汉化插件")
BAT=r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"
pat=re.compile(r'@UUID\[([^\]]*)\](?:\{([^}]*)\})?')
packs=["ember.adventure.json","ember.crucible-adventure.json"]
cn={p:json.load(open(os.path.join(REPO,'compendium','cn',p),encoding='utf-8')) for p in packs}
def get(root,p):
    cur=root.get('entries',root)
    for seg in p.split('.'):
        if isinstance(cur,dict) and seg in cur: cur=cur[seg]
        else: return None
    return cur
def setv(root,p,val):
    cur=root.get('entries',root); segs=p.split('.')
    for seg in segs[:-1]: cur=cur[seg]
    cur[segs[-1]]=val
# collect the (target, old->new) edits
edits=collections.defaultdict(set)
after=copy.deepcopy(cn)
for p in packs:
    bf=os.path.join(BAT,f"U2__ember__{p}")
    if not os.path.isfile(bf): continue
    b=json.load(open(bf,encoding='utf-8'))
    for path,new in b.items():
        old=get(cn[p],path)
        o=[(m.group(1),m.group(2)) for m in pat.finditer(old)]
        n=[(m.group(1),m.group(2)) for m in pat.finditer(new)]
        for (t1,l1),(t2,l2) in zip(o,n):
            if l1!=l2: edits[t1].add((l1,l2))
        setv(after[p],path,new)
# now census residual old labels in the AFTER state
print(f'{len(edits)} targets edited\n')
for tgt,pairs in edits.items():
    olds={a for a,b in pairs}; news={b for a,b in pairs}
    resid=collections.Counter(); locs=[]
    for p in packs:
        rows=[]
        def walk(o,path):
            if isinstance(o,dict):
                for k,v in o.items(): walk(v,path+[str(k)])
            elif isinstance(o,list):
                for i,v in enumerate(o): walk(v,path+[str(i)])
            elif isinstance(o,str): rows.append((".".join(path),o))
        walk(after[p],[])
        for pp,s in rows:
            if tgt not in s: continue
            for m in pat.finditer(s):
                if tgt not in m.group(1): continue
                lab=m.group(2)
                if lab in olds:
                    resid[lab]+=1; locs.append((p,pp,lab))
    if resid:
        print(f'!! {tgt}  fixed {sorted(olds)} -> {sorted(news)}   BUT residual old labels remain:')
        for l,n in resid.most_common(): print(f'     x{n}  {l!r}')
        for x in locs[:6]: print('       ',x[0],'::',x[1])
        print()
