# -*- coding: utf-8 -*-
"""Census every @UUID label used for a given target id, EN side and CN side, across both repos."""
import json,os,re,sys,collections
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPOS=["1-Ember汉化插件","2-Crucible汉化插件"]
target=sys.argv[1]
def walk(o,path,out):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,path+[str(k)],out)
    elif isinstance(o,list):
        for i,v in enumerate(o): walk(v,path+[str(i)],out)
    elif isinstance(o,str): out.append((".".join(path),o))
pat=re.compile(r'@UUID\[([^\]]*)\](?:\{([^}]*)\})?')
res={'en':collections.Counter(),'cn':collections.Counter()}
leaves={'en':[],'cn':[]}
for repo in REPOS:
    for side in ('en','cn'):
        d=os.path.join(ROOT,repo,'compendium',side)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json') or fn.startswith('_'): continue
            try: data=json.load(open(os.path.join(d,fn),encoding='utf-8'))
            except Exception: continue
            rows=[];walk(data,[],rows)
            for p,s in rows:
                if target not in s: continue
                for m in pat.finditer(s):
                    if target not in m.group(1): continue
                    lab=m.group(2) if m.group(2) is not None else '(NO LABEL)'
                    res[side][lab]+=1
                    leaves[side].append((repo[:1],fn,p,lab))
for side in ('en','cn'):
    print(f'--- {side.upper()} labels for {target}')
    for lab,n in res[side].most_common(): print(f'   {n:4d}  {lab!r}')
print('--- CN leaves')
for r in leaves['cn']: print('  ',r)
