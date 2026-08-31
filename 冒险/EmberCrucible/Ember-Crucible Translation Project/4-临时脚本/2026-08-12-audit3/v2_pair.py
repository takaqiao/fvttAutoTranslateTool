# -*- coding: utf-8 -*-
"""Positionally pair EN label -> CN label for a target across the whole repo."""
import json,os,re,sys,collections
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
target=sys.argv[1]
pat=re.compile(r'@UUID\[([^\]]*)\](?:\{([^}]*)\})?')
pairs=collections.Counter()
for repo in ["1-Ember汉化插件","2-Crucible汉化插件"]:
    ed=os.path.join(ROOT,repo,'compendium','en'); cd=os.path.join(ROOT,repo,'compendium','cn')
    if not os.path.isdir(ed): continue
    for fn in sorted(os.listdir(ed)):
        if not fn.endswith('.json') or fn.startswith('_'): continue
        en=json.load(open(os.path.join(ed,fn),encoding='utf-8'))
        cp=os.path.join(cd,fn)
        cn=json.load(open(cp,encoding='utf-8')) if os.path.isfile(cp) else {}
        rows={}
        def walk(o,c,path):
            if isinstance(o,dict):
                for k,v in o.items(): walk(v,c.get(k) if isinstance(c,dict) else None,path+[str(k)])
            elif isinstance(o,list):
                for i,v in enumerate(o): walk(v,c[i] if isinstance(c,list) and i<len(c) else None,path+[str(i)])
            elif isinstance(o,str): rows[".".join(path)]=(o,c if isinstance(c,str) else '')
        walk(en,cn,[])
        for p,(e,c) in rows.items():
            if target not in e: continue
            el=[m.group(2) for m in pat.finditer(e) if target in m.group(1)]
            cl=[m.group(2) for m in pat.finditer(c) if target in m.group(1)]
            if len(el)!=len(cl): 
                pairs[('LEN-MISMATCH '+p[:60],'')]+=1; continue
            for a,b in zip(el,cl): pairs[(a,b)]+=1
for (a,b),n in pairs.most_common(): print(f'  x{n:3d}  EN {{{a}}}  ->  CN {{{b}}}')
