# -*- coding: utf-8 -*-
"""Positionally align @UUID labels EN vs CN for a leaf, restricted to one target id."""
import json,os,re,sys
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
repo,pack,path=sys.argv[1],sys.argv[2],sys.argv[3]
target=sys.argv[4] if len(sys.argv)>4 else None
def get(root,p):
    cur=root.get('entries',root)
    for seg in p.split('.'):
        if isinstance(cur,dict) and seg in cur: cur=cur[seg]
        else: return None
    return cur
pat=re.compile(r'@UUID\[([^\]]*)\](?:\{([^}]*)\})?')
out={}
for side in ('en','cn'):
    d=json.load(open(os.path.join(repo,'compendium',side,pack),encoding='utf-8'))
    v=get(d,path) or ''
    out[side]=[(m.group(1),m.group(2)) for m in pat.finditer(v) if (target is None or target in m.group(1))]
e,c=out['en'],out['cn']
print(f'EN n={len(e)}  CN n={len(c)}  {"LEN-MISMATCH" if len(e)!=len(c) else ""}')
for i in range(max(len(e),len(c))):
    ee=e[i] if i<len(e) else ('---','---')
    cc=c[i] if i<len(c) else ('---','---')
    tgtok='' if ee[0]==cc[0] else '  <<TARGET DIFFERS>>'
    print(f'  [{i}] EN {{{ee[1]}}}   ->  CN {{{cc[1]}}}{tgtok}')
