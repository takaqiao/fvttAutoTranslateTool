# -*- coding: utf-8 -*-
import os,sys,json,re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from rebuild import *
items=json.load(open(os.path.join(WORK,'items_scope.json'),encoding='utf-8'))
delta=load_delta(); CNJ={f:jload(os.path.join(CN,f)) for f in FILES}
W=int(os.environ.get('W','200'))
for a in sys.argv[1:]:
    n=int(a); scope,ptr=items[n-1]
    f='ember.adventure.json' if scope in ('both','ember.adventure.json') else scope
    old,new=delta[f]['changed'][ptr]
    cn=get_at(CNJ[f],split_path(ptr))
    parts,slots=rebuild(old,new,cn)
    r=render(parts)
    print(f"\n#### ITEM {n} [{scope}] {ptr.split('/',3)[-1]}")
    for m in re.finditer(r'⟦(\d+):(\w+)⟧', r):
        print(f"  --- slot {m.group(1)} {m.group(2)} ---")
        print("   …",r[max(0,m.start()-W):m.end()+W].replace('\n',' '))
