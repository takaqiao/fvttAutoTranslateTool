# -*- coding: utf-8 -*-
import json, io, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk import load, walk
sys.stdout.reconfigure(encoding='utf-8')

PAIRS = [
 ('2-Crucible汉化插件/compendium/en/crucible.talent.json','2-Crucible汉化插件/compendium/cn/crucible.talent.json'),
 ('1-Ember汉化插件/compendium/en/ember.crucible-adventure.json','1-Ember汉化插件/compendium/cn/ember.crucible-adventure.json'),
]
PAT = re.compile(r'fluency in weaving spellcraft')

def collect(obj, prefix=''):
    out={}
    def rec(node, p):
        if isinstance(node, dict):
            for k,v in node.items():
                np = p+'/'+k
                if isinstance(v,str): out[np]=v
                else: rec(v,np)
        elif isinstance(node,list):
            for i,v in enumerate(node):
                np=p+'/'+str(i)
                if isinstance(v,str): out[np]=v
                else: rec(v,np)
    rec(obj,prefix)
    return out

rows=[]
for enp,cnp in PAIRS:
    en=collect(load(enp)); cn=collect(load(cnp))
    for k,v in en.items():
        if PAT.search(v):
            rows.append((enp.split('/')[-1], k, v, cn.get(k,'<MISSING>')))
print('total EN leaves matching:', len(rows))
for f,k,v,c in rows:
    print('='*100)
    print(f, k)
    print('EN:', v)
    print('CN:', c)
