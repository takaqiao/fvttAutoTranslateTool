# -*- coding: utf-8 -*-
import json, io, os, sys, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk import load
sys.stdout.reconfigure(encoding='utf-8')
PAIRS = [
 ('2-Crucible汉化插件/compendium/en/crucible.talent.json','2-Crucible汉化插件/compendium/cn/crucible.talent.json'),
 ('1-Ember汉化插件/compendium/en/ember.crucible-adventure.json','1-Ember汉化插件/compendium/cn/ember.crucible-adventure.json'),
 ('2-Crucible汉化插件/compendium/en/crucible.adversary-talents.json','2-Crucible汉化插件/compendium/cn/crucible.adversary-talents.json'),
]
PAT = re.compile(r'skill, providing a')
def collect(obj):
    out={}
    def rec(node,p):
        if isinstance(node,dict):
            for k,v in node.items():
                np=p+'/'+k
                if isinstance(v,str): out[np]=v
                else: rec(v,np)
        elif isinstance(node,list):
            for i,v in enumerate(node):
                np=p+'/'+str(i)
                if isinstance(v,str): out[np]=v
                else: rec(v,np)
    rec(obj,'')
    return out
rows=[]
for enp,cnp in PAIRS:
    if not os.path.exists(os.path.join(r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project',enp)): continue
    en=collect(load(enp)); cn=collect(load(cnp))
    for k,v in en.items():
        if PAT.search(v): rows.append((os.path.basename(enp),k,v,cn.get(k,'<MISSING>')))
print('leaves:',len(rows), Counter(r[0] for r in rows))
# show raw EN variety
enn=Counter()
for f,k,en,c in rows:
    e=re.sub(r'<[^>]+>','',en); e=re.sub(r'\s+',' ',e).strip()
    enn[e]+=1
print('distinct EN (unmasked):',len(enn))
for s,c in enn.most_common(): print('  ',c,'|',s)
