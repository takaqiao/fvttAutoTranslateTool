# -*- coding: utf-8 -*-
import json, io, os, sys, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk import load
sys.stdout.reconfigure(encoding='utf-8')
PAIRS = [
 ('2-Crucible汉化插件/compendium/en/crucible.talent.json','2-Crucible汉化插件/compendium/cn/crucible.talent.json'),
 ('1-Ember汉化插件/compendium/en/ember.crucible-adventure.json','1-Ember汉化插件/compendium/cn/ember.crucible-adventure.json'),
]
PAT = re.compile(r'You gain the (\w+) rank in the ([A-Za-z]+) skill, providing a \+(\d) Skill Bonus\.')
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
    en=collect(load(enp)); cn=collect(load(cnp))
    for k,v in en.items():
        m=PAT.search(v)
        if m: rows.append((os.path.basename(enp),k,m.groups(),v,cn.get(k,'<MISSING>')))
print('leaves matched by strict pattern:',len(rows))
# build cn masks
RANK={'Journeyman':None,'Adept':None,'Master':None}
c=Counter(); ex=defaultdict(list)
raw=Counter()
for f,k,(rank,skill,bonus),en,cn in rows:
    s=re.sub(r'<[^>]+>','',cn)
    s=re.sub(r'\s+','',s)
    raw[s]+=1
    # mask: remove cn skill name and rank name heuristically by replacing digits and any 2-4 char before 技能/等级
    t=s
    t=re.sub(r'\+?\d','N',t)
    ex[t].append((f,k,rank,skill,bonus,s))
    c[t]+=1
print('distinct raw CN:',len(raw))
print('distinct CN after digit-mask:',len(c))
for s,n in c.most_common():
    print('  ',n,'|',s)
    for e in ex[s][:3]:
        print('        eg',e[3],e[2],'->',e[5])
