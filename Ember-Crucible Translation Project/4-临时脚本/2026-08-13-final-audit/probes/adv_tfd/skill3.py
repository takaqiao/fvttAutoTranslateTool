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
def strip(s): return re.sub(r'\s+',' ',re.sub(r'<[^>]+>','',s)).strip()
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
        m=PAT.search(strip(v))
        if m: rows.append((os.path.basename(enp),k,m.groups(),strip(v),cn.get(k,'<MISSING>')))
print('leaves:',len(rows), Counter(r[0] for r in rows))
# EN full-text check: is it exactly the one sentence?
enfull=Counter(re.sub(r'(Journeyman|Adept|Master)','§R',re.sub(r'\b(Arcana|Athletics|Awareness|Deception|Diplomacy|Intimidation|Medicine|Performance|Science|Society|Stealth|Wilderness)\b','§S',re.sub(r'\+\d','+N',r[3]))) for r in rows)
print('EN masked variants:',len(enfull))
for s,c in enfull.most_common(): print('  ',c,'|',s)
CNSK={'Arcana':'奥术','Athletics':'运动','Awareness':'察觉','Deception':'欺瞒','Diplomacy':'外交','Intimidation':'威吓','Medicine':'医疗','Performance':'表演','Science':'科学','Society':'社交','Stealth':'潜行','Wilderness':'荒野'}
c=Counter(); ex=defaultdict(list); rawc=Counter()
unmasked=[]
for f,k,(rank,skill,bonus),en,cn in rows:
    s=re.sub(r'<[^>]+>','',cn); s=re.sub(r'\s+','',s)
    rawc[s]+=1
    t=re.sub(r'\d','N',s)
    # mask skill CN if we know it
    cns=CNSK.get(skill)
    if cns and cns in t: t=t.replace(cns,'§S')
    else: unmasked.append((skill,s))
    t=re.sub(r'(学徒|熟练|老手|大师|专家|入门|游历|精通|巧匠|行家|新手)','§R',t,count=1)
    c[t]+=1; ex[t].append((skill,rank,s))
print()
print('distinct raw CN:',len(rawc))
print('distinct CN masked:',len(c))
for s,n in c.most_common():
    print('  ',n,'|',s)
    for e in ex[s][:2]: print('        eg',e[0],e[1],'->',e[2])
if unmasked:
    print(); print('UNMASKED skill names (cn term unknown):')
    for u in unmasked[:20]: print('   ',u)
