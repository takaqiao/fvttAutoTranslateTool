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
PAT = re.compile(r'fluency in weaving spellcraft using the Rune of (\w+)')
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

RUNE_CN = {}  # discovered
rows=[]
for enp,cnp in PAIRS:
    en=collect(load(enp)); cn=collect(load(cnp))
    for k,v in en.items():
        m=PAT.search(v)
        if m:
            rows.append((os.path.basename(enp),k,m.group(1),v,cn.get(k,'<MISSING>')))

print('leaves:',len(rows))
print('distinct runes:', sorted(set(r[2] for r in rows)))
print('per-file:', Counter(r[0] for r in rows))
print('per-rune:', Counter(r[2] for r in rows))

# normalize CN: strip HTML, replace the rune CN name, replace action macro
def norm_cn(s, rune):
    s=re.sub(r'@Action\[[^\]]*\]','@A',s)
    s=re.sub(r'<[^>]+>','',s)
    s=re.sub(r'\s+','',s)
    return s
# first sentence up to first 。
first=Counter(); second=Counter()
by_first=defaultdict(list)
for f,k,rune,en,c in rows:
    n=norm_cn(c,rune)
    parts=n.split('。')
    f1=parts[0]+'。' if parts else n
    by_first[f1].append((f,k,rune))
    first[f1]+=1
print()
print('--- distinct first sentences (raw, rune name not masked) ---')
for s,c in first.most_common():
    print(c, s)
