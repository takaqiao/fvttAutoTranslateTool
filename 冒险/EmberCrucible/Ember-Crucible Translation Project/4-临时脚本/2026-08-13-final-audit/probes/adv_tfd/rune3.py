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
CNRUNE={'Control':'控制','Death':'死亡','Earth':'大地','Flame':'火焰','Frost':'霜冻','Illumination':'照明','Illusion':'幻象','Kinesis':'念力','Life':'生命','Oblivion':'湮灭','Soul':'灵魂','Storm':'风暴'}
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
        if m: rows.append((os.path.basename(enp),k,m.group(1),v,cn.get(k,'<MISSING>')))
def norm(s,rune):
    s=re.sub(r'@Action\[[^\]]*\]','@A',s)
    s=re.sub(r'<[^>]+>','',s)
    s=re.sub(r'\s+','',s)
    s=s.replace(CNRUNE[rune],'§')
    return s
def norm_en(s,rune):
    s=re.sub(r'@Action\[[^\]]*\]','@A',s)
    s=re.sub(r'<[^>]+>','',s)
    s=re.sub(r'\s+',' ',s)
    return s.replace('Rune of '+rune,'Rune of §')
enset=Counter(); c1=Counter(); c2=Counter(); c3=Counter()
byrune=defaultdict(set)
for f,k,rune,en,c in rows:
    enset[norm_en(en,rune)]+=1
    n=norm(c,rune)
    parts=[p for p in n.split('。') if p]
    c1[parts[0]+'。']+=1
    if len(parts)>1: c2[parts[1]+'。']+=1
    if len(parts)>2: c3['。'.join(parts[2:])+'。']+=1
    byrune[rune].add(n)
print('EN normalized variants:',len(enset))
for s,c in enset.most_common(): print('  ',c,s[:200])
print()
print('CN sentence1 variants:',len(c1))
for s,c in c1.most_common(): print('  ',c,s)
print()
print('CN sentence2 variants:',len(c2))
for s,c in c2.most_common(): print('  ',c,s)
print()
print('CN rest variants:',len(c3))
for s,c in c3.most_common(): print('  ',c,s)
print()
print('runes with internal divergence:', {r:len(v) for r,v in byrune.items() if len(v)>1})
