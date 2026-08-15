# -*- coding: utf-8 -*-
import json,io,os,re,sys,glob
from collections import Counter,defaultdict
sys.stdout.reconfigure(encoding='utf-8')
R=r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
SKIP={'ember.adventure.json','ember.dnd5e-effects.json','ember.dnd5e-items.json','ember.character.json'}
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
def sten(s):
    s=re.sub(r'@\w+\[[^\]]*\]','@M',s); s=re.sub(r'<[^>]+>',' ',s)
    return re.sub(r'\s+',' ',s).strip()
def stcn(s):
    s=re.sub(r'@\w+\[[^\]]*\]','@M',s); s=re.sub(r'<[^>]+>','',s)
    return re.sub(r'\s+','',s).strip()
items=[]
for repo in ['1-Ember汉化插件','2-Crucible汉化插件']:
    for f in sorted(glob.glob(os.path.join(R,repo,'compendium','en','*.json'))):
        b=os.path.basename(f)
        if b.startswith('_') or b in SKIP: continue
        cnf=os.path.join(R,repo,'compendium','cn',b)
        if not os.path.exists(cnf): continue
        en=collect(json.load(io.open(f,encoding='utf-8')))
        cn=collect(json.load(io.open(cnf,encoding='utf-8')))
        for k,v in en.items():
            c=cn.get(k)
            if c: items.append((repo,b,k,sten(v),stcn(c)))
FAMS=[
 ('F1 rune-proficiency',   r'fluency in weaving spellcraft'),
 ('F2 skill-rank JAM',     r'rank in the \w+ skill, providing'),
 ('F3 spellcraft-affix',   r'imbued with the essence of'),
 ('F4 bane',               r'especially effective against'),
 ('F5 dmg-resistance',     r'resistance by 1?3? for each Tier of this prefix'),
 ('F6 damage-suffix',      r'flat additional damage'),
 ('F7 conversion',         r'is converted to \w+ damage'),
 ('F8 condition-dot',      r'suffers additional \w+ damage to Health at the start'),
 ('F9 skillcheck-affix',   r'Enchantment Bonus for \w+ skill checks by'),
 ('F10 defense-affix',     r'Increase \w+ defense by 1 for each Tier'),
 ('F11 rune-potency',      r'Enchantment Bonus for spells using the \w+ rune'),
 ('F12 rune-mastery',      r'highly skilled in weaving the Rune of'),
]
tot=0; distinct=0; used=set()
for name,pat in FAMS:
    p=re.compile(pat)
    rs=[it for it in items if p.search(it[3])]
    for it in rs: used.add((it[0],it[1],it[2]))
    tot+=len(rs); distinct+=len(set(r[4] for r in rs))
    print('%-22s leaves=%3d distinctCN=%3d  %s'%(name,len(rs),len(set(r[4] for r in rs)),
          dict(Counter(r[1] for r in rs))))
print()
print('TOTAL leaves = %d, distinct CN strings = %d, unique leaf paths = %d'%(tot,distinct,len(used)))
