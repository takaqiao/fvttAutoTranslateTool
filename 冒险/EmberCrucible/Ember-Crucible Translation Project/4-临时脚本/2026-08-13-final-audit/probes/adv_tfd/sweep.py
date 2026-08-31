# -*- coding: utf-8 -*-
"""Template-family sweep: EN leaves that normalize to ONE skeleton but CN doesn't."""
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
def stripen(s):
    s=re.sub(r'@\w+\[[^\]]*\]','@M',s)
    s=re.sub(r'<[^>]+>','',s)
    return re.sub(r'\s+',' ',s).strip()
def stripcn(s):
    s=re.sub(r'@\w+\[[^\]]*\]','@M',s)
    s=re.sub(r'<[^>]+>','',s)
    return re.sub(r'\s+','',s).strip()

FAMS=[
 ('rune-proficiency', re.compile(r'fluency in weaving spellcraft')),
 ('skill-rank',       re.compile(r'skill, providing a')),
 ('spellcraft-affix', re.compile(r'imbued with the essence of')),
 ('bane',             re.compile(r'especially effective against')),
 ('resistance',       re.compile(r'resistance by ?\d? ?for each Tier|Increase .{0,40}resistance by')),
 ('damage-suffix',    re.compile(r'flat additional damage')),
 ('conversion',       re.compile(r'is converted to .{0,30}damage')),
 ('condition-dot',    re.compile(r'suffers additional \w+ damage to Health at the start')),
]
rows=defaultdict(list)
for repo in ['1-Ember汉化插件','2-Crucible汉化插件']:
    for f in sorted(glob.glob(os.path.join(R,repo,'compendium','en','*.json'))):
        b=os.path.basename(f)
        if b.startswith('_') or b in SKIP: continue
        cnf=os.path.join(R,repo,'compendium','cn',b)
        if not os.path.exists(cnf): continue
        en=collect(json.load(io.open(f,encoding='utf-8')))
        cn=collect(json.load(io.open(cnf,encoding='utf-8')))
        for k,v in en.items():
            e=stripen(v)
            for name,pat in FAMS:
                if pat.search(e):
                    rows[name].append((repo,b,k,e,stripcn(cn.get(k,'<MISSING>'))))
                    break
tot=0
for name,pat in FAMS:
    rs=rows[name]
    if not rs: continue
    tot+=len(rs)
    cnc=Counter(r[4] for r in rs)
    print('%-18s leaves=%3d  files=%s  distinctCN=%d'%(name,len(rs),dict(Counter(r[1] for r in rs)),len(cnc)))
print('TOTAL leaves in these 8 families:',tot)
