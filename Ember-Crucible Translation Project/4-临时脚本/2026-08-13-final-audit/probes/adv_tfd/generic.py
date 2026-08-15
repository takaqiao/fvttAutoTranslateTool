# -*- coding: utf-8 -*-
"""Generic template-family detector.
Skeleton = EN with every capitalized token (incl. sentence-initial) and every number -> §.
Family = skeleton shared by >=3 leaves whose raw EN differ (i.e. slot-only variation).
Report families where the CN, after masking Chinese chars that differ, still shows >1 shape.
Known false-positive modes:
  - masking ALL capitalized tokens also masks legit different verbs/nouns -> two genuinely
    different EN sentences can collapse into one skeleton. Every reported family is
    re-printed with its raw EN set so a human can reject that case.
  - CN shape count uses a crude 'drop all CJK chars that are not punctuation/structure'
    normalization; it OVER-reports when the CN slot word length differs. So the CN count
    here is only a screening signal; per-family manual read is required.
"""
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
def skel(e):
    e=re.sub(r'\d+','#',e)
    e=re.sub(r'\b[A-Z][a-zA-Z\'-]*\b','§',e)
    return e
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
            e=sten(v)
            if len(e)<40: continue
            c=cn.get(k)
            if not c: continue
            items.append((repo,b,k,e,stcn(c)))
fam=defaultdict(list)
for it in items: fam[skel(it[3])].append(it)
out=[]
for sk,rs in fam.items():
    rawen=set(r[3] for r in rs)
    if len(rs)<3 or len(rawen)<3: continue
    cns=set(r[4] for r in rs)
    # crude cn skeleton: drop CJK runs
    cnsk=set(re.sub(r'[\u4e00-\u9fff]+','§',c) for c in cns)
    out.append((len(rs),len(rawen),len(cns),len(cnsk),sk,rs))
out.sort(key=lambda x:-x[0])
print('families(>=3 leaves, >=3 distinct EN):',len(out))
print('%5s %5s %6s %7s  %s'%('leaf','enVar','cnVar','cnSkel','skeleton'))
for n,ne,nc,ncs,sk,rs in out:
    flag='DRIFT' if ncs>1 else '.'
    print('%5d %5d %6d %7d %-5s %s'%(n,ne,nc,ncs,flag,sk[:120]))
