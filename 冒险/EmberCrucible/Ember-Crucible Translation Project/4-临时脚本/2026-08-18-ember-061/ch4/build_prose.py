# -*- coding: utf-8 -*-
import sys,os,json,re,collections
sys.path.insert(0,r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\stage1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
HERE=os.path.dirname(os.path.abspath(__file__))
MARKUP=re.compile(r'(@[A-Za-z]+\[[^\]]*\](\{[^}]*\})?)|(\[\[[^\]]*\]\])|(<[^>]+>)|(&[a-z]+;)')
def has_letters(s): return bool(re.search(r'[A-Za-z]{2,}', MARKUP.sub(' ',s)))
CNJ={f:jload(os.path.join(CN,f)) for f in FILES}
ENJ={f:jload(os.path.join(EN060,f)) for f in FILES}
gap=[]
for f in FILES:
    a=dict(walk_leaves(ENJ[f].get('entries',{}))); b=dict(walk_leaves(CNJ[f].get('entries',{})))
    for k,v in a.items():
        if k in b or not isinstance(v,str): continue
        gap.append((f,'/entries/'+'/'.join(k),v))
prose=[(f,p,v) for f,p,v in gap if has_letters(v)]
pure=[(f,p,v) for f,p,v in gap if not has_letters(v)]
print('剩余 EN-only 叶',len(gap),'其中正文',len(prose),'纯记号',len(pure))
items=collections.OrderedDict()
for f,p,v in prose: items.setdefault(v,[]).append((f,p))
print('正文去重工作项',len(items),'字符',sum(len(k) for k in items))
json.dump([[v,locs] for v,locs in items.items()],open(os.path.join(HERE,'prose_items.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
json.dump(pure,open(os.path.join(HERE,'pure_left.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
