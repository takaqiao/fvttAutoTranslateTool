# -*- coding: utf-8 -*-
import sys,os,json,re,collections
sys.path.insert(0,r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\stage1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
HERE=os.path.dirname(os.path.abspath(__file__))
BI={'actors/*/items/*/name','journals/*/pages/*/name','actors/*/items/*/effects/*/name','items/*/name',
 'actors/*/items/*/actions/*/name','scenes/*/notes/*','actors/*/name','folders/*',
 'actors/*/items/*/actions/*/effects/#/name','scenes/*/name','items/*/effects/*/name','journals/*/name',
 'actors/*/archetype/name','actors/*/taxonomy/name','items/*/actions/*/name','macros/*/name','Shard God/name'}
CNONLY={'scenes/*/regions/*/behaviors/*/name','scenes/*/regions/*/name','journals/*/pages/*/outcomes/*/label',
 'journals/*/categories/*','actors/*/tokenName','scenes/*/levels/*','journals/*/pages/*/encounterTokens/*',
 'scenes/*/sounds/*','items/*/effects/*/adjective','actors/*/items/*/effects/*/adjective'}
CONT={'folders','journals','pages','scenes','actors','items','macros','playlists','tables','levels','notes',
 'sounds','regions','behaviors','results','effects','actions','categories','encounterTokens','outcomes',
 'tokens','lights','walls','drawings','tiles'}
def sig(path):
    segs=path.strip('/').split('/'); out=[]
    for i,s in enumerate(segs):
        if s in ('entries','Ember Early Access'): continue
        if s.isdigit(): out.append('#')
        elif i>0 and segs[i-1] in CONT: out.append('*')
        else: out.append(s)
    return '/'.join(out)
J=lambda n: json.load(open(os.path.join(HERE,n),encoding='utf-8'))
cand=J('gloss_map.json'); core=J('core_map.json'); gmd=J('gloss_md.json')
MANUAL={
 'Cat':'猫', 'Non-Player Character':'非玩家角色', 'Trickadee':'戏诈雀',
 'Primordis 2: Canny':'普里莫迪斯 2：机敏','Primordis 4: Devious':'普里莫迪斯 4：狡诈',
 'Primordis 5: Predatory':'普里莫迪斯 5：掠食',
 'Canny':'机敏','Devious':'狡诈','Predatory':'掠食',
 '+3 AC':'AC +3 加值','Reliable':'可靠','Nimbleness':'轻捷','Amulet of Nimbleness':'轻捷护符',
 'Cloak':'斗篷','Poison Conversion':'毒素转化','Venomous':'剧毒','Inflection: Determine':'屈折：限定',
}
for n in range(2,10): MANUAL[f'Torch {n}']=f'火把 {n}'
for ew,cew in (('East','东'),('West','西')):
    for lv,clv in (('Ground','地面层'),('Lower','下层'),('Upper','上层')):
        MANUAL[f'Waterfall {ew} {lv}']=f'瀑布 {cew}侧{clv}'
DIRS={'North':'北','South':'南','East':'东','West':'西','North East':'东北','North West':'西北',
      'South East':'东南','South West':'西南'}
for d,cd in DIRS.items():
    MANUAL[f'Buzzing Insects {d}']=f'昆虫嗡鸣 {cd}'
    MANUAL[f'Fleshy Movement {d}']=f'肉质蠕动 {cd}'

def strip_tail(cn,en):
    return cn[:-len(en)].rstrip() if cn.endswith(en) else cn

def resolve(en):
    """-> (core_cn, source) or (None,None)"""
    if en in MANUAL: return MANUAL[en],'manual'
    if en in cand:   return cand[en][0],'ruling'
    if en in core:   return core[en][0][0],'library'
    if en in gmd:    return strip_tail(gmd[en],en),'glossaryB'
    e2=en.replace('\u2019',"'")
    for src,d in (('manual',MANUAL),('ruling',cand)):
        if e2 in d: return (d[e2][0] if src=='ruling' else d[e2]),src+'~apos'
    if e2 in core: return core[e2][0][0],'library~apos'
    if e2 in gmd:  return strip_tail(gmd[e2],e2),'glossaryB~apos'
    return None,None

def build():
    names=J('name_leaves.json')
    out=[]; unres=[]; srcs=collections.Counter()
    for p,k,v in names:
        s=sig(k)
        c,src=resolve(v)
        if c is None: unres.append((p,k,v,s)); continue
        if s in BI: cn=f'{c} {v}'
        elif s in CNONLY: cn=c
        else: unres.append((p,k,v,'UNKNOWN-ROLE:'+s)); continue
        srcs[src]+=1
        out.append({'pack':p,'ptr':k,'en':v,'cn':cn,'sig':s,'src':src})
    return out,unres,srcs

if __name__=='__main__':
    out,unres,srcs=build()
    print('resolved',len(out),'unresolved',len(unres),dict(srcs))
    for x in unres: print('  !!',x)
    json.dump(out,open(os.path.join(HERE,'names_resolved.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
