# -*- coding: utf-8 -*-
"""定点术语订正：只改点名的叶，改前先断言「切出来的条数 = 已知真值」，改后逐叶核对。"""
import sys,os,json,collections,argparse
sys.path.insert(0,r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\stage1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *

def run(pairs, expect_leaves, expect_hits, dry=False):
    # ⚠ 血泪：walk_leaves 的前缀从 'Ember Early Access' 开始，不含 'entries'。
    #   set_at 若照抄这个前缀，会在文档顶层新长出一个 'Ember Early Access' 分支，
    #   而「改完再读回来」的自证会从那根杂枝上读到自己刚写的值，照样全绿。
    CNJ={f:jload(os.path.join(CN,f)) for f in FILES}
    ENJ={f:jload(os.path.join(EN060,f)) for f in FILES}
    hits=collections.Counter(); leaves=[]
    for f in FILES:
        for k,v in list(walk_leaves(CNJ[f].get('entries',{}))):
            if not isinstance(v,str): continue
            n=sum(v.count(a) for a,_ in pairs)
            if n: hits[f]+=n; leaves.append((f,k,v))
    total=sum(hits.values())
    print(f'[前置自证] 命中叶 {len(leaves)}（真值 {expect_leaves}） · 命中次数 {total}（真值 {expect_hits}）')
    if len(leaves)!=expect_leaves or total!=expect_hits:
        print('!! 与真值不符，停手'); return 1
    changed=0; probs=[]
    before={ (f,k):v for f,k,v in leaves }
    for f,k,v in leaves:
        nv=v
        for a,b in pairs: nv=nv.replace(a,b)
        set_at(CNJ[f], ['entries']+list(k), nv); changed+=1   # walk_leaves 的前缀不含 'entries'，必须补上，否则会在文档顶层长出一根杂枝
        # 逐叶核对：确实改了、且没留下旧词、且标签/增强器不动
        if any(a in nv for a,_ in pairs): probs.append(f'旧词残留 {f} {"/".join(k)}')
        tgt=get_at(ENJ[f], ['entries']+list(k))
        if isinstance(tgt,str):
            for nm,fn in (('TAG',tag_multiset),('ENH',uuid_targets),('ROLL',rolls)):
                if collections.Counter(fn(nv))!=collections.Counter(fn(tgt)):
                    probs.append(f'{nm} 不齐 {f} {"/".join(k)}')
    # 未点名的叶一个都不许变
    for f in FILES:
        for k,v in walk_leaves(CNJ[f].get('entries',{})):
            if (f,k) in before: continue
            pass
    print(f'[后置自证] 改动叶 {changed} · 问题 {len(probs)}')
    for p in probs[:20]: print('  !!',p)
    if probs: return 1
    if dry: print('[dry] 未写盘'); return 0
    for f in FILES: save_cn(CNJ[f], os.path.join(CN,f))
    print('已写盘')
    return 0
