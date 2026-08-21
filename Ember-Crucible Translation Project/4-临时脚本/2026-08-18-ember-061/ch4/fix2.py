# -*- coding: utf-8 -*-
"""按「叶后缀 + 片段」定点订正。每条都要求：片段在该叶里恰好出现 N 次（N 为写死的真值），
改完逐叶回读真身路径核对，并核 EN 侧的标签/增强器多重集不变。"""
import sys,os,collections
sys.path.insert(0,r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\stage1")
from common import *

# (叶路径后缀, 旧片段, 新片段, 该片段在**每一个**命中叶里应出现的次数)
EDITS = [
 ('journals/Kadra Zann/pages/Sanctuary/text',            '寰宇传讯','宇宙传讯',1),
 ('journals/Ruby Grove/pages/Where Shadows Lie/text',    '寰宇眷顾','宇宙眷顾',1),
 ('actors/Verno Kreed/biography/private',                '寰宇层面','宇宙层面',1),
 ('items/Elder God/description',                         '余烬宇宙','余烬寰宇',1),
 ('actors/Nimaelle/biography/public',                    '在宇宙各处','在寰宇各处',1),
 ('items/Amber Staff/description/private',               '如果你是术士','如果你是邪术师',1),
 ('items/Amber Staff/description/private',               '术法师、术士或法师','术士、邪术师或巫师',1),
 ('actors/Shadebranch/biography/private',                '{荆芽灵}术士','{荆芽灵}邪术师',1),
 ('actors/Shadebranch/biography/private',                '这位术士的','这位邪术师的',1),
 ('journals/Ruby Grove/pages/Local Experts/text',        '别信这些术法师','别信这些术士',1),
]
GLOBAL = [('六角格','六边格')]   # 全库零容忍旧写法

def main(dry=False):
    CNJ={f:jload(os.path.join(CN,f)) for f in FILES}
    ENJ={f:jload(os.path.join(EN060,f)) for f in FILES}
    touched=collections.defaultdict(dict)   # (f,pathtuple) -> newval
    report=[]
    for suf,old,new,cnt in EDITS:
        hits=0
        for f in FILES:
            for k,v in walk_leaves(CNJ[f].get('entries',{})):
                if not isinstance(v,str): continue
                if not '/'.join(k).endswith(suf): continue
                cur = touched[(f,k)] if (f,k) in touched else v
                if old not in cur: continue
                n=cur.count(old)
                assert n==cnt, f'{suf} 里 {old!r} 出现 {n} 次（真值 {cnt}）'
                touched[(f,k)]=cur.replace(old,new)
                hits+=1
        report.append(f'{suf} :: {old} -> {new} · 命中叶 {hits}')
        assert hits>0, f'{suf} :: {old} 一叶都没命中'
    gl=0
    for old,new in GLOBAL:
        for f in FILES:
            for k,v in walk_leaves(CNJ[f].get('entries',{})):
                if not isinstance(v,str): continue
                cur = touched[(f,k)] if (f,k) in touched else v
                if old in cur:
                    touched[(f,k)]=cur.replace(old,new); gl+=1
    report.append(f'全库 六角格->六边格 · 命中叶 {gl}')
    for r in report: print('  ',r)
    print('拟改叶总数',len(touched))
    # 落地
    for (f,k),nv in touched.items(): set_at(CNJ[f], ['entries']+list(k), nv)
    # 后置自证
    probs=[]
    for (f,k),nv in touched.items():
        got=get_at(CNJ[f], ['entries']+list(k))
        if got!=nv: probs.append(f'回读不符 {f} {"/".join(k)}')
        if '六角格' in got: probs.append(f'旧写法残留 {f} {"/".join(k)}')
        tgt=get_at(ENJ[f], ['entries']+list(k))
        if isinstance(tgt,str):
            for nm,fn in (('TAG',tag_multiset),('ENH',uuid_targets),('ROLL',rolls)):
                if collections.Counter(fn(got))!=collections.Counter(fn(tgt)): probs.append(f'{nm} 不齐 {f} {"/".join(k)}')
    # 全库确认
    left=sum(v.count('六角格') for f in FILES for _,v in walk_leaves(CNJ[f].get('entries',{})) if isinstance(v,str))
    print('后置自证：问题',len(probs),'· 全库残留「六角格」',left)
    for p in probs[:20]: print('  !!',p)
    if probs or left: print('!! 不写盘'); return 1
    if dry: print('[dry] 未写盘'); return 0
    for f in FILES: save_cn(CNJ[f], os.path.join(CN,f))
    print('已写盘'); return 0

if __name__=='__main__': sys.exit(main('--dry' in sys.argv))
