# -*- coding: utf-8 -*-
"""通用落地器：吃 [{pack,ptr,en,cn}]，写进 compendium/cn/。
前置自证 + 后置自证都在里面，任何一条不过就不写盘。
"""
import sys,os,json,collections,argparse
sys.path.insert(0,r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-18-ember-061\stage1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *

_orig_split = split_path
def split_path(ptr):
    """本单元的路径用 /effects/0/name 这种纯斜杠写法（lib4.walk 产的），
    数字段一律当数组下标切成 int。"""
    return [int(x) if x.isdigit() else x for x in ptr.strip('/').split('/')]

def en_only(cnj, enj):
    a=dict(walk_leaves(enj.get('entries',{}))); b=dict(walk_leaves(cnj.get('entries',{})))
    return [k for k in a if k not in b]

def run(items, expect_n, dry=False):
    byfile=collections.defaultdict(list)
    for it in items: byfile[it['pack']].append(it)
    CNJ={f:jload(os.path.join(CN,f)) for f in FILES}
    ENJ={f:jload(os.path.join(EN060,f)) for f in FILES}
    before={f:dict(walk_leaves(CNJ[f].get('entries',{}))) for f in FILES}
    gap0=sum(len(en_only(CNJ[f],ENJ[f])) for f in FILES)
    # ---- 前置自证 ----
    prob=[]
    assert len(items)==expect_n, f'条数不等真值: {len(items)} != {expect_n}'
    for it in items:
        parts=split_path(it['ptr'])
        if has_leaf(CNJ[it['pack']], parts): prob.append(f"[前置] CN 已有该叶 {it['pack']} {it['ptr']}")
        tgt=get_at(ENJ[it['pack']],parts)
        if not isinstance(tgt,str): prob.append(f"[前置] EN 无此叶 {it['pack']} {it['ptr']}")
        elif tgt!=it['en']: prob.append(f"[前置] EN 值不符 {it['ptr']}")
    if prob:
        print('前置自证失败',len(prob)); [print('  !!',p) for p in prob[:30]]; return 1
    print(f'[前置自证] 条数 {len(items)} == 真值 {expect_n} ✓ ; 全部落点在 CN 中不存在且 EN 侧存在且英文逐字节相符 ✓')
    print(f'[前置自证] 落地前 EN-only 叶 = {gap0}')
    # ---- 写入 ----
    for f,lst in byfile.items():
        for it in lst: set_at(CNJ[f], split_path(it['ptr']), it['cn'])
    # ---- 后置自证 ----
    bad=[]
    for it in items:
        got=get_at(CNJ[it['pack']], split_path(it['ptr']))
        if got!=it['cn']: bad.append(f"[后置] 落点值不符 {it['ptr']}: {got!r}")
    # 未点名的叶一个都不许变
    changed_other=[]
    for f in FILES:
        now=dict(walk_leaves(CNJ[f].get('entries',{})))
        touched={tuple(str(x) for x in split_path(it['ptr'])[1:]) for it in byfile.get(f,[])}  # 去掉开头的 'entries'，与 walk_leaves 的前缀对齐
        for k,v in before[f].items():
            if k in touched: continue
            if now.get(k)!=v: changed_other.append(f'{f} {"/".join(k)}')
        for k in now:
            if k not in before[f] and k not in touched: changed_other.append(f'NEW-UNASKED {f} {"/".join(k)}')
    gap1=sum(len(en_only(CNJ[f],ENJ[f])) for f in FILES)
    # 标签/增强器/占位符对拍
    par=[]
    for it in items:
        parts=split_path(it['ptr']); tgt=get_at(ENJ[it['pack']],parts); out=it['cn']
        for nm,fn in (('TAG',tag_multiset),('ENH',uuid_targets),('ROLL',rolls)):
            a,b=collections.Counter(fn(out)),collections.Counter(fn(tgt))
            if a!=b: par.append(f"{nm} 不齐 {it['ptr']} CN多={list(a-b)[:3]} EN多={list(b-a)[:3]}")
        # {…} 是 @UUID[…]{可译标签}，中英必然不同串 —— 与 edit_engine 一致，只比个数
        na,nb=len(placeholders(out)),len(placeholders(tgt))
        if na!=nb: par.append(f"PH 个数不等 {it['ptr']} CN={na} EN={nb}")
    print(f'[后置自证] 落点值不符 {len(bad)} · 误伤其他叶 {len(changed_other)} · 多重集不齐 {len(par)}')
    print(f'[后置自证] 落地后 EN-only 叶 = {gap1}（应为 {gap0}-{len(items)}={gap0-len(items)}）')
    for p in (bad+changed_other+par)[:30]: print('  !!',p)
    if bad or changed_other or par or gap1!=gap0-len(items):
        print('!! 自证不过，不写盘'); return 1
    if dry:
        print('[dry] 自证全过，未写盘'); return 0
    for f in FILES: save_cn(CNJ[f], os.path.join(CN,f))
    print('已写盘:', ', '.join(FILES))
    return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--expect',type=int,required=True)
    ap.add_argument('--dry',action='store_true')
    a=ap.parse_args()
    items=json.load(open(a.file,encoding='utf-8'))
    sys.exit(run(items,a.expect,a.dry))
