# -*- coding: utf-8 -*-
"""V1: 收尾验收。前置自证 + 逐条核对。

前置自证（两件都断言）：
  ① 切出来的条数 = 已知真值：本轮 CN 侧真正变动的叶 = 删 298 + 迁移源 16（走） + 迁移目标 16（来） + 改 334；
  ② 切对地方：变动集合与「删除桶 ∪ 迁移目标 ∪ 改动桶」逐一相等，一个多余的叶都没有。
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

BK = os.path.join(WORK, 'BACKUP_cn')
delta = load_delta()
MIG = jload(os.path.join(WORK, 'mig_map.json')) if os.path.exists(os.path.join(WORK, 'mig_map.json')) else None

def leaves(d):
    return {'/'.join(p): v for p, v in walk_leaves(d) if isinstance(v, str)}

ok = True
print("=== ① 变动面自证 ===")
tot = collections.Counter()
all_diff = {}
for f in FILES:
    before = leaves(jload(os.path.join(BK, f)))
    after = leaves(jload(os.path.join(CN, f)))
    gone = set(before) - set(after)
    new = set(after) - set(before)
    chg = {k for k in set(before) & set(after) if before[k] != after[k]}
    all_diff[f] = (gone, new, chg)
    tot['gone'] += len(gone); tot['new'] += len(new); tot['changed'] += len(chg)
    print(f"  {f:32s} CN 少了 {len(gone):4d} 叶 / 多了 {len(new):3d} 叶 / 改了 {len(chg):4d} 叶")
print(f"  合计 少 {tot['gone']} / 多 {tot['new']} / 改 {tot['changed']}")
exp_gone, exp_new = 314, 16          # 314 = 298 真删 + 16 迁移源；16 = 迁移目标
noop = 0
for f in FILES:
    b = leaves(jload(os.path.join(BK, f))); a = leaves(jload(os.path.join(CN, f)))
    noop += sum(1 for p in delta[f]['changed'] if p.lstrip('/') in a and p.lstrip('/') in b
                and a[p.lstrip('/')] == b[p.lstrip('/')])
print(f"  改动桶 334 叶里，英文改动**不影响中文**的（译文按原样保留）：{noop} 叶")
assert (tot['gone'], tot['new'], tot['changed'] + noop) == (exp_gone, exp_new, 334), (tot, noop)
print(f"  与真值 (少 {exp_gone} / 多 {exp_new} / 改动桶 {tot['changed']}+{noop}=334) 相等  OK")

print("\n=== ② 切对地方自证 ===")
for f in FILES:
    gone, new, chg = all_diff[f]
    d = delta[f]
    exp_gone_set = {p.lstrip('/') for p in d['gone']}
    exp_chg_set = {p.lstrip('/') for p in d['changed']}
    bad = []
    if gone != exp_gone_set:
        bad.append(f"少掉的叶 != 删除桶（多 {len(gone-exp_gone_set)} / 缺 {len(exp_gone_set-gone)}）")
    if chg - exp_chg_set:
        bad.append(f"改掉的叶超出改动桶 {len(chg-exp_chg_set)} 处")
    for p in new:
        if '/' + p not in d['added']:
            bad.append(f"新增的叶不在新增桶里: {p}")
    st = 'OK ' if not bad else 'BAD'
    if bad:
        ok = False
    print(f"  {st} {f:32s} {bad if bad else '删/改/增 三面逐一对上'}")

print("\n=== ③ 未触碰的叶：一个字节都没动 ===")
for f in FILES:
    before = leaves(jload(os.path.join(BK, f)))
    after = leaves(jload(os.path.join(CN, f)))
    untouched = (set(before) & set(after)) - {p.lstrip('/') for p in delta[f]['changed']}
    diff = [k for k in untouched if before[k] != after[k]]
    print(f"  {f:32s} 未触碰 {len(untouched)} 叶，其中被改动的 {len(diff)}")
    if diff:
        ok = False

print("\n=== ④ 增强器 / 标签 / 骰子记号：全库多重集对拍 ===")
print("  (a) 未触碰叶：改前 == 改后")
for f in FILES:
    before = leaves(jload(os.path.join(BK, f)))
    after = leaves(jload(os.path.join(CN, f)))
    untouched = (set(before) & set(after)) - {p.lstrip('/') for p in delta[f]['changed']}
    for name, fn in (('ENH', uuid_targets), ('TAG', tag_multiset), ('ROLL', rolls)):
        a = collections.Counter(x for k in untouched for x in fn(before[k]))
        b = collections.Counter(x for k in untouched for x in fn(after[k]))
        st = 'OK ' if a == b else 'BAD'
        if a != b:
            ok = False
        print(f"    {st} {f.split('.')[1]:20s} {name} 改前 {sum(a.values())} / 改后 {sum(b.values())}")

print("  (b) 被改动的 334 叶 + 16 迁移目标：CN 与 EN061 逐叶多重集相等")
bad = collections.Counter()
n = 0
for f in FILES:
    after = jload(os.path.join(CN, f))
    en = jload(os.path.join(EN061, f))
    gone, new, chg = all_diff[f]
    for p in sorted({q.lstrip('/') for q in delta[f]['changed']} | new):
        parts = split_path('/' + p)
        c = get_at(after, parts); e = get_at(en, parts)
        if not isinstance(e, str):
            bad['EN061缺叶'] += 1
            continue
        n += 1
        for name, fn in (('ENH', uuid_targets), ('TAG', tag_multiset), ('ROLL', rolls)):
            if collections.Counter(fn(c)) != collections.Counter(fn(e)):
                bad[name] += 1
print(f"    逐叶核了 {n} 叶，不齐：{dict(bad) if bad else '0（三项全齐）'}")
if bad:
    ok = False

print("\n=== ⑤ 中文侧不得留空 / 不得留槽标记 ===")
empt = 0
for f in FILES:
    after = leaves(jload(os.path.join(CN, f)))
    gone, new, chg = all_diff[f]
    for p in sorted({q.lstrip('/') for q in delta[f]['changed']} | new):
        v = after.get(p, '')
        if '⟦' in v or v.strip() == '':
            empt += 1
            print('    !!', f, p)
print(f"    留空/留标记 {empt} 处")
if empt:
    ok = False

print("\nVERIFY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
