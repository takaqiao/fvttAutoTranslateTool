# -*- coding: utf-8 -*-
"""A2: 改动桶落地。
读 fills/*.json（{item: {slot: 中文串}}），用 rebuild 的骨架拼出新 CN 并写回。
未填的槽：PATCH 用旧 CN、M-only PATCH 自动、NEW 未填则报错（不许留 ⟦⟧ 进产线）。
落地后逐叶核 EN061 的标签 / 增强器 / 占位符多重集。
"""
import os, sys, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from rebuild import rebuild, render, reattach_ids
from segs import seg

DRY = '--dry' in sys.argv

fills = {}
for p in sorted(glob.glob(os.path.join(WORK, 'fills', '*.json'))):
    for k, v in json.load(open(p, encoding='utf-8')).items():
        fills.setdefault(k, {}).update(v)

items = json.load(open(os.path.join(WORK, 'items_scope.json'), encoding='utf-8'))
delta = load_delta()
CNJ = {f: jload(os.path.join(CN, f)) for f in FILES}
ENJ = {f: jload(os.path.join(EN061, f)) for f in FILES}

problems, stats = [], collections.Counter()
touched = collections.defaultdict(list)
for i, (scope, ptr) in enumerate(items, 1):
    f = 'ember.adventure.json' if scope in ('both', 'ember.adventure.json') else scope
    old, new = delta[f]['changed'][ptr]
    cn = get_at(CNJ[f], split_path(ptr))
    parts, slots = rebuild(old, new, cn)
    fl = fills.get(str(i), {})
    fill = {}
    for k, s in enumerate(slots):
        if '__full__' in fl:
            fill[k] = ''
            continue
        if str(k) in fl:
            fill[k] = fl[str(k)]
            stats['filled'] += 1
            continue
        t0 = [t for kk, t in seg(s.en_old) if kk == 'T'] if s.en_old else None
        t1 = [t for kk, t in seg(s.en_new) if kk == 'T']
        if s.kind == 'PATCH' and t0 == t1:
            # 只有 @UUID/[[…]] 变了：把 CN 里的 M 记号按位换成 EN 新的
            cs = seg(s.cn_old)
            m_new = [t for kk, t in seg(s.en_new) if kk == 'M']
            m_old = [t for kk, t in seg(s.en_old) if kk == 'M']
            cm = [t for kk, t in cs if kk == 'M']
            if len(cm) == len(m_old) == len(m_new):
                it = iter(m_new)
                fill[k] = ''.join(next(it) if kk == 'M' else t for kk, t in cs)
                stats['auto-M'] += 1
            else:
                fill[k] = s.cn_old
                problems.append(f"ITEM {i} slot {k}: M 记号数不齐 cn{len(cm)} en060 {len(m_old)} en061 {len(m_new)}")
        elif s.kind == 'PATCH':
            fill[k] = s.cn_old
            stats['unfilled-PATCH'] += 1
            problems.append(f"ITEM {i} slot {k}: PATCH 未填（沿用旧 CN） {ptr.split('/',3)[-1]}")
        else:
            fill[k] = ''
            stats['unfilled-NEW'] += 1
            problems.append(f"ITEM {i} slot {k}: NEW 未填（留空） {ptr.split('/',3)[-1]}")
    if '__full__' in fl:
        out = fl['__full__']
        stats['full-override'] += 1
    else:
        out = reattach_ids(render(parts, fill), cn)
    assert '⟦' not in out, f"ITEM {i} 渲染残留槽标记"
    tgt = get_at(ENJ[f], split_path(ptr))
    for name, fn in (('TAG', tag_multiset), ('ENH', uuid_targets), ('ROLL', rolls)):
        a, b = collections.Counter(fn(out)), collections.Counter(fn(tgt))
        if a != b:
            problems.append(f"ITEM {i} {name} 不齐 CN多={list(a-b)[:4]} EN多={list(b-a)[:4]} :: {ptr.split('/',3)[-1]}")
            stats['parity-' + name] += 1
    d_cn = len(placeholders(out)) - len(placeholders(cn))
    d_en = len(placeholders(tgt)) - len(placeholders(old))
    if d_cn != d_en:
        problems.append(f"ITEM {i} PH 增量不等 CNΔ{d_cn} ENΔ{d_en} :: {ptr.split('/',3)[-1]}")
        stats['parity-PH'] += 1
    targets = FILES if scope == 'both' else [scope]
    for tf in targets:
        if ptr in delta[tf]['changed']:
            touched[tf].append((ptr, out))
    stats['items'] += 1

for f, lst in touched.items():
    if DRY:
        continue
    cnj = CNJ[f]
    for ptr, out in lst:
        set_at(cnj, split_path(ptr), out)
    save_cn(cnj, os.path.join(CN, f))
print(dict(stats))
print(f"落地叶数: {sum(len(v) for v in touched.values())} (真值 334)")
print(f"问题 {len(problems)} 条")
for p in problems[:200]:
    print('  !!', p)
