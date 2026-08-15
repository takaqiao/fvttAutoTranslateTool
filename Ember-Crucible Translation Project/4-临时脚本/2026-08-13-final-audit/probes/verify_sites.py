#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐站点核实：缺角色/类型维度的分类器各自会误伤多少。只读。"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EM, CR = os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")
CJK = re.compile(r'[\u4e00-\u9fff]')


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{p}.{i}' if p else str(i))
    elif isinstance(o, str) and o.strip():
        yield p, o


CONTAINERS = ('items', 'actions', 'effects', 'pages', 'journals', 'actors',
              'results', 'folders', 'tables', 'scenes', 'macros', 'regions',
              'levels', 'tokens', 'outcomes', 'categories')


def subrole(path):
    segs = [s for s in path.split('.') if not s.isdigit()]
    last = segs[-1] if segs else path
    if last != 'name':
        return last
    for s in reversed(segs[:-1]):
        if s in CONTAINERS:
            return f'{s}.name'
    return 'top.name'


def pairs(repo):
    en_dir, cn_dir = os.path.join(repo, 'compendium', 'en'), os.path.join(repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith('.json') or fn == '_source.json':
            continue
        cnp = os.path.join(cn_dir, fn)
        if not os.path.exists(cnp):
            continue
        en, cn = load(os.path.join(en_dir, fn)), load(cnp)
        yield fn, en, cn


# ══════════════════ 站点 A：fill_twin_names.py 的 lib ══════════════════
print('=' * 78)
print('A. fill_twin_names.py  ——  lib 只按「末段 .name + 英文」取多数')
print('=' * 78)
seen = Counter()
role_of_vote = defaultdict(lambda: defaultdict(Counter))
for repo in (EM, CR):
    for fn, en, cn in pairs(repo):
        enf = dict(walk(en.get('entries', {})))
        cnf = dict(walk(cn.get('entries', {})))
        for path, e in enf.items():
            if not path.endswith('.name'):
                continue
            c = cnf.get(path)
            if c and CJK.search(c):
                seen[(e.strip(), c)] += 1
                role_of_vote[e.strip()][c][subrole(path)] += 1
lib = {}
for (e, c), n in seen.items():
    if e not in lib or n > lib[e][1]:
        lib[e] = (c, n)
print(f'lib 条目 {len(lib)}')

# 目标：ember.adventure 里仍空的 .name 槽
ea_en = load(os.path.join(EM, 'compendium', 'en', 'ember.adventure.json')).get('entries', {})
ea_cn = load(os.path.join(EM, 'compendium', 'cn', 'ember.adventure.json')).get('entries', {})
enf, cnf = dict(walk(ea_en)), dict(walk(ea_cn))
empty = [(p, e) for p, e in enf.items() if p.endswith('.name')
         and not (cnf.get(p) and CJK.search(cnf[p]))]
print(f'ember.adventure 仍为空的 .name 槽：{len(empty)}')

BARE_ROLES = {'regions.name', 'results.name', 'levels.name', 'categories.name',
              'tokens.name', 'outcomes.name'}
bad = []
for p, e in empty:
    key = e.strip()
    if key not in lib:
        continue
    val, n = lib[key]
    src_roles = role_of_vote[key][val]
    tgt = subrole(p)
    # 误伤 = 取值来自「裸中文约定」的角色，而写入目标是文档 name（需双语）
    if set(src_roles) & BARE_ROLES and tgt not in BARE_ROLES:
        bad.append((p, e, val, dict(src_roles), tgt))
print(f'  其中会从「裸中文约定」角色取值、写进需双语的 name 槽：{len(bad)}')
for r in bad[:20]:
    print(f'   {r[0][:88]}\n      EN={r[1]!r} -> 取 {r[2]!r}  来源角色 {r[3]}  目标角色 {r[4]}')

# lib 里有多少条目的取值来自裸中文角色（＝被污染的 TM 条目）
poisoned = [e for e, (c, n) in lib.items()
            if set(role_of_vote[e][c]) & BARE_ROLES]
print(f'  lib 中取值来自裸中文角色的条目总数：{len(poisoned)}')
print('   样例:', [(e, lib[e][0], dict(role_of_vote[e][lib[e][0]])) for e in poisoned[:8]])


# ══════════════════ 站点 B：fill_missing.py 留一法 ══════════════════
print()
print('=' * 78)
print('B. fill_missing.py  ——  shape_of 白名单缺 adjective 等；shape 未命中即退到纯英文键')
print('=' * 78)
STRUCT = {'entries', 'actors', 'items', 'actions', 'effects', 'pages', 'journals',
          'results', 'folders', 'biography', 'description', 'name', 'text',
          'label', 'public', 'private', 'appearance', 'condition', 'tokenName'}


def shape_of(path):
    return '.'.join(p for p in path.split('.') if p in STRUCT or p.isdigit())


tm_shape, tm_plain = defaultdict(Counter), defaultdict(Counter)
allleaf = []
for repo in (EM, CR):
    for fn, en, cn in pairs(repo):
        enf, cnf = dict(walk(en)), dict(walk(cn))
        for path, src in enf.items():
            tgt = cnf.get(path)
            if tgt and CJK.search(tgt):
                tm_shape[(shape_of(path), src)][tgt] += 1
                tm_plain[src][tgt] += 1
                allleaf.append((repo, fn, path, src, tgt))
print(f'TM: shape 键 {len(tm_shape)} / 纯英文键 {len(tm_plain)}；叶 {len(allleaf)}')

wrong = []
for repo, fn, path, src, tgt in allleaf:
    sk = (shape_of(path), src)
    c = Counter(tm_shape[sk])
    c[tgt] -= 1                       # 留一
    if c[tgt] <= 0:
        del c[tgt]
    cands = c if c else Counter(tm_plain[src])
    if cands is not c:                # 退到了纯英文键
        cands = Counter(cands)
        cands[tgt] -= 1
        if cands[tgt] <= 0:
            del cands[tgt]
    if not cands:
        continue
    best = cands.most_common(1)[0][0]
    if best != tgt:
        wrong.append((fn, path, subrole(path), src[:50], tgt[:40], best[:40]))
print(f'留一法：会被 TM 填成**与实际译文不同**的叶 {len(wrong)}')
byrole = Counter(w[2] for w in wrong)
print('  按角色:', byrole.most_common(15))
for w in wrong[:15]:
    print(f'   {w[2]:<14} {w[1][:70]}\n       EN={w[3]!r}  实际={w[4]!r}  TM会填={w[5]!r}')


# ══════════════════ 站点 C：resolve_generic_fallback.py 的类型盲 ══════════════════
print()
print('=' * 78)
print('C. resolve_generic_fallback.py  ——  resolvable 集合合并了所有文档类型')
print('=' * 78)
by_type = defaultdict(set)
for repo in (EM, CR):
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(cn_dir)):
        if not fn.endswith('.json'):
            continue
        doc = load(os.path.join(cn_dir, fn))
        for k, v in (doc.get('entries') or {}).items():
            if isinstance(v, dict) and isinstance(v.get('name'), str) and CJK.search(v['name']):
                by_type[fn].add(k)
allnames = set().union(*by_type.values())
print(f'resolvable（类型盲，合并全部包）：{len(allnames)} 个名字，来自 {len(by_type)} 个包')
# 按包类型粗分
JOURNALISH = {'crucible.rules.json', 'crucible._packs-folders.json'}
print('  各包贡献：', sorted(((len(v), k) for k, v in by_type.items()), reverse=True)[:8])
# 首匹配问题：路径里 .actors. 出现在 .items. 之前的比例
EMBEDDED = re.compile(r'\.(items|effects|actors)\.([^.]+)')
tot = firstactor = 0
for repo, fn, path, src, tgt in allleaf:
    m = EMBEDDED.search('.' + path)
    if not m:
        continue
    tot += 1
    if m.group(1) == 'actors' and re.search(r'\.(items|effects)\.', path[m.end():]):
        firstactor += 1
print(f'  含内嵌段的路径 {tot}；其中首匹配落在 actors 而真正的内嵌文档在更深处：{firstactor}'
      f'（{firstactor/max(tot,1):.1%}）')
