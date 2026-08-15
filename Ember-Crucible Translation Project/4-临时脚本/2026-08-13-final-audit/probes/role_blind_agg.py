#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：找出「聚合键里缺了字段角色」的分类器/归一器，并量化它会误伤多少。

判据（可机械化）
----------------
一个聚合站点 = (keyfn, 作用域, 产出)。给全库每片叶算 role = 路径里最后一段非数字的键名。
对某个 keyfn 分组后：
    role_hetero  = 组内出现 >= 2 种 role
    cn_split_by_role = 组内不同 role 的中文集合不同（即「角色决定写法」）
`cn_split_by_role` 的组，就是「缺角色维度会给出错误单值/错误归一建议」的那些。

只读，不写库。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
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


def role_of(path):
    for seg in reversed(path.split('.')):
        if not seg.isdigit():
            return seg
    return path


def corpus():
    """[(repo, pack, path, role, en, cn|None)] —— 覆盖 entries + folders 两个顶层。"""
    rows = []
    for repo in REPOS:
        rn = os.path.basename(repo)
        en_dir, cn_dir = os.path.join(repo, 'compendium', 'en'), os.path.join(repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json') or fn == '_source.json':
                continue
            cnp = os.path.join(cn_dir, fn)
            if not os.path.exists(cnp):
                continue
            en_doc, cn_doc = load(os.path.join(en_dir, fn)), load(cnp)
            for top in ('entries', 'folders'):
                en_t, cn_t = en_doc.get(top) or {}, cn_doc.get(top) or {}
                cn_flat = dict(walk(cn_t))
                for path, e in walk(en_t):
                    c = cn_flat.get(path)
                    rows.append((rn, fn, f'{top}.{path}', role_of(path), e, c))
    return rows


# ---- fill_missing.py / fill_twin.py 的 shape_of（逐字复刻） ---------------
STRUCT_FM = {'entries', 'actors', 'items', 'actions', 'effects', 'pages', 'journals',
             'results', 'folders', 'biography', 'description', 'name', 'text',
             'label', 'public', 'private', 'appearance', 'condition', 'tokenName'}
STRUCT_FT = {'journals', 'pages', 'actors', 'items', 'actions', 'effects', 'folders',
             'macros', 'scenes', 'tables', 'results', 'text', 'content', 'system',
             'description', 'public', 'private', 'name', 'biography', 'prototypeToken',
             'overview', 'exposition', 'summary', 'terrain', 'gamemaster', 'subtitle',
             'pronunciation', 'caption', 'label', 'outcomes', 'notes', 'journal'}


def shape_fm(path):   # fill_missing 用文档根路径（含 entries./folders. 前缀）
    return '.'.join(p for p in path.split('.') if p in STRUCT_FM or p.isdigit())


def shape_ft(path):   # fill_twin 用 entries 之下的路径
    q = path.split('.', 1)[1] if path.startswith(('entries.', 'folders.')) else path
    return '.'.join(p for p in q.split('.') if p in STRUCT_FT or p.isdigit())


def report(title, groups, note='', show=14):
    """groups: key -> list[(role, cn, pack, path, en)]"""
    hetero, split = {}, {}
    for k, mem in groups.items():
        roles = {r for r, *_ in mem}
        if len(roles) < 2:
            continue
        hetero[k] = mem
        by_role = defaultdict(set)
        for r, c, *_ in mem:
            by_role[r].add(c)
        # 角色决定写法：存在两个角色，其中文集合不相交
        vals = list(by_role.items())
        disjoint = any(not (a[1] & b[1]) for i, a in enumerate(vals) for b in vals[i + 1:])
        if disjoint:
            split[k] = mem
    print(f'\n=== {title} ===')
    if note:
        print(f'    {note}')
    print(f'    组数 {len(groups)} | 跨角色组 {len(hetero)} | **角色决定中文写法** {len(split)}')
    for k, mem in sorted(split.items(), key=lambda kv: -len(kv[1]))[:show]:
        by_role = defaultdict(Counter)
        for r, c, pack, path, en in mem:
            by_role[r][c] += 1
        parts = ' || '.join(f'{r}: ' + ', '.join(f'{c!r}×{n}' for c, n in cc.most_common(3))
                            for r, cc in by_role.items())
        print(f'  - {k!r:60.60}  {parts[:200]}')
    return split


def main():
    rows = corpus()
    trans = [r for r in rows if r[5] and CJK.search(r[5])]
    print(f'叶子总数 {len(rows)}（含中文 {len(trans)}）')
    print('role 分布 top20:', Counter(r[3] for r in trans).most_common(20))

    # ---- 站点 1：仅按英文原文聚合（build_glossary.harvest / fill_*.plain 回退）
    g = defaultdict(list)
    for rn, pack, path, role, en, cn in trans:
        if len(en) <= 60 and '\n' not in en and '<' not in en:
            g[en].append((role, cn, pack, path, en))
    s1 = report('S1  key=英文原文（无角色）  —— build_glossary.harvest / fill_*.tm_plain 回退',
                g, 'glossary_ec.json 取多数；fill_missing/fill_twin 在 shape 键未命中时退到这里')

    # ---- 站点 2：fill_missing.shape_of
    g = defaultdict(list)
    for rn, pack, path, role, en, cn in trans:
        g[(shape_fm(path), en)].append((role, cn, pack, path, en))
    s2 = report('S2  key=(fill_missing.shape_of, 英文)', g,
                'STRUCT 白名单外的字段名全被丢弃 → adjective/subtitle/pronunciation/overview… 塌成同一 shape')

    # ---- 站点 3：fill_twin.shape_of
    g = defaultdict(list)
    for rn, pack, path, role, en, cn in trans:
        g[(shape_ft(path), en)].append((role, cn, pack, path, en))
    s3 = report('S3  key=(fill_twin.shape_of, 英文)', g, 'STRUCT 较全，但同样没有 adjective')

    # ---- 站点 4：末段 .name + 英文（fill_twin_names.lib / scan_bare_english_names.DICT）
    g = defaultdict(list)
    for rn, pack, path, role, en, cn in trans:
        if role == 'name':
            g[en].append(('.'.join(p for p in path.split('.') if not p.isdigit())[:0] or _nrole(path),
                          cn, pack, path, en))
    s4 = report('S4  key=英文，作用域=所有 .name（末段键）—— fill_twin_names.lib', g,
                'fill_twin.py 的 docstring 明写「按末段键会把三类 name 塌回一堆」，本站点正是如此',
                show=14)
    return s1, s2, s3, s4


def _nrole(path):
    """name 的**子角色**：它挂在什么容器下（item / action / effect / page / folder / …）"""
    segs = [p for p in path.split('.') if not p.isdigit()]
    # segs[-1] == 'name'
    for s in reversed(segs[:-1]):
        if s in ('items', 'actions', 'effects', 'pages', 'journals', 'actors',
                 'results', 'folders', 'tables', 'scenes', 'macros', 'regions',
                 'levels', 'tokens', 'outcomes', 'entries'):
            return s
    return 'other'


if __name__ == '__main__':
    main()
