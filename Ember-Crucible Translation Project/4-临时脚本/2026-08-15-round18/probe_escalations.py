#!/usr/bin/env python3
"""定位 Y2 升报的 4 条真缺陷在两仓里的**全路径与孪生落点**，并打印 EN/CN 同块。

反空转：打印遍历叶数、命中路径数。
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']

# split_blocks 与 R-arcturel-arcturian-blocks / R-rank-sense-blocks 同口径：只按块级标签切
BLOCK_TAGS = r'(?:p|div|section|h[1-6]|li|ul|ol|table|tr|td|th|blockquote|aside|figure|figcaption|hr|br)'
BLOCK_SPLIT = re.compile(r'(?=<\s*/?\s*' + BLOCK_TAGS + r'\b)', re.I)
INLINE = re.compile(r'<\s*/?\s*(?!' + BLOCK_TAGS + r'\b)[a-zA-Z][^>]*>')


def split_blocks(s):
    return [b for b in BLOCK_SPLIT.split(s) if b.strip()]


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(o, pre=''):
    if isinstance(o, str):
        yield pre, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{pre}.{k}' if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{pre}.{i}')


TARGETS = [
    ('E1', 'actors.Sadri Zhalimorne.biography.private', None),
    ('E2', 'actors.Constructed Companion.biography.private', None),
    ('E2ref', 'actors.Woven Construct.biography.private', None),
    ('E3', 'Unfinished Business.pages.Shine On.text', None),
    ('E4', 'Unfinished Business.pages.The Old Flame.text', None),
]

leaves = 0
hits = {t[0]: [] for t in TARGETS}
for repo in REPOS:
    cnd = os.path.join(ROOT, repo, 'compendium', 'cn')
    end = os.path.join(ROOT, repo, 'compendium', 'en')
    for fn in sorted(os.listdir(cnd)):
        if not fn.endswith('.json'):
            continue
        cn = load(os.path.join(cnd, fn))
        en = load(os.path.join(end, fn)) if os.path.exists(os.path.join(end, fn)) else {}
        cmap = dict(walk(cn.get('entries', {})))
        emap = dict(walk(en.get('entries', {})))
        for p, s in cmap.items():
            leaves += 1
            for tag, suffix, _ in TARGETS:
                if p.endswith(suffix):
                    hits[tag].append((repo, fn, p, s, emap.get(p)))

print(f'遍历中文叶数={leaves}')
for tag, suffix, _ in TARGETS:
    print('=' * 78)
    print(f'{tag}  suffix={suffix}  命中={len(hits[tag])}')
    for repo, fn, p, cns, ens in hits[tag]:
        bc = split_blocks(cns)
        be = split_blocks(ens) if ens else []
        print(f'  -- {repo}/{fn}')
        print(f'     path={p}')
        print(f'     CN块={len(bc)} EN块={len(be)}')
        if tag in ('E1',):
            idxs = [21]
        elif tag == 'E2':
            idxs = [7]
        elif tag == 'E2ref':
            idxs = [3]
        elif tag == 'E3':
            idxs = [4, 14, 31]
        else:
            idxs = [264, 268]
        for i in idxs:
            if i < len(be):
                print(f'     [EN {i}] {be[i]}')
            if i < len(bc):
                print(f'     [CN {i}] {bc[i]}')
        # 关键词扫描（不依赖块号）
        for kw in ['阿克图里安高原', 'Rank 1', 'Rank 2', '级魂印', '级魂缚', '名匠']:
            n = cns.count(kw)
            if n:
                print(f'     CN含 {kw!r} ×{n}')
