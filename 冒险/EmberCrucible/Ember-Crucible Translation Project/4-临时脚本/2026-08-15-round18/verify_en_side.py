#!/usr/bin/env python3
"""对每处改动核英文侧：取同块英文，再全库数英文闸计数。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPO = os.path.join(ROOT, '1-Ember汉化插件')
TAGSPLIT = re.compile(r'(?=<)')

def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)

def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict): node = node.get(p)
        elif isinstance(node, list):
            try: node = node[int(p)]
            except (ValueError, IndexError): return None
        else: return None
    return node

def split_path(root, path):
    naive = path.split('.')
    if get_at(root, naive) is not None: return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + '.')]
            if cands:
                k = max(cands, key=len); parts.append(k); node = node[k]; rest = rest[len(k)+1:]; continue
        head, _, rest = rest.partition('.')
        parts.append(head); node = get_at(node, [head])
    return parts

def blocks(s): return [b for b in TAGSPLIT.split(s) if b]

TARGETS = [
    ('ember.adventure.json', "Ember Early Access.journals.The Book Of Tales.pages.The Signborn's Secret.text", [43]),
    ('ember.adventure.json', "Ember Early Access.journals.Disgraced House.pages.To Copy a Key.text", [123, 205]),
]
for pack, path, idxs in TARGETS:
    en = load(os.path.join(REPO, 'compendium', 'en', pack))
    cn = load(os.path.join(REPO, 'compendium', 'cn', pack))
    parts = split_path(en['entries'], path)
    be = blocks(get_at(en['entries'], parts))
    bc = blocks(get_at(cn['entries'], parts))
    print('=' * 70)
    print(f'{pack} :: {path}  EN块={len(be)} CN块={len(bc)}  结构相同={len(be)==len(bc)}')
    for i in idxs:
        print('-' * 70)
        print(f'[块 {i}] EN: {be[i]}')
        print(f'[块 {i}] CN: {bc[i]}')

# ---- 全库英文闸计数 ----
print('=' * 70)
print('英文闸计数（全库遍历，含 crucible 仓）')
import itertools
def walk(o, pre=''):
    if isinstance(o, str):
        yield pre, o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, pre + '.' + k if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from walk(v, f'{pre}.{i}')

GATES = {
    'Jahud':      re.compile(r'\bJahud\w*\b'),
    'assassin(小写)': re.compile(r'\bassassins?\b'),
    'Assassin(大写)': re.compile(r'\bAssassins?\b'),
    'the party':  re.compile(r'\bthe party\b'),
    'Any character who': re.compile(r'\bAny characters? who\b', re.I),
}
counts = {k: 0 for k in GATES}
leaves = 0
for repo in ['1-Ember汉化插件', '2-Crucible汉化插件']:
    d = os.path.join(ROOT, repo, 'compendium', 'en')
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.json') or fn == '_source.json': continue
        o = load(os.path.join(d, fn))
        for p, s in walk(o):
            leaves += 1
            for k, rx in GATES.items():
                counts[k] += len(rx.findall(s))
print(f'扫描英文叶数={leaves}')
for k, v in counts.items(): print(f'  {k}: {v}')

# 中文侧计数
cn_counts = {'贾胡德': 0, '刺客': 0}
cn_leaves = 0
for repo in ['1-Ember汉化插件', '2-Crucible汉化插件']:
    d = os.path.join(ROOT, repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.json'): continue
        o = load(os.path.join(d, fn))
        for p, s in walk(o):
            cn_leaves += 1
            for k in cn_counts: cn_counts[k] += s.count(k)
print(f'扫描中文叶数={cn_leaves}')
for k, v in cn_counts.items(): print(f'  CN {k}: {v}')
