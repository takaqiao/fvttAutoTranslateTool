#!/usr/bin/env python3
"""落盘前核查：批次的整叶新值 vs 库里当前值，逐块对齐报出每一处差异。

反空转：打印扫了多少 key / 多少块。
"""
import json, os, re, sys, difflib

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPO = os.path.join(ROOT, '1-Ember汉化插件')
BATCHDIR = os.path.join(ROOT, '4-临时脚本', '2026-08-15-round18', 'batches')

BATCHES = {
    'ember.adventure.json': 'r18-drop.ember.adventure.json',
    'ember.crucible-adventure.json': 'r18-drop.ember.crucible-adventure.json',
}
if len(sys.argv) > 1 and sys.argv[1] == '--escalations':
    BATCHES = {
        'ember.adventure.json': 'r18-escalations.ember.adventure.json',
        'ember.crucible-adventure.json': 'r18-escalations.ember.crucible-adventure.json',
    }

TAGSPLIT = re.compile(r'(?=<)')


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def split_path(root, path):
    naive = path.split('.')
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + '.')]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition('.')
        parts.append(head)
        node = get_at(node, [head])
    return parts


def blocks(s):
    """按 HTML 标签切块（round16 probes/split_dives.py 的口径）。"""
    return [b for b in TAGSPLIT.split(s) if b]


out = []
n_keys = 0
n_blocks_cmp = 0
n_blocks_diff = 0
n_shape_mismatch = 0
n_char_repl = 0

for pack, bfn in BATCHES.items():
    batch = load(os.path.join(BATCHDIR, bfn))
    items = batch.get('items', batch)
    en = load(os.path.join(REPO, 'compendium', 'en', pack))
    cn = load(os.path.join(REPO, 'compendium', 'cn', pack))
    out.append('=' * 70)
    out.append(f'PACK {pack}  batch={bfn}  keys={len(items)}')
    for path, newv in items.items():
        n_keys += 1
        parts = path.split('.')
        root_en = en.get('entries', {})
        root_cn = cn.get('entries', {})
        if parts[0] == '(folders)':
            root_en, root_cn = en.get('folders', {}), cn.get('folders', {})
            parts = parts[1:]
        parts = split_path(root_en, '.'.join(parts))
        src = get_at(root_en, parts)
        old = get_at(root_cn, parts)
        out.append('-' * 70)
        out.append(f'KEY {path}')
        out.append(f'  EN present: {isinstance(src, str)}  CN present: {isinstance(old, str)}')
        if not isinstance(old, str):
            out.append('  !! 库里没有旧中文，整叶新增')
            continue
        if old == newv:
            out.append('  == 新值与库里完全一致（空批次项）')
            continue
        bo, bn = blocks(old), blocks(newv)
        out.append(f'  blocks old={len(bo)} new={len(bn)}')
        if len(bo) != len(bn):
            n_shape_mismatch += 1
            out.append('  !! 块数不同，无法逐块对齐 —— 需人工核')
            for i, l in enumerate(difflib.unified_diff(bo, bn, lineterm='', n=0)):
                out.append('    ' + l)
                if i > 60:
                    out.append('    ...')
                    break
            continue
        for i, (a, b) in enumerate(zip(bo, bn)):
            n_blocks_cmp += 1
            if a == b:
                continue
            n_blocks_diff += 1
            out.append(f'  [块 {i}] 改动:')
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    continue
                n_char_repl += 1
                out.append(f'     {tag}: {a[i1:i2]!r} -> {b[j1:j2]!r}')
            out.append(f'     OLD: {a}')
            out.append(f'     NEW: {b}')

out.append('=' * 70)
out.append(f'扫描统计: keys={n_keys}  逐块比较块数={n_blocks_cmp}  有差异块={n_blocks_diff}  '
           f'字符级改动段={n_char_repl}  块数不匹配叶={n_shape_mismatch}')

sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(out))
