#!/usr/bin/env python3
"""从旧英→新英的 diff 里，只挑**译文层面看得见**的实质改动。

上游这一波改动里大量是编辑部规范化（towards→toward、大小写、句末加句号、
`succeeds on a check`→`makes a successful check`、把 `<ul><li>` 换成
`<ul class="complex-check">`）——这些**不影响中文该怎么写**，或者只影响标记
（标记已由 drift_marker_side.py 三方计数确认中文跟的是新英文）。

真正可能把中文留在旧版的，只有三类：
  NUM   数字变了（DC / 距离 / 数量）——中文若还写旧数字就是错的
  SEG   整句/整段增删（净变化 ≥ --minlen 字符的纯文本）
  NAME  首字母大写的专名 token 变了（改名类 drift，port_orphans 的老坑）

用法同 drift_diff.py，另加 --minlen。
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

TAG = re.compile(r'<[^>]*>')
NUM = re.compile(r'\d+')
PROP = re.compile(r'\b[A-Z][a-zA-Z\']{2,}\b')
STOP = {'The', 'This', 'That', 'They', 'There', 'These', 'Those', 'If', 'Any', 'One',
        'A', 'An', 'It', 'Its', 'You', 'Your', 'When', 'While', 'Upon', 'On', 'In',
        'At', 'As', 'And', 'But', 'For', 'Some', 'Something', 'Characters', 'Character',
        'Result', 'Critical', 'Success', 'Failure', 'Once', 'Each', 'Both', 'All',
        'After', 'Before', 'Then', 'Their', 'Her', 'His', 'She', 'He', 'Who', 'What',
        'Search', 'Use', 'Cast', 'DC', 'GM', 'Advantage', 'Disadvantage'}


def load_json(path):
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


def tok(s):
    out, buf = [], []
    for ch in s:
        buf.append(ch)
        if ch.isspace():
            out.append(''.join(buf))
            buf = []
    if buf:
        out.append(''.join(buf))
    return out


def plain(s):
    return TAG.sub(' ', s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--ctx', type=int, default=7)
    ap.add_argument('--minlen', type=int, default=35)
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    items = items[a.start:a.start + a.limit]

    old_map = baseline_packs(a.baseline)
    cache = {}

    def pack_leaves(pack):
        if pack in cache:
            return cache[pack]
        o, n, c = {}, {}, {}
        if pack in old_map:
            leaves(load_json(old_map[pack]).get('entries', {}), [], o)
        pe = os.path.join(a.repo, 'compendium', 'en', pack)
        pc = os.path.join(a.repo, 'compendium', 'cn', pack)
        if os.path.exists(pe):
            leaves(load_json(pe).get('entries', {}), [], n)
        if os.path.exists(pc):
            leaves(load_json(pc).get('entries', {}), [], c)
        cache[pack] = (o, n, c)
        return cache[pack]

    for i, it in enumerate(items, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        old_en, new_en = o.get(p), n.get(p)
        print(f'\n{"="*96}\n[{i}] {p}\nEN {it["en_len_old"]} -> {it["en_len_new"]} | CN {it["cn_len"]}')
        if not old_en or not new_en:
            print('  !! 缺一侧英文')
            continue
        A, B = tok(old_en), tok(new_en)
        sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
        kept = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            ot, nt = ''.join(A[i1:i2]), ''.join(B[j1:j2])
            op, npl = plain(ot), plain(nt)
            kinds = []
            if NUM.findall(op) != NUM.findall(npl):
                kinds.append('NUM')
            po = {w for w in PROP.findall(op) if w not in STOP}
            pn = {w for w in PROP.findall(npl) if w not in STOP}
            if po != pn:
                kinds.append('NAME')
            if abs(len(op.strip()) - len(npl.strip())) >= a.minlen:
                kinds.append('SEG')
            if not kinds:
                continue
            kept += 1
            pre = plain(''.join(A[max(0, i1 - a.ctx):i1]))
            post = plain(''.join(A[i2:i2 + a.ctx]))
            print(f'  --{",".join(kinds)}-- …{pre.strip()}  ‖  {post.strip()}…')
            print(f'    OLD: {ot[:900]}')
            print(f'    NEW: {nt[:900]}')
        if not kept:
            print('  (无实质改动：全是规范化/大小写/标点/标记重构)')


if __name__ == '__main__':
    main()
