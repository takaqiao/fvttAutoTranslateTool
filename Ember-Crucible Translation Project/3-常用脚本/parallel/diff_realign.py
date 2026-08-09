#!/usr/bin/env python3
"""Show, block by block, exactly what a realign unit changed.

  python diff_realign.py <单元目录> [<页 id>] [--stat]

The reviewer's hardest question is not "is the new Chinese good" -- it is
**"did the translator quietly reword the Chinese that was already vetted?"**
Reading two 16 K-char pages side by side cannot answer that reliably; a diff
against the pristine original can, and it answers it in seconds.

`_original_cn.json` holds the byte-exact Chinese as it stood before the unit was
touched, so this is a true before/after.

Read the output as:
  +  a block the translator ADDED   (should correspond to a missing 8c block)
  -  a block the translator REMOVED (should correspond to a surplus 8j block)
  ~  a block the translator MODIFIED

**`~` lines are the ones to scrutinise.** A legitimate `~` means the English for
that block changed (a marker, an id, a number). A `~` that only reshuffles
wording is the rewrite this batch is designed to prevent -- revert it.
"""
from __future__ import annotations
import difflib
import json
import os
import re
import sys

BLOCK_OPEN = re.compile(
    r'(?=<(?:p|li|h[1-6]|section|blockquote|ul|ol|div|figure|figcaption|table|tr)\b)',
    re.I)


def blocks(s):
    return [b for b in BLOCK_OPEN.split(s) if b.strip()]


def squash(s, n=110):
    s = re.sub(r'\s+', ' ', s).strip()
    return s if len(s) <= n else s[:n - 1] + '…'


def expected(page):
    """(应补, 应删)，兼容两套 index.json 键名。

    `prep_realign.py` 写 `missing_blocks`/`extra_blocks`（块数），
    `prep_sigfix.py` 写 `missing`/`surplus`（标记数）。硬编码前一套会让本脚本
    在 sigfix 单元上直接 KeyError —— crucible 那批的审校 agent 就是被这个挡住、
    自己在 scratchpad 打了补丁才跑通的。
    """
    return (page.get('missing_blocks', page.get('missing', 0)),
            page.get('extra_blocks', page.get('surplus', 0)))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    stat = '--stat' in sys.argv
    if not args:
        raise SystemExit('usage: diff_realign.py <单元目录> [<页 id>] [--stat]')
    unit = args[0]
    only = args[1] if len(args) > 1 else None

    index = json.load(open(os.path.join(unit, 'index.json'), encoding='utf-8'))
    original = json.load(open(os.path.join(unit, '_original_cn.json'), encoding='utf-8'))

    tot_add = tot_del = tot_mod = 0
    for page in index['pages']:
        i = str(page['id'])
        if only is not None and i != str(only):
            continue
        f = os.path.join(unit, 'pages', f'{i}.cn.html')
        if not os.path.exists(f):
            print(f'[{i}] 页面文件不见了')
            continue
        with open(f, encoding='utf-8-sig', newline='') as fh:
            new = fh.read()
        if new.endswith('\n'):
            new = new[:-1]
        old = original.get(i, '')
        a, b = blocks(old), blocks(new)
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        add = dele = mod = 0
        lines = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            if tag == 'replace':
                # pair them up so a marker-only change reads as one ~ line
                for k in range(max(i2 - i1, j2 - j1)):
                    o = a[i1 + k] if i1 + k < i2 else None
                    n = b[j1 + k] if j1 + k < j2 else None
                    if o is not None and n is not None:
                        mod += 1
                        lines.append(f'    ~ 旧 {squash(o)}')
                        lines.append(f'      新 {squash(n)}')
                    elif n is not None:
                        add += 1
                        lines.append(f'    + {squash(n)}')
                    else:
                        dele += 1
                        lines.append(f'    - {squash(o)}')
            elif tag == 'insert':
                for n in b[j1:j2]:
                    add += 1
                    lines.append(f'    + {squash(n)}')
            elif tag == 'delete':
                for o in a[i1:i2]:
                    dele += 1
                    lines.append(f'    - {squash(o)}')
        tot_add += add
        tot_del += dele
        tot_mod += mod
        exp_add, exp_del = expected(page)
        flag = ''
        if not (add or dele or mod):
            flag = '   ← 未动过'
        elif mod and not exp_del and not exp_add:
            flag = '   ← 只有修改，可疑'
        head = (f'[{i}] +{add} -{dele} ~{mod}  '
                f'(应补 {exp_add} / 应删 {exp_del})'
                f'  {page["path"].split(".journals.")[-1][:60]}{flag}')
        print(head)
        if not stat:
            print('\n'.join(lines))

    print(f'\n合计 +{tot_add} -{tot_del} ~{tot_mod}')
    print('~ 是要重点看的：英文没变却改了措辞 = 把校对过的译文洗掉了，应还原')


main()
