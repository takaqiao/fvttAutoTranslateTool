#!/usr/bin/env python3
"""Compare the block-tag SKELETON of en.html vs cn.html, position by position.

  python tagseq.py <单元目录> [<页 id>] [--stat]

Why this exists (and why the gate cannot replace it)
----------------------------------------------------
`apply_translations.py` compares markup as an **order-insensitive multiset**.
That is the right call for the gate -- it must not care whether a translator
moved a `<strong>` inside a sentence. But it means a补 block inserted in the
WRONG PLACE passes cleanly: the counts are identical, only the order is wrong.
The player then reads the paragraphs of a room in scrambled order.

This walks the two skeletons with difflib and reports the first position where
they diverge, so a misplaced block shows up as a `replace`/`insert` opcode.

Origin: written by the realign-8 review agent during batch 6, which reported it
as "本批唯一能抓出『补的块位置错了』的手段——闸门看不出来". Promoted from that
agent's scratchpad into the permanent toolset, with a --stat mode added so the
main controller can sweep all 12 units at once before landing.

A `MISMATCH` is not automatically a defect: EN and CN legitimately differ where
the Chinese renders a `<sup class="system-swap-inline">` pair differently, and
some pages carry upstream's own broken tags. Read the opcode, then judge.
"""
from __future__ import annotations
import difflib
import json
import os
import re
import sys

BLOCK_OPEN = re.compile(
    r'(?=<(?:p|li|h[1-6]|section|blockquote|ul|ol|div|figure|figcaption|table|tr)\b)', re.I)


def blocks(s):
    return [b for b in BLOCK_OPEN.split(s) if b.strip()]


def sig(b):
    """tag + class + trailing closing-tag sequence -- structure without prose."""
    m = re.match(r'<(\w+)([^>]*)>', b)
    if not m:
        return '?'
    tag = m.group(1).lower()
    cls = re.search(r'class="([^"]*)"', m.group(2))
    out = tag + ('.' + cls.group(1).replace(' ', '.') if cls else '')
    tail = re.findall(r'</(\w+)>', b)
    return out + '|' + ','.join(tail[-4:])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    stat = '--stat' in sys.argv
    if not args:
        raise SystemExit('usage: tagseq.py <单元目录> [<页 id>] [--stat]')
    unit = args[0]
    only = args[1] if len(args) > 1 else None

    index = json.load(open(os.path.join(unit, 'index.json'), encoding='utf-8'))
    bad = 0
    for page in index['pages']:
        i = str(page['id'])
        if only is not None and i != str(only):
            continue
        p_en = os.path.join(unit, 'pages', f'{i}.en.html')
        p_cn = os.path.join(unit, 'pages', f'{i}.cn.html')
        if not (os.path.exists(p_en) and os.path.exists(p_cn)):
            continue
        en = open(p_en, encoding='utf-8-sig').read()
        cn = open(p_cn, encoding='utf-8-sig').read()
        a, b = blocks(en), blocks(cn)
        sa, sb = [sig(x) for x in a], [sig(x) for x in b]
        ops = [o for o in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes()
               if o[0] != 'equal']
        if ops:
            bad += 1
        if stat and not ops:
            continue
        status = 'OK' if not ops else f'MISMATCH x{len(ops)}'
        tail = '.'.join(page['path'].split('.')[-3:-1])
        print(f'--- [{i}] en={len(a)} cn={len(b)} {status}  {tail}')
        if stat:
            continue
        for tag, i1, i2, j1, j2 in ops:
            print(f'   {tag} en[{i1}:{i2}] cn[{j1}:{j2}]')
            for k in range(i1, i2):
                print(f'     EN{k} {sa[k]}  :: {re.sub(chr(10), " ", a[k])[:120]}')
            for k in range(j1, j2):
                print(f'     CN{k} {sb[k]}  :: {re.sub(chr(10), " ", b[k])[:120]}')

    print(f'\n{index["unit"]}: {bad} 页骨架对不上 / 共 {len(index["pages"])} 页')


main()
