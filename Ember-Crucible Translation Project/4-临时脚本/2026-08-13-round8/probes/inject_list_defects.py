#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""灵敏度回测：往一份**临时副本**里注入四类已知错位缺陷。

绝不碰真库 —— `--src` 读真库，`--dst` 写副本目录，两者必须不同。

注入的四类：
  R  某个 `<li>`/`<td>` 列表的中文条目整体**轮转一格**（SHIFT 应报）
  M  把一个 `<ul>` 的最后一项**挪到下一个 `<ul>`** 开头，总条数不变（SHAPE 应报，BLOCK 不该报）
  D  删掉一个中文 `<dt>`（DL_COUNT 应报；dt 不在 BLOCK 的标签表里，BLOCK 不该报）
  X  把某个 `<dt>` 的中文换成**同列表另一条**的中文（LEX_SHIFT/SHIFT 视锚点而定）

用法：
  python inject_list_defects.py --src <repo> --dst <临时目录>
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LI = re.compile(r'(<li\b[^>]*>)((?:(?!</?li[\s>]).)*)(</li>)', re.S | re.I)
TD = re.compile(r'(<td\b[^>]*>)((?:(?!</?td[\s>]).)*)(</td>)', re.S | re.I)
DT = re.compile(r'<dt\b[^>]*>(?:(?!</?dt[\s>]).)*</dt>', re.S | re.I)
UL = re.compile(r'<ul\b[^>]*>.*?</ul>', re.S | re.I)


def leaves(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f'{prefix}.{k}' if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f'{prefix}.{i}')
    elif isinstance(obj, str) and obj:
        yield prefix, obj


def setpath(obj, path, val):
    parts = path.split('.')
    cur = obj
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    if isinstance(cur, list):
        cur[int(parts[-1])] = val
    else:
        cur[parts[-1]] = val


def rotate(s, rx):
    ms = list(rx.finditer(s))
    if len(ms) < 3:
        return None
    bodies = [m.group(2) for m in ms]
    rot = bodies[1:] + bodies[:1]
    out, last = [], 0
    for m, b in zip(ms, rot):
        out.append(s[last:m.start()] + m.group(1) + b + m.group(3))
        last = m.end()
    return ''.join(out) + s[last:]


def move_li(s):
    """把第一个 <ul> 的最后一项挪进第二个 <ul> 的开头（总 li 数不变）。"""
    uls = list(UL.finditer(s))
    if len(uls) < 2:
        return None
    a, b = uls[0], uls[1]
    ai = list(LI.finditer(a.group(0)))
    bi = list(LI.finditer(b.group(0)))
    if len(ai) < 2 or not bi:
        return None
    item = a.group(0)[ai[-1].start():ai[-1].end()]
    na = a.group(0)[:ai[-1].start()] + a.group(0)[ai[-1].end():]
    nb = b.group(0)[:bi[0].start()] + item + b.group(0)[bi[0].start():]
    return s[:a.start()] + na + s[a.end():b.start()] + nb + s[b.end():]


def drop_dt(s):
    ms = list(DT.finditer(s))
    if len(ms) < 3:
        return None
    m = ms[1]
    return s[:m.start()] + s[m.end():]


def swap_dt(s):
    ms = list(DT.finditer(s))
    if len(ms) < 3:
        return None
    a, b = ms[0], ms[1]
    return s[:a.start()] + b.group(0) + s[a.end():b.start()] + a.group(0) + s[b.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--pack', action='append', required=True)
    a = ap.parse_args()
    assert os.path.abspath(a.src) != os.path.abspath(a.dst), '副本目录必须与源不同'
    assert 'compendium' not in os.path.abspath(a.dst).replace('\\', '/').split('/')[-3:-1] \
        or True

    for sub in ('en', 'cn'):
        os.makedirs(os.path.join(a.dst, 'compendium', sub), exist_ok=True)
    log = []
    for pack in a.pack:
        shutil.copy(os.path.join(a.src, 'compendium', 'en', pack),
                    os.path.join(a.dst, 'compendium', 'en', pack))
        cn_p = os.path.join(a.src, 'compendium', 'cn', pack)
        doc = json.load(open(cn_p, encoding='utf-8-sig'))
        ent = doc['entries']
        flat = dict(leaves(ent))
        used = {'R_li': 0, 'R_td': 0, 'M': 0, 'D': 0, 'X': 0}
        for path, s in flat.items():
            if used['R_li'] < 2 and len(LI.findall(s)) >= 6 and re.search(r'\d', s):
                n = rotate(s, LI)
                if n and n != s:
                    setpath(ent, path, n)
                    used['R_li'] += 1
                    log.append(('R_li', pack, path))
                    continue
            if used['R_td'] < 2 and len(TD.findall(s)) >= 6:
                n = rotate(s, TD)
                if n and n != s:
                    setpath(ent, path, n)
                    used['R_td'] += 1
                    log.append(('R_td', pack, path))
                    continue
            if used['M'] < 2 and len(UL.findall(s)) >= 2:
                n = move_li(s)
                if n and n != s:
                    setpath(ent, path, n)
                    used['M'] += 1
                    log.append(('M', pack, path))
                    continue
            if used['D'] < 2 and len(DT.findall(s)) >= 3:
                n = drop_dt(s)
                if n and n != s:
                    setpath(ent, path, n)
                    used['D'] += 1
                    log.append(('D', pack, path))
                    continue
            if used['X'] < 2 and len(DT.findall(s)) >= 3:
                n = swap_dt(s)
                if n and n != s:
                    setpath(ent, path, n)
                    used['X'] += 1
                    log.append(('X', pack, path))
                    continue
        json.dump(doc, open(os.path.join(a.dst, 'compendium', 'cn', pack), 'w',
                            encoding='utf-8'), ensure_ascii=False)
        print(pack, used)
    for k, p, path in log:
        print(f'  [{k}] {p}  {path[:100]}')


if __name__ == '__main__':
    main()
