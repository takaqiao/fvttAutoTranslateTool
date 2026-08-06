#!/usr/bin/env python3
"""How much untranslated content hides inside pages that already count as done?

`validate_translations.py` scores a leaf as covered the moment it holds any
Chinese, so a page translated against an older English shows 100% while missing
whole blocks. Stage 13 found the campaign pack short 2207 <li> and 7458 <p>.

This puts a character number on it: for every TRANSLATED leaf, compare the
English block count with the Chinese one and, where the Chinese is short,
attribute the missing blocks' share of the English text length.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
BLOCK = re.compile(r'<(p|li)\b[^>]*>')
EMPTY_P = re.compile(r'<p>\s*</p>')


def textlen(s):
    return len(TAG.sub(' ', s))


def blocks(s):
    return Counter(m.group(1).lower() for m in BLOCK.finditer(EMPTY_P.sub('', s)))


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        if not (isinstance(cn, str) and CJK.search(cn)):
            return                                    # 未译，属常规待译量，不在本项内
        be, bc = blocks(en), blocks(cn)
        short = {k: be[k] - bc.get(k, 0) for k in be if be[k] > bc.get(k, 0)}
        if not short:
            return
        missing = sum(short.values())
        total = sum(be.values()) or 1
        out.append({'path': '.'.join(path), 'missing_blocks': missing,
                    'en_blocks': total, 'short': short,
                    'est_chars': int(textlen(en) * missing / total)})


for pack in ("ember.crucible-adventure.json",):
    en = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", pack), encoding="utf-8"))
    out = []
    walk(en.get("entries", {}), cn.get("entries", {}), [], out)
    out.sort(key=lambda r: -r['est_chars'])
    tot_blocks = sum(r['missing_blocks'] for r in out)
    tot_chars = sum(r['est_chars'] for r in out)
    print(f"== {pack}")
    print(f"   受影响条目 {len(out)}，缺失区块 {tot_blocks}，估算字符 {tot_chars}\n")
    print(f"   {'est chars':>10}{'missing':>9}/{'en':<5} path")
    for r in out[:25]:
        print(f"   {r['est_chars']:>10}{r['missing_blocks']:>9}/{r['en_blocks']:<5} {r['path'][:88]}")
    dump = os.path.join(P, "5-其他内容", "reports", "ember", "missing_blocks.json")
    json.dump({'_meta': {'entries': len(out), 'missing_blocks': tot_blocks, 'est_chars': tot_chars},
               'items': out}, open(dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n   -> {dump}")
