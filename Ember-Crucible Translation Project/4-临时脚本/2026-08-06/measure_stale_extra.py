#!/usr/bin/env python3
"""The mirror of item 8c: translations that carry content the English no longer has.

Upstream shortens or rewrites a page; the Chinese keeps the old paragraphs. The
player then reads rules or scenery that were deliberately deleted. Coverage
cannot see it (there is Chinese there), and `measure_8c.py` cannot either --
it only looks for the Chinese being SHORT.

Attributes characters to the surplus blocks, same method as measure_8c.py.
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
            return
        be, bc = blocks(en), blocks(cn)
        over = {k: bc[k] - be.get(k, 0) for k in bc if bc[k] > be.get(k, 0)}
        if not over:
            return
        extra = sum(over.values())
        total = sum(bc.values()) or 1
        out.append({'path': '.'.join(path), 'extra_blocks': extra, 'cn_blocks': total,
                    'over': over, 'est_chars': int(textlen(cn) * extra / total)})


pack = "ember.crucible-adventure.json"
en = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", pack), encoding="utf-8"))
cn = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", pack), encoding="utf-8"))
out = []
walk(en.get("entries", {}), cn.get("entries", {}), [], out)
out.sort(key=lambda r: -r['est_chars'])
print(f"受影响条目 {len(out)}，多出区块 {sum(r['extra_blocks'] for r in out)}，"
      f"估算多出字符 {sum(r['est_chars'] for r in out)}\n")
print(f"{'est chars':>10}{'extra':>7}/{'cn':<5} path")
for r in out[:20]:
    print(f"{r['est_chars']:>10}{r['extra_blocks']:>7}/{r['cn_blocks']:<5} {r['path'][:86]}")
dump = os.path.join(P, "5-其他内容", "reports", "ember", "stale_extra_blocks.json")
json.dump({'_meta': {'entries': len(out),
                     'extra_blocks': sum(r['extra_blocks'] for r in out),
                     'est_chars': sum(r['est_chars'] for r in out)},
           'items': out}, open(dump, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"\n-> {dump}")
