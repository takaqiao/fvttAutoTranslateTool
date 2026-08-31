#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""库真实状态计数（落盘前后各跑一次；本轮是恒等落盘，两次必须相同）。
反空转：先打印扫了几个包、几个文件字节数，再报叶数。"""
import json, os, re, sys, hashlib

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CJK = re.compile('[一-鿿]')


def leaves(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from leaves(v)
    elif isinstance(node, list):
        for v in node:
            yield from leaves(v)


for repo in ('1-Ember汉化插件', '2-Crucible汉化插件'):
    cndir = os.path.join(P, repo, 'compendium', 'cn')
    packs = sorted(f for f in os.listdir(cndir) if f.endswith('.json'))
    tot = cjk = 0
    for f in packs:
        d = json.load(open(os.path.join(cndir, f), encoding='utf-8'))
        for s in leaves(d.get('entries', {})):
            tot += 1
            if CJK.search(s):
                cjk += 1
    lang = os.path.join(P, repo, 'lang', 'cn.json')
    lk = len(json.load(open(lang, encoding='utf-8')))
    print(f'{repo}: cn 包 {len(packs)} 个 · entries 字符串叶 {tot} · 其中含中文 {cjk} · lang/cn.json 顶层键 {lk}')

for pack in ('crucible.rules.json', 'crucible.equipment.json'):
    p = os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn', pack)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()
    print(f'  {pack}: {os.path.getsize(p)} bytes · md5 {h}')
