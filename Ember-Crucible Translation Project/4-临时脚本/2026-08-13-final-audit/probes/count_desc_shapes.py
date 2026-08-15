#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 en 抽取基线里 **Item 文档级 system.description** 的形状：str vs {public,private}。

判据：叶路径形如  ....(entries|items).<文档名>.description
  - path[-3] in {'entries','items'}  → 这条 description 就是某个 Item 的 system.description
  - 排除 path[-3] 为条目名的情形（如 entries.<Actor>.ancestry.description，
    那是 system.details.ancestry.description，走 crucibleNested，不是 Item）
  - 排除 actions.<id>.description / effects.<n>.description（另一个字段）
只读。假阳性模式：
  1) `entries.<X>.description` 里 X 也可能是 ActiveEffect（affixes/effects 包），
     ActiveEffect.description 是核心文档字段而非 system.description —— 已单列。
  2) 抽取只导出「有可译文本」的字段，空描述不出现，所以这是**下界**。
"""
import json, os, collections

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']
# 这些包的顶层条目是 ActiveEffect，不是 Item
AE_PACKS = {'crucible.affixes.json', 'ember.crucible-affixes.json',
            'ember.crucible-effects.json', 'ember.dnd5e-effects.json'}

def scan(doc):
    s, o = [], []
    def rec(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                p = path + [k]
                if k == 'description' and len(p) >= 3 and p[-3] in ('entries', 'items'):
                    (s if isinstance(v, str) else o if isinstance(v, dict) else []).append('.'.join(p))
                rec(v, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                rec(v, path + [str(i)])
    rec(doc, [])
    return s, o

grand = collections.Counter()
for repo in REPOS:
    d = os.path.join(ROOT, repo, 'compendium', 'en')
    print(f'--- {repo}')
    tot = collections.Counter()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.json') or fn == '_source.json':
            continue
        doc = json.load(open(os.path.join(d, fn), encoding='utf-8-sig'))
        s, o = scan(doc)
        if not (s or o):
            continue
        tag = ' (ActiveEffect 包，不计)' if fn in AE_PACKS else ''
        if not tag:
            tot['STRING'] += len(s); tot['OBJECT'] += len(o)
        print(f'  {fn:<40} str {len(s):>5}  obj {len(o):>5}{tag}   e.g. {(s[:1] or [""])[0][:70]}')
    print(f'  === Item 级 STRING {tot["STRING"]}   OBJECT {tot["OBJECT"]}')
    grand += tot
print(f'\n全库 Item 级 system.description： STRING {grand["STRING"]}  OBJECT {grand["OBJECT"]}')
