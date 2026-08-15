#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一条判据的第三个落点：i18n 合并也是一条**写入路径**。

Foundry 把模块的 lang 文件 mergeObject 进 game.i18n.translations（递归、overwrite）。
如果本项目在某个 key 上给的**类型**与上游不同（上游是命名空间对象，我们给字符串，
或反过来），合并会把上游那一整棵子树顶掉 —— 与 register.js 那条一样，是
「类型判据缺失 + 无差别写入别人的结构」。

对照来源：
  crucible 0.10.1  systems/crucible/lang/en.json
  ember 0.6.0      modules/ember/lang/en.json
  core v14         resources/app/public/lang/en.json（若存在）

只读。
"""
from __future__ import annotations
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = r'C:\Users\Taka\AppData\Local\FoundryVTT\Data'
CORE = r'C:\Program Files\Foundry Virtual Tabletop\resources\app'

UPSTREAM = {
    'crucible': os.path.join(DATA, 'systems', 'crucible', 'lang', 'en.json'),
    'ember': os.path.join(DATA, 'modules', 'ember', 'lang', 'en.json'),
    'core': os.path.join(CORE, 'public', 'lang', 'en.json'),
}
OURS = {
    'crucible-cn': os.path.join(ROOT, '2-Crucible汉化插件', 'lang', 'cn.json'),
    'ember_cn': os.path.join(ROOT, '1-Ember汉化插件', 'lang', 'cn.json'),
}


def nodes(o, p=''):
    """产出 (path, 'obj'|'leaf')。obj 也要产出，才能发现 leaf-vs-obj 冲突。"""
    if isinstance(o, dict):
        if p:
            yield (p, 'obj')
        for k, v in o.items():
            yield from nodes(v, f'{p}.{k}' if p else k)
    else:
        yield (p, 'leaf')


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


up = {}
for name, p in UPSTREAM.items():
    d = load(p)
    print(f'上游 {name:<9} {"缺失" if d is None else str(len(list(nodes(d)))) + " 节点"}  {p}')
    if d is not None:
        for path, kind in nodes(d):
            up.setdefault(path, {})[name] = kind

total = 0
for name, p in OURS.items():
    d = load(p)
    if d is None:
        print(f'{name}: lang/cn.json 缺失')
        continue
    ours = list(nodes(d))
    leaves = sum(1 for _, k in ours if k == 'leaf')
    clash = []
    for path, kind in ours:
        for owner, ukind in up.get(path, {}).items():
            if ukind != kind:
                clash.append((path, kind, owner, ukind))
    print(f'\n{name}: {len(ours)} 节点 / {leaves} 叶  类型冲突 {len(clash)}')
    for c in clash[:40]:
        print(f'   {c[0]}  ours={c[1]}  {c[2]}={c[3]}')
    total += len(clash)
print(f'\n合计类型冲突 {total}')
