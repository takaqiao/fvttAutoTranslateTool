#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：多态字段被单态代码强行改类型（只读库）

同一类问题的第三条通道：项目里对 `system.description` 有**两套互相矛盾的认识**。
  - babele 侧（runtime-converters.js / mappings.mjs:40-49）明确写着它是 POLYMORPHIC：
    talent/ancestry/archetype/background/taxonomy/spell 是**纯字符串**，
    equipment 与 actor 才是 `{public, private}`；转换器专门做了「源是什么形状就还什么形状」。
  - register.js 侧（normalizeDescriptionValue / migrateLegacyDescriptionShape /
    sanitizeItemDataShape）把「字符串」一律当成**旧版脏数据**，无条件改成 `{public, private}`。

本探针用**本库自己的译文文件**统计每个包里 description 的实际形状，
给出「会被 register.js 改坏」的叶子数量下界。

判据：cn 包里 `*.description` 为 str（而不是 dict）的条目，即 crucible 侧
schema 为 HTMLField（字符串）的条目；register.js 会把它们写成 {public, private}，
Foundry 的 StringField._cast 再 String(obj) 成字面量 "[object Object]"。

两个口径：
  LOOSE  所有 `*.description` 叶子（含 `actions.*.description` /
         `effects.*.description` / actor 的 `details.*.description`）——
         **过宽**，register.js 只碰 Item 根上的 `system.description`。
  STRICT 只数 register.js 真正会碰的：Item 包的顶层条目 description
         ＋ 任意层级的 `items.<name>.description`（＝ actor 内嵌 item）。
         报结论用 STRICT。

假阳性：
  - cn 文件是译文快照，不是 live 文档；但形状由抽取器按源文档形状产生
    （crucibleDescription.extract：源是 str 就出 str，源是 obj 就出 {public,private}），
    所以形状与 live 数据一致。
  - STRICT 口径把 `crucible.affixes` 排除了：那是 ActiveEffect 包，
    `description` 是文档顶层字段不是 `system.description`，register.js 不碰。
  - 只统计 compendium 内容；世界里玩家自建的同类型 item 同样受影响，不在计数内。
"""
from __future__ import annotations
import json
import os
import sys
import collections

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']


def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'description':
                out.append((path + '.description', type(v).__name__))
            walk(v, f'{path}.{k}', out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f'{path}.{i}', out)


ITEM_PACKS = {
    'crucible.talent.json', 'crucible.spell.json', 'crucible.ancestry.json',
    'crucible.archetype.json', 'crucible.background.json', 'crucible.taxonomy.json',
    'crucible.adversary-talents.json', 'crucible.equipment.json',
    'crucible.adversary-equipment.json', 'ember.crucible-items.json',
    'ember.dnd5e-items.json',
}


def strict_hits(fn, out):
    """register.js 真正会碰的那一层：Item.system.description。"""
    c = collections.Counter()
    for p, t in out:
        seg = p.split('.')
        top = (len(seg) == 3 and fn in ITEM_PACKS)          # ['', <entry>, 'description']
        emb = '.items.' in p and p.count('.', p.index('.items.') + 7) == 1
        if top or emb:
            c[t] += 1
    return c


def main():
    grand = collections.Counter()
    strict = collections.Counter()
    strict_pack = {}
    per_pack = {}
    samples = collections.defaultdict(list)
    for repo in REPOS:
        d = os.path.join(ROOT, repo, 'compendium', 'cn')
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(d, fn), encoding='utf-8-sig') as f:
                doc = json.load(f)
            out = []
            walk(doc.get('entries', doc), '', out)
            sc = strict_hits(fn, out)
            if sc:
                strict_pack[f'{repo}/{fn}'] = dict(sc)
                strict.update(sc)
            c = collections.Counter(t for _, t in out)
            if c:
                per_pack[f'{repo}/{fn}'] = dict(c)
                grand.update(c)
                for p, t in out:
                    if t == 'str' and len(samples[f'{repo}/{fn}']) < 3:
                        samples[f'{repo}/{fn}'].append(p)

    print('=== 每包 description 叶子形状 ===')
    for k, v in per_pack.items():
        s = v.get('str', 0)
        o = v.get('dict', 0)
        flag = '  <-- 字符串形态，会被 register.js 改成对象' if s else ''
        print(f'  {k:<62} str={s:<6} dict={o:<6}{flag}')
        for p in samples[k][:2]:
            print(f'        e.g. {p}')

    print(f'\nLOOSE 合计（过宽，仅供对照）: {dict(grand)}')

    print('\n=== STRICT：Item.system.description（结论口径）===')
    for k, v in strict_pack.items():
        print(f'  {k:<58} {v}')
    print(f'\nSTRICT 合计: {dict(strict)}')
    print(f'其中字符串形态 {strict["str"]} 处 —— crucible schema 为 HTMLField(纯字符串)；'
          f'register.js 的 normalizeDescriptionValue 会把它们写成 {{public,private}}，'
          f'Foundry StringField._cast (common/data/fields.mjs:1705) 再 String(obj) '
          f'变成字面量 "[object Object]"。对照 {strict["dict"]} 处 {{public,private}}，'
          f'全部是装备类（CruciblePhysicalItem，schema 确为 SchemaField）。')

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'polymorphic_description_shapes.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump({'loose_per_pack': per_pack, 'loose_total': dict(grand),
                   'strict_per_pack': strict_pack, 'strict_total': dict(strict)},
                  f, ensure_ascii=False, indent=1)
    print('->', dst)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
