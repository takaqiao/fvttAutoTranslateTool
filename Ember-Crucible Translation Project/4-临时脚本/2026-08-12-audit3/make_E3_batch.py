#!/usr/bin/env python3
"""E3 单元（drift stale 桶 [102,153)）的批次生成器。

每条改动都是**在现有中文上做定点替换**，不重写整叶 —— 重写会洗掉已校对的译文。
脚本自带断言：替换次数必须与预期完全相符，否则抛错（防止库变动后静默改错地方）。
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

REPO = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件'
PACK = 'ember.crucible-adventure.json'
OUT = (r'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/'
       r'e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches/'
       r'E3__ember__ember.crucible-adventure.json')

OPPORTUNIST = ('<p>你会抓住一切机会，趁敌人背对你时加以利用。'
               '即使你原本处于<strong>完全交战</strong>状态，'
               '你仍可以进行一次@Action[default reactiveStrike]。</p>')

# path -> [(old, new, 预期次数), ...]；new 为 None 表示整叶替换成 old（此时 old 是新值）
EDITS = {
    # ① Infirmary：页 name 与同卷四页 + 场景针脚都是「医务室」，正文却写「病房」
    "Ember Early Access.journals.Traveler's Rest.pages.Infirmary.text": [
        ('病房', '医务室', 8),
    ],
    # ② 上游已删掉 Upper（Upper Arcturel 今天的英文里不存在），中文残留「上层」
    'Ember Early Access.journals.Glitter in the Dark.pages.Poolside Predicaments.text': [
        ('上层阿克图瑞尔', '阿克图瑞尔', 1),
    ],
    # ③ 场景已改名 Arcturel Upper - Tradeway -> Arcturel Tradeway，标签仍是旧名
    "Ember Early Access.journals.Gamemaster's Guide.pages.Patch 0.2.0.text": [
        ('{阿克图瑞尔上层 - 贸易道}', '{阿克图瑞尔贸易道}', 1),
    ],
    # ④⑤ Aedir 全库 496 处「艾迪尔」，只有 Wellstone 一条写「艾狄尔」
    'Ember Early Access.journals.Signal of Intent.pages.Well Enough Alone.text': [
        ('艾狄尔井石', '艾迪尔井石', 2),
    ],
    'Ember Early Access.items.Aedir Wellstone.name': [
        ('艾狄尔井石', '艾迪尔井石', 1),
    ],
    # ⑥ 同一 actor 内 charge 的两种译法（Throw Fire 是「消耗一次…的充能」）
    'Ember Early Access.actors.Kynryth.items.Burst of Speed.description': [
        ('消耗1份@UUID[Actor.yqcFzCs8eJXXNKiR.Item.MFcSuAhMSQo009S5]{内在之光}，',
         '消耗一次@UUID[Actor.yqcFzCs8eJXXNKiR.Item.MFcSuAhMSQo009S5]{内在之光}的充能，', 1),
    ],
}

# 5 条英文完全相同的 Opportunist，中文却是 5 种写法 —— 整叶统一
FULL = {f'Ember Early Access.actors.{a}.items.Opportunist.description': OPPORTUNIST
        for a in ['Bassa the Firebug', 'Jurtak Hunter', 'Jahud Assassin',
                  'Otherhood Brigand', 'Otherhood Raider']}


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


def main():
    cn = {}
    leaves(load_json(os.path.join(REPO, 'compendium', 'cn', PACK)).get('entries', {}), [], cn)
    batch = {}
    for p, ops in EDITS.items():
        if p not in cn:
            raise SystemExit(f'缺中文叶: {p}')
        v = cn[p]
        for old, new, want in ops:
            got = v.count(old)
            if got != want:
                raise SystemExit(f'{p}: 预期 {want} 次 "{old}"，实得 {got}')
            v = v.replace(old, new)
        assert v != cn[p], p
        batch[p] = v
    for p, v in FULL.items():
        if p not in cn:
            raise SystemExit(f'缺中文叶: {p}')
        batch[p] = v
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=1)
    print(f'写出 {len(batch)} 条 -> {OUT}')
    for k in batch:
        print('  ', k)


if __name__ == '__main__':
    main()
