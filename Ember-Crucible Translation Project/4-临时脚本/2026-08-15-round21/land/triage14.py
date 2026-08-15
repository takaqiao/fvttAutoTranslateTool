#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三方对照 dump：旧英文(0.8.9.1) / 新英文(现役 compendium/en) / 中文(现役 compendium/cn)
只为把 README 末尾那 14 条「中文更贴合旧英文」的可疑叶摆到眼前逐条读，不产任何批次。

⚠ 反空转：本脚本必须自证「三方都真的读到了」——
  每条打印 old/new/cn 三侧的字符数，任何一侧为 0 就打 **读不到** 而不是静静跳过。
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OLD = os.path.join(P, '5-其他内容', 'english-baseline', 'crucible-cn-0.8.9.1-shipped-en')
NEW = os.path.join(P, '2-Crucible汉化插件', 'compendium', 'en')
CN = os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn')

TARGETS = [
    ('crucible.equipment.json', 'Cloak of Kindly Visage.description.private'),
    ('crucible.equipment.json', 'Common Clothing.description.public'),
    ('crucible.rules.json', 'Character Mechanics.pages.Defenses.text'),
    ('crucible.rules.json', 'Combat.pages.Engagement and Flanking.text'),
    ('crucible.rules.json', 'Conditions.pages.Broken.text'),
    ('crucible.rules.json', 'Conditions.pages.Incapacitated.text'),
    ('crucible.rules.json', 'Conditions.pages.Stunned.text'),
    ('crucible.rules.json', 'Conditions.pages.Weakened.text'),
    ('crucible.rules.json', 'Crafting.pages.Tradeskills Overview.text'),
    ('crucible.rules.json', 'Equipment.pages.Weapons.text'),
    ('crucible.rules.json', 'Spellcraft.pages.Inflections.text'),
    ('crucible.rules.json', 'Welcome To Crucible.pages.Module Recommendations.text'),
    ('crucible.rules.json', 'Welcome To Crucible.pages.Providing Feedback.text'),
    ('crucible.rules.json', 'Welcome To Crucible.pages.What is Crucible.text'),
]

TAG = re.compile(r'<[^>]+>')


def load(d, fn):
    for cand in (fn, fn.replace('.json', '-en.json')):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return json.load(open(p, encoding='utf-8')), cand
    return None, None


def dig(obj, path):
    """entries 下按 '.' 路径取值；路径段本身可能含空格但不含 '.'。"""
    cur = obj.get('entries', obj)
    for seg in path.split('.'):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur if isinstance(cur, str) else None


def plain(s):
    return TAG.sub(' ', s or '')


def main():
    read_ok = 0
    for fn, path in TARGETS:
        o, ofn = load(OLD, fn)
        n, nfn = load(NEW, fn)
        c, cfn = load(CN, fn)
        ov = dig(o, path) if o else None
        nv = dig(n, path) if n else None
        cv = dig(c, path) if c else None
        print('=' * 100)
        print(f'{fn} :: {path}')
        print(f'  [读到] old={len(ov or "")} 字 ({ofn}) · new={len(nv or "")} 字 ({nfn}) · cn={len(cv or "")} 字 ({cfn})')
        if not (ov and nv and cv):
            print('  ** 三方有一侧读不到 —— 本条无从判 **')
            continue
        read_ok += 1
        print(f'  old==new ? {ov == nv}')
        print('--- OLD EN ---')
        print(plain(ov))
        print('--- NEW EN ---')
        print(plain(nv))
        print('--- CN ---')
        print(plain(cv))
    print('=' * 100)
    print(f'三方齐备 {read_ok} / {len(TARGETS)} 条')


if __name__ == '__main__':
    main()
