#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：i18n 命名空间外溢（只读库）

同一类问题的另一条通道：`lang/*.json` 里的键**不是本模块的私有命名空间**，
它被整棵合并进全局 `game.i18n.translations`。任何与 Foundry 本体 / crucible 本体 /
另一个汉化包同名的键，都会**静默覆盖**对方的字符串，覆盖范围是整个客户端，
而且谁也不会报错 —— 与 register.js 那条「补丁作用域外溢 + 静默丢字段」同构。

判据：
  A. 本模块 cn.json 的键（拍平到叶子）∩ Foundry 本体 public/lang/en.json 的键
     → 覆盖核心 UI 字符串
  B. ember_cn 的键 ∩ crucible 系统自己的 lang 键 → 越过 crucible-cn 改系统字符串
  C. ember_cn 的键 ∩ crucible-cn 的键，且**值不同** → 两个包抢同一个键，
     结果取决于模块加载顺序（不确定行为）
  D. 顶层键为通用词（Name/Save/Sort/Delete/…）→ 高危命名空间污染

假阳性：
  - 一个汉化模块**有意**补译本体缺失的键，是合法用法（但必须与本体值一致或更好）
  - crucible-cn 覆盖 crucible 系统自己的键正是它的职责，B 只对 ember_cn 有意义
"""
from __future__ import annotations
import json
import os
import sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
FOUNDRY = r'C:\Program Files\Foundry Virtual Tabletop\resources\app\public\lang\en.json'
DATA = r'C:\Users\Taka\AppData\Local\FoundryVTT\Data'

EMBER_CN = os.path.join(ROOT, '1-Ember汉化插件', 'lang', 'cn.json')
CRUC_CN = os.path.join(ROOT, '2-Crucible汉化插件', 'lang', 'cn.json')
CRUC_EN = os.path.join(DATA, 'systems', 'crucible', 'lang', 'en.json')
EMBER_EN = os.path.join(DATA, 'modules', 'ember', 'lang', 'en.json')


def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f'{p}.{k}' if p else k)
    else:
        yield p, o


def load(path):
    if not os.path.exists(path):
        print('MISSING', path)
        return {}
    with open(path, encoding='utf-8-sig') as f:
        return dict(flat(json.load(f)))


def main():
    core = load(FOUNDRY)
    ecn = load(EMBER_CN)
    ccn = load(CRUC_CN)
    cen = load(CRUC_EN)
    een = load(EMBER_EN)
    print(f'core={len(core)}  ember_cn={len(ecn)}  crucible_cn={len(ccn)}  '
          f'crucible_en={len(cen)}  ember_en={len(een)}')

    report = {}

    for name, mod in (('ember_cn', ecn), ('crucible_cn', ccn)):
        hit = sorted(set(mod) & set(core))
        report[f'A:{name}_overrides_core'] = [
            {'key': k, 'core_en': core[k], 'mod_cn': mod[k]} for k in hit]
        print(f'\nA. {name} 覆盖 Foundry 本体键: {len(hit)}')
        for k in hit[:40]:
            print(f'   {k:<40} core={core[k]!r:<34} mod={mod[k]!r}')

    hit = sorted(set(ecn) & set(cen))
    report['B:ember_cn_overrides_crucible_system'] = [
        {'key': k, 'crucible_en': cen[k], 'ember_cn': ecn[k],
         'crucible_cn': ccn.get(k)} for k in hit]
    print(f'\nB. ember_cn 覆盖 crucible 系统键: {len(hit)}')
    for k in hit[:60]:
        same = '同' if ccn.get(k) == ecn[k] else ('缺' if k not in ccn else '⚠异')
        print(f'   [{same}] {k:<44} ember_cn={ecn[k]!r:<24} crucible_cn={ccn.get(k)!r}')

    both = sorted(set(ecn) & set(ccn))
    diff = [k for k in both if ecn[k] != ccn[k]]
    report['C:ember_cn_vs_crucible_cn_conflict'] = [
        {'key': k, 'ember_cn': ecn[k], 'crucible_cn': ccn[k]} for k in diff]
    print(f'\nC. ember_cn 与 crucible_cn 同键: {len(both)}，其中**取值不同**: {len(diff)}')
    for k in diff[:80]:
        print(f'   {k:<44} ember_cn={ecn[k]!r:<30} crucible_cn={ccn[k]!r}')

    # D. 顶层裸通用词
    for name, mod in (('ember_cn', ecn), ('crucible_cn', ccn)):
        top = sorted({k for k in mod if '.' not in k})
        report[f'D:{name}_bare_toplevel'] = [{'key': k, 'value': mod[k]} for k in top]
        print(f'\nD. {name} 顶层裸键（无命名空间前缀）: {len(top)}')
        for k in top[:60]:
            mark = ' ⚠核心同名' if k in core else ''
            print(f'   {k!r} = {mod[k]!r}{mark}')

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i18n_overreach.json')
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('\n->', dst)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
