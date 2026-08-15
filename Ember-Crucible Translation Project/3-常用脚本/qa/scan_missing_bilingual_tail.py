#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**应带双语尾巴的 `name` 却是裸中文** —— 2026-08-14 第十四轮新增。

这一维度此前全库无判据负责
--------------------------
`scan_same_en_split.py` 的归一注释把它交接给 `scan_bare_english_names` /
`scan_token_name`，实查两个接盘方都看不见：

* `scan_bare_english_names.py` 主扫描循环第一行就是
  `if path.endswith('.name') or not CJK.search(v): continue`，整类 `.name` 被排除
  （该文件里另一处 `.name` 用法是**反向**的：从 EN 侧建「英文专名→中文译名」词典）；
* `scan_token_name.py` 管的是 tokenName 该不该**剥掉**尾巴，不是 name 该不该**带**。

漏网实例（本脚本的第一条判据来源）：`2-Crucible汉化插件/compendium/cn/crucible.rules.json`
的 `entries.Combat.pages` 共 12 页，11 页是「先攻与回合顺序 Initiative and Turn Order」
这种双语，唯 `Engagement and Flanking` 的 name = 「交战与夹击」，没有英文尾巴。

判据
----
只看 `.name` 叶，且按**结构路径**分桶（复用 `tm/fill_twin.py::shape_of`，
末段键不够用：`tables.…results.<k>.name` 与 `journals.pages.<k>.name` 的角色不同）：

* 应双语桶：entries 顶层 / journals / pages / items / actions / effects /
  actors / scenes / tables / macros 的 `name`；
* 应裸中文桶（本脚本不管）：`results` / `regions` / `levels` / `categories` /
  `tokens` / `behaviors` 底下的 `name`，以及 `tokenName` / `adjective` / `label`。

对「应双语」桶报出：
    len(en) <= 60  且 en 含拉丁字母  且 cn 含汉字  且 en 不作为子串出现在 cn 里

内建的假阳性排除（每条都是实测教训，不排除就淹没在噪声里）
---------------------------------------------------------
1. **英文不是专名**：`+1 to AC` / `2d6 Fire` 这类 effects.name 是数值描述，
   没有「译名 + 英文原名」的写法。凡英文首字符不是大写字母或 `(`、或含 `+` `%`
   等运算符号的，一律不报。
2. **英文自带句点或连续两个空格**：`Burning.` / `Altar of  Aura` —— 上游原文就是
   那样，中文侧接不接尾巴都对不齐，属于 `scan_en_drift` 的辖区。
3. **英文里含产品名**：`What is Crucible` / `Ember Early Access` —— 这些名字里
   本来就有英文，接尾巴会变成「什么是 Crucible What is Crucible」。
4. **中文里已经有拉丁串**：说明尾巴以别的形态（倒装、括号、`@UUID{}` 标签）在了，
   接不接由人判，不进本报告。

用法：
  python scan_missing_bilingual_tail.py --repo <pluginRepo> [--repo <另一个>] \
         [--out <json>] [--limit N]
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
import sys

CJK = re.compile(r'[一-鿿]')
LATIN = re.compile(r'[A-Za-z]')
MAX_EN_LEN = 60

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    '_fill_twin', os.path.join(_HERE, '..', 'tm', 'fill_twin.py'))
_fill_twin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fill_twin)
shape_of = _fill_twin.shape_of

# 结构段一旦出现在路径里，其下的 name 就是裸中文约定，本脚本不管。
BARE_PARENTS = {'results', 'regions', 'levels', 'categories', 'tokens', 'behaviors'}
# 产品名 / 上游本来就是英文的名字。
PRODUCT_WORDS = ('Crucible', 'Ember', 'Foundry', 'dnd5e', 'D&D', 'Babele')
BAD_EN_CHARS = re.compile(r'[+%*=<>|]')


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def wants_bilingual(path: str) -> bool:
    return not (set(shape_of(path).split('.')[:-1]) & BARE_PARENTS)


def name_like(en: str) -> bool:
    """英文看起来像个专名（而不是 `+1 to AC` 这类数值描述）。"""
    en = en.strip()
    if not en or not LATIN.search(en):
        return False
    if BAD_EN_CHARS.search(en):
        return False
    if not (en[0].isupper() or en[0] == '('):
        return False
    return True


def excluded(en: str) -> str | None:
    """返回排除理由；None 表示不排除。"""
    if len(en) > MAX_EN_LEN:
        return 'en too long'
    if not name_like(en):
        return 'en not name-like'
    if '.' in en or '  ' in en:
        return 'en has period / double space (upstream drift)'
    if any(w in en for w in PRODUCT_WORDS):
        return 'en contains a product name'
    return None


def scan_repo(repo):
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    en_dir = os.path.join(repo, 'compendium', 'en')
    hits, skipped = [], {}
    if not os.path.isdir(cn_dir):
        return hits, skipped
    for pack in sorted(os.listdir(cn_dir)):
        if not pack.endswith('.json'):
            continue
        en_p = os.path.join(en_dir, pack)
        if not os.path.exists(en_p):
            continue
        rows = []
        walk(load(en_p).get('entries', {}), load(os.path.join(cn_dir, pack)).get('entries', {}),
             [], rows)
        for path, en, cn in rows:
            if not path.endswith('.name') or not cn:
                continue
            if not wants_bilingual(path):
                continue
            if not CJK.search(cn):
                continue
            if LATIN.search(cn):
                continue          # 尾巴以某种形态已经在了
            why = excluded(en.strip())
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            hits.append({'repo': os.path.basename(os.path.normpath(repo)),
                         'pack': pack, 'path': path, 'shape': shape_of(path),
                         'en': en.strip(), 'cn': cn.strip()})
    return hits, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--limit', type=int, default=40)
    a = ap.parse_args()

    all_hits, all_skipped = [], {}
    for repo in a.repo:
        hits, skipped = scan_repo(repo)
        all_hits.extend(hits)
        for k, v in skipped.items():
            all_skipped[k] = all_skipped.get(k, 0) + v
        print(f'{os.path.basename(os.path.normpath(repo)):<24} 缺双语尾巴 {len(hits)}')

    by_shape = {}
    for h in all_hits:
        by_shape.setdefault(h['shape'], 0)
        by_shape[h['shape']] += 1
    print(f'\n合计 {len(all_hits)} 条')
    for s, n in sorted(by_shape.items(), key=lambda kv: -kv[1]):
        print(f'  {s:<40}{n:>6}')
    print('\n内建排除计数（不是缺陷，只是本判据不管）：')
    for k, v in sorted(all_skipped.items(), key=lambda kv: -kv[1]):
        print(f'  {k:<48}{v:>6}')
    print()
    for h in all_hits[:a.limit]:
        print(f'  {h["pack"]}::{h["path"]}\n     EN {h["en"]}\n     CN {h["cn"]}')
    if len(all_hits) > a.limit:
        print(f'  ... 还有 {len(all_hits) - a.limit} 条')

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'_meta': {'count': len(all_hits), 'byShape': by_shape,
                                 'excluded': all_skipped},
                       'hits': all_hits}, f, ensure_ascii=False, indent=2)
        print(f'\n-> {a.out}')
    raise SystemExit(1 if all_hits else 0)


if __name__ == '__main__':
    main()
