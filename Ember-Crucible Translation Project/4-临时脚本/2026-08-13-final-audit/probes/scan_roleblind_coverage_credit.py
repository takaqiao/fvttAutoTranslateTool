# -*- coding: utf-8 -*-
r"""
scan_roleblind_coverage_credit.py
—— 「对异质集合施加同一处判断、不按成员做类型/字段判据」这一类，落在**工作量核销**上。

被查对象：`qa/resolve_generic_fallback.py`
  它把「还没译的叶子」判成「babele 会自动从别的包按名字取到，无需翻译」，
  判据只有两条（:28 / :36-53 / :88-93）：
     EMBEDDED = r'\.(items|effects|actors)\.([^.]+)'      ← 取路径里**第一个**内嵌段的名字
     resolvable = 全部 cn 包里**所有** entries 的键（只要该条目自己的 name 是中文）
  两条都**没有类型维度**，也**没有字段维度**：
     · resolvable 把 Item / ActiveEffect / Actor / JournalEntry / Macro 的条目名
       混进同一个集合，而 babele 的 generic 回落**先按 documentType 过滤**
       （document-converter.js:_genericTranslationSource -> runtime.translatedPackFor(documentType, data)）；
     · 判的是「这个**文档**有同名译文」，核销的却是「这一条**字段**不用翻」——
       同名条目里未必有这个字段（例如 actions.<id>.description 的 id 不同）。

上一次运行的结论落在 5-其他内容/reports/*/todo/_residual_after_fallback.json：
  crucible  todo 161 / auto 161 / residual 0
  ember     todo 436 / auto 436 / residual 0
也就是说 597 条未译叶子被判成「不用翻」。

本探针复刻三层，逐层收窄，只读：
  L0 todo         —— 逐字复刻 validate_translations.py:walk 的 todo 定义
  L1 工具判据      —— 复刻 resolve_generic_fallback 的 EMBEDDED + 扁平 name 集合
  L2 忠实判据      —— 加上 babele 真正会用的两个维度：
                      (a) 候选包的 documentType 必须与内嵌字段对应
                      (b) 候选条目里必须**真的有这个相对字段路径**且是中文
  差额 L1-L2 就是被错误核销掉的翻译工作量。

假阳性模式
----------
1. babele 还有 exact-source（_stats.compendiumSource）与 owner-package 两档回落，
   本探针看不到 compendiumSource（英文基线里不含该字段），所以 L2 只覆盖 generic 一档；
   若某条叶子其实走 exact-source 命中，L2 会低估覆盖 —— 但 exact-source 命中的前提
   同样是「候选条目里有这个字段」，(b) 这条判据对两档都成立，只有 (a) 可能过严。
   因此脚本把「只因 documentType 不符而落选」与「因为字段根本不存在而落选」分开计数，
   后者对任何一档回落都成立，是**下界**。
2. _sourceAwareData 会用源文档的原名替换内嵌名，本探针无法模拟（假阴性方向）。
3. todo 的定义随库变化；脚本会把自己算出的 todo 总数与报告里的 161/436 一起打印，
   对不上说明库已变，结论按当前数为准。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {'ember': "1-Ember汉化插件", 'crucible': "2-Crucible汉化插件"}

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')
LATIN = re.compile(r'[A-Za-z]')
EMBEDDED = re.compile(r'\.(items|effects|actors)\.([^.]+)')   # 与工具逐字相同

# 内嵌字段 -> babele 的 documentType（mappings.mjs / babele default-mappings）
FIELD_TYPE = {'items': 'Item', 'effects': 'ActiveEffect', 'actors': 'Actor',
              'journals': 'JournalEntry', 'pages': 'JournalEntryPage',
              'results': 'TableResult', 'scenes': 'Scene', 'tables': 'RollTable',
              'macros': 'Macro', 'regions': 'Region', 'behaviors': 'RegionBehavior',
              'sounds': 'PlaylistSound', 'playlists': 'Playlist'}


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def translatable(en):
    rest = MARKUP.sub(' ', TAG.sub(' ', en)).replace('&amp;', ' ')
    return bool(LATIN.search(rest))


def walk(en, cn, path, todo):
    """逐字复刻 validate_translations.walk 的 todo 分支。"""
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], todo)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], todo)
    elif isinstance(en, str):
        if not en.strip() or not translatable(en):
            return
        if not (isinstance(cn, str) and CJK.search(cn)):
            todo.append(('.'.join(path), en))


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def main():
    # ---- 全库 cn 条目索引：name -> {documentType: {pack: entry}} ----
    by_name = defaultdict(lambda: defaultdict(dict))
    flat_names = set()                      # 工具用的扁平集合
    pack_type = {}
    for key, repo in REPOS.items():
        src = load(os.path.join(P, repo, 'compendium', 'en', '_source.json'))
        pkg = src['packageId']
        for p in src['packs']:
            pack_type[f'{pkg}.{p["pack"]}.json'] = p['documentType']
        cn_dir = os.path.join(P, repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith('.json') or fn.startswith('_'):
                continue
            doc = load(os.path.join(cn_dir, fn))
            dt = pack_type.get(fn, '?')
            for k, v in (doc.get('entries') or {}).items():
                if isinstance(v, dict) and isinstance(v.get('name'), str) and CJK.search(v['name']):
                    flat_names.add(k)
                    by_name[k][dt][fn] = v

    print(f'全库可按名字命中的条目名：{len(flat_names)}（工具用的就是这一个扁平集合）')
    print(f'其中同名跨 documentType 的：'
          f'{sum(1 for n, d in by_name.items() if len(d) > 1)}\n')

    grand = Counter()
    examples = defaultdict(list)
    for key, repo in REPOS.items():
        en_dir = os.path.join(P, repo, 'compendium', 'en')
        cn_dir = os.path.join(P, repo, 'compendium', 'cn')
        todo_all = []
        for fn in sorted(f for f in os.listdir(en_dir) if f.endswith('.json') and not f.startswith('_')):
            en = load(os.path.join(en_dir, fn))
            cnp = os.path.join(cn_dir, fn)
            cn = load(cnp) if os.path.exists(cnp) else {'entries': {}}
            t = []
            walk(en.get('entries', {}), cn.get('entries', {}), [], t)
            walk(en.get('folders', {}), cn.get('folders', {}), ['(folders)'], t)
            todo_all += [(fn, p, s) for p, s in t]

        auto = resid = 0
        miss_type = miss_field = ok = 0
        for fn, path, src in todo_all:
            m = EMBEDDED.search('.' + path)
            if not (m and m.group(2) in flat_names):
                resid += 1
                continue
            auto += 1
            field, name = m.group(1), m.group(2)
            dt = FIELD_TYPE[field]
            # 相对字段路径 = 内嵌文档名之后的部分
            i = ('.' + path).index(f'.{field}.{name}.') if f'.{field}.{name}.' in ('.' + path) else -1
            rel = ('.' + path)[i + len(f'.{field}.{name}.'):] if i >= 0 else ''
            cands = by_name[name].get(dt, {})
            if not cands:
                miss_type += 1
                if len(examples[f'{key}:type']) < 8:
                    examples[f'{key}:type'].append(
                        (fn, path, f'{name} 只作为 {list(by_name[name])} 存在，需要 {dt}'))
                continue
            hit = False
            for pk, entry in cands.items():
                v = get_at(entry, rel.split('.')) if rel else entry.get('name')
                if isinstance(v, str) and CJK.search(v):
                    hit = True
                    break
            if hit:
                ok += 1
            else:
                miss_field += 1
                if len(examples[f'{key}:field']) < 8:
                    examples[f'{key}:field'].append(
                        (fn, path, f'{list(cands)} 里的 {name} 没有字段 {rel!r}'))
        print(f'== {key} ({repo})')
        print(f'   L0 todo（复刻 validate_translations）            {len(todo_all)}')
        print(f'   L1 工具判成「babele 自动取到，不用翻」           {auto}')
        print(f'   L1 工具判成残余（真的要人翻）                    {resid}')
        print(f'   L2 其中 documentType 就对不上（generic 取不到）  {miss_type}')
        print(f'   L2 其中同名条目里**没有这个字段**（任何回落都取不到）{miss_field}')
        print(f'   L2 真的能取到                                    {ok}')
        grand['todo'] += len(todo_all); grand['auto'] += auto
        grand['type'] += miss_type; grand['field'] += miss_field; grand['ok'] += ok

    print(f'\n合计：被核销 {grand["auto"]} 条；其中 documentType 不符 {grand["type"]} 条、'
          f'字段根本不存在 {grand["field"]} 条、确实能自动取到 {grand["ok"]} 条')
    for k, rows in examples.items():
        print(f'\n  样例 [{k}]')
        for fn, path, why in rows:
            print(f'    {fn} :: {path}\n        {why}')


main()
