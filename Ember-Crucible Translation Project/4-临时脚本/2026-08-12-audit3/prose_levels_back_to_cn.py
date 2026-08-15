#!/usr/bin/env python3
"""把正文里回填的**英文层名**改回中文 —— 管线补上之后的收尾。

来历（PROJECT.md 第 7 节第 13 项）：
  `Scene.levels[].name` 此前根本没被抽取/翻译，玩家在 Foundry 的层级选择器里看到的
  永远是英文。第三轮采取的是**安全侧**做法：把正文里被译成中文的层名回填成英文
  （110 处），这样 GM 按正文里的名字能在 UI 里找到那一层。

  2026-08-13 管线补上了（`mappings.mjs` 的 Scene 层加 `levels`/`tokens`/`navName`），
  层名从此会被翻译，所以**安全侧那一步要反过来**：正文重新写中文，且必须与
  层名译文**逐字一致**，否则 GM 还是对不上。

判据：正文里形如 `“<英文层名>”层级` 的地方，把引号内换成该层名的中文译文。
只认映射表里确实存在的英文层名，认不出的原样不动并打印出来。

用法：
  python prose_levels_back_to_cn.py --repo <repo> --mapping <levels.json> \
         --out-dir <批次目录> [--show 20]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 正文里引用层名的固定句式。第三轮把「配置」「构图」也规范成了「层级」，
# 但孪生包里可能还有旧写法，一并认。
QUOTED = re.compile(r'([“"])([^”"]{1,60})([”"])(\s*)(层级|配置|构图)')


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--mapping', required=True, help='英文层名 -> 中文层名')
    ap.add_argument('--out-dir')
    ap.add_argument('--show', type=int, default=20)
    a = ap.parse_args()

    with open(a.mapping, encoding='utf-8') as f:
        M = json.load(f)

    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    hit, miss = collections.Counter(), collections.Counter()
    batches = {}
    for pack in sorted(os.listdir(cn_dir)):
        if not pack.endswith('.json'):
            continue
        with open(os.path.join(cn_dir, pack), encoding='utf-8') as f:
            cn = dict(walk(json.load(f).get('entries', {})))
        out = {}
        for path, v in cn.items():
            def sub(m):
                name = m.group(2)
                if name in M:
                    hit[name] += 1
                    # 句式一律规范成「层级」：`Level` 就是层级，
                    # 「构图」是 Composition 的译名、「配置」是错译（§8 2026-08-13）
                    return f'{m.group(1)}{M[name]}{m.group(3)}{m.group(4)}层级'
                if not re.search(r'[一-鿿]', name):
                    miss[name] += 1          # 还是英文、却不在映射表里 —— 要人看
                return m.group(0)
            nv = QUOTED.sub(sub, v)
            if nv != v:
                out[path] = nv
        if out:
            batches[pack] = out

    print(f'{a.repo}: 命中 {sum(hit.values())} 处 / {len(hit)} 个层名，'
          f'涉及 {sum(len(v) for v in batches.values())} 叶')
    for n, c in hit.most_common(a.show):
        print(f'    {c:3d}×  {n} -> {M[n]}')
    if miss:
        print(f'\n  ⚠ 引号里是英文、但不在层名映射表里的 {len(miss)} 个（**没动**，要人看）：')
        for n, c in miss.most_common(15):
            print(f'    {c:3d}×  {n}')

    if a.out_dir and batches:
        os.makedirs(a.out_dir, exist_ok=True)
        tag = 'ember' if a.repo.startswith('1') else 'crucible'
        for pack, items in batches.items():
            p = os.path.join(a.out_dir, f'PLV__{tag}__{pack[:-5]}.json')
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=1)
            print(f'-> {os.path.basename(p)}  ({len(items)} 叶)')


if __name__ == '__main__':
    main()
