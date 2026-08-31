#!/usr/bin/env python3
"""把重抽出来的 `scenes.*.levels` 子树并进现有英文基准，**其余一律不动**。

为什么不整份覆盖：`5-其他内容/english-baseline/LOCAL-PATCHES.md` 记着对上游英文笔误
打的本地补丁（例：`@Condition[exhaustion` 缺右方括号，不补就有条目永远过不了闸门）。
`extract_en.mjs` 重抽会把它们静默回退 —— PROJECT.md 5.0 第 3 步专门写了这一条。

本轮的改动面本来就是外科式的：新增 `levels` 之后，重抽相对现有基准
**只多出 259 个层名、0 删除**，另有 3 条（×2 包）差异正是那些本地补丁。
所以只搬 `levels`，把补丁留在原地。

用法：
  python merge_levels_into_en.py --repo <repo> --new <重抽输出目录> [--write]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')


#: 本轮新加进 Scene 层的三个字段（见 mappings.mjs 的 SCENE_LEVELS 注释）。
#: `levels`/`tokens` 是 name 集合（子字典），`navName` 是单个字符串。
COLLECTIONS = ('levels', 'tokens')
SCALARS = ('navName',)


def collect_levels(node, path, out):
    """产出 (到集合键为止的路径元组, 该集合的字典)；标量字段包成单键字典。"""
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if k in COLLECTIONS and isinstance(v, dict) and all(isinstance(x, str) for x in v.values()):
            out[tuple(path + [k])] = v
        elif k in SCALARS and isinstance(v, str):
            # 标量：把父节点当成"集合"，键就是字段名本身
            out.setdefault(tuple(path), {})[k] = v
        elif isinstance(v, dict):
            collect_levels(v, path + [k], out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    total_files = total_keys = 0
    for f in sorted(os.listdir(a.new)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        cur_p = os.path.join(a.repo, 'compendium', 'en', f)
        if not os.path.exists(cur_p):
            print(f'  ! 现有基准没有 {f}，跳过')
            continue
        with open(cur_p, encoding='utf-8') as fh:
            cur = json.load(fh)
        with open(os.path.join(a.new, f), encoding='utf-8') as fh:
            new = json.load(fh)

        found = {}
        collect_levels(new.get('entries', {}), [], found)
        if not found:
            continue

        added = 0
        for segs, levels in found.items():
            node = cur.setdefault('entries', {})
            for s in segs[:-1]:
                node = node.setdefault(s, {})
            tgt = node.setdefault(segs[-1], {})
            for k, v in levels.items():
                if k not in tgt:
                    tgt[k] = v
                    added += 1
        print(f'  {f:34s} levels 组 {len(found):3d}  新增键 {added:4d}')
        total_files += 1
        total_keys += added
        if a.write and added:
            with open(cur_p, 'w', encoding='utf-8') as fh:
                json.dump(cur, fh, ensure_ascii=False, indent=2)
                fh.write('\n')

    print(f'{a.repo}: {total_files} 个包，合计新增 {total_keys} 个层名键'
          + ('（已写回）' if a.write else '（未加 --write，未改动）'))


if __name__ == '__main__':
    main()
