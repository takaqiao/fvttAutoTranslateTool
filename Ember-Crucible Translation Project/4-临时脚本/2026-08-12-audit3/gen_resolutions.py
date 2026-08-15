#!/usr/bin/env python3
"""为 `merge_batches.py` 的冲突自动生成 `resolutions.json`（取超集 + 叠加其余编辑）。

并行单元改同一个叶子时，`merge_batches` 只会在编辑区间**重叠**时报冲突。实测这批冲突
几乎全是**超集关系**：A 做了 B 的全部编辑再加几处，于是两者在同一区间都有动作。

做法：
  1. 取编辑数最多的那个变体当基（chosen）；
  2. 其余变体的每一处编辑，若在 chosen 里尚未体现，就作为**带上下文的字符串替换**叠加：
     - 替换型（old 非空）：直接 old -> new，要求 chosen 里恰好能找到 old；
     - 插入型（old 为空）：用 base 的左右各 12 字符定位，left+right -> left+new+right。
  3. 每一步都验证「能找到、且只改预期的量」，找不到就**不生成**该条并打印出来交人工。

输出的 `then` 列表会被 `merge_batches.py` 逐条 `str.replace` 应用到 chosen 上，
且它自己也会校验 old 存在（不存在就 SystemExit），所以这是双保险。
"""
from __future__ import annotations
import argparse
import difflib
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

CTX = 12


def ops_of(base, val):
    return [o for o in difflib.SequenceMatcher(None, base, val, autojunk=False).get_opcodes()
            if o[0] != 'equal']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conflicts', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    conflicts = json.load(open(a.conflicts, encoding='utf-8'))
    res, manual = {}, []
    for idx, c in enumerate(conflicts):
        base = c['base']
        variants = [(v['batch'], v['value']) for v in c['variants']]
        # 基 = 编辑数最多的那个
        chosen_name, chosen = max(variants, key=lambda nv: len(ops_of(base, nv[1])))
        then, skipped = [], []
        cur = chosen
        for name, val in variants:
            if name == chosen_name:
                continue
            for tag, i1, i2, j1, j2 in ops_of(base, val):
                old, new = base[i1:i2], val[j1:j2]
                if old and old in cur and new not in cur:
                    then.append([old, new])
                    cur = cur.replace(old, new, 1)
                elif not old:                       # 纯插入，用 base 上下文定位
                    left, right = base[max(0, i1 - CTX):i1], base[i2:i2 + CTX]
                    anchor = left + right
                    if left + new + right in cur:
                        continue                    # 基已经做过同一处插入
                    if anchor and anchor in cur:
                        then.append([anchor, left + new + right])
                        cur = cur.replace(anchor, left + new + right, 1)
                    else:
                        skipped.append(f'{name}: 插入 {new!r} 定位不到')
                elif old and old not in cur:
                    # chosen 里已经不存在 old —— 多半是 chosen 已经做过同一处编辑
                    if new in cur:
                        continue
                    skipped.append(f'{name}: {old!r}->{new!r} 在基里找不到')
        entry = {'take': chosen_name, 'path': c['path'][-70:],
                 'why': f'取编辑最多的 {chosen_name.split("__")[0]} 为基，'
                        f'叠加其余单元的 {len(then)} 处编辑（自动合成，见 gen_resolutions.py）'}
        if then:
            entry['then'] = then
        if skipped:
            entry['manual_check'] = skipped
            manual.append((idx, c['path'][-60:], skipped))
        res[str(idx)] = entry

    json.dump(res, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'生成 {len(res)} 条裁决 -> {a.out}')
    if manual:
        print(f'\n⚠ 有 {len(manual)} 条含定位不到的编辑，需人工看：')
        for idx, p, s in manual:
            print(f'  [{idx}] {p}')
            for x in s:
                print(f'        {x}')


if __name__ == '__main__':
    main()
