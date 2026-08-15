# -*- coding: utf-8 -*-
"""逐叶对照「当前 compendium/en 的某包」与「刚从上游 LevelDB 重抽的同包」。

    python diff_leaves.py <当前包.json> <重抽包.json>

输出：仅在一侧存在的路径数、两侧都有但值不同的路径（逐条打印，值截断）。
反空转：先打印两侧各扫到多少叶，任一侧为 0 直接非零退出。
"""
from __future__ import annotations
import io
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(p):
    return json.load(io.open(p, encoding="utf-8-sig"))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def main():
    cur_p, new_p = sys.argv[1], sys.argv[2]
    cur, new = {}, {}
    leaves(load(cur_p).get("entries", {}), [], cur)
    leaves(load(new_p).get("entries", {}), [], new)
    print(f"当前包叶数 {len(cur)}   重抽包叶数 {len(new)}")
    if not cur or not new:
        print("!! 一侧为 0 叶，判据空转，退出")
        return 2
    only_cur = sorted(set(cur) - set(new))
    only_new = sorted(set(new) - set(cur))
    both = sorted(set(cur) & set(new))
    diff = [k for k in both if cur[k] != new[k]]
    print(f"共有路径 {len(both)}   仅当前有 {len(only_cur)}   仅重抽有 {len(only_new)}   共有但值不同 {len(diff)}")
    for k in only_cur:
        print(f"[仅当前] {k}")
    for k in only_new:
        print(f"[仅重抽] {k}")
    for k in diff:
        print(f"\n=== 值不同: {k}")
        a, b = cur[k], new[k]
        # 找第一处分歧，前后各截 120 字
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        lo, hi = max(0, i - 120), i + 120
        print(f"  当前: ...{a[lo:hi]}...")
        print(f"  重抽: ...{b[lo:hi]}...")
    print(f"\n合计差异叶 {len(only_cur) + len(only_new) + len(diff)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
