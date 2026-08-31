# -*- coding: utf-8 -*-
"""与 diff_baseline_dirs.py 同口径，但对「上游/旧版写坏的 JSON（尾逗号）」做一次容错修复再比。

    python diff_tolerant.py <A目录> <B目录> [--show]

反空转：每包打印两侧叶数；修复过的文件会打印 [repaired]；任一侧合计 0 叶 → 非零退出。
"""
from __future__ import annotations
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAILING = re.compile(r",(\s*[}\]])")


def load(p):
    raw = io.open(p, encoding="utf-8-sig").read()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        fixed = TRAILING.sub(r"\1", raw)
        return json.loads(fixed), True


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def pack_leaves(p):
    d, rep = load(p)
    out = {}
    leaves(d.get("entries", {}), [], out)
    return out, rep


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    show = "--show" in sys.argv
    names = lambda d: sorted(
        f for f in os.listdir(d)
        if f.endswith(".json") and not f.startswith("_")
    )
    a_names, b_names = names(a_dir), names(b_dir)
    print(f"A = {a_dir}  ({len(a_names)} 包)")
    print(f"B = {b_dir}  ({len(b_names)} 包)")
    if set(a_names) - set(b_names):
        print(f"仅 A 有的包: {sorted(set(a_names) - set(b_names))}")
    if set(b_names) - set(a_names):
        print(f"仅 B 有的包: {sorted(set(b_names) - set(a_names))}")
    tot = [0, 0, 0, 0, 0]
    for n in sorted(set(a_names) & set(b_names)):
        A, ra = pack_leaves(os.path.join(a_dir, n))
        B, rb = pack_leaves(os.path.join(b_dir, n))
        oa, ob = set(A) - set(B), set(B) - set(A)
        df = [k for k in (set(A) & set(B)) if A[k] != B[k]]
        tag = ("[A repaired]" if ra else "") + ("[B repaired]" if rb else "")
        print(f"{n:<38} A叶={len(A):<7} B叶={len(B):<7} 仅A={len(oa):<5} 仅B={len(ob):<5} 值不同={len(df):<5} {tag}")
        for i, v in enumerate((len(A), len(B), len(oa), len(ob), len(df))):
            tot[i] += v
        if show:
            for k in sorted(df)[:40]:
                print(f"    值不同 {k}\n      A: {A[k][:150]}\n      B: {B[k][:150]}")
    print(f"{'合计':<38} A叶={tot[0]:<7} B叶={tot[1]:<7} 仅A={tot[2]:<5} 仅B={tot[3]:<5} 值不同={tot[4]:<5}")
    if tot[0] == 0 or tot[1] == 0:
        print("!! 一侧 0 叶，判据空转，退出")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
