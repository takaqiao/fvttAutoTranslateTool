# -*- coding: utf-8 -*-
"""逐包对照两个 english-baseline 目录（或 baseline 目录 vs 插件仓 compendium/en）。

    python diff_baseline_dirs.py <A目录> <B目录>

输出每包：A 叶数 / B 叶数 / 仅 A 有 / 仅 B 有 / 共有但值不同。
反空转：合计叶数任一侧为 0 → 非零退出；每包都打印自己扫到多少叶。
"""
from __future__ import annotations
import io
import json
import os
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


def pack_leaves(p):
    out = {}
    leaves(load(p).get("entries", {}), [], out)
    return out


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    show = "--show" in sys.argv
    names = lambda d: sorted(
        f for f in os.listdir(d) if f.endswith(".json") and f != "_source.json"
    )
    a_names, b_names = names(a_dir), names(b_dir)
    print(f"A = {a_dir}  ({len(a_names)} 包)")
    print(f"B = {b_dir}  ({len(b_names)} 包)")
    only_a_pack = sorted(set(a_names) - set(b_names))
    only_b_pack = sorted(set(b_names) - set(a_names))
    if only_a_pack:
        print(f"仅 A 有的包: {only_a_pack}")
    if only_b_pack:
        print(f"仅 B 有的包: {only_b_pack}")
    tot_a = tot_b = tot_only_a = tot_only_b = tot_diff = 0
    print(f"\n{'包':<38}{'A叶':>8}{'B叶':>8}{'仅A':>7}{'仅B':>7}{'值不同':>8}")
    for n in sorted(set(a_names) & set(b_names)):
        A = pack_leaves(os.path.join(a_dir, n))
        B = pack_leaves(os.path.join(b_dir, n))
        oa = set(A) - set(B)
        ob = set(B) - set(A)
        both = set(A) & set(B)
        df = [k for k in both if A[k] != B[k]]
        tot_a += len(A)
        tot_b += len(B)
        tot_only_a += len(oa)
        tot_only_b += len(ob)
        tot_diff += len(df)
        print(f"{n:<38}{len(A):>8}{len(B):>8}{len(oa):>7}{len(ob):>7}{len(df):>8}")
        if show and df:
            for k in sorted(df):
                print(f"    值不同 {k}")
                print(f"      A: {A[k][:160]}")
                print(f"      B: {B[k][:160]}")
        if show and oa:
            for k in sorted(oa)[:20]:
                print(f"    仅A {k}")
        if show and ob:
            for k in sorted(ob)[:20]:
                print(f"    仅B {k}")
    print(f"\n{'合计':<38}{tot_a:>8}{tot_b:>8}{tot_only_a:>7}{tot_only_b:>7}{tot_diff:>8}")
    if tot_a == 0 or tot_b == 0:
        print("!! 一侧 0 叶，判据空转，退出")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
