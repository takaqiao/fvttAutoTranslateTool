# -*- coding: utf-8 -*-
"""Leaf-level diff between two Babele-shaped translation JSONs.

Usage: python leafdiff.py <a.json> <b.json> [--max N]

Prints, per pack: leaves only in A, leaves only in B, leaves whose value differs.
A "leaf" is any string value, addressed by its full dotted/indexed path.
"""
import json, sys, io

sys.stdout.reconfigure(encoding="utf-8")


def leaves(node, prefix=""):
    if isinstance(node, str):
        yield prefix, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, f"{prefix}[{i}]")


def load(p):
    return dict(leaves(json.load(io.open(p, encoding="utf-8"))))


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    mx = 40
    if "--max" in sys.argv:
        mx = int(sys.argv[sys.argv.index("--max") + 1])
    A, B = load(a_path), load(b_path)
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    diff = sorted(k for k in set(A) & set(B) if A[k] != B[k])
    print(f"A={a_path}")
    print(f"B={b_path}")
    print(f"leaves A={len(A)} B={len(B)} | onlyA={len(only_a)} onlyB={len(only_b)} valueDiff={len(diff)}")
    for k in only_a[:mx]:
        print(f"  ONLY-A {k} = {A[k][:160]!r}")
    for k in only_b[:mx]:
        print(f"  ONLY-B {k} = {B[k][:160]!r}")
    for k in diff[:mx]:
        print(f"  DIFF   {k}\n     A: {A[k][:220]!r}\n     B: {B[k][:220]!r}")


if __name__ == "__main__":
    main()
