#!/usr/bin/env python3
"""Leaf-level workload measurement: how many translatable strings / CJK chars are
present in the new EN extraction, and how many are already covered by CN."""
import json, os, sys, re, argparse

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')

def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield (prefix, obj)

def textlen(s):
    return len(TAG.sub(' ', s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-en', required=True)
    ap.add_argument('--cn', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    print(f"{'pack':<38}{'EN str':>8}{'EN chars':>10}{'covered':>9}{'cov%':>6}{'todo str':>9}{'todo chars':>11}")
    T = [0,0,0,0,0]
    for fn in sorted(f for f in os.listdir(a.new_en) if f.endswith('.json')):
        ne = load(os.path.join(a.new_en, fn)).get('entries', {})
        cp = os.path.join(a.cn, fn)
        ce = load(cp).get('entries', {}) if os.path.exists(cp) else {}

        n_str = n_ch = cov = todo_s = todo_c = 0
        for k, v in ne.items():
            cn_entry = ce.get(k)
            cn_leaves = {p: s for p, s in leaves(cn_entry)} if cn_entry else {}
            for p, s in leaves(v):
                if not s.strip():
                    continue
                n_str += 1; L = textlen(s); n_ch += L
                c = cn_leaves.get(p)
                if c and CJK.search(c):
                    cov += 1
                else:
                    todo_s += 1; todo_c += L
        pct = (100.0*cov/n_str) if n_str else 0.0
        print(f"{fn:<38}{n_str:>8}{n_ch:>10}{cov:>9}{pct:>5.0f}%{todo_s:>9}{todo_c:>11}")
        T = [T[0]+n_str, T[1]+n_ch, T[2]+cov, T[3]+todo_s, T[4]+todo_c]
    pct = (100.0*T[2]/T[0]) if T[0] else 0
    print(f"{'TOTAL '+a.label:<38}{T[0]:>8}{T[1]:>10}{T[2]:>9}{pct:>5.0f}%{T[3]:>9}{T[4]:>11}")

main()
