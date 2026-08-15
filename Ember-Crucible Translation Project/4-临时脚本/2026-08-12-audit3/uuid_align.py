#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复核用：按叶路径把 EN / CN 的 @UUID[target]{label} 按出现顺序对齐打印。
用法: uuid_align.py <repo> <pack> <leafpath> [--target SUBSTR]
      uuid_align.py <repo> <pack> --scan-target <target-id>   # 全库找该目标的所有标签
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
UU = re.compile(r'@UUID\[([^\]]*)\]\{([^}]*)\}')
UU_NOLBL = re.compile(r'@UUID\[([^\]]*)\](?!\{)')

def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def resolve(root, dotted):
    parts = dotted.split('.')
    def rec(node, i):
        if i == len(parts):
            return node, True
        if not isinstance(node, dict):
            return None, False
        for j in range(len(parts), i, -1):
            key = '.'.join(parts[i:j])
            if key in node:
                r, ok = rec(node[key], j)
                if ok:
                    return r, True
        return None, False
    return rec(root, 0)

def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))

def main():
    repo, pack = sys.argv[1], sys.argv[2]
    cn = load(f"{P}/{repo}/compendium/cn/{pack}")
    en = load(f"{P}/{repo}/compendium/en/{pack}")
    cnE = cn.get('entries', cn); enE = en.get('entries', en)
    if sys.argv[3] == '--scan-target':
        tgt = sys.argv[4]
        leaves_en, leaves_cn = [], []
        walk(enE, [], leaves_en); walk(cnE, [], leaves_cn)
        cnmap = dict(leaves_cn)
        from collections import Counter
        pairs = Counter()
        for p, s in leaves_en:
            if tgt not in s:
                continue
            e = UU.findall(s)
            c = UU.findall(cnmap.get(p, ''))
            if len(e) != len(c):
                for (t, l) in e:
                    if tgt in t:
                        pairs[(l, '<对齐失败>')] += 1
                continue
            for (te, le), (tc, lc) in zip(e, c):
                if tgt in te:
                    pairs[(le, lc)] += 1
        for (le, lc), n in pairs.most_common():
            print(f"  {n:4d}  EN «{le}»  ->  CN «{lc}»")
        return
    leaf = sys.argv[3]
    ens, ok1 = resolve(enE, leaf)
    cns, ok2 = resolve(cnE, leaf)
    if not (ok1 and ok2):
        print("路径解析失败", ok1, ok2); return
    e = UU.findall(ens); c = UU.findall(cns)
    print(f"EN 链接数 {len(e)} / CN 链接数 {len(c)}")
    print(f"EN 无标签链接 {len(UU_NOLBL.findall(ens))} / CN 无标签链接 {len(UU_NOLBL.findall(cns))}")
    filt = sys.argv[4] if len(sys.argv) > 4 else None
    for i in range(max(len(e), len(c))):
        te, le = e[i] if i < len(e) else ('-', '-')
        tc, lc = c[i] if i < len(c) else ('-', '-')
        if filt and filt not in te and filt not in tc:
            continue
        flag = '  ' if te == tc else '!!'
        print(f"{flag}[{i:3d}] {te}")
        print(f"        EN «{le}»")
        print(f"        CN «{lc}»")

main()
