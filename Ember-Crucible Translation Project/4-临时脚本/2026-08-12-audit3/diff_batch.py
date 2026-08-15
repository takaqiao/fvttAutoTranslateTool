#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复核用：把批次值与 compendium/cn 现值逐叶做字符级 diff，
输出每处改动的 (旧片段 -> 新片段)，并核对标记序列是否与 EN 侧一致。"""
import json, sys, re, difflib, io
sys.stdout.reconfigure(encoding='utf-8')

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"

def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def get(d, path):
    cur = d
    for seg in path:
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur

def split_path(bp):
    # batch path: dotted, but names may contain dots. Resolve greedily against tree.
    return bp

def resolve(root, dotted):
    """按点号路径解析，段名可能含点：贪婪回溯。"""
    parts = dotted.split('.')
    def rec(node, i):
        if i == len(parts):
            return node, True
        if not isinstance(node, dict):
            return None, False
        # try longest first
        for j in range(len(parts), i, -1):
            key = '.'.join(parts[i:j])
            if key in node:
                r, ok = rec(node[key], j)
                if ok:
                    return r, True
        return None, False
    return rec(root, 0)

MARK = re.compile(r'@UUID\[[^\]]*\]|@Embed\[[^\]]*\]|@Check\[[^\]]*\]|\[\[[^\]]*\]\]|<[^>]+>|&[a-zA-Z#0-9]+;')

def marks(s):
    return MARK.findall(s or '')

def main(batch_file, repo, pack):
    batch = load(batch_file)
    cn = load(f"{P}/{repo}/compendium/cn/{pack}")
    en = load(f"{P}/{repo}/compendium/en/{pack}")
    cnE = cn.get('entries', cn)
    enE = en.get('entries', en)
    total_changed = 0
    maxrun = 0
    for bp, newv in batch.items():
        oldv, ok = resolve(cnE, bp)
        env, oken = resolve(enE, bp)
        print("=" * 100)
        print("PATH:", bp)
        if not ok:
            print("  !!! 现 cn 里找不到该路径")
            continue
        if not isinstance(oldv, str):
            print("  !!! 现值不是字符串:", type(oldv))
            continue
        if oldv == newv:
            print("  ~~ 无变化（空改）")
            continue
        # markup check vs EN
        if oken and isinstance(env, str):
            mo, mn, me = marks(oldv), marks(newv), marks(env)
            if mn != me:
                print("  !!! 标记序列与 EN 不一致  new-vs-en")
                for l in difflib.unified_diff(me, mn, 'EN', 'NEW', lineterm='', n=0):
                    print("      ", l)
            if mn != mo:
                print("  !!! 标记序列相对旧中文有变化")
                for l in difflib.unified_diff(mo, mn, 'OLD', 'NEW', lineterm='', n=0):
                    print("      ", l)
        sm = difflib.SequenceMatcher(None, oldv, newv, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            total_changed += 1
            run = max(i2 - i1, j2 - j1)
            maxrun = max(maxrun, run)
            ctx0 = oldv[max(0, i1 - 25):i1]
            ctx1 = oldv[i2:i2 + 25]
            print(f"  [{tag}] len(old)={i2-i1} len(new)={j2-j1}")
            print(f"     …{ctx0}«{oldv[i1:i2]}»{ctx1}…")
            print(f"     …{ctx0}«{newv[j1:j2]}»{ctx1}…")
    print("=" * 100)
    print(f"改动片段总数 {total_changed}，最长单片段 {maxrun} 字")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
