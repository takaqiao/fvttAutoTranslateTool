# -*- coding: utf-8 -*-
"""把「整叶口径报了、块级重判压掉」的那些叶逐条摊开，人核过滤器是不是压错了。

输入两份报告（--no-block-filter 的与带过滤器的），对差集里的每一叶重算候选块，
打印 旧英文对应段 / 新英文块 / 中文块 三方原文。反空转：报出扫了多少叶多少块。
"""
import json, os, re, sys, difflib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/3-常用脚本/qa")
import scan_dropped_terms as S


def main(off_rep, on_rep, repo, baseline, limit=99, cut=260):
    off = json.load(open(off_rep, encoding="utf-8"))
    on = {f["path"] for f in json.load(open(on_rep, encoding="utf-8"))["findings"]}
    idmap = S.load_idmap(off["meta"]["bindings"])
    gloss = S.load_glossary(off["meta"]["glossary"])
    packs = S.baseline_packs(baseline)
    cache = {}
    nleaf = nblk = 0
    for f in off["findings"]:
        if f["path"] in on:
            continue
        nleaf += 1
        if nleaf > limit:
            continue
        pk, pa = f["pack"], f["path"]
        if pk not in cache:
            o, n, c = {}, {}, {}
            S.leaves(S.load_json(packs[pk]).get("entries", {}), [], o)
            S.leaves(S.load_json(os.path.join(repo, "compendium", "en", pk)).get("entries", {}), [], n)
            S.leaves(S.load_json(os.path.join(repo, "compendium", "cn", pk)).get("entries", {}), [], c)
            cache[pk] = (o, n, c)
        o, n, c = cache[pk]
        oe, ne, cn = o[pa], n[pa], c[pa]
        ow, _oo, o_parts = S.block_tokens(oe, idmap)
        nw, n_owner, n_parts = S.block_tokens(ne, idmap)
        cn_parts = S.TAG_SPLIT.split(cn)
        sm = difflib.SequenceMatcher(None, [w.lower() for w in ow], [w.lower() for w in nw],
                                     autojunk=False)
        anchor = S.anchor_old_to_new_blocks(sm.get_opcodes(), len(ow), n_owner)
        del_idx = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("delete", "replace"):
                for k in range(i1, i2):
                    del_idx.setdefault(S.stem(ow[k]), set()).add(k)
        print(f"\n########## {pa}")
        for h in f["dropped"]:
            st = S.stem(h["en"])
            term = gloss.get(st)
            cand = set()
            for k in del_idx.get(st, ()):
                cand |= anchor[k]
            nblk += len(cand)
            print(f"  === {h['en']} 全叶 {h['en_old_n']}->{h['en_new_n']} 中文{term}×{h['cn_count']}"
                  f"  候选块 {sorted(cand)}")
            for b in sorted(cand):
                o_l = sum(1 for k, w in enumerate(ow) if S.stem(w) == st and b in anchor[k])
                n_l = sum(1 for j, w in enumerate(nw) if S.stem(w) == st and n_owner[j] == b)
                c_l = cn_parts[b].count(term)
                print(f"    块{b}: 局部 旧{o_l} 新{n_l} 中{c_l}")
                print(f"      新英文: {n_parts[b][:cut]!r}")
                print(f"      中文  : {cn_parts[b][:cut]!r}")
    print(f"\n压掉 {nleaf} 叶（打印前 {min(nleaf, limit)} 叶）· 摊开候选块 {nblk} 个")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 99)
