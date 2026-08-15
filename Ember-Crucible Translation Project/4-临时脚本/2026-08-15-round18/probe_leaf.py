# -*- coding: utf-8 -*-
"""把某一叶的候选块三方（旧英文对应段 / 新英文块 / 中文块）打印出来，人核用。"""
import json, os, re, sys, difflib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/3-常用脚本/qa")
import scan_dropped_terms as S

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"


def main(report, repo, baseline, needle, term_filter=None):
    rep = json.load(open(report, encoding="utf-8"))
    idmap = S.load_idmap(rep["meta"]["bindings"])
    packs = S.baseline_packs(baseline)
    for f in rep["findings"]:
        if needle not in f["path"]:
            continue
        pk, pa = f["pack"], f["path"]
        o, n, c = {}, {}, {}
        S.leaves(S.load_json(packs[pk]).get("entries", {}), [], o)
        S.leaves(S.load_json(os.path.join(repo, "compendium", "en", pk)).get("entries", {}), [], n)
        S.leaves(S.load_json(os.path.join(repo, "compendium", "cn", pk)).get("entries", {}), [], c)
        oe, ne, cn = o[pa], n[pa], c[pa]
        ow, o_owner, o_parts = S.block_tokens(oe, idmap)
        nw, n_owner, n_parts = S.block_tokens(ne, idmap)
        cn_parts = S.TAG_SPLIT.split(cn)
        sm = difflib.SequenceMatcher(None, [w.lower() for w in ow], [w.lower() for w in nw],
                                     autojunk=False)
        anchor = S.anchor_old_to_new_blocks(sm.get_opcodes(), len(ow), n_owner)
        print(f"\n########## {pa}")
        print(f"块数 旧{len(o_parts)} 新{len(n_parts)} 中{len(cn_parts)}")
        for h in f["dropped"]:
            if term_filter and h["en"] != term_filter:
                continue
            st = S.stem(h["en"])
            print(f"\n=== {h['en']} 全叶 {h['en_old_n']}->{h['en_new_n']} cn={h['cn_count']}"
                  f" | local {h['local']}")
            for b in h["local"]["blocks"]:
                old_here = " ‖ ".join(w for k, w in enumerate(ow)
                                      if b in anchor[k] and S.stem(w) == st)
                print(f"  --- 块 {b} ---")
                print(f"   旧英文锚到此块的该词: [{old_here}]")
                print(f"   新英文块: {n_parts[b][:300]!r}")
                print(f"   中文块  : {cn_parts[b][:300]!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         sys.argv[5] if len(sys.argv) > 5 else None)
