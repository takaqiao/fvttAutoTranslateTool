# -*- coding: utf-8 -*-
"""G4 回测：scan_cross_channel.py 的 A/C 段词对齐重写前后对照。

两个方向都要过，缺一个这次重写就不算数：

  ① **特异度**（不再造假警报）：PROJECT.md 点名的三个坏例 Ward / Aspect / Shoddy，
     旧版分别报成「神殿区」「会替」「品质」。新版这三个词的 compendium 候选里
     不得再出现它们。
  ② **灵敏度**（真漂移仍抓得到）：把 lang 值换回历史上真出过问题的写法
     （Fan=扇子、Stride=跨步、Steading=庄园 …），新版必须仍然报 LANG_ORPHAN
     并给出正确的对家。只做 ① 不做 ②，"修好"的方法可以是把什么都判成 AGREE。

只读，不写任何译文。用法：
    python g4_backtest.py [--out <json>]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path("C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project")
REPOS = [str(P / "1-Ember汉化插件"), str(P / "2-Crucible汉化插件")]
MJS = P / "1-Ember汉化插件" / "scripts" / "ember-hardcoded-cn.mjs"
PKGS = ["C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible",
        "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"]

# (英文, 喂给它的 lang 值, 期待的对家, 该不该报)
CASES = [
    # ① 特异度：这三条是 PROJECT.md 点名的坏例，坏候选**不得出现**
    ("Ward",    "防护", None, "specificity", "神殿区"),
    ("Aspect",  "化相", None, "specificity", "会替"),
    ("Shoddy",  "粗糙", None, "specificity", "品质"),
    ("Blast",   "爆破", None, "specificity", "爆炸瓶"),   # Blast Flask 的译名
    ("Fan",     "扇形", None, "specificity", "裂的弧形"),
    ("Cone",    "锥形", None, "specificity", "尺锥"),
    ("Surge",   "涌动", None, "specificity", "奔涌"),
    # ② 灵敏度：把 lang 换回历史上的错值，必须重新报出来
    #
    #    ⚠ 这一组的 `expect` 是**当前 compendium 的多数写法**，会随译文修订而变 ——
    #    它不是常量。2026-08-13 的 G4-1 把 `Steading` 的 14 叶由「安居」统一成「耕耘」之后，
    #    这条的期待值就从 安居 变成了 耕耘；不跟着改的话回测会红，而那不是判据坏了。
    #    **改译文时记得回来改这里**，否则下一个人会把一条正确的报告当成回归。
    ("Fan",      "扇子", "扇形", "sensitivity", None),
    ("Stride",   "跨步", "步幅", "sensitivity", None),
    ("Steading", "庄园", "耕耘", "sensitivity", None),   # 2026-08-13 前为「安居」
    ("Reckless", "莽撞", "鲁莽", "sensitivity", None),
    ("Aspect",   "样貌", "化相", "sensitivity", None),
    ("Ward",     "病房", "防护", "sensitivity", None),
]


def load_module(legacy=False):
    spec = importlib.util.spec_from_file_location(
        "scc", P / "3-常用脚本" / "qa" / "scan_cross_channel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    m = load_module()

    leaves = m.load_leaves(REPOS, [])
    cn_all = [lf[4] for lf in leaves if lf[4]]
    lang_all = {}
    for repo in REPOS:
        en_flat, cn_raw, _src, _keep = m.load_lang_pairs(repo, PKGS)
        pairs = {}
        for k, en in en_flat.items():
            cn = m.foundry_lookup(cn_raw, k)
            if cn is not None:
                pairs[k] = (en, cn)
        lang_all[Path(repo).name] = pairs
    mjs_entries = [(str(MJS), t, en, cn)
                   for t, pairs in m.parse_mjs_tables(MJS).items() for en, cn in pairs]
    glossary = m.load_json(P / "5-其他内容" / "glossary" / "glossary_ec.json")

    terms = sorted({c[0] for c in CASES})
    gated = m.build_gate(leaves, set(terms), False)
    ev = m.build_evidence(leaves, lang_all, glossary, mjs_entries)
    lex = m.LexIndex(set(ev.lex), leaves)
    en_cache = {}

    # 旧的 n-gram 挖掘（对照组）
    bg = {}
    legacy_rank = {}
    for t in terms:
        idxs = [i for i in gated.get(t, []) if leaves[i][4]]
        if not idxs:
            continue
        cands, texts = m.mine_preselect(idxs, leaves, 500)
        from collections import Counter
        bgdf = Counter()
        probe = {g for g, _ in cands}
        for cn in cn_all:
            hit = probe & m.cjk_ngrams(cn)
            if hit:
                bgdf.update(hit)
        legacy_rank[t] = m.rank_candidates(cands, bgdf, idxs, leaves, texts, t, {}, top=10)

    rows, failed = [], 0
    for en, lang_cn, expect, kind, banned in CASES:
        idxs = [i for i in gated.get(en, []) if leaves[i][4]]
        rx = m._free_regex(en)
        free = {i for i in idxs if m.occurs_free(en, leaves[i][3], rx)}
        cands = m.build_candidates(en, idxs, leaves, ev, lex, len(cn_all), 3, 8.0,
                                   free, en_cache, top=10, max_common=800)
        corpus_df = {lang_cn: sum(1 for c in cn_all if lang_cn in c)}
        row = m.analyse_term(en, lang_cn, [("test", "test")], idxs, cands, leaves,
                             corpus_df, 3, 3, 4, False, free_idxs=free)
        top = row["compendium_top"]
        newtop = top["cn"] if top else None
        allc = [c["cn"] for c in row["candidates"]]
        legacy_top = (legacy_rank.get(en) or [{}])[0].get("cn")
        ok = True
        if kind == "specificity":
            ok = banned not in allc
        else:
            ok = (row["verdict"] in ("LANG_ORPHAN", "LANG_MINORITY", "LANG_SPLIT")
                  and newtop == expect)
        failed += (not ok)
        rows.append({"en": en, "lang_cn": lang_cn, "kind": kind, "pass": ok,
                     "verdict": row["verdict"], "new_top": newtop,
                     "new_candidates": allc, "legacy_top": legacy_top,
                     "banned": banned, "expect": expect,
                     "gated": row["gated_with_cn"], "gated_free": row["gated_free"],
                     "lang_support": row["lang_support"],
                     "attributed": row["attributed"]})
        print(f"[{'PASS' if ok else 'FAIL'}] {kind:12} {en:<10} lang=「{lang_cn}」 "
              f"闸 {row['gated_with_cn']:>4}(自由{row['gated_free']:>4}) 支持 {row['lang_support']:>4}"
              f"  旧头名=「{legacy_top}」 → 新头名=「{newtop}」  [{row['verdict']}]")
        if not ok:
            print(f"        候选={allc}  期待={expect} 禁止={banned}")

    print(f"\n{len(rows) - failed}/{len(rows)} 通过")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print("→", a.out)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
