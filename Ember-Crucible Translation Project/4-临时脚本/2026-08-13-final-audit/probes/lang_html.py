# -*- coding: utf-8 -*-
"""对 lang/en.json vs lang/cn.json 做同样的 HTML 结构解析对照。

lang 文件里有一部分值是带标记的（提示条、对话框正文、规则摘要），
但历轮的 lang 判据只查「四项 + 拍平三数相等」（键覆盖/占位符/顺序），
从来没有人把 lang 的值当 HTML 解析过。
"""
import json, re, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from html_wellformed import analyze, leaves, raw_amp_count, bracket_sig  # noqa
import tree_shape as T  # noqa

PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\-]*\}")

counts = collections.Counter()
rows = []
for repo in sys.argv[1:]:
    repo = Path(repo)
    en = dict(leaves(json.loads((repo / "lang" / "en.json").read_text(encoding="utf-8-sig"))))
    cn = dict(leaves(json.loads((repo / "lang" / "cn.json").read_text(encoding="utf-8-sig"))))
    n_html = sum(1 for v in cn.values() if "<" in v)
    print(f"{repo.name}: en {len(en)} 键 / cn {len(cn)} 键，cn 侧含 < 的 {n_html} 个")
    for k, s in cn.items():
        e = en.get(k, "")
        if "<" not in s and "<" not in e and "&" not in s and "&" not in e:
            continue
        errs, ids, _ = analyze(s)
        eerrs, _, _ = analyze(e) if e else ([], [], None)
        ecodes = collections.Counter(c for c, _ in eerrs)
        for code, det in errs:
            mine = collections.Counter(c for c, _ in errs)
            if ecodes.get(code, 0) >= mine.get(code, 0) and e:
                counts[code + "_upstream"] += 1
                continue
            counts[code] += 1
            rows.append((code, repo.name, k, det, e[:200], s[:200]))
        # 树形
        if "<" in s and "<" in e:
            bc, be = T.block_paths(s), T.block_paths(e)
            if bc is not None and be is not None and bc != be:
                counts["T1"] += 1
                rows.append(("T1", repo.name, k, f"CN={bc} EN={be}", e[:200], s[:200]))
        # 实体
        if e:
            a_en, a_cn = raw_amp_count(e), raw_amp_count(s)
            if a_cn > a_en:
                counts["P7"] += 1
                rows.append(("P7", repo.name, k, f"裸& EN={a_en} CN={a_cn}", e[:200], s[:200]))
            if bracket_sig(e) != bracket_sig(s):
                counts["P9"] += 1
                rows.append(("P9", repo.name, k, f"括号 EN={bracket_sig(e)} CN={bracket_sig(s)}", e[:200], s[:200]))

print(counts)
seen = collections.Counter()
for code, rn, k, det, e, s in rows:
    seen[code] += 1
    if seen[code] > 25:
        continue
    print("-" * 96)
    print(f"[{code}] {rn} | {k}")
    print("    det:", str(det)[:260])
    print("    EN :", e)
    print("    CN :", s)
