# -*- coding: utf-8 -*-
"""外部第二实现对照：我们的 crucible 英文基准 vs Padhiver/Crucible-FR 的。

2026-08-06 版（`4-临时脚本/2026-08-06/crosscheck_vs_crucible_fr.py`）的重跑，
改了三处：
  1. 路径不再指向某个 session 的 scratchpad，改成 round8 的 fr-ref clone；
  2. 基准取项目仓库里现役的 `2-Crucible汉化插件/compendium/en`，
     不是 `5-其他内容/english-baseline` 的快照 —— 要对的是**现在发出去的东西**；
  3. 出 JSON（--out），能进回测记录。

只对 crucible：Crucible-FR 没有公开发布 Ember 译文，它的 compendium/en 只有
crucible 系统包。Ember 侧的同类盲区靠 `census_pack_fields.mjs` 直接枚举 LevelDB。

用法: python crosscheck_vs_crucible_fr_r8.py --out out.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OURS = os.path.join(ROOT, r"2-Crucible汉化插件\compendium\en")
THEIRS = os.path.join(ROOT, r"4-临时脚本\2026-08-13-round8\fr-ref\compendium\en")
TAG = re.compile(r"<[^>]+>")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, str) and o.strip():
        yield o


def stats(entries):
    n = ch = 0
    for s in leaves(entries):
        n += 1
        ch += len(TAG.sub(" ", s))
    return n, ch


def by_name(entries):
    """两侧建键方式不同（我们按 name，FR 按 id-ish slug），重建成按 name 才能比。"""
    out = {}
    for v in entries.values():
        if isinstance(v, dict) and isinstance(v.get("name"), str):
            out.setdefault(v["name"], v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    report = {"ours": OURS, "theirs": THEIRS, "packs": [], "totals": {}}
    tot = [0, 0, 0, 0]
    print(f"{'pack':<34}{'ours':>7}{'theirs':>8}{'onlyO':>7}{'onlyT':>7}"
          f"{'ourStr':>8}{'thrStr':>8}{'ourCh':>10}{'thrCh':>10}")
    for fn in sorted(f for f in os.listdir(OURS) if f.endswith(".json") and not f.startswith("_")):
        tp = os.path.join(THEIRS, fn)
        row = {"pack": fn}
        if not os.path.exists(tp):
            row["missing_in_fr"] = True
            report["packs"].append(row)
            print(f"{fn:<34}  --- not in Crucible-FR ---")
            continue
        ae = load(os.path.join(OURS, fn)).get("entries", {})
        be = load(tp).get("entries", {})
        an, ac = stats(ae)
        bn, bc = stats(be)
        tot = [tot[0] + an, tot[1] + bn, tot[2] + ac, tot[3] + bc]

        na, nb = by_name(ae), by_name(be)
        shared = set(na) & set(nb)
        fa, fb, char_gap = Counter(), Counter(), Counter()
        for k in shared:
            fa.update(na[k].keys())
            fb.update(nb[k].keys())
            for f in set(nb[k]) - set(na[k]):
                char_gap[f] += sum(len(TAG.sub(" ", s)) for s in leaves(nb[k][f]))
        row.update({
            "entries_ours": len(ae), "entries_theirs": len(be),
            "names_only_ours": sorted(set(na) - set(nb))[:20],
            "names_only_theirs": sorted(set(nb) - set(na))[:20],
            "n_names_only_ours": len(set(na) - set(nb)),
            "n_names_only_theirs": len(set(nb) - set(na)),
            "shared_entries": len(shared),
            "strings_ours": an, "strings_theirs": bn, "chars_ours": ac, "chars_theirs": bc,
            "fields_only_theirs": {k: v for k, v in fb.items() if k not in fa},
            "fields_only_ours": {k: v for k, v in fa.items() if k not in fb},
            "chars_we_miss_by_field": dict(char_gap.most_common(8)),
        })
        report["packs"].append(row)
        print(f"{fn:<34}{len(ae):>7}{len(be):>8}{row['n_names_only_ours']:>7}"
              f"{row['n_names_only_theirs']:>7}{an:>8}{bn:>8}{ac:>10}{bc:>10}")

    report["totals"] = {"strings_ours": tot[0], "strings_theirs": tot[1],
                        "chars_ours": tot[2], "chars_theirs": tot[3]}
    print(f"{'TOTAL':<34}{'':>7}{'':>8}{'':>7}{'':>7}{tot[0]:>8}{tot[1]:>8}{tot[2]:>10}{tot[3]:>10}")

    print("\n=== 字段级差异（只看共有条目）===")
    any_gap = False
    for r in report["packs"]:
        d = {k: r.get(k) for k in ("fields_only_theirs", "fields_only_ours", "chars_we_miss_by_field")}
        if any(d.values()):
            any_gap = True
            print(f"-- {r['pack']}: {json.dumps(d, ensure_ascii=False)}")
    if not any_gap:
        print("(无：两侧在共有条目上的字段名集合完全一致)")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
