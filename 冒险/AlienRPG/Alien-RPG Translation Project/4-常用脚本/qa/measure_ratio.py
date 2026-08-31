# -*- coding: utf-8 -*-
"""测量本语料自己的 中文/英文 长度比，得出一条**临时**带宽。

    python measure_ratio.py                         # 全部三段都跑
    python measure_ratio.py --only lang             # 只跑 lang 段
    python measure_ratio.py --out <report.json>

为什么必须重测：EC 的三个常数一个都不能搬
------------------------------------------
Ember/Crucible 那套用的是
    · 0.31  中位 CN/EN 字符比
    · 0.22  低于此判 TRUNCATED（译文被截断）
    · 0.25  catch-up 补译的下限
这三个数是**从 EC 自己的语料反推出来的**，不是语言学常数。它们至少踩三条：

1. **register 不同。** EC 的比值样本大量来自散文段落；Alien 侧现有的中文
   只有系统 UI 的 349 条标签（`lang/cn.json`），标签的压缩比和散文差一大截，
   两者不能互相当阈值用。

2. **分母不同 —— 这条最致命。** 本语料的待译体量绝大部分是 journal 页的
   **HTML 富文本**（151 页里 59 页有 `text.content`，共 2,230,280 字符）。
   拿「原始串长」当分母时，标签、属性、`@UUID[...]{...}` 这些**不译的字节**
   全算进去了。本脚本第三段实测（2026-08-29 重测，用 `extract/mappings.mjs`
   的 project 层抽出的 **5023 条叶 / 2,705,264 原始字符**；此前一版数字
   3401 叶是 Babele 默认映射的产物，已作废）：
     · 整包 可见/原始  78.0% / 58.7% / 49.3%（系统 / 新手包 / 核心书）
     · journal 页正文 56 条长叶：中位 52.5%，**最低 13.0%**，最高 83.8%
     · RollTable 结果描述 30 条长叶更极端：中位 12.0%，最低 6.8%
   长叶整体（n=217）min 6.8% / max 100.0%——同一段译文按两种分母算，
   比值最多差 **14.7 倍**。任何阈值都必须说清自己的分母是哪一个。

3. **样本量。** 349 条不足以定一条会用来判「截断」的硬闸。

⇒ **本脚本不写死任何阈值，也不引入 EC 的任何数。** 它只把本语料测到的
分布印出来，给一条标注为 PROVISIONAL 的带宽。

Phase-2 退出项（必须完成才能把任何比值闸接上）
----------------------------------------------
    [ ] 真中文语料落地后（`compendium/cn/*.json` 有实质译文），
        跑 `python measure_ratio.py --only corpus` 重测，
        用**可见字符**作分母，按 register（name / 短字段 / 散文）分桶给带宽，
        再决定 TRUNCATED 判据。在此之前任何比值闸都**不得启用**。
"""
from __future__ import annotations
import argparse
import html
import io
import json
import os
import re
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
SYSTEM_LANG = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\alienrpg\lang"
REPOS = ["1-系统汉化插件", "2-新手包汉化插件", "3-核心书汉化插件"]

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TAG = re.compile(r"<[^>]+>")
ENRICHER = re.compile(r"@(?:UUID|TEXTDRAW|DRAW|Compendium|Item|JournalEntry)\[[^\]]*\]"
                      r"(?:\{([^}]*)\})?")
WS = re.compile(r"\s+")


def load(p):
    return json.load(io.open(p, encoding="utf-8-sig"))


def flatten(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            flatten(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            flatten(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def visible(s):
    """Characters a player actually reads: enricher labels kept, ids/tags dropped."""
    s = ENRICHER.sub(lambda m: m.group(1) or "", s)
    s = TAG.sub(" ", s)
    s = html.unescape(s)
    return WS.sub(" ", s).strip()


def describe(values):
    if not values:
        return None
    v = sorted(values)
    n = len(v)

    def pct(q):
        if n == 1:
            return v[0]
        i = q * (n - 1)
        lo, hi = int(i), min(int(i) + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)

    return {
        "n": n,
        "min": round(v[0], 4),
        "p05": round(pct(0.05), 4),
        "p10": round(pct(0.10), 4),
        "p25": round(pct(0.25), 4),
        "median": round(statistics.median(v), 4),
        "p75": round(pct(0.75), 4),
        "p90": round(pct(0.90), 4),
        "p95": round(pct(0.95), 4),
        "max": round(v[-1], 4),
        "mean": round(statistics.fmean(v), 4),
    }


def table(title, d):
    if not d:
        print(f"  {title}: (no samples)")
        return
    print(f"  {title}  n={d['n']}")
    print(f"    min {d['min']:.3f}   p05 {d['p05']:.3f}   p10 {d['p10']:.3f}"
          f"   p25 {d['p25']:.3f}   median {d['median']:.3f}")
    print(f"    p75 {d['p75']:.3f}   p90 {d['p90']:.3f}   p95 {d['p95']:.3f}"
          f"   max {d['max']:.3f}   mean {d['mean']:.3f}")


# --------------------------------------------------------------- section 1
def section_lang(report):
    print("=" * 74)
    print("[1] 系统 lang/cn.json 已有中文  ——  唯一现存的真中文样本")
    print("=" * 74)
    en = {}
    cn = {}
    flatten(load(os.path.join(SYSTEM_LANG, "en.json")), [], en)
    flatten(load(os.path.join(SYSTEM_LANG, "cn.json")), [], cn)
    raw_cn = load(os.path.join(SYSTEM_LANG, "cn.json"))
    flat_all = {}

    def flat_any(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                flat_any(v, path + [k])
        else:
            flat_all[".".join(path)] = node
    flat_any(raw_cn, [])

    pairs = []
    for k, c in cn.items():
        if not CJK.search(c):
            continue
        e = en.get(k)
        if not isinstance(e, str) or not e.strip():
            continue
        pairs.append((k, e, c))

    ratios = [len(c) / len(e) for _, e, c in pairs]
    d = describe(ratios)

    # register buckets: UI labels are overwhelmingly short. Split so the band is
    # not silently dominated by 2-word button captions.
    short = [len(c) / len(e) for _, e, c in pairs if len(e) <= 20]
    mid = [len(c) / len(e) for _, e, c in pairs if 20 < len(e) <= 60]
    long_ = [len(c) / len(e) for _, e, c in pairs if len(e) > 60]

    print(f"  en.json 叶 {len(en)}   cn.json 叶 {len(flat_all)}"
          f"   （其中 JSON null {sum(1 for v in flat_all.values() if v is None)}，"
          f"回落英文 {len(en) - len(cn)} 键）")
    print(f"  含 CJK 且英文侧非空的配对: {len(pairs)}")
    print()
    table("全部", d)
    print()
    table("英文 <= 20 字符（按钮/标签）", describe(short))
    table("英文 21-60 字符", describe(mid))
    table("英文 > 60 字符（唯一接近散文的一档）", describe(long_))

    report["lang"] = {
        "source": os.path.join(SYSTEM_LANG, "{en,cn}.json"),
        "en_leaves": len(en), "cn_leaves": len(flat_all),
        "pairs": len(pairs),
        "all": d,
        "by_en_length": {"le20": describe(short), "21_60": describe(mid), "gt60": describe(long_)},
        "denominator": "raw characters (lang values carry no HTML)",
    }
    return d, describe(long_)


# --------------------------------------------------------------- section 2
def section_corpus(report):
    print()
    print("=" * 74)
    print("[2] compendium 语料 CN/EN  ——  Phase-2 才有数")
    print("=" * 74)
    pairs = []
    seen = 0
    for repo in REPOS:
        en_dir = os.path.join(P, repo, "compendium", "en")
        cn_dir = os.path.join(P, repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            continue
        for f in sorted(os.listdir(en_dir)):
            if not f.endswith(".json") or f == "_source.json":
                continue
            seen += 1
            cnp = os.path.join(cn_dir, f)
            if not os.path.exists(cnp):
                continue
            e, c = {}, {}
            flatten(load(os.path.join(en_dir, f)).get("entries", {}), [], e)
            flatten(load(cnp).get("entries", {}), [], c)
            for k, cv in c.items():
                ev = e.get(k)
                if isinstance(ev, str) and ev.strip() and CJK.search(cv):
                    pairs.append((k, ev, cv))

    print(f"  英文包文件 {seen}，其中有对应中文包的贡献了 {len(pairs)} 组配对")
    if not pairs:
        print("  -> 中文语料尚未落地。这一段是 Phase-2 的退出项，见文件头。")
        report["corpus"] = {"pairs": 0, "raw": None, "visible": None,
                            "status": "empty - phase-2 exit item"}
        return None
    raw = describe([len(c) / len(e) for _, e, c in pairs])
    vis = describe([len(visible(c)) / max(1, len(visible(e))) for _, e, c in pairs])
    table("按原始字符", raw)
    table("按可见字符", vis)
    report["corpus"] = {"pairs": len(pairs), "raw": raw, "visible": vis, "status": "measured"}
    return vis


# --------------------------------------------------------------- section 3
def section_visible(report):
    print()
    print("=" * 74)
    print("[3] 英文语料 可见/原始  ——  比值闸的分母之争，实测")
    print("=" * 74)
    per_pack = []
    page_ratios = []
    by_kind = {"journal page text": [], "table result description": [], "other long leaves": []}
    for repo in REPOS:
        en_dir = os.path.join(P, repo, "compendium", "en")
        if not os.path.isdir(en_dir):
            continue
        for f in sorted(os.listdir(en_dir)):
            if not f.endswith(".json") or f == "_source.json":
                continue
            leaves = {}
            flatten(load(os.path.join(en_dir, f)).get("entries", {}), [], leaves)
            raw = sum(len(v) for v in leaves.values())
            vis = sum(len(visible(v)) for v in leaves.values())
            longs = [(len(visible(v)) / len(v)) for v in leaves.values() if len(v) >= 500]
            page_ratios.extend(longs)
            for k, v in leaves.items():
                if len(v) < 500:
                    continue
                r = len(visible(v)) / len(v)
                if ".pages." in k and k.endswith(".text"):
                    by_kind["journal page text"].append(r)
                elif ".results." in k and k.endswith(".description"):
                    by_kind["table result description"].append(r)
                else:
                    by_kind["other long leaves"].append(r)
            per_pack.append({"file": f, "leaves": len(leaves), "raw_chars": raw,
                             "visible_chars": vis,
                             "visible_of_raw": round(vis / raw, 4) if raw else None,
                             "long_leaves": len(longs)})
            print(f"  {f}")
            print(f"    叶 {len(leaves)}   原始 {raw}   可见 {vis}"
                  f"   可见/原始 {vis / raw:.1%}" if raw else "")
    d = describe(page_ratios)
    print()
    print(f"  长叶（>=500 原始字符，主要就是 journal 页正文）逐叶 可见/原始:")
    table("", d)
    if d:
        print(f"    -> 最低 {d['min']:.1%}，最高 {d['max']:.1%}。"
              f"同一段译文按两种分母算，比值差 {d['max'] / d['min']:.1f} 倍。")
    print()
    print("  按字段种类拆开（长叶）:")
    kinds = {}
    for name, vals in by_kind.items():
        kd = describe(vals)
        kinds[name] = kd
        table(f"  {name}", kd)
    report["visible_of_raw"] = {"per_pack": per_pack, "long_leaf_distribution": d,
                                "by_kind": kinds}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lang", "corpus", "visible"], action="append", default=[])
    ap.add_argument("--out", default=os.path.join(P, "7-其他内容", "reports", "measure_ratio.json"))
    a = ap.parse_args()
    want = set(a.only) if a.only else {"lang", "corpus", "visible"}

    report = {"measuredAt": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
              "measuredBy": "4-常用脚本/qa/measure_ratio.py",
              "refusedImports": {
                  "note": "EC 的三个常数一个都没有被引用，它们只在这里作为反例出现。",
                  "ec_median": 0.31, "ec_truncated": 0.22, "ec_catchup_floor": 0.25,
                  "why_not": "register 不同（EC 散文 vs 本项目 UI 标签）；"
                             "分母不同（本语料是 HTML 富文本，可见/原始最低 13%）；"
                             "样本量 349 不足以定截断闸。"}}

    lang_all = lang_long = corpus_vis = None
    if "lang" in want:
        lang_all, lang_long = section_lang(report)
    if "corpus" in want:
        corpus_vis = section_corpus(report)
    if "visible" in want:
        section_visible(report)

    print()
    print("=" * 74)
    print("PROVISIONAL BAND  （临时带宽，不是闸门阈值）")
    print("=" * 74)
    if lang_all:
        band = (lang_all["p05"], lang_all["p95"])
        print(f"  来源 : 系统 lang/cn.json 的 {lang_all['n']} 条 CJK 配对，分母＝原始字符")
        print(f"  中位 : {lang_all['median']:.3f}")
        print(f"  带宽 : [{band[0]:.3f}, {band[1]:.3f}]  (p05..p95)")
        if lang_long and lang_long["n"]:
            print(f"  仅看最接近散文的一档（英文 >60 字符，n={lang_long['n']}）: "
                  f"中位 {lang_long['median']:.3f}，带宽 "
                  f"[{lang_long['p05']:.3f}, {lang_long['p95']:.3f}]")
        report["provisional_band"] = {
            "source": "system lang/cn.json CJK pairs",
            "denominator": "raw characters",
            "n": lang_all["n"],
            "median": lang_all["median"],
            "band_p05_p95": [band[0], band[1]],
            "prose_proxy": lang_long,
            "status": "PROVISIONAL - do not wire a gate to this",
        }
    print()
    print("  ⚠ 这条带宽**不得**接成闸门：样本是 UI 标签，分母是原始字符，"
          "而真正要判的是 HTML 散文。")
    print("  ⚠ EC 的 0.31 / 0.22 / 0.25 本脚本一个都没用，也不要用。")
    print()
    print("PHASE-2 EXIT ITEM")
    print("  [ ] 真中文语料落地后跑 `measure_ratio.py --only corpus`，"
          "用**可见字符**作分母，")
    print("      按 register 分桶重测，再决定是否设 TRUNCATED 判据。"
          "在此之前比值闸保持关闭。")
    if corpus_vis is None:
        print("  当前状态: 未满足（compendium/cn 尚无 CJK 配对）")
    report["phase2_exit_item"] = {
        "id": "measure-ratio-on-real-cn-corpus",
        "satisfied": bool(corpus_vis),
        "requirement": "re-measure CN/EN on compendium/cn with VISIBLE characters as the "
                       "denominator, bucketed by register, before enabling any ratio gate",
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
