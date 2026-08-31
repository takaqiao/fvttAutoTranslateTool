# -*- coding: utf-8 -*-
"""硬标记跟进闸：上游改了**标记**，中文没跟着换 —— 尤其是「中文还留着旧英文那一版标记」。

    python scan_marker_followup.py --repo <repo> --baseline <旧英文基准目录>
                                   [--out <json>] [--md <md>] [--show 30] [--include-extra]
                                   [--strict-coverage]

⚠ 基准缺包 = 静默不扫（第十九轮 Y5）
------------------------------------
本闸只扫**基准目录里存在**的包。基准缺哪个包，那个包就一条也不看 —— 而报告长得
和「扫过、干净」一模一样。实测 2026-08-15：ember 侧基准
`ember-cn-v1.0.15-shipped-en/` 只有 **3** 个包，仓里 10 个 en 包 / 9 个有中文，
另外 **6 个包 15 147 条有中文的叶**从未进过本闸。
现在每次运行都会印「本次扫 N 个包 / 仓里共 M 个包 / 缺 K 个」以及缺掉的中文叶数，
并逐包印进闸对数；`--strict-coverage` 让缺包直接以退出码 3 结束。

**这个洞补不了「过去」，只补得了「将来」**：基准是历史快照，那 6 个包发版当时就
没被捕获，「上游从那时到现在改了什么」对它们无法回答。所以第十九轮改为
**升级前先截全包快照**（`3-常用脚本/qa/capture_baseline.py`）：
`english-baseline/ember-0.6.0-preupgrade-2026-08-15/`（10 包）、
`crucible-0.10.1-preupgrade-2026-08-15/`（15 包）。
拿这两份跑**今天**恒为 0 告警（它们就是当前英文），那是对的、不是闸坏了；
它们的用途是下一次上游升级之后当「旧英文」。

和 `scan_markup_drift.py` 的分工（别把两者当重复）
--------------------------------------------------
`scan_markup_drift.py` 的 LINK 档比的是**当前英文 vs 中文的标记多重集**，
只要计数不等就报。它管得住「链接坏了」，但两件事它做不到：

1. **它没有旧英文，分不清成因。** 中文缺一个 `@UUID` 到底是「从没译到」还是
   「上游换了目标、中文停在旧目标上」，报告里长得一模一样。后者是本闸的
   `MARKER_STALE`：**该标记出现在旧英文里、不在新英文里、却还留在中文里** ——
   这是「中文没跟上上游改动」的**直接证据**，不是启发式。
2. **它只看 `@X[…]` / `[[…]]` / 块级与行内标签，不看属性。** 撑着 590 处目录锚点的
   `id="…"`、决定版式的 `class="…"`（`<section class="block readaloud">`）、
   上标 `<sup>` 全在它的视野之外 —— 而 `id` 一旦跟着变、中文没跟，跳转就断了。

所以本闸按**存在性**（0 ↔ 非 0）判，不按计数判；计数差异留给 `scan_markup_drift`。
两边都报到的条目属于故意的交叉验证，不算重复劳动。

判据
----
对**基准与当前英文里路径相同、且有中文**的每一条，先抹掉「本来就该译」的部分
（见下），再取三份标记集合：`OLD`（旧英文）/ `NEW`（当前英文）/ `CN`（中文）。

* `MARKER_STALE`   —— 标记 ∈ OLD、∉ NEW、∈ CN。**中文停在旧英文上**，最强的一档。
* `MARKER_MISSING` —— 标记 ∈ NEW、∉ CN。中文没跟上（也可能是从来没跟上）。
* `MARKER_EXTRA`   —— 标记 ∈ CN、∉ NEW、∉ OLD（译者自造）。`--include-extra` 才报。

管的标记种类
------------
* `@Word[…]`（`@UUID` / `@Embed` / `@Condition` / `@Check` / `@CriticalSuccess` …）
* `&Reference[…]`（dnd5e 侧）
* `[[…]]` 行内指令（`[[/r]]` / `[[/check]]` / `[[lookup …]]`）
* `id="…"`（标题锚点，590 处目录跳转靠它）
* `class="…"`（`<section class="block readaloud">` 这类版式）
* `<sup>` 的有无

已抹掉的「本来就该译」的部分（否则每翻一句就多报一条）
------------------------------------------------------
* 标记尾随的 `{标签}` —— 给玩家看的文字
* `label=` / `readaloud=` / `caption=` 参数值（带引号与不带引号两种写法）
* `[[/r 1d6#分身被摧毁]]` 里 `#` 之后的骰点说明

已知假阳性
----------
* `[[/item 战镐]]` —— dnd5e 按**中文**物品名解析，中文形式才是运行时存在的那个。
  这类会同时报一条 MISSING（英文名）和一条 EXTRA（中文名），属 by design，
  已按 `scan_markup_targets.py` 的 `BY_DESIGN_CMD` 白名单放行。
* 中文把两条相邻链接合并成一条（计数差）不会被本闸看见 —— 那是
  `scan_markup_drift` LINK 档的定义域，本闸只判有无。
* `class="…"` 里出现 Foundry 自动生成的随机后缀时两边必然不同 —— 本库未见，
  真出现时按 pack 整体豁免即可。

回测（2026-08-15，`4-临时脚本/2026-08-15-round16/qa/backtest_marker_followup.py`）
----------------------------------------------------------------------------
双向 6/6 PASS。灵敏度 3/3：上游把 `@UUID` 指向新文档（报 STALE + MISSING 一对）、
上游把标题锚点 `id="overview"` 改成 `id="basics"`（同上）、上游新增
`@Condition[exhaustion]`（报 MISSING）。
特异度 3/3 静默：只改 `{标签}`、只改 `readaloud="…"` 参数值、
`[[/item Warhammer]]` 对 `[[/item 战锤]]`。

当前库基线（2026-08-15）
------------------------
* crucible（基准 crucible-0.9.1-legacy）：英文变过的 295 条，标记 295/295 全一致，
  **STALE 0 / MISSING 0**。
* ember（基准 ember-cn-v1.0.15-shipped-en）：英文变过的 933 条，933/933 全一致，
  **STALE 0 / MISSING 0**。
* `--all-leaves --include-extra` 全库跑（ember 9214 条）另有 446 条 `MARKER_EXTRA`，
  全部是同一个已知良性形态：**我方本地补的目录锚点**（`id="lore"`、`id="culture"`，
  英文侧没有）。这一类别当缺陷看。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CJK = re.compile(r"[一-鿿]")

AT_MARKUP = re.compile(r"@[A-Za-z]+\[[^\]]*\]|&(?:amp;)?[A-Za-z]+\[[^\]]*\]")
INLINE_CMD = re.compile(r"\[\[[^\]]*\]\]")
ID_ATTR = re.compile(r'\bid\s*=\s*"([^"]*)"')
CLASS_ATTR = re.compile(r'\bclass\s*=\s*"([^"]*)"')
SUP = re.compile(r"<\s*sup\b", re.I)

# 与 scan_markup_drift.py / apply_translations.py 保持同一套「这是散文」的白名单
QUOTED_PARAM = re.compile(r'=\s*"[^"]*"')
BARE_TEXT_PARAM = re.compile(r'\b(label|readaloud|caption)\s*=\s*(?!["\'])([^\s\]<>"\']+)')
LABEL_AFTER = re.compile(
    r"(?:@[A-Za-z]+\[[^\]]*\]|&(?:amp;)?[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])\s*\{[^}]*\}")
# `[[/item 战锤]]` / `[[/r 1d6#说明]]`：dnd5e 按中文名解析、`#` 后是骰点说明，都算 by design
BY_DESIGN_CMD = re.compile(r"^\[\[/(?:item|r|roll|damage|check|save)\b")


def normalize(s: str) -> str:
    s = BARE_TEXT_PARAM.sub(r'\1="~"', s)
    s = QUOTED_PARAM.sub('="~"', s)
    return LABEL_AFTER.sub(lambda m: m.group(0).split("{")[0], s)


def markers(s: str) -> set[str]:
    """一条文本里的硬标记集合（**存在性**，不计次数）。"""
    z = normalize(s)
    out = set()
    for m in AT_MARKUP.findall(z):
        out.add("AT:" + m)
    for m in INLINE_CMD.findall(z):
        if BY_DESIGN_CMD.match(m):
            continue                      # dnd5e 按中文名解析，两边本来就不同
        out.add("CMD:" + re.sub(r"#.*$", "#~", m))
    for v in ID_ATTR.findall(s):
        out.add("ID:" + v.strip())
    for v in CLASS_ATTR.findall(s):
        out.add("CLASS:" + " ".join(sorted(v.split())))
    if SUP.search(s):
        out.add("TAG:sup")
    return out


# --------------------------------------------------------------------- 载入
def load_json(path):
    raw = open(path, encoding="utf-8-sig").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith(".json") or f == "_source.json":
            continue
        key = ("ember.crucible-adventure.json" if f == "_repaired.json"
               else f.replace("-en.json", ".json"))
        out.setdefault(key, os.path.join(bdir, f))
    return out


# ----------------------------------------------------------- 基准覆盖了几个包
def cjk_leaf_count(path):
    """一个 cn 包里**含中文**的叶数 —— 本闸的定义域就是这些叶。"""
    if not os.path.exists(path):
        return 0
    d = {}
    leaves(load_json(path).get("entries", {}), [], d)
    return sum(1 for v in d.values() if CJK.search(v))


def coverage(repo, baseline):
    """扫了几个包 / 仓里共几个包 / 基准缺几个包（连带缺掉多少条中文叶）。

    ⚠ **反空转第 (d) 型的探针**。前三种空转形态（判据写坏 / 压根不读库 /
    读旧报告快照）的防法对它全部无效：判据没问题、库也读了、报告是当场生成的 ——
    只是基准目录里**压根没有那个包**，`scan()` 里 `if not os.path.exists(cur_p):
    continue` 会**静默**跳过，报告上一片绿，实情是没扫。

    第十九轮实测：ember 侧基准 `ember-cn-v1.0.15-shipped-en/` 只装了 3 个包
    （`_repaired.json` + adversary + character），而仓里有 10 个 en 包 / 9 个有中文，
    另外 6 个包 **15 147 条有中文的叶**从未进过这三条闸 —— 三条闸却都报 0 告警。
    所以这三个数必须每次跟着结果一起印，不能只印告警数。

    补法在 `capture_baseline.py`（第十九轮）：**上游升级之前**从当前 `compendium/en`
    截一份全包快照，命名里写死上游版本号。已截：
    `english-baseline/ember-0.6.0-preupgrade-2026-08-15/`（10 包）与
    `english-baseline/crucible-0.10.1-preupgrade-2026-08-15/`（15 包）。
    """
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    bmap = baseline_packs(baseline)
    repo_packs = sorted(f for f in os.listdir(en_dir)
                        if f.endswith(".json") and f != "_source.json")
    scanned, missing = [], []
    for f in repo_packs:
        p = bmap.get(f)
        if p and os.path.exists(p):
            scanned.append(f)
        else:
            cnp = os.path.join(cn_dir, f)
            missing.append({"pack": f, "cn_present": os.path.exists(cnp),
                            "cn_cjk_leaves": cjk_leaf_count(cnp)})
    return {
        "baseline": os.path.abspath(baseline),
        "repo_en_packs": len(repo_packs),
        "scanned_packs": scanned,
        "missing_packs": missing,
        "cn_cjk_leaves_uncovered": sum(m["cn_cjk_leaves"] for m in missing),
        # 基准里有、仓里已经没有的包（上游删包 / 改名）。不影响本次结果，但要看得见。
        "baseline_only_packs": [f for f in sorted(bmap) if f not in repo_packs],
    }


def print_coverage(cov):
    print(f"  包覆盖：本次扫 {len(cov['scanned_packs'])} 个包 / 仓里 en 共 "
          f"{cov['repo_en_packs']} 个包 / 基准缺 {len(cov['missing_packs'])} 个")
    if cov["missing_packs"]:
        print("  ⚠ 下列包**基准里没有，本闸一条也没看** —— 它们的 0 告警不是「干净」，是「没扫」：")
        for m in cov["missing_packs"]:
            tail = "" if m["cn_present"] else "（无 cn 文件）"
            print(f"       {m['pack']}  含中文的叶 {m['cn_cjk_leaves']}{tail}")
        print(f"     合计未进闸的中文叶 {cov['cn_cjk_leaves_uncovered']}")
        print("     历史快照补不回来（那些包发版当时就没捕获）；将来靠 "
              "3-常用脚本/qa/capture_baseline.py 在**升级前**截全包基准。")
    if cov["baseline_only_packs"]:
        print(f"  ⚠ 基准里有、仓里 en 已没有的包 {len(cov['baseline_only_packs'])} 个："
              f"{'、'.join(cov['baseline_only_packs'])}")


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------- 主流程
def scan(repo, baseline, include_extra=False, changed_only=True, per_pack=None):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    findings = []
    stats = collections.Counter()

    for pack, oldpath in baseline_packs(baseline).items():
        cur_p = os.path.join(en_dir, pack)
        if not os.path.exists(cur_p):
            continue
        o, n, c = {}, {}, {}
        leaves(load_json(oldpath).get("entries", {}), [], o)
        leaves(load_json(cur_p).get("entries", {}), [], n)
        cnp = os.path.join(cn_dir, pack)
        if os.path.exists(cnp):
            leaves(load_json(cnp).get("entries", {}), [], c)
        if per_pack is not None:
            per_pack[pack] = {"baseline_leaves": len(o), "cur_en_leaves": len(n),
                              "cn_leaves": len(c), "pairs": 0, "findings": 0}

        for path, new_en in n.items():
            old_en = o.get(path)
            cn = c.get(path)
            if old_en is None or not cn or not CJK.search(cn):
                continue
            if changed_only and norm_ws(old_en) == norm_ws(new_en):
                continue
            stats["pairs"] += 1
            if per_pack is not None:
                per_pack[pack]["pairs"] += 1
            OLD, NEW, CN = markers(old_en), markers(new_en), markers(cn)
            stale = sorted((OLD - NEW) & CN)
            missing = sorted(NEW - CN)
            extra = sorted(CN - NEW - OLD)
            if not (stale or missing or (include_extra and extra)):
                stats["clean"] += 1
                continue
            for verdict, items in (("MARKER_STALE", stale), ("MARKER_MISSING", missing),
                                   ("MARKER_EXTRA", extra if include_extra else [])):
                if not items:
                    continue
                stats[verdict] += 1
                if per_pack is not None:
                    per_pack[pack]["findings"] += 1
                findings.append({
                    "verdict": verdict, "repo": os.path.basename(repo), "pack": pack,
                    "path": path, "markers": items,
                    "old_en": old_en[:300], "new_en": new_en[:300], "cn": cn[:300],
                })
    return findings, stats


def main():
    ap = argparse.ArgumentParser(description="硬标记跟进闸：英文标记变了、中文没跟上")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--include-extra", action="store_true",
                    help="连弱档 MARKER_EXTRA（译者自造、旧英文也没有）一起报")
    ap.add_argument("--all-leaves", action="store_true",
                    help="不限于「英文变过」的条目，全库比一遍（会大量撞上 scan_markup_drift）")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="基准缺包时以退出码 3 结束（默认只告警：回测脚本会拿单包小仓跑，"
                         "默认非零会把它们全打红）")
    a = ap.parse_args()

    cov = coverage(a.repo, a.baseline)
    per_pack = {}
    findings, stats = scan(a.repo, a.baseline, a.include_extra, not a.all_leaves, per_pack)
    print(f"{os.path.basename(a.repo)}  基准 {os.path.basename(a.baseline)}")
    print(f"  baseline = {cov['baseline']}")
    print_coverage(cov)
    for pk, d in per_pack.items():
        print(f"     · {pk}  基准叶 {d['baseline_leaves']}  当前英文叶 {d['cur_en_leaves']}"
              f"  中文叶 {d['cn_leaves']}  进闸对 {d['pairs']}  告警 {d['findings']}")
    print(f"  纳入比对的条目 {stats['pairs']}（{'仅英文变过的' if not a.all_leaves else '全部'}），"
          f"其中标记完全一致 {stats['clean']}")
    print(f"  **MARKER_STALE {stats['MARKER_STALE']}**  "
          f"MARKER_MISSING {stats['MARKER_MISSING']}  MARKER_EXTRA {stats['MARKER_EXTRA']}")
    for f in findings[:a.show]:
        print(f"  [{f['verdict']}] {f['pack']} :: {f['path'][-70:]}")
        for m in f["markers"][:4]:
            print(f"       {m[:150]}")
    # 报告里必须带覆盖口径：只写「告警 0」的报告，事后分不清是「干净」还是「没扫」。
    meta = {"tool": os.path.basename(__file__), "repo": os.path.abspath(a.repo),
            "coverage": cov, "per_pack": per_pack, "argv": sys.argv}
    if a.out:
        json.dump({"meta": meta, "stats": dict(stats), "findings": findings},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  -> {a.out}")
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(f"# 硬标记跟进（scan_marker_followup.py）— {os.path.basename(a.repo)}\n\n")
            fh.write(f"- baseline：`{cov['baseline']}`\n")
            fh.write(f"- 包覆盖：扫 {len(cov['scanned_packs'])} / 仓里 {cov['repo_en_packs']} / "
                     f"缺 {len(cov['missing_packs'])}"
                     f"（未进闸的中文叶 {cov['cn_cjk_leaves_uncovered']}）\n")
            fh.write(f"- 纳入比对 {stats['pairs']} 条，**告警 {len(findings)}**\n\n")
            for f in findings:
                fh.write(f"## `{f['pack']}` `{f['path']}`  ({f['verdict']})\n\n")
                for m in f["markers"]:
                    fh.write(f"- `{m}`\n")
                fh.write(f"\n- 旧英文：{f['old_en']}\n- 新英文：{f['new_en']}\n- 中文：{f['cn']}\n\n")
        print(f"  -> {a.md}")
    if a.strict_coverage and cov["missing_packs"]:
        print(f"✗ --strict-coverage：基准缺 {len(cov['missing_packs'])} 个包，"
              f"本次结果不构成全库结论。")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
