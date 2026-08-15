# -*- coding: utf-8 -*-
"""缺包告警的**双向**回测 + **探针自证** —— 针对第二十轮新装闸的两条脚本：
`scan_en_drift.py` 与 `scan_renamed_terms.py`。

    python backtest_coverage_warning_y5b.py --work <临时目录> [--tools ...] [--repos ...]

与第十九轮 `backtest_coverage_warning.py` 的关系
------------------------------------------------
判据完全同源（A/B/C 三态），只是换成本轮这两条脚本，并**多加一段探针自证**。

为什么要双向（A/B/C）
----------------------
单向只证明「告警能亮」，不能排除「它一直亮着」：

    A 全包        → 缺 0，不出 ⚠，--strict-coverage 退出码 0
    B 删掉一个包  → 缺 1，⚠ 里点名该包并给出它的中文叶数，--strict-coverage 退出码 3
    C 把包补回来  → 与 A 一致（证明告警不是粘住的）

为什么还要**探针自证**（第十九轮第 6 条教训）
----------------------------------------------
「验证工具本身也会空转」：第十九轮用 bash heredoc 写的探针，heredoc 把反斜杠吃了、
被测正则根本没被弄坏，探针却假绿地报「全过」。所以本脚本的 D 段反过来验自己：

    D 变异体      → 把被测脚本里「判定缺包」的那一行改成恒不缺包，再跑 B 态。
                    此时回测**必须报 FAIL**。若它仍报 PASS，说明 B 态的判据是死的。

变异是在 Python 里对源码字符串做替换，并**断言替换次数恰为 1**（替换没发生就当场
报错退出）—— 这正是第十九轮栽的那一跤：变异没生效而探针不知道。
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
QA = P + "/3-常用脚本/qa"

# 每条工具：基准参数名 + 让它闭嘴的参数 + 「判定缺包」那一行的变异
TOOLS = {
    "scan_en_drift.py": {
        "baseline_flag": "--baseline",
        "quiet": ["--top", "0"],
        # coverage() 里认包的那一行；改成恒真 = 永远不缺包
        "mutate": ("        if p and os.path.exists(p):\n",
                   "        if True:  # MUTANT\n"),
    },
    "scan_renamed_terms.py": {
        "baseline_flag": "--old",
        # `--mode en-tail` 纯粹为了跑得完：`cn-term` 在 ember 上是 O(候选名 × 3.5 万
        # 中文叶)，一次十几分钟，三态 × 两仓跑不动。覆盖数是在两个探测器**之前**
        # 算的、与 --mode 无关（coverage() 只看 old_packs 与仓里的包），所以这不削弱
        # 本回测 —— 被验的就是覆盖行与退出码。
        "quiet": ["--show", "0", "--mode", "en-tail"],
        "mutate": ("        if f in old_packs:\n",
                   "        if True:  # MUTANT\n"),
    },
}

TARGETS = [
    # (仓, 全包基准, 故意删掉的包)
    ("1-Ember汉化插件", "5-其他内容/english-baseline/ember-0.6.0-preupgrade-2026-08-15",
     "ember.crucible-effects.json"),
    ("2-Crucible汉化插件", "5-其他内容/english-baseline/crucible-0.10.1-preupgrade-2026-08-15",
     "crucible.rules.json"),
]


def run(tool_path, spec, repo, baseline, out, strict):
    cmd = [sys.executable, tool_path, "--repo", f"{P}/{repo}",
           spec["baseline_flag"], baseline, "--out", out] + spec["quiet"]
    if strict:
        cmd += ["--strict-coverage"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if not os.path.exists(out):
        raise SystemExit(f"!! {tool_path} 没写出 {out}\n{r.stdout}\n{r.stderr}")
    cov = json.load(open(out, encoding="utf-8"))["meta"]["coverage"]
    return r.returncode, r.stdout + r.stderr, cov


def make_mutant(tool, spec, work):
    """把被测脚本复制一份并弄坏它的缺包判定；**断言替换真的发生了**。"""
    src = open(f"{QA}/{tool}", encoding="utf-8").read()
    old, new = spec["mutate"]
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"!! 变异锚点在 {tool} 里出现 {n} 次（要求恰好 1 次）"
                         f"—— 变异没生效的探针等于没验，直接停。")
    dst = os.path.join(work, "MUTANT_" + tool)
    open(dst, "w", encoding="utf-8").write(src.replace(old, new))
    return dst


def check_A_or_C(rc, so, cov, expect_n):
    return (not cov["missing_packs"] and rc == 0
            and "基准缺 0 个" in so and "一条也没看" not in so
            and len(cov["scanned_packs"]) == expect_n)


def check_B(rc, so, cov, victim, expect_n):
    miss = {m["pack"]: m for m in cov["missing_packs"]}
    return (victim in miss and rc == 3 and "基准缺 1 个" in so
            and victim in so and cov["cn_cjk_leaves_uncovered"] > 0
            and len(cov["scanned_packs"]) == expect_n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tools", nargs="*", default=sorted(TOOLS))
    ap.add_argument("--repos", nargs="*", default=[t[0] for t in TARGETS])
    a = ap.parse_args()
    shutil.rmtree(a.work, ignore_errors=True)
    os.makedirs(a.work, exist_ok=True)

    results = []
    for repo, base_rel, victim in TARGETS:
        if repo not in a.repos:
            continue
        work_base = os.path.join(a.work, os.path.basename(base_rel) + "-" + repo)
        shutil.copytree(os.path.join(P, base_rel), work_base)
        stash = os.path.join(a.work, "stash_" + victim)
        out = os.path.join(a.work, "r.json")
        for tool in a.tools:
            spec = TOOLS[tool]
            path = f"{QA}/{tool}"
            # --- A 全包 ---
            rc, so, cov = run(path, spec, repo, work_base, out, True)
            nA = len(cov["scanned_packs"])
            okA = check_A_or_C(rc, so, cov, nA) and nA == cov["repo_en_packs"]
            results.append((f"{repo}|{tool}|A 全包", "静默",
                            "静默" if okA else "**异常**",
                            f"扫 {nA} 包 / 仓里 {cov['repo_en_packs']} / 缺 0 / 退出码 {rc}"))
            # --- B 删掉一个包 ---
            shutil.move(os.path.join(work_base, victim), stash)
            rc, so, cov = run(path, spec, repo, work_base, out, True)
            okB = check_B(rc, so, cov, victim, nA)
            results.append((f"{repo}|{tool}|B 删 {victim}", "告警",
                            "告警" if okB else "**没告警**",
                            f"扫 {len(cov['scanned_packs'])} 包 / 缺 {len(cov['missing_packs'])}"
                            f" / 未进闸中文叶 {cov['cn_cjk_leaves_uncovered']} / 退出码 {rc}"))
            # --- D 探针自证：同一个 B 态，跑弄坏了缺包判定的变异体 ---
            mut = make_mutant(tool, spec, a.work)
            rc_m, so_m, cov_m = run(mut, spec, repo, work_base, out, True)
            okD = not check_B(rc_m, so_m, cov_m, victim, nA)   # 必须**测不出**
            results.append((f"{repo}|{tool}|D 变异体（应被判 FAIL）", "测不出",
                            "测不出" if okD else "**探针是死的**",
                            f"变异体：缺 {len(cov_m['missing_packs'])} / 退出码 {rc_m}"))
            # --- C 补回来 ---
            shutil.move(stash, os.path.join(work_base, victim))
            rc, so, cov = run(path, spec, repo, work_base, out, True)
            okC = check_A_or_C(rc, so, cov, nA)
            results.append((f"{repo}|{tool}|C 补回", "静默",
                            "静默" if okC else "**还在告警**",
                            f"扫 {len(cov['scanned_packs'])} 包 / 缺 0 / 退出码 {rc}"))

    bad = 0
    for name, expect, got, detail in results:
        ok = "PASS" if ("**" not in got) else "**FAIL**"
        bad += ok != "PASS"
        print(f"{ok:8} {name:66} 期望={expect:6} 实得={got:8}  {detail}")
    print(f"\n{len(results) - bad}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
