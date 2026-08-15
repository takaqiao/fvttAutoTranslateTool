# -*- coding: utf-8 -*-
"""缺包告警的**双向**回测：删掉基准里一个包必须告警，补回必须不告警。

    python backtest_coverage_warning.py --work <临时目录>

为什么要双向
------------
单向只证明「告警能亮」，不能排除「它一直亮着」（第 (d) 型空转的孪生形态：
豁免表空了 / 告警恒亮，两者在报告上都看不出来）。所以每个脚本跑三态：

    A 全包        → 缺 0，不出 ⚠，--strict-coverage 退出码 0
    B 删掉一个包  → 缺 1，⚠ 里点名该包并给出它的中文叶数，--strict-coverage 退出码 3
    C 把包补回来  → 与 A 逐字相同（证明告警不是粘住的）

三个脚本 × 两个仓 × 三态 = 18 例。
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
BIND = P + "/4-临时脚本/2026-08-15-round18/bind3.json"

TOOLS = ["scan_dropped_terms.py", "scan_number_drift.py", "scan_marker_followup.py"]

TARGETS = [
    # (仓, 全包基准, 故意删掉的包)
    ("1-Ember汉化插件", "5-其他内容/english-baseline/ember-0.6.0-preupgrade-2026-08-15",
     "ember.crucible-effects.json"),
    ("2-Crucible汉化插件", "5-其他内容/english-baseline/crucible-0.10.1-preupgrade-2026-08-15",
     "crucible.rules.json"),
]


def run(tool, repo, baseline, out, strict):
    cmd = [sys.executable, f"{QA}/{tool}", "--repo", f"{P}/{repo}",
           "--baseline", baseline, "--out", out]
    if tool == "scan_dropped_terms.py":
        cmd += ["--bindings", BIND, "--show", "0"]
    if strict:
        cmd += ["--strict-coverage"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    cov = json.load(open(out, encoding="utf-8"))["meta"]["coverage"]
    return r.returncode, r.stdout, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    shutil.rmtree(a.work, ignore_errors=True)
    os.makedirs(a.work, exist_ok=True)

    results = []
    for repo, base_rel, victim in TARGETS:
        work_base = os.path.join(a.work, os.path.basename(base_rel))
        shutil.copytree(os.path.join(P, base_rel), work_base)
        stash = os.path.join(a.work, "stash_" + victim)
        out = os.path.join(a.work, "r.json")
        for tool in TOOLS:
            # --- A 全包 ---
            rc, so, cov = run(tool, repo, work_base, out, strict=True)
            okA = (not cov["missing_packs"] and rc == 0
                   and "基准缺 0 个" in so and "本闸一条也没看" not in so)
            nA = len(cov["scanned_packs"])
            results.append((f"{repo}|{tool}|A 全包", "静默", "静默" if okA else "**异常**",
                            f"扫 {nA} 包 / 缺 0 / 退出码 {rc}"))
            # --- B 删掉一个包 ---
            shutil.move(os.path.join(work_base, victim), stash)
            rc, so, cov = run(tool, repo, work_base, out, strict=True)
            miss = {m["pack"]: m for m in cov["missing_packs"]}
            okB = (victim in miss and rc == 3 and "基准缺 1 个" in so
                   and victim in so and cov["cn_cjk_leaves_uncovered"] > 0
                   and len(cov["scanned_packs"]) == nA - 1)
            results.append((f"{repo}|{tool}|B 删 {victim}", "告警",
                            "告警" if okB else "**没告警**",
                            f"扫 {len(cov['scanned_packs'])} 包 / 缺 {len(cov['missing_packs'])}"
                            f" / 未进闸中文叶 {cov['cn_cjk_leaves_uncovered']} / 退出码 {rc}"))
            # --- C 补回来 ---
            shutil.move(stash, os.path.join(work_base, victim))
            rc, so, cov = run(tool, repo, work_base, out, strict=True)
            okC = (not cov["missing_packs"] and rc == 0
                   and "基准缺 0 个" in so and "本闸一条也没看" not in so
                   and len(cov["scanned_packs"]) == nA)
            results.append((f"{repo}|{tool}|C 补回", "静默",
                            "静默" if okC else "**还在告警**",
                            f"扫 {len(cov['scanned_packs'])} 包 / 缺 0 / 退出码 {rc}"))

    bad = 0
    for name, expect, got, detail in results:
        ok = "PASS" if ("**" not in got) else "**FAIL**"
        bad += ok != "PASS"
        print(f"{ok:8} {name:62} 期望={expect:4} 实得={got:6}  {detail}")
    print(f"\n{len(results) - bad}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
