# -*- coding: utf-8 -*-
"""ATTRIBUTION: is the current gate/selftest red caused by MY unit or another one?

The main gate came back 67/2 BEFORE this unit touched anything, and --selftest
354/357. Both point at R-patterns-translate-cases ("PATTERNS 28 -> 29").
That table lives in 1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs, which this
unit is forbidden to touch and which `git status` shows as modified by another
unit in flight.

This script proves the attribution mechanically instead of asserting it:
it runs the SAME node executor the gate runs (translate_cases_runner.mjs) with
the SAME case list from RESOLUTIONS.assertions.json, once against the WORKING
TREE copy of that file and once against its `git show HEAD:` copy.

SELF-PROOF, two separate assertions printed before any conclusion:
  A. COUNT   : the rule pulled from the assertions file must have exactly the
               case counts the gate reports (negative 66 / positive 81 /
               notify_negative 45 / notify_positive 32) -> right rule, right file.
  B. IDENTITY: the two source files must actually differ, and the HEAD copy must
               contain the stub_import line the runner needs -> we really are
               comparing two different revisions of the right file, not the same
               bytes twice.
"""
from __future__ import annotations
import json, io, os, subprocess, sys, tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPO = os.path.join(P, "1-Ember汉化插件")
SRC = os.path.join(REPO, "scripts", "ember-hardcoded-cn.mjs")
RUNNER = os.path.join(P, "3-常用脚本", "qa", "translate_cases_runner.mjs")
RULES = os.path.join(P, "5-其他内容", "RESOLUTIONS.assertions.json")
OUT = os.path.join(P, "4-临时脚本", "2026-08-22-ui-round3")

rule = next(r for r in json.load(io.open(RULES, encoding="utf-8"))["assertions"]
            if r["id"] == "R-patterns-translate-cases")

EXPECT = {"negative": 66, "positive": 81, "notify_negative": 45, "notify_positive": 32}
got = {k: len(rule.get(k, [])) for k in EXPECT}
A_ok = got == EXPECT
print(f"A-case-counts  {got}  expected {EXPECT}  -> {'OK' if A_ok else 'FAIL'}")

head = subprocess.run(["git", "show", "HEAD:scripts/ember-hardcoded-cn.mjs"],
                      cwd=REPO, capture_output=True)
head_src = head.stdout.decode("utf-8")
work_src = io.open(SRC, encoding="utf-8").read()
B_ok = (head_src != work_src) and (rule["stub_import"] in head_src) and (rule["stub_import"] in work_src)
print(f"B-two-revisions  HEAD {len(head_src)} chars vs worktree {len(work_src)} chars, "
      f"differ={head_src != work_src}, stub_import present in both="
      f"{rule['stub_import'] in head_src and rule['stub_import'] in work_src}  -> {'OK' if B_ok else 'FAIL'}")
if not (A_ok and B_ok):
    print("SELF-PROOF FAILED - not drawing any conclusion")
    sys.exit(2)

tmp = tempfile.mkdtemp(prefix="attribute.")
head_path = os.path.join(tmp, "ember-hardcoded-cn.HEAD.mjs")
io.open(head_path, "w", encoding="utf-8", newline="").write(head_src)

summary = {}
for label, src in (("worktree(other unit's in-flight edit)", SRC), ("HEAD(last committed)", head_path)):
    d = tempfile.mkdtemp(prefix="tc.")
    spec = {
        "src": src,
        "ember_mjs": os.path.join(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember", "scripts", "ember.mjs"),
        "harness": os.path.join(d, "_harness.mjs"),
        "out": os.path.join(d, "result.json"),
        "stub_import": rule["stub_import"],
        "negative": rule.get("negative", []),
        "positive": rule.get("positive", []),
        "notify_negative": rule.get("notify_negative", []),
        "notify_positive": rule.get("notify_positive", []),
        "coverage": rule.get("coverage") or {},
        "arrangements": rule.get("arrangements"),
    }
    sp = os.path.join(d, "spec.json")
    json.dump(spec, io.open(sp, "w", encoding="utf-8"), ensure_ascii=False)
    pr = subprocess.run(["node", RUNNER, sp], capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if pr.returncode != 0 or not os.path.exists(spec["out"]):
        print(f"[{label}] runner exit {pr.returncode}: {(pr.stderr or pr.stdout)[-300:]}")
        summary[label] = {"runner_failed": True}
        continue
    res = json.load(io.open(spec["out"], encoding="utf-8"))
    viol = res.get("violations", [])
    counts = res.get("counts", {})
    print(f"[{label}] violations={len(viol)}  patterns_size={counts.get('patterns_size')} "
          f"prefixed_size={counts.get('prefixed_size')} np_size={counts.get('np_size')}")
    for v in viol[:6]:
        print("    ", str(v)[:220])
    summary[label] = {"violations": len(viol), "counts": counts, "sample": viol[:6]}

io.open(os.path.join(OUT, "attribute.json"), "w", encoding="utf-8").write(
    json.dumps(summary, ensure_ascii=False, indent=2))
print("-> attribute.json")
