# -*- coding: utf-8 -*-
"""probe_twin_sensitivity 的精确版：直接调 a_twin_files，不靠 grep 主输出。"""
import io, os, shutil, sys

PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
QA = os.path.join(PROJ, "3-常用脚本", "qa")
OUT = os.path.join(PROJ, "4-临时脚本", "2026-08-16-round24")
FAKE = os.path.join(OUT, "fakeroot2")
sys.path.insert(0, QA)

A_REL = os.path.join("1-Ember汉化插件", "scripts", "ember-cn-selfcheck.mjs")
B_REL = os.path.join("2-Crucible汉化插件", "selfcheck", "cn-selfcheck.mjs")

if os.path.isdir(FAKE):
    shutil.rmtree(FAKE)
for rel in (A_REL, B_REL):
    dst = os.path.join(FAKE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(PROJ, rel), dst)
with io.open(os.path.join(FAKE, B_REL), "a", encoding="utf-8") as f:
    f.write("\n// ROUND24 注入漂移（只在副本里）\n")

import assert_resolutions as A

rule = None
import json
rules = json.load(io.open(A.DEFAULT_RULES, encoding="utf-8"))
for r in rules["assertions"]:
    if r["id"] == "R-selfcheck-twin":
        rule = r
print("规则里的 repo_a =", repr(rule["pairs"][0]["repo_a"]))
print("REPOS 的键     =", list(A.REPOS.keys()))
print("=> ctx.repos.get(repo_a) 能否取到 =",
      rule["pairs"][0]["repo_a"] in A.REPOS, "（False 即退化成裸相对路径，按 cwd 解析）")
print()


class FakeCtx:
    pass


for label, repos in [
    ("ctx.repos 按真实实现（键=ember/crucible）", {"ember": os.path.join(FAKE, "1-Ember汉化插件"),
                                                "crucible": os.path.join(FAKE, "2-Crucible汉化插件")}),
]:
    c = FakeCtx(); c.repos = repos
    os.chdir(PROJ)   # cwd = 真实项目根，副本在 FAKE
    bad, detail = A.a_twin_files(rule, c)
    print("%s\n   cwd=项目根, --root=副本(已注入漂移)" % label)
    print("   detail =", detail)
    print("   violations =", len(bad), bad if bad else "(无 —— 断言没响)")
    print("   结论 =", "灵敏度回测有效" if bad else
          "**空转**：漂移在副本里，判据比的是真实树")

shutil.rmtree(FAKE)
