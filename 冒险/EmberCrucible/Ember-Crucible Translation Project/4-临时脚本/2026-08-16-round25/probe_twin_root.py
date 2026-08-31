# -*- coding: utf-8 -*-
"""落盘探针：用真正的 CLI `--root <副本>` 证明 R-selfcheck-twin 现在能做灵敏度回测。

造两棵副本树（干净 / 注入漂移），各跑一次 `assert_resolutions.py --root <树>`，
只看 R-selfcheck-twin 那一条的结论。**cwd 固定在项目根**——旧实现在那儿必然报绿。
"""
import os
import shutil
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCRIPT = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")
A = ("1-Ember汉化插件", os.path.join("scripts", "ember-cn-selfcheck.mjs"))
B = ("2-Crucible汉化插件", os.path.join("selfcheck", "cn-selfcheck.mjs"))


def build(dst, drift):
    for repo, rel in (A, B):
        src = os.path.join(ROOT, repo, rel)
        out = os.path.join(dst, repo, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(src, out)
    if drift:
        p = os.path.join(dst, B[0], B[1])
        with open(p, "ab") as fh:
            fh.write(b"\n// injected drift\n")


def run(tree, label):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, SCRIPT, "--root", tree, "--verbose"],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    lines = (r.stdout or "").splitlines()
    hit = []
    for i, l in enumerate(lines):
        if "R-selfcheck-twin" in l:
            hit.extend(lines[i:i + 6])
    print(f"=== {label} ===")
    for l in hit[:8]:
        print("   ", l.strip())
    if not hit:
        print("    !! 报告里一个字都没提 R-selfcheck-twin")
    print()


with tempfile.TemporaryDirectory() as tmp:
    clean = os.path.join(tmp, "clean")
    dirty = os.path.join(tmp, "dirty")
    build(clean, False)
    build(dirty, True)
    run(clean, "副本树·干净 —— 期望 ok")
    run(dirty, "副本树·注入漂移 —— 期望 FAIL（旧实现在这里是绿的）")
