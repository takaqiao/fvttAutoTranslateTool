# -*- coding: utf-8 -*-
"""
旁证：R-selfcheck-twin 的灵敏度回测是否真的作用在 `--root <副本>` 上。

线索：REPOS 的键是 "ember"/"crucible"（assert_resolutions.py:145），
     而规则里 repo_a/repo_b 写的是目录名 "1-Ember汉化插件"/"2-Crucible汉化插件"。
     a_twin_files 用 ctx.repos.get(pair["repo_a"], pair["repo_a"]) 取 —— **永远取不到**，
     每次都退化成裸目录名这个**相对路径**，于是按 cwd 解析。

本探针把两份 twin 复制到一棵副本树，**故意在副本里注入漂移**，再用 --root 指向副本跑，
看断言会不会响。不响 ⇒ 该断言无法做灵敏度回测（空转），当前的绿是 cwd 巧合。
落盘再跑。
"""
import io, os, shutil, subprocess, sys

PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(PROJ, "4-临时脚本", "2026-08-16-round24")
FAKE = os.path.join(OUT, "fakeroot")
QA = os.path.join(PROJ, "3-常用脚本", "qa", "assert_resolutions.py")

A_REL = os.path.join("1-Ember汉化插件", "scripts", "ember-cn-selfcheck.mjs")
B_REL = os.path.join("2-Crucible汉化插件", "selfcheck", "cn-selfcheck.mjs")

if os.path.isdir(FAKE):
    shutil.rmtree(FAKE)
for rel in (A_REL, B_REL):
    dst = os.path.join(FAKE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(PROJ, rel), dst)

# 注入漂移：只改副本的 b 侧
b = os.path.join(FAKE, B_REL)
with io.open(b, "a", encoding="utf-8") as f:
    f.write("\n// ROUND24 SENSITIVITY PROBE — 故意注入的漂移，只存在于副本树\n")

sa = os.path.getsize(os.path.join(FAKE, A_REL))
sb = os.path.getsize(b)
print("副本树:", FAKE)
print("副本 a 大小 =", sa, " 副本 b 大小 =", sb, " -> 副本里两份已不同 =", sa != sb)
print("真实树两份相同 =",
      open(os.path.join(PROJ, A_REL), "rb").read() == open(os.path.join(PROJ, B_REL), "rb").read())
print()

env = dict(os.environ, PYTHONIOENCODING="utf-8")
r = subprocess.run([sys.executable, QA, "--root", FAKE],
                   cwd=PROJ, capture_output=True, text=True, encoding="utf-8", env=env)
out = (r.stdout or "") + (r.stderr or "")
lines = [l for l in out.split("\n") if "selfcheck-twin" in l or "比对" in l]
print("--root 指向注入了漂移的副本树后，R-selfcheck-twin 的表现：")
for l in lines:
    print("   ", l.strip())
tail = [l for l in out.split("\n") if l.startswith("通过 ")]
print("   ", tail[0] if tail else "(没抓到汇总行)")
print()
fired = any("selfcheck-twin" in l and "FAIL" in l for l in out.split("\n"))
print("断言是否响 =", fired)
print("=> " + ("灵敏度回测有效" if fired else
      "**没响**：注入的漂移在副本里，判据却去比了真实树 —— 该断言无法做灵敏度回测"))

shutil.rmtree(FAKE)
print("\n副本树已清理")
