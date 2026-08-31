# -*- coding: utf-8 -*-
"""第三十轮 · A 的验收：H1 / H2 / H3 **对着真主闸**重跑一遍，看是红是绿。

⚠ 跑的是 `assert_resolutions.py` 的**真身**（子进程 + `--rules <变异副本>`），
  不是在进程里调 `a_ruleset_shape` —— 后者只能证明「那条断言会响」，
  证明不了「主闸的返回码会变」，而 H1/H3 的要害恰恰在**返回码**上
  （断言打了 `??` 一行、`skipped += 1`，`return 1 if failed else 0` 照样给 0）。

⚠ 副本能这么用，靠的是本轮 B 那一处修复：`a_tracked_inputs` 此前**自己去磁盘重读
  写死的那份规则文件**，`--rules <副本>` 对它这半边永远是空转（形态 (g)）。
  修之前这个探针本身就是形态 (h)。
"""
import copy
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GATE = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")
SRC = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
sys.stdout.reconfigure(encoding="utf-8")

base = json.load(io.open(SRC, encoding="utf-8"))


def pick(r, kind):
    return next(i for i, x in enumerate(r["assertions"]) if x.get("kind") == kind)


def h1(r):
    """把任意一条断言的 kind 拼错一个字母。"""
    r["assertions"][pick(r, "cn_absent")]["kind"] = "cn_absentt"


def h2(r):
    """删掉一条真断言，再复制另一条换个 id 当填充 —— 条数不变、kind 数不变。"""
    r["assertions"].pop(pick(r, "cn_absent"))
    filler = copy.deepcopy(r["assertions"][pick(r, "term_gated")])
    filler["id"] = "R-filler-h2"
    r["assertions"].append(filler)


def h3(r):
    """删一条，塞一个 `{"kind": "nope"}` 空壳。"""
    r["assertions"].pop(pick(r, "cn_absent"))
    r["assertions"].append({"kind": "nope"})


def baseline(r):
    """对照：什么都不改 —— 必须是绿的（否则下面的红说明不了任何事）。"""


CASES = [("baseline 原样", baseline, 0), ("H1 kind 拼错一个字母", h1, 1),
         ("H2 删一条 + 复制一条换 id 填充", h2, 1), ("H3 删一条 + 塞空壳", h3, 1)]

env = dict(os.environ, PYTHONIOENCODING="utf-8")
print("=" * 78)
for note, fn, want_rc_nonzero in CASES:
    r = copy.deepcopy(base)
    fn(r)
    out = os.path.join(HERE, "rules.%s.json" % note.split()[0])
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(r, fh, ensure_ascii=False, indent=1)
    p = subprocess.run([sys.executable, GATE, "--rules", out], cwd=ROOT,
                       capture_output=True, env=env)
    txt = p.stdout.decode("utf-8", "replace")
    tail = [ln for ln in txt.splitlines() if ln.startswith("通过 ")]
    fails = [ln.split("——")[0].strip() for ln in txt.splitlines() if ln.startswith("  FAIL")]
    got_red = p.returncode != 0
    verdict = "红" if got_red else "绿"
    ok = got_red == bool(want_rc_nonzero)
    print(f"\n▸ {note}")
    print(f"    计数：{tail[0] if tail else '?'}    rc={p.returncode}  ⇒ **{verdict}**"
          f"   {'ok' if ok else '←←← 不符合期望'}")
    for f in fails:
        print(f"      {f}")
print("\n" + "=" * 78)
