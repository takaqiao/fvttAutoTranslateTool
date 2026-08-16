# -*- coding: utf-8 -*-
"""第三十轮 · 「新加的这三道自己能不能被单点绕掉」—— 逐个试，别等复核来抓。

试的都是**判据文件本身**上的单点动作（不是规则文件）。对每一个，报两件事：
  · 主闸（`assert_resolutions.py`）会不会红；
  · `--selftest` 会不会红。
两个都绿 = 真的能单点绕过，必须如实报出来。
"""
import copy
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "3-常用脚本", "qa"))
sys.stdout.reconfigure(encoding="utf-8")

import assert_resolutions as AR       # noqa: E402

BASE = json.load(io.open(AR.DEFAULT_RULES, encoding="utf-8"))
RULE = next(r for r in BASE["assertions"] if r.get("kind") == "ruleset_shape")


def gate_red(rules=None):
    """主闸里 ruleset_shape 那一条会不会红（其余断言与本轮改动无关）。"""
    b, _ = AR.a_ruleset_shape(RULE, AR._RulesCtx(copy.deepcopy(rules or BASE)))
    return bool(b)


def shape_selftest_red():
    """`--selftest` 的 ruleset_shape 组会不会红（吞掉打印）。"""
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        nbad, _n = AR.run_ruleset_shape_selftest()
    return nbad > 0


def shape_count_red(counts):
    bad, _notes = AR._selftest_shape_check(counts)
    return bool(bad)


LIVE_COUNTS = {k: v for k, v in AR.SELFTEST_SHAPE.items()}

print("=" * 92)
print(f"{'判据文件上的单点动作':56s} {'主闸':>6s} {'--selftest':>12s}")
print("=" * 92)


def row(note, gate, self_):
    verdict = "ok（有人接）" if (gate or self_) else "←←← **两边都绿：真能单点绕过**"
    print(f"{note:56s} {'红' if gate else '绿':>6s} {'红' if self_ else '绿':>12s}   {verdict}")


# ① 把 REGISTERED_ASSERTIONS 删剩一条
saved = AR.REGISTERED_ASSERTIONS
AR.REGISTERED_ASSERTIONS = {"R-ruleset-shape": "ruleset_shape"}
row("REGISTERED_ASSERTIONS 删剩 1 条（登记表变空壳）", gate_red(), shape_selftest_red())
AR.REGISTERED_ASSERTIONS = saved

# ② min_by_kind 退回只覆盖 4 种（本轮之前的样子）
saved2 = AR.RULESET_SHAPE["min_by_kind"]
AR.RULESET_SHAPE["min_by_kind"] = {"translate_cases": 1, "panel_liveness": 1,
                                   "tracked_inputs": 1, "ruleset_shape": 1}
row("min_by_kind 退回只覆盖 4 种 kind", gate_red(), shape_selftest_red())
AR.RULESET_SHAPE["min_by_kind"] = saved2

# ③ containers 整块清空（复核当时判它「纯冗余」的那一刀）
saved3 = AR.RULESET_SHAPE["containers"]
AR.RULESET_SHAPE["containers"] = {}
row("RULESET_SHAPE['containers'] 整块清空", gate_red(), shape_selftest_red())
AR.RULESET_SHAPE["containers"] = saved3

# ④ tracked_must_include 清空
saved4 = AR.RULESET_SHAPE["tracked_must_include"]
AR.RULESET_SHAPE["tracked_must_include"] = []
row("RULESET_SHAPE['tracked_must_include'] 清空", gate_red(), shape_selftest_red())
AR.RULESET_SHAPE["tracked_must_include"] = saved4

# ⑤ min_assertions 改小（历史地板层）
saved5 = AR.RULESET_SHAPE["min_assertions"]
AR.RULESET_SHAPE["min_assertions"] = 1
row("min_assertions 66 → 1", gate_red(), shape_selftest_red())
AR.RULESET_SHAPE["min_assertions"] = saved5

# ⑥ 自检里删掉一组用例（分母悄悄变小）—— 用 SELFTEST_SHAPE 那道检查代跑
c = dict(LIVE_COUNTS)
c["ruleset_shape"] = c["ruleset_shape"] - 6
row("`--selftest` 的 ruleset_shape 组删掉 6 条用例", False, shape_count_red(c))
c2 = dict(LIVE_COUNTS)
c2.pop("main 返回码")
row("`--selftest` 整组摘掉（main 返回码那一组）", False, shape_count_red(c2))
c3 = dict(LIVE_COUNTS)
c3["tracked_inputs"] = c3["tracked_inputs"] - 4
row("`--selftest` 的 tracked_inputs 摘掉 4 条（形态 (g) 那一批）", False, shape_count_red(c3))

print("=" * 92)
print("""
⚠ 照实说清楚边界（与 `_two_layer_floor` / `RULESET_SHAPE` 的口径一致）：
  上面每一行**单点**动作都有人接。要真绕过去，得**同时**改若干处：
  比如「删断言 + 摘登记表那一行 + 改 min_assertions + 改 min_by_kind 的那一类」——
  四处改动**全部落在 assert_resolutions.py 的 diff 上**，每一处都写着自己干了什么。
  判据能做到的就是这个：**把无痕的单点动作变成一串互相印证的自证改动**，
  不是「不可能被绕过」。
""")
