# -*- coding: utf-8 -*-
"""第三十轮 · B 的验收：`tracked_inputs` 的 `--rules <副本>` 灵敏度回测，**逐条**跑。

背景（本轮 B 要修的那件事）
-------------------------
`a_tracked_inputs` 此前推「运行时输入」用的是**自己去磁盘重读**的
`5-其他内容/RESOLUTIONS.assertions.json`（写死路径），而同一条断言的
`sweep` / `must_include` / `min_checked` 用的是**传进来的 rule 对象**。半读半不读。
后果：任何往 `--rules <副本>` 里注入违规的探针，对「推导 + 反查 + git 入库」这半边
**永远是空转**（形态 (g)），跑出来的绿是假绿。

本文件干两件事
-------------
  ① `new` 口径（修好之后的真身）：逐条注入违规，`R-assertion-inputs-tracked` 必须红；
  ② `old` 口径（把老的半读半不读**原样复现**）：同一批注入，看它绿成什么样 ——
     这就是「这一半至今从没被真正回测过」的**当场证据**，不是嘴上说。
     复现方式：在进程内把 ctx.rules 换回磁盘上那一份再调断言，
     其余（sweep / must_include / min_checked 读传进来的 rule）保持不变。

⚠ 注入的都是**路径层面**的违规（指向不存在的文件 / 点名一个推不出来的路径 /
  扫一个不存在的目录），不往别人独占的仓里写任何文件 —— 本轮硬约束 1。
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
DISK = json.load(io.open(AR.DEFAULT_RULES, encoding="utf-8"))     # 「磁盘上那一份」


class Ctx:
    """跑 tracked_inputs 要的最小 ctx（与 main() 组装的那个同形）。"""

    def __init__(self, rules, rules_path=None):
        self.root = AR.ROOT
        self.here = os.path.join(AR.ROOT, "3-常用脚本", "qa")
        self.repos = {"ember": os.path.join(AR.ROOT, AR.REPOS["ember"]),
                      "crucible": os.path.join(AR.ROOT, AR.REPOS["crucible"])}
        self.rules = rules
        self.rules_path = rules_path or AR.DEFAULT_RULES


def ti_rule(rules):
    return next(r for r in rules["assertions"] if r.get("kind") == "tracked_inputs")


def pick(rules, kind):
    return next(i for i, r in enumerate(rules["assertions"]) if r.get("kind") == kind)


# ------------------------------------------------------------------ 注入用例
# 每条：(名字, 改规则集的函数, 期望在违规里看到的关键串)
def m_src(r):
    r["assertions"][pick(r, "translate_cases")]["src"] = "scripts/no-such-src.mjs"


def m_panel(r):
    r["assertions"][pick(r, "panel_liveness")]["panel"] = "scripts/no-such-panel.mjs"


def m_glossary(r):
    r["assertions"][pick(r, "glossary_value")]["glossary"] = "5-其他内容/no-such-glossary.json"


def m_exclusions(r):
    r["assertions"][pick(r, "exclusions_closed")]["exclusions"] = "5-其他内容/no-such-exc.json"


def m_scanner(r):
    r["assertions"][pick(r, "exclusions_closed")]["scanner"] = "no_such_scanner.py"


def m_pack(r):
    r["assertions"][pick(r, "leaf_literal")]["pack"] = "no.such.pack.json"


def m_runner(r):
    r["assertions"][pick(r, "translate_cases")]["runner"] = "no_such_runner.mjs"


def m_files(r):
    r["assertions"][pick(r, "twin_files")]["files"] = [
        {"repo": "ember", "path": "scripts/no-such-twin.mjs"}]


def m_must(r):
    ti_rule(r)["must_include"] = list(ti_rule(r)["must_include"]) + [
        "5-其他内容/no-such-must.json"]


def m_sweep(r):
    ti_rule(r)["sweep"] = list(ti_rule(r)["sweep"]) + ["3-常用脚本/no-such-dir"]


def m_min_checked(r):
    ti_rule(r)["min_checked"] = 99999


def m_pathish(r):
    # 反查那一路：给某条断言加一个**推导器不认识的字段**，值是库里真实存在的文件。
    # `5-其他内容/EXCLUSIONS.json` 实测**不在**推导结果里，正合用。
    r["assertions"][pick(r, "cn_absent")]["mystery_input"] = "5-其他内容/EXCLUSIONS.json"


def m_baseline(r):
    """对照组：什么都不改。两个口径都必须 0 违规，否则下面的红说明不了任何事。"""


CASES = [
    ("baseline 原样（对照）", m_baseline, None),
    ("src 指向不存在的文件", m_src, "no-such-src.mjs"),
    ("panel 指向不存在的文件", m_panel, "no-such-panel.mjs"),
    ("glossary 指向不存在的文件", m_glossary, "no-such-glossary.json"),
    ("exclusions 指向不存在的文件", m_exclusions, "no-such-exc.json"),
    ("scanner 指向不存在的执行体", m_scanner, "no_such_scanner.py"),
    ("pack 指向不存在的分包", m_pack, "no.such.pack.json"),
    ("runner 指向不存在的执行体", m_runner, "no_such_runner.mjs"),
    ("files[] 指向不存在的文件", m_files, "no-such-twin.mjs"),
    ("must_include 点名一个推不出来的路径", m_must, "no-such-must.json"),
    ("sweep 指向不存在的目录", m_sweep, "no-such-dir"),
    ("min_checked 抬到 99999（清单够不够长）", m_min_checked, "只查了"),
    ("反查：新字段里写一个库里真有、推导器不认识的文件", m_pathish, "EXCLUSIONS.json"),
]


def run(mut, mode):
    r = copy.deepcopy(BASE)
    mut(r)
    rule = ti_rule(r)
    if mode == "old":
        # 老口径：`sweep` / `must_include` / `min_checked` 照读传进来的 rule，
        # 而**推导**那一半读磁盘上那份写死的规则文件。
        ctx = Ctx(copy.deepcopy(DISK), rules_path=AR.DEFAULT_RULES)
    else:
        ctx = Ctx(r, rules_path=AR.DEFAULT_RULES)
    return AR.a_tracked_inputs(rule, ctx)


print("=" * 96)
print(f"{'用例':52s} {'old 口径':>12s} {'new 口径':>12s}   命中期望串")
print("=" * 96)
rows = []
for note, mut, want in CASES:
    ob, _ = run(mut, "old")
    nb, _ = run(mut, "new")
    hit_o = want and any(want in str(x[2]) or want in str(x[3]) for x in ob)
    hit_n = want and any(want in str(x[2]) or want in str(x[3]) for x in nb)
    if want is None:
        verdict = "ok" if (not ob and not nb) else "←← 对照组不干净"
    else:
        verdict = "ok" if (hit_n and not hit_o) else (
            "（两边都响：不属于被换走的那一半）" if hit_n and hit_o else "←← new 没响")
    rows.append((note, len(ob), len(nb), bool(hit_o), bool(hit_n), verdict))
    print(f"{note:52s} {len(ob):5d} 违规 {len(nb):5d} 违规   "
          f"old命中={str(bool(hit_o)):5s} new命中={str(bool(hit_n)):5s}  {verdict}")

print("=" * 96)
silent = [r for r in rows if r[4] and not r[3]]
print(f"\n结论：{len(silent)} / {len(CASES) - 1} 条注入**在老口径下是静默的**"
      f"（探针空转、跑出假绿），修完之后全部当场响。")
for r in silent:
    print(f"   · {r[0]}")
both = [r for r in rows if r[3] and r[4]]
if both:
    print(f"\n老口径下也会响的（这些走的是「读传进来的 rule」那一半，本来就没坏）：")
    for r in both:
        print(f"   · {r[0]}")
