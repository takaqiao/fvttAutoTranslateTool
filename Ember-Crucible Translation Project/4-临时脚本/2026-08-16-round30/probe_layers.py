# -*- coding: utf-8 -*-
"""第三十轮 · C：把 `RULESET_SHAPE` 的每一层逐个变异，看**是哪一层响的**。

用途有二：
  ① 给 `run_ruleset_shape_selftest` 的分层归因填正确的期望层名；
  ② 回答验收里那句「`containers` 到底承不承重」—— 对每个容器块 / 点名键做**单点删除**，
     看除了 `containers` 之外还有没有别的层响。只有 `containers` 响的，它就是承重的。
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

base = json.load(io.open(AR.DEFAULT_RULES, encoding="utf-8"))
rule = next(r for r in base["assertions"] if r.get("kind") == "ruleset_shape")


def layers(mutate):
    r = copy.deepcopy(base)
    mutate(r)
    bad, _d = AR.a_ruleset_shape(rule, AR._RulesCtx(r))
    out = {}
    for x in bad:
        out.setdefault(x[1], []).append(x[2])
    return out


def pick(rules, kind):
    return next(i for i, r in enumerate(rules["assertions"]) if r.get("kind") == kind)


def show(note, mutate):
    L = layers(mutate)
    print(f"\n▸ {note}")
    if not L:
        print("    （没有任何一层响 —— 全绿）")
    for k in sorted(L):
        print(f"    {k:28s} × {len(L[k])}   e.g. {L[k][0]}")
    return L


print("=" * 74)
print("第一部分：容器层逐个单点删除 —— `containers` 承不承重")
print("=" * 74)
only_containers = []
also_others = []
for kind, spec in AR.RULESET_SHAPE["containers"].items():
    for block, keys in spec.items():
        def kill_block(r, kind=kind, block=block):
            r["assertions"][pick(r, kind)].pop(block, None)
        L = show(f"删整块 {kind}.{block}", kill_block)
        (only_containers if set(L) <= {"配置·containers"} else also_others).append(
            (f"{kind}.{block}", sorted(L)))
        for k in keys:
            def kill_key(r, kind=kind, block=block, k=k):
                r["assertions"][pick(r, kind)][block].pop(k, None)
            L = show(f"只摘键 {kind}.{block}.{k}", kill_key)
            (only_containers if set(L) <= {"配置·containers"} else also_others).append(
                (f"{kind}.{block}.{k}", sorted(L)))

print("\n" + "=" * 74)
print("小结：只有 containers 一层响的（= containers 是唯一承重层）")
for name, L in only_containers:
    print(f"   ONLY  {name}")
print("\n还有别的层一起响的（= containers 在这里是冗余）")
for name, L in also_others:
    print(f"   ALSO  {name:52s} {L}")

print("\n" + "=" * 74)
print("第二部分：把 containers 整块清空，看哪些用例会掉")
print("=" * 74)
saved = AR.RULESET_SHAPE["containers"]
AR.RULESET_SHAPE["containers"] = {}
for name, L in only_containers:
    print(f"   清空后失守：{name}")
AR.RULESET_SHAPE["containers"] = saved

print("\n" + "=" * 74)
print("第三部分：现有 8 个 C-用例分别由哪一层响")
print("=" * 74)
i_tc, i_pl, i_ti = pick(base, "translate_cases"), pick(base, "panel_liveness"), pick(base, "tracked_inputs")
show("C-① 删 arrangements 整块", lambda r: r["assertions"][i_tc].pop("arrangements", None))
show("C-② channels 砍成一个",
     lambda r: r["assertions"][i_tc]["arrangements"].__setitem__("channels", ["Music"]))
show("C-③ 面板 min/max 整块删", lambda r: [r["assertions"][i_pl].pop(k, None) for k in ("min", "max")])
show("C-④ tracked.min_checked → 0",
     lambda r: r["assertions"][i_ti].__setitem__("min_checked", 0))
show("C-⑤ tracked.must_include 清空",
     lambda r: r["assertions"][i_ti].__setitem__("must_include", []))
show("C-⑥ arrangements.min_labels 改小",
     lambda r: r["assertions"][i_tc]["arrangements"].__setitem__("min_labels", 2))
show("C-⑦ 删掉一条 cn_absent 断言",
     lambda r: r["assertions"].pop(pick(r, "cn_absent")))
show("C-⑨ min.checkedDistinct → 1",
     lambda r: r["assertions"][i_pl]["min"].__setitem__("checkedDistinct", 1))
show("C-⑩ max.missDistinct → 999",
     lambda r: r["assertions"][i_pl]["max"].__setitem__("missDistinct", 999))
show("C-⑪ must_include 摘掉默认执行体",
     lambda r: r["assertions"][i_ti].__setitem__(
         "must_include", [x for x in r["assertions"][i_ti]["must_include"]
                          if not x.endswith("selfcheck_panel_runner.mjs")]))

print("\n" + "=" * 74)
print("第四部分：H1 / H2 / H3 分别由哪一层响（断言侧；main() 侧另跑）")
print("=" * 74)
show("H1 把一条 cn_absent 的 kind 拼错成 cn_absentt",
     lambda r: r["assertions"][pick(r, "cn_absent")].__setitem__("kind", "cn_absentt"))


def h2(r):
    i = pick(r, "cn_absent")
    r["assertions"].pop(i)
    filler = copy.deepcopy(r["assertions"][pick(r, "term_gated")])
    filler["id"] = "R-filler-30"
    r["assertions"].append(filler)


show("H2 删一条 cn_absent，复制一条 term_gated 换 id 当填充", h2)


def h3(r):
    i = pick(r, "cn_absent")
    r["assertions"].pop(i)
    r["assertions"].append({"kind": "nope"})


show("H3 删一条，塞 {\"kind\":\"nope\"} 空壳", h3)


def dup(r):
    i = pick(r, "cn_absent")
    d = copy.deepcopy(r["assertions"][i])
    r["assertions"].append(d)          # 连 id 一起复制
    r["assertions"].pop(pick(r, "term_gated"))


show("H2' 连 id 一起复制（id 重复）", dup)


print("\n" + "=" * 74)
print("第五部分：tracked_must_include 清空 —— 是不是冗余")
print("=" * 74)
saved_tmi = AR.RULESET_SHAPE["tracked_must_include"]
AR.RULESET_SHAPE["tracked_must_include"] = []
show("（护栏已清空）+ 规则侧摘掉 pack 那条",
     lambda r: r["assertions"][i_ti].__setitem__(
         "must_include", [x for x in r["assertions"][i_ti]["must_include"]
                          if "crucible-adventure" not in x]))
AR.RULESET_SHAPE["tracked_must_include"] = saved_tmi
show("（护栏在）+ 规则侧摘掉 pack 那条",
     lambda r: r["assertions"][i_ti].__setitem__(
         "must_include", [x for x in r["assertions"][i_ti]["must_include"]
                          if "crucible-adventure" not in x]))
