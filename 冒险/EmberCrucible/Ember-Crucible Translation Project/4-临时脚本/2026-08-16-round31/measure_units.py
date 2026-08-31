# -*- coding: utf-8 -*-
"""探针：跑一遍全部 66 条断言，量出每条「判了几条规矩」，并生成 PAYLOAD_FLOORS 源码。

⚠ 硬约束 4（前置自证）：
  · 断言条数必须 == 66，kind 数 == 22，否则 abort；
  · 22 个 kind 的执行体必须**每个都被调用过**（_KIND_CALLS），否则说明我根本没跑到它，
    量出来的 0 是假的；
  · 每条断言的规矩数必须 > 0 —— 出现 0 就当场打出来给人看，不许静默写进地板。
"""
import importlib.util, json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
MOD = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")

spec = importlib.util.spec_from_file_location("ar", MOD)
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

rules = json.load(open(ar.DEFAULT_RULES, encoding="utf-8"))
A = rules["assertions"]
assert len(A) == 66, f"条数 {len(A)}"
assert len({r["kind"] for r in A}) == 22, "kind 数"

repos = {}
for name, rel in ar.REPOS.items():
    d = os.path.join(ar.ROOT, rel)
    if os.path.isdir(d):
        repos[name] = d
assert len(repos) == 2, f"仓数 {len(repos)}"
ctx = ar.Ctx(repos, rules.get("meta"))
ctx.rules = rules
ctx.rules_path = ar.DEFAULT_RULES

units, errs = {}, []
for r in A:
    fn = ar.KINDS[r["kind"]]
    ar._UNITS.clear()
    try:
        bad, detail = fn(r, ctx)
    except Exception as exc:
        errs.append((r["id"], repr(exc)))
        units[r["id"]] = 0
        continue
    units[r["id"]] = len(ar._UNITS)

assert len(units) == 66, f"量到 {len(units)} 条"
missing = sorted(set(ar.KINDS) - set(ar._KIND_CALLS))
print(f"前置自证：66 条全跑过；执行体调用覆盖 {len(ar._KIND_CALLS)}/22"
      f"{'，缺 ' + str(missing) if missing else '，一个不缺'}")
if errs:
    print("⚠ 执行出错：", errs)
zero = [k for k, v in units.items() if v == 0]
print(f"⚠ 规矩数为 0 的：{zero or '无'}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "units.json"),
          "w", encoding="utf-8") as fh:
    json.dump(units, fh, ensure_ascii=False, indent=1)
for k, v in units.items():
    print(f"  {k:36s} {v}")
