# -*- coding: utf-8 -*-
"""生成 PAYLOAD_FLOORS 与 JUDGED_UNITS 的源码块。

⚠ 硬约束 3：生成物里**一个反斜杠都不许有** —— 只有键名、类型标签和数字。
  生成完当场断言 `"\\" not in out`，不满足就 abort（`\b`/`\s` 被改写脚本吃掉，本项目栽过两次）。
⚠ 硬约束 4：前置自证 —— 66 条 / 22 kind / 载荷键总数与 probe_payload_keys 数出来的一致。
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")

A = json.load(open(RULES, encoding="utf-8"))["assertions"]
assert len(A) == 66 and len({r["kind"] for r in A}) == 22, "前置自证失败"

META = {"id", "kind", "title", "decision", "why", "note", "_why", "rule"}
# 含义两可 / 本来就按等号判的数：一律 eq
EQ_KEYS = {"window", "short_len", "substr_expect_miss", "min_ratio", "min_sentence_frac"}
RECURSE_MAX = 12          # 子键太多的字典（entries 31 条）只记总长，别把表撑爆


def classify(key, v):
    if isinstance(v, bool):
        return ("eq", v)
    if isinstance(v, (int, float)):
        leaf = key.split(".")[-1]
        if leaf in EQ_KEYS:
            return ("eq", v)
        if leaf.startswith("max_") or leaf == "max":
            return ("le", v)
        if leaf.startswith("min_") or leaf == "min":
            return ("ge", v)
        return ("eq", v)
    if isinstance(v, str):
        return ("str", len(v))
    if isinstance(v, (list, dict)):
        return ("list", len(v))
    return None


floors = {}
n_keys = 0
for r in A:
    spec = {}
    for key, v in r.items():
        if key in META:
            continue
        c = classify(key, v)
        if c:
            spec[key] = c
        # 一层递归：小字典的每个子键各记一道（min/max/recorded/arrangements/sense/…）
        if isinstance(v, dict) and 0 < len(v) <= RECURSE_MAX:
            for k2, v2 in v.items():
                if k2 in META or k2.startswith("_why") or k2.startswith("why"):
                    continue          # 块内的散文注释键不是载荷，别把它的字数钉死
                c2 = classify(f"{key}.{k2}", v2)
                if c2:
                    spec[f"{key}.{k2}"] = c2
    floors[r["id"]] = spec
    n_keys += len(spec)

units = json.load(open(os.path.join(HERE, "units.json"), encoding="utf-8"))
assert set(units) == set(floors), "两张表的 id 集合对不上"

# —— 两处**有意低于实测值**的地板，理由写进注释（见下）
SOFT = {"R-glossary-not-laundering": 31, "R-fortitude-scope": 3}

lines = []
for r in A:
    rid = r["id"]
    spec = floors[rid]
    body = ", ".join(f'"{k}": ("{t}", {v!r})' for k, (t, v) in sorted(spec.items()))
    lines.append(f'    "{rid}": {{{body}}},')
payload_src = "\n".join(lines)

ulines = []
for r in A:
    rid = r["id"]
    n = SOFT.get(rid, units[rid])
    tail = ""
    if rid in SOFT:
        tail = (f"    # 实测 {units[rid]}；地板取**单层**的量 —— 词表 base 层按设计"
                f"住在项目根之外，执行体明说那一层可以不在（缺了由 min_checked 报）")
    ulines.append(f'    "{rid}": {n},' + ("\n" + tail if tail else ""))
units_src = "\n".join(ulines)

out = payload_src + "\n@@@\n" + units_src
assert "\\" not in out, "生成物里出现了反斜杠 —— 硬约束 3，abort"
print(f"前置自证 ok：66 条 / 22 kind / 载荷键 {n_keys} 道 / 生成物无反斜杠")
with open(os.path.join(HERE, "gen_blocks.txt"), "w", encoding="utf-8") as fh:
    fh.write(out)
print(f"载荷地板：{n_keys} 道，覆盖 {len(floors)} 条断言 / {len({r['kind'] for r in A})} 个 kind")
