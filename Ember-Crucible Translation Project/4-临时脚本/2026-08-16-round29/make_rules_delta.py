# -*- coding: utf-8 -*-
"""第二十九轮·**增量**补丁候选（在断言方已落地的第二十九轮补丁之上再加一层）。

背景：本轮两路并行 —— 断言三文件那一路已经把「np_size 31→27 · 删 5 条通知正例 ·
hex 正例 12.4→s.12.4」落进了 `5-其他内容/RESOLUTIONS.assertions.json`，与本路独立算出的
候选（RESOLUTIONS.assertions.round29-candidate.json）在这 6 处**逐字一致**。

但对比之后还差两样，**都是本轮验收条款直接要求的**，本文件把它们做成增量补丁候选：

  ① 已落地的 `notify_negative`（39 条）里**没有**本轮坐实的那两条现网越界现场：
       '"myLayer" is already registered for this layer.'
       '"crimson" is already registered for this color.'
     —— 验收条款写明「至少要覆盖这两条」。它们钉的是「**这条正则不许被加回来**」：
     只要有人把 `^"(.+)" is already registered for this (layer|color)\\.$` 放回表里，
     这两条当场红。没有它们，删除这个动作**没有任何常设判据守着**（下一轮谁加回来都无声）。
  ② 另三条同型的反向证据（被删的休息族两条 + group actor 一条）与 hex 收紧的一条，
     理由同上；外加一条 hex 正例覆盖**另一个 slice 前缀与负偏移**（`p.0.0` / `s.-3.7`），
     出处：ember.mjs:119780 `prefix: "s"` · :120120 `prefix: "p"` · :955-956 i/j 只要求整数。

⚠ 幂等：已经在里面的条目不会重复加；加完把 `recorded` 的两个数记成新值（两层判法要求）。
"""
import io
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "RESOLUTIONS.assertions.round29-delta.json")

d = json.load(io.open(SRC, encoding="utf-8"))
rule = next(a for a in d["assertions"] if a.get("kind") == "translate_cases")

ADD_NEG = [
    # ★ 本轮**用真身实跑坐实**的两条现网越界现场（改动前会被整句重写成中文）
    '"myLayer" is already registered for this layer.',
    '"crimson" is already registered for this color.',
    # 被删的休息族（dnd5e 系休息模块会发的近似句子）
    "Your rest was interrupted after 3 hours by the Ambush event!",
    "Your rest was interrupted by the Ambush event!",
    # 被删的 group actor（dnd5e 核心就有 type:"group"）
    'You cannot create multiple tokens for the "Bandits" group actor.',
    # hex 收紧的反向证据（上游 getKey 恒带 slice 前缀，这两种形状只可能是别人发的）
    "Adjacent hex 12.4 is not directly reachable from current hex 12.3.",
    "Adjacent hex A is not directly reachable from current hex B.",
]
ADD_POS = [
    ["Adjacent hex p.0.0 is not directly reachable from current hex s.-3.7.",
     "相邻六边格 p.0.0 无法从当前六边格 s.-3.7 直达。"],
]

n0, p0 = len(rule["notify_negative"]), len(rule["notify_positive"])
have_neg = set(rule["notify_negative"])
rule["notify_negative"] = rule["notify_negative"] + [s for s in ADD_NEG if s not in have_neg]
have_pos = {p[0] for p in rule["notify_positive"]}
rule["notify_positive"] = rule["notify_positive"] + [p for p in ADD_POS if p[0] not in have_pos]

rule["recorded"]["notify_negative"] = len(rule["notify_negative"])
rule["recorded"]["notify_positive"] = len(rule["notify_positive"])
rule["recorded"]["_why_29b"] = (
    "第二十九轮增量：通知反例 +7 / 通知正例 +1，**只加不删**，两个记录值随之上调。"
    "加的 7 条反例是本轮 4 条删除 + 1 条收紧的**反向证据**，其中前两条"
    "（`\"myLayer\" … layer.` / `\"crimson\" … color.`）是复核用真身实跑坐实的**现网越界现场**："
    "改动前它们会被 `^\"(.+)\" is already registered for this (layer|color)\\.$` 整句重写成"
    "「「myLayer」已在该层上注册过了。」。⚠ 它们钉的不是某条现存表项，而是"
    "「**这条正则不许被加回来**」—— 删除这个动作本身此前没有任何常设判据守着，"
    "下一轮谁把它放回表里都是无声的。加的 1 条正例覆盖 hex 的另一个 slice 前缀与负偏移"
    "（`p.0.0` / `s.-3.7`，出处 ember.mjs:119780 / :120120 / :955-956）。"
)

with io.open(OUT, "w", encoding="utf-8") as fh:
    json.dump(d, fh, ensure_ascii=False, indent=1)
    fh.write("\n")

print("写出：", OUT)
print(f"notify_negative: {n0} -> {len(rule['notify_negative'])}")
print(f"notify_positive: {p0} -> {len(rule['notify_positive'])}")
sys.exit(0)
