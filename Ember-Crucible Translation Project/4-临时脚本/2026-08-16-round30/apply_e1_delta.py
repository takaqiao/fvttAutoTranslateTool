# -*- coding: utf-8 -*-
"""第三十轮 · D：落第二十九轮备好的 E1 增量补丁（**就地改规则文件**，幂等）。

与第二十九轮 `make_rules_delta.py` 的差别，两处：
  ① 它写到 `4-临时脚本/…/RESOLUTIONS.assertions.round29-delta.json`（只是候选），
     本文件**就地改** `5-其他内容/RESOLUTIONS.assertions.json` —— 本轮验收要求它落地；
  ② **`_why_29b` 的措辞按第三十轮复核的实测口径重写**。第二十九轮那版写的是
     「删除这个动作本身此前没有任何常设判据守着，下一轮谁把它放回表里都是无声的」——
     **这句话经实测不成立**：把那条正则原样加回被判文件，主闸当场违规 4 处、红。
     所以这 6 条不是「补洞」，是**加固**（把「合法地加回来」那条路也堵上）。照实写。

⚠ 幂等：已在表里的条目不重复加；`recorded` 的两个数每次按实际长度重算。
⚠ 只加不删。
"""
import io
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
BAK = os.path.join(HERE, "RESOLUTIONS.assertions.before-e1.json")
sys.stdout.reconfigure(encoding="utf-8")

if not os.path.exists(BAK):
    shutil.copyfile(SRC, BAK)

d = json.load(io.open(SRC, encoding="utf-8"))
rule = next(a for a in d["assertions"] if a.get("kind") == "translate_cases")

ADD_NEG = [
    # ★ 第二十九轮复核**用真身实跑坐实**的两条现网越界现场：改动前它们会被
    #   `^"(.+)" is already registered for this (layer|color)\.$` 整句重写成中文。
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
new_neg = [s for s in ADD_NEG if s not in have_neg]
rule["notify_negative"] = rule["notify_negative"] + new_neg
have_pos = {p[0] for p in rule["notify_positive"]}
new_pos = [p for p in ADD_POS if p[0] not in have_pos]
rule["notify_positive"] = rule["notify_positive"] + new_pos

rule["recorded"]["notify_negative"] = len(rule["notify_negative"])
rule["recorded"]["notify_positive"] = len(rule["notify_positive"])
rule["recorded"]["_why_29b"] = (
    "第二十九轮备好、第三十轮落地的增量：通知反例 +6 / 通知正例 +1，**只加不删**，"
    "两个记录值随之上调（notify_negative 39→45 · notify_positive 31→32）。"
    "⚠ **措辞按第三十轮复核的实测口径更正过**：第二十九轮原稿写的是「删除这个动作本身"
    "此前没有任何常设判据守着，下一轮谁把它放回表里都是无声的」—— **这句话经实测不成立**。"
    "把第二十九轮删掉的那条正则（`^\"(.+)\" is already registered for this (layer|color)\\.$`）"
    "原样加回被判文件，主闸当场**违规 4 处、红**：覆盖归因、结构护栏、`np_size` 三道一起响；"
    "要让它变绿必须 4 处协同改动，而每一处都在 diff 里自证。"
    "⇒ 这 6 条**不是补一个无人看守的洞，是加固**：那三道守的是「表变长了却没人记账」，"
    "而**合法的加法**——把正则加回去、同时补上它自己的正例与近似反例并把记录值一起上调——"
    "本来是一条走得通的路；这 6 条反例把那条路也钉死了。"
    "⚠ 第三十轮就地实测过这件事（临时副本树加回正则、跑发布中的这条规则，被判文件本体没动）："
    "落 E1 **之前** 4 处违规全是记账类（coverage / negative_structure / np_size / np_neg_matched），"
    "逐条都能靠「补正例 + 补专属反例 + 上调 np_size」合法消掉；"
    "落 E1 **之后**同样 4 处，但其中 2 处换成了这 6 条里的 `\"myLayer\"` 与 `\"crimson\"` "
    "**自己被整句重写成中文**（实得「「myLayer」已在该层上注册过了。」），"
    "而 negative_structure / np_neg_matched 反倒被这 6 条**满足**了 —— "
    "也就是说它们一边充当那条表项的专属近似反例、一边把它钉死。"
    "它们钉的不是某条现存表项，而是「**这条正则不许被加回来**」，"
    "且**改记录值救不了**（反例判的是「这句不许被翻译」，不是数量）；要变绿只能把这 6 条删掉，"
    "而那是作弊路径 A，第二十七轮已堵。"
    "其中前两条（`\"myLayer\" … layer.` / `\"crimson\" … color.`）是复核用真身实跑坐实的"
    "**现网越界现场**：改动前它们确实会被整句重写成「「myLayer」已在该层上注册过了。」。"
    "加的 1 条正例覆盖 hex 的另一个 slice 前缀与负偏移"
    "（`p.0.0` / `s.-3.7`，出处 ember.mjs:119780 `prefix: \"s\"` · :120120 `prefix: \"p\"` · "
    ":955-956 i/j 只要求整数）。"
)

with io.open(SRC, "w", encoding="utf-8") as fh:
    json.dump(d, fh, ensure_ascii=False, indent=1)
    fh.write("\n")

print(f"就地改：{SRC}")
print(f"备份（首次运行时留的）：{BAK}")
print(f"notify_negative: {n0} -> {len(rule['notify_negative'])}   本次新加 {len(new_neg)} 条")
for s in new_neg:
    print(f"    + {s}")
print(f"notify_positive: {p0} -> {len(rule['notify_positive'])}   本次新加 {len(new_pos)} 条")
for p in new_pos:
    print(f"    + {p[0]}")
