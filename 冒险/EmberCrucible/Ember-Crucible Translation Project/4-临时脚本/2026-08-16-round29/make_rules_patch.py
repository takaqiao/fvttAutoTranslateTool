# -*- coding: utf-8 -*-
"""第二十九轮：生成 RESOLUTIONS.assertions.json 的**候选补丁副本**（不改真文件）。

断言三文件不归本轮的执行者动，所以补丁先做成副本、用
`assert_resolutions.py --rules <副本>` 跑**真脚本**证明能回绿，再随升报交出去。

改动只有一处结构性的：`R-patterns-translate-cases`。
  · notify_positive：删 5 条（被删掉的 4 条正则的用例）+ 改 1 条（hex 收紧后带前缀）
    + 加 1 条（hex 的另一个 slice 前缀 / 负偏移）
  · notify_negative：加 7 条（4 条删除 + 1 条收紧各自的**反向证据**）
  · recorded：np_size / notify_positive / notify_negative 三个数跟着记成新值
  · title 里的 31 → 27；why 末尾加第二十九轮脚注（沿用第二十八轮「不改旧记法、加脚注」的做法）
"""
import io
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "RESOLUTIONS.assertions.round29-candidate.json")

d = json.load(io.open(SRC, encoding="utf-8"))
rule = next(a for a in d["assertions"] if a.get("kind") == "translate_cases")

# ── ① notify_positive：删 5 条 ────────────────────────────────────────────────
DROP_POS = {
    'Your rest was interrupted after 3 hours by the Ambush event!',   # 删 :37970
    'Your rest was interrupted by the Ambush event!',                 # 删 :38009
    '"horns" is already registered for this layer.',                  # 删 :51766
    '"red" is already registered for this color.',                    # 删 :51766
    'You cannot create multiple tokens for the "Bandits" group actor.',  # 删 :60902
}
before_pos = len(rule["notify_positive"])
rule["notify_positive"] = [p for p in rule["notify_positive"] if p[0] not in DROP_POS]
assert len(rule["notify_positive"]) == before_pos - 5, "要删的 5 条正例没对上"

# ── ② hex 那条正例：改成上游真会产出的形状（getKey 恒带 slice 前缀）+ 补一条 ──
OLD_HEX = "Adjacent hex 12.4 is not directly reachable from current hex 12.3."
NEW_HEX = "Adjacent hex s.12.4 is not directly reachable from current hex s.12.3."
hits = [i for i, p in enumerate(rule["notify_positive"]) if p[0] == OLD_HEX]
assert len(hits) == 1, f"hex 正例找到 {len(hits)} 条，期望 1 条"
rule["notify_positive"][hits[0]] = [
    NEW_HEX, "相邻六边格 s.12.4 无法从当前六边格 s.12.3 直达。"]
rule["notify_positive"].insert(hits[0] + 1, [
    "Adjacent hex p.0.0 is not directly reachable from current hex s.-3.7.",
    "相邻六边格 p.0.0 无法从当前六边格 s.-3.7 直达。"])

# ── ③ notify_negative：加 7 条反向证据 ───────────────────────────────────────
ADD_NEG = [
    # 本轮已坐实的两条**现网越界现场**（改前会被整句重写成中文）
    '"myLayer" is already registered for this layer.',
    '"crimson" is already registered for this color.',
    # 删掉的休息族两条：dnd5e 系休息模块会发的近似句子
    "Your rest was interrupted after 3 hours by the Ambush event!",
    "Your rest was interrupted by the Ambush event!",
    # 删掉的 group actor：dnd5e 核心就有 type:"group"
    'You cannot create multiple tokens for the "Bandits" group actor.',
    # hex 收紧的反向证据：上游 getKey 恒带前缀，这两种形状只可能是别人发的
    "Adjacent hex 12.4 is not directly reachable from current hex 12.3.",
    "Adjacent hex A is not directly reachable from current hex B.",
]
for s in ADD_NEG:
    assert s not in rule["notify_negative"], f"反例重复：{s}"
rule["notify_negative"] = rule["notify_negative"] + ADD_NEG

# ── ④ recorded：三个数记成新值 ───────────────────────────────────────────────
rule["recorded"]["np_size"] = 27
rule["recorded"]["notify_positive"] = len(rule["notify_positive"])
rule["recorded"]["notify_negative"] = len(rule["notify_negative"])

# ── ⑤ 标题里的表长 + why 脚注 ────────────────────────────────────────────────
rule["title"] = rule["title"].replace("NOTIFICATION_PATTERNS 31", "NOTIFICATION_PATTERNS 27")
rule["decision"] = rule["decision"] + "· 2026-08-16d（第二十九轮：V9 裁决，裁掉 4 条现网越界的通知正则 + 收紧 hex 那条）"
rule["why"] = rule["why"] + (
    "｜**第二十九轮（V9 裁决，主控拍板）**：本表挂在**全局** `ui.notifications.notify` 上，"
    "所以每条正则必须 ① 句内自带厂商锚点（Ember 专有词），或 ② 插值位的定义域封闭且形状足够特异；"
    "**两者都没有的删掉**。代价不对称：少翻我们自己一条通知 = 一句英文；把别人的通知重写成错的中文 ="
    "用户看到一句语义完全错的中文**且查不出是谁干的**。⚠ 上面 (D) 段与第二十八轮那段脚注写的"
    "「31 条」是当时的记法，**不改它们，在这里加脚注**：本轮 31 → **27**。"
    "｜删掉 4 条（复核用真身实跑坐实了前两条的现网越界）："
    "① `^\"(.+)\" is already registered for this (layer|color)\\.$`（ember.mjs:51766）—— 实测"
    "`'\"myLayer\" is already registered for this layer.'` → 「「myLayer」已在该层上注册过了。」、"
    "`'\"crimson\" is already registered for this color.'` → 「「crimson」已在该颜色上注册过了。」；"
    "整句一个 Ember 专有词都没有，而 `layer` 恰是 Foundry 的通用概念（canvas layer / sheet layer），"
    "任何做注册去重的模块都会中招。⚠ **与它同出一个函数**（`#importValue`，:51757-51771）的 :51760 "
    "那条反而安全，因为句子里带 \"Token Maker\" —— 这两条的对照本身就是「锚点」这个判据的证据。"
    "收紧插值位帮不上忙：`value` 来自 `#getImportValue`（:51742-51746），是 partId 串 / 颜色串 / "
    "字面量 \"unused\" 三选一，收紧它只是把误伤面缩小，**句子仍然没有锚点**。"
    "② `^Your rest was interrupted after (\\d+) hours by the (.+) event!$`（:37970）与 "
    "③ `^Your rest was interrupted by the (.+) event!$`（:38009）—— 回上游查了 `event` 那个位："
    "`restEvent.label` ← `ember.events.nextEvent()`（:4713 → :4731 `getHexOutcomes`）← "
    "`_prepareEventData()`（:2483）把 `label` 赋成 **`this.parent.name`**，即事件所在 "
    "JournalEntryPage 的**页名**（EmberNarrativeNode 构造器 :19042 默认空串，391 处 `addChildEvent` "
    "一个 label 都没写死）⇒ 开放集合 + 自然语言形状，**封不住**；句内零锚点，而 dnd5e 系休息模块"
    "发近似句子完全可能。④ `^You cannot create multiple tokens for the \"(.+)\" group actor\\.$`"
    "（:60902）—— 插值位是 `document.actor.name`（任意角色名），而 \"group actor\" 在 dnd5e 里是"
    "**核心角色类型**（type: \"group\"）⇒ 无锚点 + 开放，删。"
    "｜**收紧 1 条**：`^Adjacent hex (.+) is not directly reachable from current hex (.+)\\.$`"
    "（:61049）→ `([a-z]{1,2}\\.-?\\d+\\.-?\\d+)`。两个位是 `h1.key` / `h0.key`；`get key()`（:738）"
    "→ `getKey()`（:908）恒返回 `` `${prefix}.${i}.${j}` ``，`validateString()`（:955-956）要求 i/j "
    "是 `Number.isInteger`，slice 前缀现有且仅有 `\"s\"`（:119780）与 `\"p\"`（:120120），"
    "`[a-z]{1,2}` 是给上游加 slice 留的余量。⚠ **顺带订正一条用例**：旧的通知正例写的是"
    "「Adjacent hex 12.4 …」（**没有前缀**），而 `getKey` 恒带前缀 —— 那是个上游根本产不出的形状，"
    "改成 `s.12.4` / `s.12.3`，并补一条 `p.0.0` / `s.-3.7` 覆盖另一个前缀与负偏移。"
    "｜**MED 4 条本轮不动**（:37966 `Completed resting for (\\d+) hours…` · :120582 · :120627 · "
    "已收紧的 Mirror :98432），判据写在被判文件的注释里：它们的插值位是 `(\\d+)` 或闭合集合，"
    "**位置上不携带专名** —— 即便别的模块逐字发出同一句，译出来的中文对它**也是对的**；"
    "被删那 4 条的插值位是**名字**（事件名 / 角色名 / partId / 颜色串），「错的中文框 + 真的专名」"
    "= 一句关于某个具名事物的**错误陈述**（Mirror Image → 「镜子 Image 不存在！」就是现场）。"
    "⇒ HIGH / MED 的分界不是「有没有锚点」，是「**插值位上是不是专名**」。"
    "｜用例随之改：通知正例 36 → 32（删 5 改 1 加 1），通知反例 14 → **21**（4 条删除 + 1 条收紧"
    "各自的反向证据，含两条现网越界现场）；`recorded` 的 np_size / notify_positive / notify_negative "
    "三个数同步记成新值 —— 按第二十八轮两层判法，**这三处改小/改大都必须留在 diff 里自证**。"
)

with io.open(OUT, "w", encoding="utf-8") as fh:
    json.dump(d, fh, ensure_ascii=False, indent=1)
    fh.write("\n")

print("写出：", OUT)
print("notify_positive:", before_pos, "->", len(rule["notify_positive"]))
print("notify_negative:", len(rule["notify_negative"]))
print("recorded:", {k: rule["recorded"][k] for k in
                    ("np_size", "notify_positive", "notify_negative")})
sys.exit(0)
