# 本目录里两个批次已作废（第十六轮收口段，2026-08-15）

落盘人在第十六轮收口段按主控裁决作废，**任何会话都不要再落它们**。
作废方式：文件名加 `.rejected` 后缀（保留内容作证据，不删）。

## `z1b-lang-ember-token.json.rejected`

把 ember `lang/cn.json` 的 5 个 `Token` 键改成「令牌」
（`ACTOR.CONTROLS.EmberToken` / `EMBER.CONTROLS.Teleport` /
`EMBER.ACTOR_FLAGS.FIELDS.disableDynamicToken.label` / `.hint` /
`EMBER.ACTOR_FLAGS.SHEET.TOKEN`）。

**与既定裁决 1 方向完全相反**：`Token` 的定译是**指示物**，「令牌/代币」零容忍
（唯一豁免是 4 叶故事内信物）。落这个批次会把 5 个正确的键改坏。

实测：落盘前后两仓 compendium + lang 全库「令牌」计数均为 **0**，这个批次从未落过盘。

## `z3-arcturian.json.rejected`

把 `Ember Early Access.actors.Arcturian` 的 `.name` / `.tokenName`
由「阿克图里安人 Arcturian」改成「阿克图里安 Arcturian」。

**主控裁决 R-A 已判这是改错了**：`Arcturian` 是**已留证的合法分裂** ——
定语／文化标签（`Arcturian dwellings`、`Arcturian Wirrun`，英文侧 472 次）＝**阿克图里安**；
指人的名词（单 86 + 复 137 = 223 次）＝**阿克图里安人**。
这张角色卡是一个「人」，所以 name/tokenName 必须留「阿克图里安人」。

实测：库里现值就是「阿克图里安人 Arcturian」（这个批次同样从未落过盘），
`scan_label_vs_name` 现为 **2**（两处 Maziran 孪生，与本轮无关），
断言 `R-arcturian-actor-card` 绿。
