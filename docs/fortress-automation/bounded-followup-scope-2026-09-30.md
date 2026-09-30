# 后续复杂流程的范围决定

本轮自动完成激发武器的指定武器下一次消费、原次延后伤害与重掷快照。以下范围依据已捕获的安装源和现行自制 provider 评估；不引入通用聊天推断或第二套付款、伤害应用者。

| 范围 | 源码证据与限制 | 当前决定与后续验收条件 |
|---|---|---|
| 任意外部混合伤害 / 逐来源 IWR | 当前 Toolbelt **3.56.5** source map 的 `src/tools/better-chat/merge/process.ts:133–155` 仍按 damage type + persistent 合并及 materials union；`:188–199` 把 ignoredResistances 放入整体 roll、仅取 messageRoll 的 critRule；`:212–225` 的重建 DamageInstance 没有逐来源 degree/IWR 归属。`merge/_utils.ts:32–64、77–80` 保留原 source/outcome 到 mergeFlags，但静态源未证明 native apply 会逐来源使用这些 flags。该证据复核了先前 3.56.2 的手工范围，不表示第三方完全丢失原来源。 | 保留已验证的自制组合入口。任意 Toolbelt merge / inject / 外部宏继续保留原卡供 GM 核对；不将整体 critical marker 推广为每一来源的正确 IWR。后续需普通命中+暴击、critical immunity、来源独立 bypass/material、basic save倍伤/持续伤害的真实 PF2e 对照，再设计逐来源证据。 |
| 回算单次攻击/豁免的复杂反应 | 现有 reaction、Scar、Glimpse 等入口保存精确 actor/source/card/application 或原生 check 作用域。旧 settlement 决策已保留飞弹偏斜护腕、三角齿、反应掩护、反制表演的 GM 结果回算；攻击卡后再推测一个新 AC 或全世界重做 saving throw 缺乏单次身份。 | 保留原生主动反应选择和当前已支持的特定 provider。未支持者由 GM 对原次结果裁定；新增某能力时只绑定其原生 invocation 和 native reroll 后最终结果，证明一次付款、取消和来源回合到期。不能依照最后一张聊天卡、名字或几何推断自动回算。 |
| 法袭箭、弹药与第三方组合器 | Ranged Combat 8.0.7 `fire-weapon-processor.js:17–88` 已在 weapon-attack 下消费 loaded/inventory ammo；`ammunition-effects.js:28–49` 替换 native prepareSiblingData 并拥有 reload/fire/damage 的弹药规则生命周期。当前 custom spell-combination 处理所选 spell/weapon 的实际 native attack/damage，未提供储存法术弹药通用入口。 | 原有 native/Ranged Combat 拥有装填、弹药消耗和效果；本插件不再付款或复制其生命周期。法袭箭保持手工核对储存 spell、save/persistent 和资源，直到真实角色需求绑定具体 module API、弹药 source 和单次消费并测试网络不确定状态。 |
| 召唤师 / 幻想伙伴归属 | 已检查的 Companion Compendia 7.7.2 `module.json:123` 提供 Eidolon pack，`:364–365` 仅加载 warnings.js；pack 并不证明共享 HP/行动的运行时自动化。现有 companion provider 管理特定已知 pair 的 support，不是通用 Summoner 主从账本。本轮未取得其他 Summoner 运行时的完整安装/启用与 API 证据。 | 保留当前具体 support 入口。后续先查真实启用 module 和主从 Actor/Token 权限，确定共享 HP、action、effect 的唯一写入者，再验证 player/GM双client和独立 synthetic Token。不得把同 ownership、名字或 party 关系当成幻想伙伴绑定，也不在证据不全时新增自动同步。 |

其他安装源位于 `C:\Users\Taka\Desktop\fvtt\tmp\fortress-gap-audit-20260925\code\modules`。当前 Toolbelt 3.56.5 证据为 [QA 安装 source map](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/qa/runtime/Data/modules/pf2e-toolbelt/scripts/main.js.map)；Toolbelt 内容行号为 map 内原 TypeScript 源码行号，其他为各包原文件行号。以上是范围决定与静态证据，未声称复杂项已现场验收或全部自动化。移动、距离、视线与触碰事实不进入新增监控。
