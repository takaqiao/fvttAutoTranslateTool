# 原生自动化补全执行记录

本轮 Task 1–7 的实现和回归已完成，Task 8 的全量测试及最终独立审核已确认。最后发布候选、版本化发版包和正式部署仍待完成；下述测试与隔离 QA 结果不代表已部署。

计划：[2026-09-30-automation-native-followups.md](../superpowers/plans/2026-09-30-automation-native-followups.md)。工作区为 `C:/Users/Taka/.codex/worktrees/automation-native-20260930/fvtt`，分支 `codex/automation-native-20260930`。用户最新明确不要移动、路线、距离、视线、触碰检测或重复确认；必要的真实反应选择、规则分支和 GM 知识裁定保留。已授权实施，不再重复请求方案批准。

## Task 1–7 的实现与证据

| 任务 | 已完成的行为 | 验证与证据 |
|---|---|---|
| 1：生产基线与测试核对 | 只读捕获 CN 0.9.18.6 的 137 个运行文件，在隔离工作区保留 canonical 测试。过时版本/源码锁断言改为真实接口及业务能力断言；Voltage mock 补齐原生批量更新和部分已提交回执，未恢复运行时版本锁。 | 原基线 1758 项：1623 pass、67 fail、68 skip，外部 `baseline-tests.log`。原 67 个失败已核对消除；11 个归属测试 227 项、224 pass、0 fail、3 环境 skip。见 [基线核对](baseline-test-reconciliation.md)，提交 `aca21ea0`。后续完整真实依赖运行无 skip。 |
| 2：名称权限及原生流程 | 目标名称按接收者权限显示。医疗、列盾突进、恐惧、Eat Fortune、维持等保留原生入口及实际结果，去除额外空间检测和动作完成确认；必要的原操作者原生 UI 保留。魔宠角色卡监听在重渲染/关闭时释放。 | `c4059ce6`；[针对性验证](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/native-flow-focused-verification.txt) 246 项、244 pass、0 fail、2 当时的环境 skip。隐藏名字、来源/权限改变、原生取消和 DOM 释放有回归；完整真实依赖运行涵盖这些测试。 |
| 3：索引维护及合并 | Scar 启动恢复一次后索引相关效果和原伤害消息；普通聊天不扫物品。Party/AV/Glimpse 与 main 按相关字段和受影响角色合并维护，异步维护期间新增 dirty 请求仍补一次；真实动作保持独立。Party 不监听 Token 移动，Guardian 保留真实盾牌及来源条件。 | [性能验证](performance-followups-verification.md) RED→GREEN，53/53、0 skip，提交 `737ec67a`。真实 provider hook 计数：普通 Scar chat 零 inventory 扫描，八事件 burst 一次维护，被阻塞维护期间新增请求共两次。此为操作计数，非浏览器耗时 profile。 |
| 4：统一回忆知识 | 原操作者执行 Workbench 风格的一次真实主检定；比较技能使用本地绑定的无骰探测，所有候选共享该次保留骰点。固定技能、Lore、Assurance、Automatic Knowledge 及一次机械收益保留；GM 裁定沿用保存原卡。目标 Actor 绑定和丢失 RPC 回复均以持久结果验证。 | RK bridge、entrypoint/HUD/hotbar 接线和 `c504414f` 的无副作用探测已回归；主检定重新进入当前 public Check wrapper，避免重用已失效的 libWrapper continuation。`b710aaab` 补强原卡输出失败后的单次规则消费；`cacd945d`、`c1fd032a`、`4e293522` 保留原生 Assurance 和相抵后完整加值。最终独立 **52/52、0 skip**，执行 PF2e 8.5.1 StatisticCheck、Modifier、Check parser、FlatModifier、RollTwice、SubstituteRoll 源；普通、无目标、隐藏目标、fortune/substitution 和最新 Assurance 两分支都有实机绿色证据，见下表。 |
| 5：使用状态及凭据 | 真实观察的付款凭据不再因五秒延迟失效；认领绑定 item/user/原消息并保持单次。等待、取消、完成分开显示；删除源文档及关闭角色卡释放相应引用。native-sheet 和 metapower 共用一条 libWrapper 路径和真实 awaited Use。 | usage 初次 RED→GREEN 29/29。实际 main callback、shared factories 和真实 ledger 回归证明 begin/finish 各一次、native Use 一次，hotbar 同样单层；提交 `25aa0183`。最后与 owner/lifecycle 联测 57/57、0 skip。 |
| 6：原操作者组合流程与重掷 | 组合攻击、武器/法术伤害在原操作者端调用原生 API，遵守其窗口设置；GM 只接收绑定原卡 nonce 的持久结果。首次原生接受才付款，取消不冒充结果，丢回复不重复投骰或付款。玩家/GM 保留重掷使旧合伤失效；原生已付款法术卡可继续，合伤按保留结果由 GM 核对，已应用伤害用原生撤销。 | combo+owner 曾联测 119/119。最终独立 owner/lifecycle/shared-sheet/usage 五文件联测 57/57、0 skip，涵盖断线、目标重绑、durable done/committed 丢回复、付款/取消、重掷同步拦截及等待反应后的末端拦截。已进入原生 HP 更新的操作不做补偿，始终提示如旧伤害已应用先原生撤销。 |
| 7：激发武器及复杂范围决定 | 下一次指定武器实际 Strike 命中或未命中消费原 Surge；别的武器、取消、后来替换的效果保留。原/空快照存在 PF2e context，延迟伤害及英雄点重掷不能借用或消费后来新 Surge。并发真实动作不合并；付款重prepare 时按 variant 对象防止重复包装及 gate 自等。 | [Surge 验证](weapon-surge-verification.md)：RED→GREEN，最新 68/68、0 skip；真实 DamageDice 源执行 1/5/9 环 spirit d6 为 1/2/3。提交 `21fdad2c`、`93f8550e`、`a1f6e9c6`。任意外部逐来源 IWR 合伤、复杂反应、法袭箭/弹药及 Summoner 边界见 [范围决定](bounded-followup-scope-2026-09-30.md)，未无证新增通用自动化。 |

范围文档中的 Toolbelt 3.56.2 证据已另用当前 QA/生产版本 **3.56.5** 复核：[当前安装 source map](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/qa/runtime/Data/modules/pf2e-toolbelt/scripts/main.js.map) 内 `src/tools/better-chat/merge/process.ts:133–155` 仍按 damage type/persistent 聚合并合并 materials；`:188–199` 仍把 ignoredResistances 放入整体新 roll、仅保留 messageRoll 的 critRule，`:212–225` 重建 DamageInstance 未保存逐来源 degree/IWR 归属。`merge/_utils.ts:32–64、77–80` 确实保存原 source/outcome 到 Toolbelt merge flags，不能说原来源完全丢失；但当前静态源未证明原生 IWR 应用会逐来源消费这些 flags。因此维持原范围决定，不将任意外部 merge 推定为逐来源机械结算已经正确。上述行号为 map 中原 TypeScript 源，非单行 JSON map 的物理行号。

## 独立审查发现与修复

- shared sheet 曾让 usage 和 metapower 各观察一次，导致第二个 lease 冲突、原生 Use 未运行。main 的 `entry:'native-sheet'` 只做相应 beforeUse，保留内层 metapower，hotbar 仍走一次观察；真实 factories+ledger 回归已通过。
- 真实 QA 发现关闭检定窗口后，付款更新 Actor 重建 Strike，Knowledge 外层 wrapper 遮住 Surge 函数身份，重新包装后等待自身 gate。Surge 改按 variant 对象及 strike+damage method 去重；原生付款重prepare+真实 Knowledge wrapper 的 RED→GREEN 和现场 fresh case 均通过。已经付款的旧不确定卡保留，未重试。
- kept reroll 原先先 await GM linkage 保存才拦旧伤害，玩家也等待 GM 广播。现所有客户端识别原来源后在任何 await 前同步标记 stale，GM 持久写入仍独立排队；同时绑定 target Token 和 Actor。被阻塞的 linkage 写入与玩家广播空窗回归已通过。
- 原操作者在 started 原生窗口期间断线、RPC 不回复时，GM 等待曾无法结束并保留三个 hook。现未完成操作检查原操作者 online/OWNER、来源与目标，断线释放等待；已经保存的 done 仍权威有效。无正常原生 UI 超时限制、退款或自动重试。
- 旧伤害进入 pipeline 后可能等待真实反应，期间重掷；仅在 pipeline 开头检查不足。main 的 scar/no-scar 最深原生入口共用 `activityResults.applyNativeDamage`，实际调用前重查，回归证明 native HP application 为零。
- PF2e 先 await HP 更新、后发布 damage-taken 回执，回执可能晚于重掷失效标记。失效卡和活动状态始终说明“如旧伤害已应用，请先原生撤销”，不尝试回滚或补偿 HP。

最后 Assurance 实机对照发现：提前发送 `substitute:assurance` 让原生 AdjustModifier 抑制能力值，而空 predicate 的 Modifier 在 fortune/misfortune 抵消后仍保留 ignored 状态。实际旧分支为 `1d20 + 12`，同角色普通原生对照为 `2d20kl + 16`。提交 `c1fd032a`、`4e293522` 改先捕获完整普通原生 check，仅未抵消的主检定加入 Assurance 标记；没有显式 DC 的固定 Lore 保留数值给 GM，不编造识别 DC。直接执行真实 Modifier 整 class 的回归涵盖这个持久抑制状态。

上述最终源码独立复测为 ROOT 接线 57/57 和 RK/Assurance 源测试 **52/52**，均无 skip；后者在 `4e293522` 上重新执行，含真实 Check parser、Modifier 和规则 afterRoll。无冲突检定严格验证零骰 `10 + prepared proficiency`、排除其他加值；原生 unconditional 效果经持久 claim 消费一次，未启用的非熟练 if-enabled 效果保留，DoS/DC 在消费前冻结。misfortune 抵消保留完整 check，由原生 parser 投一个普通骰，状态保存 `assurance=false / assuranceRequested=true`。最新双客户端两分支均通过，独立审查未发现当前同范围未处理阻断；最后发布候选确认仍由 Task 8 收口。

## 隔离双客户端 QA 已证明的边界

现场为 loopback 的独立 Foundry 14.368 / PF2e 8.5.1，克隆 world 与实际玩家、GM 两客户端；依赖包括 Workbench 7.7.5、HUD 2.55.2、Toolbelt 3.56.5、Patreon 3.2.29、Trigger Engine 1.35.1、libWrapper 1.13.5.1 和 socketlib。最后候选 smoke 启用 Dailies 4.20.0；生产为 4.20.1，两个版本有差别。以下均为隔离验证，无正式服务器部署结论。

报告目录：`C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/qa/`。`dual-client-report.json` 会被新的 attempt 更新；固定历史报告用于保留以下证据。

| 已验证案例 | 实际证据 | 覆盖限制 |
|---|---|---|
| 原操作者窗口取消和接受 | `dual-client-report-1790734600815.json`：`owner-close-proof` 为 cancelled、payment 0、资源仍 3、GM dialog 0；`owner-accept-result` 为玩家作者真实出卡、payment 1、资源 3→2。 | 以 owner helper 和英雄点资源替代凭据验证真实原生边界；未逐步点击完整 SpellStrike UI，也不代表每个组合能力的全部分支均已验收。 |
| 关闭检定窗口，付款更新 Actor 后出卡 | `dual-client-report-1790735200527.json`：`owner-false-fixed-proof/ui` 为 rolled、payment 1、资源 3→2、dialog 0；旧 paid started 卡仍保留。 | 证明 Surge gate 修复后的 fresh invocation；未重放旧不确定动作。 |
| Surge 未命中、后来新效果及延迟原伤害 | 同一固定报告的 `surge-native-case`：Longsword outcome failure，旧 1 环效果删除、新 9 环效果保留，原快照 rank 1，延迟 native Damage 为 `(1d8 + 3) slashing + 1d6 spirit`。 | 现场直接证明该 MISS/延迟案例；其他武器、各命中级别、并发、null outcome 和 1/5/9 环缩放另有离线/真实规则源回归，不能把这些都说成现场逐例通过。 |
| 玩家真实英雄点重掷及旧合伤拦截 | `dual-client-report-1790735691070.json`：`hero-genuine-d20-native-reroll` 使用真实 CheckRoll，旧卡删除、保留卡 evaluated/isReroll、Workbench auto-damage 禁用，英雄点 2→1；`hero-reroll-gm-final-state` 重新绑定原活动、paid 保留、旧合伤 superseded、usage waiting、无 autoDamageCards。GM 与 player 的旧 damage apply 均 blocked/markedUnapplied、目标 HP 200→200。 | 验证玩家真实 reroll 与两端拦截；GM 自己执行 reroll、已经应用伤害的原生 undo、英雄点+Surge 延迟伤害组合及网络掉线现场未据此宣称全覆盖。相关绑定/取消/丢回复/竞态已有回归。 |
| RK 普通、无目标及一次性加值 | `dual-client-report-1790736537814.json`：`rk-no-target-fresh-native-frame` 真实 CheckRoll 仅一次 `1d20 + 6`，FlatModifier afterRoll/delete 各一次；七个候选均保留本次消费前加值。 | 无目标保留 blind 比较，不编造生物 DC 或机械识别成功。一次性加值不是每个探测各消费一次。 |
| RK 隐藏目标及玩家 DOM | 同一报告 `rk-hidden-humanoid-native` 只选适用 Society，保留骰 5、modifier 6、total 11、DC 20、degree 1，原玩家作者、blind、仅 GM whisper、resolved；`rk-privacy-player-render` 的 actual DOM `isContentVisible=false`，无隐藏真名/DC。 | 证明该隐藏 humanoid 的实际渲染隐私，不替代所有第三方 UI 入口的逐例验收。 |
| RK fortune 与原生替代值 | 同一报告 `rk-fortune-native-correct-expiry` 仅一次真实 `2d20kh + 6`，8 discarded、12 active，七候选共享 12，FlatModifier/RollTwice 各 afterRoll/delete 一次；`rk-native-substitution` 仅一次 `17 + 6`、零骰，原卡 constant 17，Society 23/DC20/degree2，FlatModifier/SubstituteRoll 各 afterRoll/delete 一次。 | 使用真实选中 dice/substitution；正常对照不是强行固定一次裸 d20。Assurance 另有后续报告，不能把此 17 替代值案例说成 Assurance。 |
| Assurance 固定值与真实规则消费 | [固定报告 1790739052554](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/qa/dual-client-report-1790739052554.json) `assurance-clean-probe-fixed-native`：真实 `10 + 9`、零骰、total19/DC20/degree1、原卡 raw constant10，die:null；unconditional FlatModifier 删除一次，未启用 if-enabled 保留，probeUse done。 | 真实 Society expert/proficiency9 专长与两 FlatModifier 效果，使用 `4e293522` 后的运行文件；PWL 的 prepared 值与其他技能另由回归覆盖。 |
| Assurance 与 misfortune 抵消 | 同一报告 `assurance-clean-probe-misfortune-native`：真实 `1d20 + 16`（INT4+prof9+circ3），die10/total26/DC20/degree2，固定 Society、assurance:false/requested:true，各 rule afterRoll 一次、两 FlatModifier 各删除一次；原卡保留真实一个骰。`assurance-clean-probe-source-and-privacy`：GM 原卡说明相抵普通检定，context 无 Assurance/substitute 标记，玩家 DOM 为私骰占位、isContentVisible:false、HP200。 | 证明最新源码的该冲突分支及玩家隐私。misfortune RollTwice 未用于最终单骰，效果保留；不声称所有其他 fortune/misfortune 组合已逐例现场测试。 |

随后固定报告 `dual-client-report-1790736893638.json` 再次证明 owner no-dialog payment 一次、持久 operation done/committed 和玩家作者；新的真实 hero reroll 后，GM/player 均从 main 最终接线阻止旧 damage，标记 unapplied、HP 200→200。`dual-client-report-1790737083525.json` 对 **Assurance 补丁前的 `ebdeb2c7`** 精确 payload 做 normal RK 与 privacy smoke：单次 Society `1d20 + 6`、FlatModifier 删除一次、保留骰 15/total21/DC20/success，玩家 DOM 无隐藏真名/DC、HP仍200。这是旧候选的隔离 smoke，不是最后补丁后的发布确认。

历史 attempt `1790736016698` 的只读 `Set.first` 赋值、过期 libWrapper continuation 以及 harness 接口/`findLast` 问题保留为失败证据；后续 fresh normal RK 报告已通过。`1790738093248` 保留已关闭的 Assurance 相抵缺 INT4 问题及普通原生 +16 对照；其中首个 instrumentation command 给无 afterRoll 的规则包了错误 wrapper，不作为检定结论，后续独立 operation 的固定值 GREEN 有效。最新 `1790739052554` 证明两分支修复，不重试旧 claimed 卡。新发布候选仍须独立证据，不能从旧 smoke 推定最后结果。

## 全量测试与待收口事项

首个完整真实依赖绿色运行记录：**1907 tests、1907 pass、0 fail、0 skip**。命令为 `node --test modules/pf2e-third-party-automation/tests/*.test.mjs`，使用实际 PF2e bundle、Foundry 原生源及已安装依赖所需的测试环境。完整外部日志为 `C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/final-tests.log`。

1907 和普通 RK 后的 1914 均为历史绿色计数。最后 `4e293522` 的完整日志 footer 已核对为 **1921 tests、1921 pass、0 fail、0 skip**，耗时 2822.6118 ms；root 确认 [verify-final.ps1](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/verify-final.ps1) 的全部 runtime syntax 检查与 `git diff --check` 同样 exit 0。后续若有运行代码改动，仍以 [final-tests.log](C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/final-tests.log) 最新完整 footer 为准。

Task 8 的最后全量测试、syntax 检查、源完整性和独立审核已确认；fresh Assurance 现场两分支也已独立核对。版本化发布包、manifest、回滚说明和最后精确 payload smoke 仍 pending。正式发版/部署、远端备份和部署后文件 hash 检查仍 pending；此记录不声称已执行这些动作。浏览器性能 profile 和未列出的复杂模块流程没有新增成功声明。
