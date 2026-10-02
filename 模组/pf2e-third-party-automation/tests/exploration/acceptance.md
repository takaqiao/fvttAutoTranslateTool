# 探索恢复验收

本清单对应已批准设计第 8 节的十二项场景。已发布基线为 `0.9.23`，固定源码 `200bc400a439b58dd2d82ddab6a588e7816a1ed1`；0.9.22 已合入发布的 0.9.19／0.9.20／0.9.21，0.9.23 完善了历史凭据界面。探索自动恢复限定已核验的 PF2e 8.5.1；其他既有能力继续遵循各自源码与原生桥校验。下文保留早期候选运行证据，并明确叠加后续修复和发行证据；旧报告不改写。

**十二项完整场景尚未全部通过。** 下表的“部分”仅表示某些子项已有证据；`ok:true`、当前 HP、世界时间已达到目标、命令名称或以前的测试总数，都不能替代实际原生结果与来源回执。

## 三层验证

### 通用行动登记检查点

通用登记只声明耗时与执行者占用，不证明规则效果已完成。源码验收需覆盖默认自动运行中的原生完成边界、认证 OWNER 表单、并行共同结束、预算冲突、空登记继续，以及 Stop／失权／重载后旧窗口拒绝和未知提交只读回查。独立 Foundry 验收应保存 GM 与玩家的原始提交、六字段窗口绑定、唯一时间回执和刷新读回；既有 Workbench 首轮固定600秒证据不能替代这项验收。本节尚无新增原生通过回执。

### 原生／Patreon 手动免疫小批（0.9.30）

源码 `579a4bef62f5945dfe3ec2f1da334b6fa1e248ab` 的[完整回归](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/native-immunity-source-check-1790861026368.json)为3249／3249、172测试文件、182运行脚本和5工具语法通过，零跳过。一轮独立产品审查发现停止录入与共享池改绑的异步窗口；追加的重复 HP 应用反例与六个原审查反例在合并后的代码通过，新测试覆盖真实 revision 摘要完成后的提交条件。原失败记录保留。

[真实GM＋玩家核验](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/native-manual-immunity-accepted-1790862049312.json)，SHA `97786fa98197d6f2ba52c562c09d2b7661416551880f4758a396679b2727141d`：A由玩家直接创建原一小时免疫，真实治疗20、HP从1到21；B由GM代建持续恢复的原十分钟免疫，真实治疗19、HP从1到20。各有一项检定、治疗结果、HP应用和免疫，原创建完成凭据绑定实际creator、useId、患者、recording会话及start=0。世界时间始终0，没有clock。GM整页刷新前后十三类原生文档、持久化会话／活动／时钟一致，玩家持续观察无新检定、HP、免疫或时间写入。

原验收脚本先后被旧式合法Compendium UUID、画布Token临时加载状态和异步默认Party基线断言中止；核验报告绑定三份原失败报告及原始trace，以首轮已保存Party、持久化账本和完整原生文档补做只读验证。A仅在第一份报告执行一次，B仅在第三份执行一次，没有重投或重新应用。[保存关闭报告](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/native-manual-immunity-preserve-2026-10-01T13-41-02-327Z-83e6f31a.json)，SHA `cc1cb46e1dbff4d60c25c3f4b57f33edbbf97d6cf73676cd41e9160fd1da8bf0`：新GM只读回完整来源后原生关闭世界，全部LOCK释放，十三个旧世界和192份候选运行文件保持。

本批限普通单患者、非共享池手动记录，依赖固定PF2e 8.5.1和Patreon 3.2.29源码及手动免疫补丁；自动首轮检查点仍仅Workbench。共享主人完成、Risky Surgery、多患者、延长及十二场景完整矩阵继续待完成。

0.9.30 已发布并与 Patreon 手动免疫补丁配对部署。[部署交接](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0930/handoff.json)，SHA `490dbcc242a9ba1d8fd00de19640469d7faec8c2de811fc932aa0d91f268f01b`，核对192份本模组运行文件和48份Patreon文件，保留3857个世界文件、其他模组、Core／PF2e、配置与服务。219份可服务文件HTTP字节一致，21份Patreon数据库文件由原生规则返回403，磁盘完整性仍全部核对。原部署回执为installed、safeToReplay=false，不能重放；该发行不表示完整goal或十二场景已完成。

### 首轮手动治疗检查点（0.9.29）

本批只接固定600秒首轮、单患者非共享池的 Workbench 治疗及另一角色的普通再聚能。[真实GM＋玩家报告](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/manual-checkpoint-native-2026-10-01T10-42-18-168Z-fc1b427c.json) SHA `71ad7614229575e0ba4a0354b37ce2214ee34fbb52b1bbb7c206fb847d9bf27a`：开启后时间0，玩家原生检定大成功、真实治疗26，原按钮应用后HP从1到27；同源一小时免疫保存start=0、expiresAt=3600。点击“继续本轮”只提交一次0→600时钟，手动活动确认、普通Refocus恢复一聚能，两项活动共用本轮。检定／结果／HP回执／免疫均各一项，manual不再次调用治疗执行器；十二个旧世界及187份候选运行文件保持。

[新鲜完整源码检查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/manual-checkpoint-source-check-1790851403559.json)为3074／3074、166测试文件、180运行脚本与2工具语法通过，零失败／跳过。一次独立产品复核发现封口可能覆盖晚到第二HP回执；[修复证据](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/manual-checkpoint-review-fix.json)保留旧源码两项失败及修复后208项定向回归。通过报告和完整13类原生来源已在新GM客户端只读复核并正常保存关闭：[停止库存](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/patreon-current-preserve-2026-10-01T10-44-45-290Z-a6439f61.json)，SHA `9459fe5ab3b81ce980bdcf4e7fcfa5059b9c8ce48a5f297221665b68fec1f13d`，全部LOCK释放、源码和旧世界保持。此前两个UI操作失败的世界与已投骰结果原样保留，没有重放旧尝试。

原验收候选的版本字段为私有0.9.28。这批首轮检查点代码已于0.9.29发布部署，源码 `f77177f8b2c255a93f6ab44df7d27374770ff966`；[发行证据](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0929/release/evidence-verification.json)绑定上述八个检查点运行文件与真实验收源码，最终集成测试3098／3098、零跳过。[部署交接](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0929/handoff.json)核对187个服务文件，保留3857个世界文件、Patreon时间补丁及其他模组／服务／配置。原生action缺免疫、共享主人回执、群体／延长及其他时长检查点仍待适配；本批不改变十二场景整体状态。

### Patreon时间效果完成回执（0.9.28）

[真实GM双标签页报告](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/patreon-v2-workbench-continuation-2026-10-01T08-50-13-127Z-8391b34b.json)，SHA `e5a6f31859bb740315f779b74997547ac0c39ab247f697d9da5b67f084d466e1`，保存两个已结算活动。A轮一次真实快速治疗骰、原生HP写入和效果倒计数完成后普通再聚能恢复一点；B轮没有符合治疗规则的成员，零治疗骰／HP应用，仍等待原生效果倒计数完成。每轮只调用一次提供者并计600秒，另一标签页零调用，累计1200秒。

这批以源码 `faa7e9157a550ca5fb648c530899459ca985d461` 发布部署，3067／3067、零跳过；[发行证据](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0928-handoff/release-final/evidence-verification.json)绑定实际运行文件及Patreon 3.2.29准确源码补丁，[部署交接](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0928-handoff/handoff.json)核对187个服务文件与独立Patreon补丁。付费源码未加入发行包。原生／Patreon手动治疗免疫的来源关联是后续独立适配，不由这项时间验收证明。

### 手动时间声明小批（0.9.25）

此批基于已部署0.9.24，仅增加通用手动声明、时长出处、声明时段、同会话前置关系和面板展示。[实际GM＋玩家报告](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/manual-time-native-2026-10-01T05-54-12-301Z-005f1f95.json)及[一次结果复核](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/manual-time-native-result-review.md)通过：GM真实对话框登记A十分钟，玩家独立客户端登记B同时五分钟和A后续五分钟，重建900秒、certainty为incomplete。作者身份、出处、时段、依赖及空原生proof持久化，刷新后相同。未拥有角色、错误会话及不完整伪治疗凭据拒绝且无新增。

前后各13集合原生GET ACK确认Actor／Item／ChatMessage／Scene／Combat与世界时间0不变，无clock commit；180项安装／HTTP源码一致，五个已停QA世界完整保留。此项仅启用本模组和三个必要依赖，不覆盖Patreon手动免疫、自动检查点接入或治疗效率的新分支。[本批完整源码回归](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0925/source-check.json)为2807／2807、0fail／skip／cancel／todo，152测试文件及176运行脚本语法检查通过。旧报告及十二场景状态保留。

| 层级 | 入口 | 能证明什么 |
| --- | --- | --- |
| 调度／状态单元测试 | `node --test tests/exploration/*.test.mjs` | 用确定输入和替身验证时间关系、状态转换、权限、预算、资源和拒绝重放 |
| 原生源码测试与已有模块回归 | 配齐真实来源环境后运行 `node --test tests/*.test.mjs tests/exploration/*.test.mjs` | 执行已核验源码或检查接口契约；替身 Actor、Hook、socket 的结果不能当作运行世界验收 |
| 独立 Foundry GM＋玩家 QA | 本次独立世界的 `qa/dual-client-report-*.json` | 实际准备后的角色、原生骰子、ChatMessage、HP／聚能写入、免疫与世界时间事件 |

源码测试需要提供真实已核验的 `PF2E_NATIVE_BUNDLE`、`FVTT_PF2E_BUNDLE`、`FVTT_NATIVE_APP`、`FVTT_COUNTERACT_MAIN`、`FVTT_REACTION_BUNDLE`、`FVTT_FORCE_BARRAGE_MACRO`、`FVTT_WORKBENCH_MACRO`、`FVTT_WORKBENCH_RECALL_MACRO` 路径。结果必须保存本次完整 TAP 输出和来源哈希；缺少来源而跳过测试不算通过。这里没有沿用以前版本的 pass 数，也没有把测试命令列出当作已经执行。

0.9.23 的 [完整源码测试](C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/verification/release-history-full-suite.log) 为2294 tests／2294 pass／0 fail／0 skipped；[最终发行验证](C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/release-tools-0923/release/evidence-verification.json) 绑定固定源码、实际客户端报告与文件表，并保留 `completeTwelveScenarioMatrixPassed:false`。早期 [pre-review-full-suite.log](C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/verification/pre-review-full-suite.log) 的2008项属于修复前候选；不能作为当前计数。源码测试不代表十二项真实世界验收通过；后续产品修改需重新验证相应源码。

质量 goal 的第一批未发版修复另有 [新鲜集成测试](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/integrated-full-suite-20260930T161050829Z.log)：2338 tests／2338 pass／0 fail／0 skipped。覆盖异步时间取消、命能资格、患者接收上下文、再聚能资格、手动回执授权和免疫恢复时重验 HP 证据。后续候选已补原子账本、私有 driver／OWNER 执行权和原生窗口取消；[最终冻结源码验证](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/owner-candidate-source-verification-20261001T011021862Z.json) 为2767／2767、331个语法检查与diff检查通过。这些阶段一结果已于2026-10-01以0.9.24发布部署；[发行验证](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0924/release/evidence-verification.json)绑定2767／2767及下述原生报告，[部署后验证](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/release-0924/postcheck.json)核对179项HTTP文件和世界／服务／配置／其他模组。它们不能回写为已发布 .23 的验收；完整goal与十二场景仍未完成。

## 质量 goal 阶段一的实际原生证据

2026-10-01 的冻结候选为179个运行文件，来源清单 [source-wire-candidate-1790817157404-35b2f732.json](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/source-wire-candidate-1790817157404-35b2f732.json)。QA执行时版本字符串仍为0.9.23；该冻结运行代码随后以0.9.24独立提交、发布并部署，发行清单和说明按新版本更新。Foundry14.368／PF2e8.5.1使用真实原生骰子；同一GM用户和同一OWNER用户各有两个独立标签页。它证明这些客户端间的唯一执行，不能替代不同GM用户交接、掉线、遭遇和外部时间的完整矩阵。

| 子项 | 已保存的实际证据 |
| --- | --- |
| 本地原生治疗、传奇治疗与普通再聚能 | [独立实际审查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/installed-native-settled-run-independent-review.md)：三个已结算会话，唯一原生结果、应用和各次600秒时钟；首次治疗的旧失败报告保留，只读复验后才执行新的活动 |
| 玩家OWNER原生治疗、普通再聚能及GM治疗 | [权限完成报告](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/installed-native-wire-permission-completion-2026-10-01T03-33-49-540Z-4ecb4adb.json)：四个已结算活动、四次时间提交累计2400秒，来源绑定的回执与实际刷新复验；OWNER偏好恢复到最初raw flags。旧报告的错误断言仍保留，不重投旧治疗 |
| OWNER／患者失权及离线拒绝边界 | 同一权限完成报告：施治者失权拒绝，患者失权会话暂停且没有活动／时钟／授权／原生效果，离线OWNER拒绝；[独立审查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/wire-permission-actual-review.md)。患者暂停不被改称已完成治疗 |
| 非driver GM中止玩家窗口 | [S1独立实际审查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/wire-fresh-s1-actual-review.md)：一个OWNER Check窗口、两个认证的私有retired ACK、窗口关闭；没有投骰或HP／资源结果，时间仅提交600秒，未完成执行域保留 |
| 实际生产超时 | [S2独立实际审查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/wire-fresh-s2-actual-review.md)：实际到期60005毫秒、driver本地60129毫秒；一个窗口关闭，未触发Stop或投骰。活动与Actor执行保持uncertain／granted，无结果，600秒只计一次 |
| 远端仙露拒绝边界 | [R6独立实际审查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/wire-r6-actual-review.md)：真实神秘技能仙露提案在remote-three-pecks-unsupported处暂停；没有执行或时间域。这证明安全拒绝，远端仙露适配仍待收尾 |

原生ObjectField会展开UUID键，候选将physical explorationRevision保存为严格canonical JSON字符串；实际账本合法链、完整文档源与回执已复核。[独立codec审查](/C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/stage1/revision-wire-independent-review.md) 保留合法legacy与非法输入测试。三个历史世界和S1世界已按原生shutdown、锁释放及完整停止库存保全；S2仍保存未知执行域。旧failed报告、claim和未知结果不被清除，以上各次新报告不授权重放旧活动。

阶段一的定向修复与这些实际子项已闭合。后续功能适配、完整十二场景以及全模组审计／实测性能仍须分别完成；本表不把这些待办列为通过。

本次 ownQA 使用 Foundry 14.368、PF2e 8.5.1、Workbench 7.7.5、Toolbelt 3.56.5、Patreon 3.2.29，GM 和普通玩家为不同客户端。证据位于主工作区 `output/exploration-recovery-20260930/`：

- [JSON证据索引](/C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/acceptance-evidence-index.json) 保留实际 command value、来源哈希、去重 session、check／result／receipt／immunity ID、时间 nonce 与禁止重放记录。
- [可读索引](/C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/acceptance-evidence-index.md) 列出完整场景及子项状态。
- `qa/roster-fixture.json` 为稳定远端备份导出的完整角色，SHA256 为 `78aa37476ff7cb3c6675209b083ebb6c5c8958e434503f7754ae637a53b639f4`。二十一名名册成员已经包含贝肯，即二十普通 PC＋一幻灵；完整源角色及付费依赖正文不进入发布包。

后续验收使用新文件记录，不覆盖这些报告。每次验收固定捕获时间、文件哈希和候选源码哈希；同一版本字符串不能证明后来修改的候选已通过旧运行。源报告保持原样，索引按 sessionId 去重。无 value 的 arrow 命令不计为执行，空数组查询只证明没有查询结果。0.9.23 的实际发行报告为 [release-qa-0923-report-1790771223350.json](C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/qa/release-qa-0923-report-1790771223350.json)，不能把早期副本的失败或成功改称这份发行验收。

## 已保存的核心真实结果

| 实际运行 | 可复核来源 |
| --- | --- |
| 原生治疗及激进手术 | `physical-native-result` 返回检定、两个结果、两个应用回执和免疫 ID；共享血池新测试另保存手术／治疗两次主人写入前后值 |
| 盾哥同时治疗三患者 | session `530173d5-4261-4d30-b613-007ee413ade4`；一个 confirmed 活动、三 check、六 result、六 receipt，时钟 3882→4482 |
| 极雷满聚能仙露治疗 | session `1e1a3b9d-1f27-47ba-ab36-d7ccd23bc1a8`；focus 3→3，治疗与再聚能共计十分钟，时钟 3282→3882 |
| 四次圣疗＋三次再聚能 | session `b22a7d72-eb5c-4a89-ac7d-002442645568`；四个 `focus-healing` 各 6 HP、三个 Refocus 各 0→1；8682→10506，共 1824 秒 |
| 共用血池等待真实主人写入 | session `1190ce3a-a368-4d48-8bf3-4a374110cde7`；canonical master pool 的两次写入带 activityId、actorUUID、before／after 和应用回执 |
| 医师情境与共享患者主人完成 | session `4fc52d11-052d-4ccf-bc7c-56ba03f46f31`：prepared医药rank2／Medic，专家DC、真实大成功；原生治疗42、Medic bonus5，主人应用receipt `fCXECnmvjQnS3oZM`／`DguwVPSrirr7sMK5`，血池50→49→73，免疫绑定幻灵患者，nonce为56301→56901 |
| Assurance 14 实际失败 | session `ed2ac8ad-2ed4-477f-a6d8-ac3519e6c5f8`；total 14、dice 0、native failure，已有免疫回执，600 秒后因预算暂停 |
| 普通再聚能每段只恢复一点 | session `32e4610b-21b7-4a66-97f9-765a5bc3e273`；0→1→2，各段 600 秒、各自 provider 回执与时间 nonce |
| 普通／持续恢复三次治疗 | 普通 session `abe3fe5e-992e-4b96-aba2-f5e527640fca`：7800秒；持续恢复 session `a48b5811-835a-46e6-b608-82cc31c4d263`：1800秒；均为同患者三个真实原生检定、合法冷却与 matching nonce。Assurance failure 不表示成功恢复HP |
| 已有免疫先等待183秒 | session `a8b47de6-79b7-43ff-bab5-8c07a103a005`：legacy Effect 初始未到期／remaining183，24918→25101等待，再→25701治疗，有check、HP应用及新免疫ID |
| 来源绑定 Workbench 玩家手动治疗 | session `8d0c8d6f-40ea-467c-a2a6-881c53961618`：check `SwzdOpYa1RkLb6gF`、result `3kRr5AROhLsMA6dx`、HP receipt `UQV1I4XKkhwfLfKi`、canonical immunity `Actor.8y61Mqd8KiZIwUx9.Item.WRX2Y4CTMhYHc4F5`，活动 confirmed；该独立副本关闭 Patreon 治疗免疫handler，手动时钟保持10506 |
| 成功延长一小时 | session `93924a12-716d-438b-ae0a-eaafa3db0756`：21318→24918共3600秒；原活动与延长活动共用check `J5XjvvwhFMjyGFVp`／result `OXVTaX6NuZxB8FOm`，追加应用receipt `5cQeTTUMAY1TkWiR` |

以上是子项运行证据，不是十二项全部验收。早期缺 pack、共享转发未确认和 Refocus 来源变化的失败 session 继续保留；新副本成功不会授权重放旧尝试。

[面板实际截图](/C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/qa/panel-actual-screenshot.png) 及 `panel-native-v14-render` 的文字／控件返回值证明该次共享血池目标和停止状态实际显示。历史报告包含 browser pageerror：`native-refocus-source-changed`，以及源角色西里斯▪格林遗留 Light 效果 `CLEw7MvZYCNrz49J` 的重复删除错误；原始错误均保留，不能宣称浏览器零错误。

## 十二项完整场景

| # | 场景 | 单元／源码入口 | ownQA 当前证据及剩余输入 |
| --- | --- | --- | --- |
| 1 | 单治疗者三患者、双治疗者二／三患者、群体治疗 | `timeline.test.mjs`、`policy.test.mjs` | 部分：Ward Medic 三患者十分钟、双治疗者三患者二十分钟已确认；补单治疗者三患者及双治疗者二患者真实计划、完成点与回执 |
| 2 | 普通免疫／持续恢复三次治疗为130／30分钟；尊重已有冷却 | `timeline.test.mjs`、`treatment.test.mjs` | 已保存当前QA场景通过证据：同患者三个普通治疗检定7800秒、持续恢复三个1800秒、fresh legacy183秒先等待。保留源副本资格修改、check／免疫ID及各段nonce；不宣称三次均治疗成功 |
| 3 | 四级真实随机结果与激进手术，原生应用顺序、成功度一次调整 | `native-treatment.test.mjs`、`treatment.test.mjs` | 部分：成功／大成功、手术和控制医药情境修正后的真实d20大失败已有结果；Assurance failure 是确定性结果。补随机失败及调整前后成功度、全分支应用顺序 |
| 4 | 仙露＋再聚能十分钟、subscriber 不重复、满聚能治疗、圣疗循环 | `refocus.test.mjs`、`refocus-quota.test.mjs` | 部分：满聚能仙露、圣疗四次／再聚能三次及普通 Refocus 每次一点已保存；补每次施法聚能消费 receipt、subscriber 全部消息／写入去重查询及完整事件 nonce |
| 5 | Ward Medic 容量及能力／房规／变体来自 prepared 数据 | `capabilities.test.mjs`、`efficiency.test.mjs` | 部分：实际准备后的盾哥 rank4／容量8、科索斯 rank1、普莱德 QA rank2 已记录；补来源与所有要求的变体／房规资格变化。源JSON熟练度不能替代这些值 |
| 6 | 召唤师／幻灵独立活动；共用 HP 不重复计算 | `hp-pool.test.mjs`、`policy.test.mjs` | 部分：共用血池真实主人 Promise 写入已确认；补独立并行动作及同效果两目标的去重 |
| 7 | 手动 Workbench／原生治疗只有一套骰点、HP、免疫与准确来源 | `manual-events.test.mjs`、`manual-proof.test.mjs`、`manual-player.test.mjs`、`patreon-manual-immunity.test.mjs`、`patreon-manual-immunity-owner.test.mjs`、`manual-native-commit-boundary.test.mjs` | 部分：Workbench玩家来源绑定已保存；0.9.30候选普通单患者、非共享池原生／Patreon手动免疫已保存玩家直接创建、GM代建、唯一检定／HP／免疫、原始开始／期限及GM reload不重放的实际证据，世界时间保持0，见本页小批报告。旧待核对轮保留；共享主人完成、Risky Surgery、多患者及完整去重矩阵未通过，首轮自动检查点仍仅Workbench。 |
| 8 | 目标、连续失败、预算、资格失效后停止，显示真实缺口 | `coordinator.test.mjs`、`panel.test.mjs` | 部分：目标完成和 Assurance failure 后预算暂停已保存；补连续随机失败及资格失效。当前独立连续失败阈值尚未实现，失败循环由时间／活动预算约束 |
| 9 | 刷新、GM交接、owner离线、遭遇、外部时钟、未知时间不重放 | `clock.test.mjs`、`owner-operations.test.mjs`、`coordinator.test.mjs` | 部分：最终审查补证已保存 GM 接手与整页刷新后同检查／HP／时间不重放，并记录重载期间玩家手动回执追加；详见 final-review-resolution.md。owner离线、遭遇、外部时间、未知时间来源及多客户端竞态的完整实际矩阵仍需补齐；reload 请求返回本身不算通过 |
| 10 | Patreon 全Party被动恢复在检查点稳定；Calendaria休息一次计时 | `time-effects.test.mjs`、`patreon-time-completion.test.mjs` | 部分：0.9.28已保存有符合成员／无符合成员的原生异步效果完成，GM双标签页每轮唯一600秒且无符合成员时零治疗骰／HP，见本页时间小批。保留早期另一Party来源缺失暂停；Calendaria原生休息保存一个28800秒外部时间事件、探索clocks为空并停止。完整Party／来源／被动恢复矩阵和休息调用nonce归因仍待补。 |
| 11 | 开始／完成上下文、免疫截止与段内到期；拒绝过去回填 | `native-treatment.test.mjs`、`refocus.test.mjs`、`clock.test.mjs` | 部分：活动上下文与多个来源 nonce 已保存；普通结果 expiresAt 为 start＋3600。补原生 Effect start／duration读回、段内到期与拒绝回填 |
| 12 | 延长失败十分钟；成功一小时，用同结果追加一次，无重掷／重复手术 | `treatment.test.mjs`、`native-treatment.test.mjs`、`coordinator.test.mjs` | 部分：成功3600秒、同check／healing结果及追加receipt已保存；补延长失败十分钟、独立全消息／应用计数和开启手术后未重复手术 |

## 支持和资源策略

默认使用可重复治疗及聚能恢复；激进治疗关闭，目标为最大 HP，默认预算120分钟／最多100次活动。面板允许正数分钟与1–100次活动。圣疗计六秒，普通再聚能每十分钟恢复一聚能；“结束时补满聚能”通过多次活动实现。每日次数、法术位和消耗品保持手动，不能为了达到目标伪造资源补充。

特殊聚能恢复专长尚未适配；检测到相关来源时拦截自动再聚能并说明需手动处理，避免按每次一点计算错误耗时。普通Refocus仍按每次一点执行。

固定 DC 只使用当前熟练度允许的选项。“自动”仅在无条件原生上下文和患者治疗修正已核验时比较期望有效治疗，未知情境回落 DC15；预计值用于调度，产品结果仍由真实原生骰子产生。Assurance 使用实际原生熟练加值与替代骰规则，14不能被提升成DC15成功。

手动活动实时观察后写账本，显示的是带并行假设的最早可行时间。相同角色按来源次序排列；不同角色是否同时开始仍需要真实事实。事后治疗不能倒推唯一实际耗时、反向推进世界时间，或重新调用已消耗的原生治疗。通用“登记其他行动”是带用户来源的声明，不获得原生治疗凭据。

## 停止、未知提交与复核

目标完成或预算耗尽后停止。资格／提供者不可用、owner响应未知、遭遇开始、外部时间变化和GM交接也停止新声明。已确认结果保留；待核对、执行未知及时间来源未知的活动禁止自动重掷骰、重应用或再次 advance。

一个时间段须同时拥有原生 advance 完成和对应 `updateWorldTime` 事件，事件需绑定 sessionId、checkpointId、expectedFrom／expectedTo、GM身份。当前 worldTime 达到目标不能证明是哪次提交造成，也不能使丢失回应的尝试变成可重放。原生声明与确切回执补齐后只继续确能证明尚未执行的后续步骤。

未知尝试索引保留 check／result／receipt／immunity 和 from／to／nonce。`native-hp-forward-unconfirmed`、`refocus-evidence-uncertain-no-retry` 及手动缺免疫回执的旧 session 不因新 session 成功而清除。停止、刷新或关闭面板都不倒退已经确认的 HP、资源或世界时间。

Patreon 被动恢复检查覆盖所有 Party 成员，并包含当前Party。没有可等待的异步完成证明时停止推进；不凭等待时长或当前 HP 推测稳定。Calendaria整夜休息作为外部时间authority，恢复会话停止后由原生休息处理；入口和时间事件无法准确归因时人工核对。

## 补齐验收时应保存的输入

每个真实子场景应保存：最终候选提交与关键源码哈希、依赖版本、实际prepared角色／共享血池／资源、sessionId和activityId、每个check／result／application／resource／immunity receipt、Effect开始与截止、时间提交from／to与完整来源nonce、明确停止原因和余下缺口。异常用例另保存错误发生前已有结果及刷新／交接后的同一记录计数。

报告仅写入本次独立QA输出，不打开正式或其他QA的活动数据库，不更改源报告。安装与正式世界启用是独立发布步骤；本清单不授予部署或重放未知尝试的权限。

## 完整审查后的回归

独立完整审查提出八项重要问题，根代理以一次TDD补丁处理：重载／GM接手发现会话及来源核对、手动索引与授权、负数世界时间、各治疗提供者持久保存主人血池回执、未适配再聚能拦截、Stitch Flesh仅增加不死患者，以及群体治疗按成功患者子集延长。测试与实际故障注入说明保存于 `output/exploration-recovery-20260930/verification/`。最终计数以 `final-full-suite.log` 和 `release-report.json` 为准；审查前2008项不是补丁后计数。

来源未知的活动仍不会重放。“核对后结束会话”要求当前GM留下核对说明，保留未知活动／时钟状态，仅解除旧会话对新会话的阻挡；这不是将未知结果改称成功。0.9.23 已实现执行者／患者、详细缺失证据、可见聊天来源跳转、未知时间区间和保存的 GM 核对说明；[独立增量审查](C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/verification/history-final-review.md) 与实际发行 QA 验证查看历史不改变 HP、聚能、时间或账本。十二项完整真实世界验收仍未全部通过。
