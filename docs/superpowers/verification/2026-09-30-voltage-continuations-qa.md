# 高电压续接：原生双客户端验收入口

范围：计划 Task 1，基线 `f45338970afec79e956f3685175758c784f9e098`，Foundry 14.368 / PF2e 8.5.1 / Eldamon 1.5.1。本文保留原生 UI 验收入口，并分开记录已执行的具体证据；源码测试或单分支通过不能替代其他双客户端结果。仅使用本任务 30426 测试世界，不启动或重连旧 30425，不写生产。

## 入口与只读状态

先通过正常原生 Use 引导一次高电压，再用另一生物对原使用者进行真实原生近战命中。不要再次 Use 高电压来续接。原卡、当前命中/豁免卡和角色动作区投影的是同一 `actorUuid + nonce + messageUuid`。

```js
const mod = game.modules.get('pf2e-third-party-automation');
const api = mod.api.eldamon; // root 的 main 接线后使用
const source = await fromUuid('Actor.SOURCE_ID');
const target = await fromUuid('Actor.TARGET_ID');
api.getContinuations(source.uuid); // 拥有者可见的步骤与中文状态
api.getContinuations(target.uuid); // claimed 后按目标 Actor 查找同一原活动
await api.continueActivity({ actorUuid: source.uuid }); // 宏/HUD：选择当前真实步骤
source.sheet.render(true); // 原角色动作区入口
target.sheet.render(true); // 目标无元素威能 feature 也应显示等待的原生豁免
```

当前卡实际按钮：`[data-message-id="CHECK_ID"] [data-voltage-action="hit"]`（高电压反击）、`metal-hit`（声明本次金属武器命中）、`save`（目标拥有者原生豁免）、`damage`（施法者投掷伤害）。角色动作区同样使用 `data-voltage-action`；刷新入口为 `[data-voltage-refresh]`，显示“刷新威能 · 2动作”。精确宏可使用查询返回的绑定：

```js
const entry = api.getContinuations(source.uuid).find(e => e.actions.length);
const step = entry.actions[0];
await api.continueActivity({
  actorUuid: entry.actorUuid, nonce: entry.nonce, messageUuid: entry.messageUuid,
  ...step // action；当前命中卡还会给出 attackUuid
});
```

## 双客户端案例与必须记录的结果

| 命令/案例 ID | 实际操作 | 应记录的结果 |
|---|---|---|
| `voltage-current-hit` | 使用者拥有者点击当前真实近战命中卡“高电压反击”；再点原卡同一入口 | 点击即声明真实触发资格，无距离/相邻/再次命中确认；原 nonce 仅认领一次，`activeNonce=null`；不自动豁免、伤害或改 HP。攻击者、原受击者绑定不变。|
| `voltage-current-kept-hit` | 对未触发的活动，将原失败攻击英雄点重掷并保留命中；点击新命中卡 | 首次看到的真实 kept reroll 也能认领同一活动，不另开确认；仅命中/暴击的真实已求值近战卡可触发。|
| `voltage-actor-target` | 目标拥有者打开其原生角色卡动作区，点击原生反射豁免 | 即使目标没有元素威能 feature，也能继续原活动；源拥有者和其他玩家不能代掷目标豁免。正常 native Check 窗口和玩家偏好不变。|
| `voltage-reload-claimed` | 停在 claimed / awaiting-save，确认 `activeNonce=null`；两端 full reload，再打开目标动作区 | 返回同一 nonce、原 messageUuid、target Actor；零自动 native roll、零资源再次支付；豁免入口仍可用。|
| `voltage-current-save-damage` | 完成豁免；先不点伤害，可英雄点重掷并保留新结果；使用者点击当前 kept save 卡伤害 | 初次豁免后零伤害卡，保留英雄点窗口；显式点击才采用唯一 kept save、生成一张绑定原目标的伤害卡；源拥有者独有伤害入口。原生应用后才以真实 damage-taken 回执 done。|
| `voltage-cancel-consumed` | 原生命中已认领后，目标关闭原生豁免窗口；从旧卡、当前卡、角色动作区尝试继续 | `status/phase=cancelled`，触发已消耗，不是回到 armed；无新豁免/伤害、无重新 Use/扣资源。对 uncertain 也无可运行按钮，仅中文主持人核对提示。|
| `voltage-private-name` | 启用隐藏名称；用未向源玩家公开名称的目标进行实际触碰选择；另以 blind/非接收者 whisper 命中检查其动作区 | 选择只显示安全名称/“目标”及序号，不显示 GM 真名；private hit 不进入该玩家候选，也不显示私密卡入口；无权限玩家查询返回空。|
| `voltage-source-change` | 活动等待中删除自有QA原channel card、删除item/token或 relink 原/目标 Token | 原channel card删除/awaiting-save的player live sheet与查询已按下方固定报告通过；其余条件仍按各自证据验收。显示来源/目标改变的中文说明，零可运行步骤；不把同一 Token 的新 Actor 作为旧目标。|

观察状态仅从源 Actor 的 `flags['pf2e-third-party-automation'].voltage.activations[nonce]` 读取；取消/不确定不能手动改为等待状态。实际检查应保留两客户端日志、原/命中/豁免/伤害 UUID、phase 变化、roll 与资源计数。不要为测试伪造聊天 flags 或手写回执。

## 已执行的原卡删除分支证据

固定报告：[1790746880258](C:/Users/Taka/Desktop/fvtt/output/automation-eldamon-ux-20260930/qa/dual-client-report-1790746880258.json)，原始 SHA256 `d733fc6f3160736b90870aed3fa1fa6fdeb63bbb5fd0c76b7196aa11aefdb614`，已 closed 于 `2026-09-30T05:59:36.226Z`。报告 runtime 为 Core14.368 / PF2e8.5.1 / 模块0.9.20；作者只读检查了原字节与返回值，没有再次执行浏览器或世界操作。

root 在本任务隔离世界通过真实新高电压 Use 生成原卡 `ChatMessage.s2N2GNCfLWJi6NSP`，nonce `660a7fc7-100d-4d3b-a323-6fb463514478`；随后声明实际触碰，仍为同活动 claimed/awaiting-save。仅删除该自有QA原卡。

| 固定成功 command | 实际读取结果 |
|---|---|
| `voltage-cn-target-ui-before-source-delete`（player） | 原生目标 actor sheet 的“绑定目标：原生反射豁免”按钮存在；查询返回同 nonce/original messageUuid、claimed/awaiting-save 与 save action。 |
| `voltage-cn-delete-own-qa-original-card`（GM） | deleted=true；查询仍指向同活动，actions=[]，中文来源改变说明出现。 |
| `voltage-cn-player-source-deleted-ui-recovery`（player） | 已打开的 live actor sheet 无须重新Use或重开sheet，按钮数为0；界面文字准确为“高电压原来源或目标已改变，等待主持人核对原活动。”；同 nonce 的 getContinuations actions=[]。 |

对应 source-field 为 `eldamon-voltage-executor.mjs:145` / `VOL-UI27`，该 producer SHA256 `b9652bc474b8357c2fc96cc7b39bcf633d1fe0734d7214fec1ef3e4b0d0ea887`；逐字段记录在 [voltage-localization-2026-09-30.json](../../fortress-automation/voltage-localization-2026-09-30.json)。这证明原source card删除的 awaiting-save/player live sheet分支，不把 awaiting-damage 删除、显式继续旧动作后的完整 roll/HP计数、取消、relink 或全部隐私条件一起标为原生通过。完整59+2源码测试另证明save/damage两种删除阶段均移除按钮且拒绝新增roll/HP。

同报告保留最早 `electricity-cn-dsn-old-proof-native-receipt-continue` 的 ok=false，原错为 `Owned native reaction Chain source missing`。root 已裁决为 harness 首个Chain来源UUID写错，后续 `...-reviewed-source` ok=true。报告 errors[]为空不等于 commands 全通过；本次只复用上述三个有明确绑定的成功command，不删除失败或把整份报告冒称clean。最终交付gate须使用root新取得的实际ZIP clean报告，早期证据按输入/consumer绑定充分时复用。metapower clear/widen字段由root维护。

## 源码验证记录

首批行为 RED：`node --test --test-name-pattern 'current real|current hit|claimed activities|native save result|two entrances|uncertain activities|local continuations|touch branch' modules/pf2e-third-party-automation/tests/eldamon-voltage.test.mjs`，8 个失败；实现后同命令 8/8。额外首次 kept reroll 当前卡测试先 RED（`armed !== claimed`），修复旧 `confirmed` 门槛后 1/1 GREEN。最终完整命令为 `node --test modules/pf2e-third-party-automation/tests/eldamon-voltage.test.mjs`；须设置 `PF2E_NATIVE_BUNDLE=C:\Users\Taka\Desktop\fvtt\tmp\fortress-gap-audit-20260925\code\systems\pf2e\pf2e.mjs` 才包括真实 PF2e 伤害应用边界，不接受 skip。

原始 Git blob：ledger `f95c7a25995e6c72b16f676dddbd34e824b20272`，executor `b1a8e77b3576af8cf5fbd790b3a794a5d53d969d`。新增 continuation helper 为局部索引，无权威状态；启动时恢复一次，后续按文档更新。可见 errors、续接状态/按钮/事实分支/刷新 caption 为中文；原物品 document name 沿用已翻译名称，不改 UUID、traits、frequency、公式、native ownership/IWR/receipt 规则。
