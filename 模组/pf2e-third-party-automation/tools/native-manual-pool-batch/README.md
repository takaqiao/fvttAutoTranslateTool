# 原生手动治疗 batch 接缝

仅适配固定 PF2e 8.5.1 bundle 的私有 `applyDamageFromMessage`。完整输入 SHA256 为 `d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157`，函数位于 UTF-16 `[1928389,1930198)`。工具同时要求输入 hash、四个唯一 loop anchor 和唯一 `jm.onInit();` API 初始化 anchor；不匹配即拒绝生成候选。

候选完整 PF2e 源只能输出到整个 checkout 之外的新私有目录，不进入 Git、公共 ZIP 或 release。工具不安装候选，不修改输入、世界或 HP。已存在的输出目录拒绝覆盖。

```powershell
node 模组/pf2e-third-party-automation/tools/native-manual-pool-batch/patch.mjs <fixed-pf2e.mjs> <private-new-output-directory>
$env:PF2E_MANUAL_POOL_BATCH_SOURCE='<fixed-pf2e.mjs>'
node --test 模组/pf2e-third-party-automation/tools/native-manual-pool-batch/seam.test.mjs
```

生成 `patched-pf2e.mjs` 和 `manifest.json`；manifest 保存输入、候选、observer、patcher 和原/候选函数区间 hash。测试在内存里从完整私有源提取原函数，执行真实的顺序循环；Foundry document、GM 选择和 native writer 边界由 fixture 替代，不证明真实 GM 持久授权或 Toolbelt 主人 Promise 已完成。

同 fixed-base 的后续组合入口是 patch.mjs 私有 `instrumentNativeManualPoolBatch(source,observer)`；公开 patch 函数先检查原完整 BASE_SHA，再调用它。未来 static receiver 必须在同一个构建函数里先核这个固定 base，再组合各自已经核验的私有 transforms，并分别核验未重叠的原/候选片段、observer 和最终组合 hash。不要把已 patch 输出交给另一个只接受 BASE_SHA 的公开 patcher，也不公开接受任意 source/transform 的绕过参数。本批尚未实现 receiver transform，仍拒绝全部相关 callbacks。

## 来源与 gate

候选在原 `jm.onInit()` 已创建 PF2e API 后安装冻结 `game.pf2e.thirdPartyManualPoolBatch`，公开面仅 `descriptor/subscribe/currentCall`，没有 mark、execute 或 terminal writer。只在原函数解析真实 message/roll、实际 activeTokens 或 actorIsTarget token、原 troop 去重之后触发 gate。不会读取稍后的 GM targets。

```js
const dispose = game.pf2e.thirdPartyManualPoolBatch.subscribe(observer, {
  authorizeBatch,
  timeoutMs: 10000 // 1..10000
});
```

`authorizeBatch` 有两个 phase。`admit` 必须同步，从本模块此前真实 native/Workbench use 建立的私有来源登记判断是否参与；普通 flags/socket 包只能索引，不能登记来源。未登记返回 `{status:'unregistered'}`，原 loop 仍按原时序准备并调用所有目标。

参与返回：

```js
{
  status: 'participating',
  sourceBinding,              // 私有真实 use/check/result/effect 身份的 JSON 快照
  isCurrent: () => boolean    // 同步、只读，必须严格返回 true
}
```

恰好一个参与 gate 可继续。未知回答、异步 admit、gate throw、多参与者都拒绝原调用，并留下该 result/roll 的来源 tombstone。已经参与的 result/roll 本客户端不能再次尝试，也不能在移除 gate 后回退为普通应用；刷新后仍须由后续 ledger/source 接线禁止重复，不靠这个内存集合恢复许可。不同 result 或 rollIndex 有独立来源身份，共同 Workbench useId 不合并独立的 per-target result。

只有参与分支才按原 loop 的 options 累积顺序准备全部实际 contextual actors 和原 params。准备不调用 applyDamage、接收 callback、Roll.evaluate 或文档更新。`select` 接收同一实际 batch 的完整候选，在第一个原应用前等待选择：

```js
await authorizeBatch({phase:'select', descriptor, batch:{
  batchId, message, roll, rollIndex, multiplier, addend, item,
  targets: [{targetOrdinal, token, patient}],
  sourceBinding,
  candidates: [{targetOrdinal, token, patient, contextualActor,
    paramsSnapshot:{damage,skipIWR,rollOptions,outcome,shieldBlockRequest}}]
}});
// 返回：
{
  status: 'selected', batchId, sourceDigest, // sourceDigest 是 64 字符 hex
  selections: [{poolUUID,effectKey,patientUUIDs,selectedOrdinal,grant}]
}
```

`grant` 保留真实私有 claim 返回的原对象身份，frame 不克隆这个许可。其纯数据快照只用于结构及原调用前的改变检查。gate 的所有 GM claim 必须已得到明确持久提交 ACK；工具本身只校验结构、本次目标/来源和选择，不能凭一个 JSON grant 验证 GM 授权。关联 pool、患者权限、跨活动唯一 claim 和最终 seal 由后续 completion/ledger/hpPools 接线负责。

超时或未知关闭该 invocation；迟到返回没有原调用。每次 await 后及首次原调用前重核真实 message/roll/item、操作人、世界时间、target/actor 对象、数据、原 params/rollOptions 和私有 isCurrent。选择只能覆盖这一个真实循环；不包含的患者、遗漏成员、重复分组或不符合 tie 的 selectedOrdinal 都拒绝。

首批模型为 `numeric-empty-reception.v1`：finite numeric healing，最多八个实际患者，所有 contextual actor 的 `healing-received` damageDice、modifiers 和 modifierAdjustments selector 均为空。未知 callbacks/adjustments 不执行也不预估；Risky 双阶段、正/零数值、动态接收、未闭合源由来源 gate 拒绝。单卡的同一 numeric healing 在本模型下全部相等，同 pool 选原顺序第一个。领域的 5/10 最大值算法可单独验证，不能据此声称此接缝支持真实同卡异值最大。

选中者仍执行那个原 `contextualActor.applyDamage(originalParams)`，没有新 HP writer。非选中成员只发 `member-linked`，先保存 poolUUID/effectKey/permitNonce 引用，后续真实 pool seal 才可附到各患者独立活动和免疫。工具不生成 damage-taken receipt、master Promise 或 HP terminal。`native-call-returned/batch-returned` 只描述原函数返回，不能当主人完成证明。

## 当前原调用帧

在唯一 applyDamage wrapper 的第一行、spread 或任何 await 之前取得：

```js
const frame = game.pf2e.thirdPartyManualPoolBatch.currentCall(this, params);
```

source WeakMap 仅存在于同步原 `actor.applyDamage(params)` 调用栈；取得返回 Promise 后先删除，再 await。只有实际 contextualActor 和原 params 同时匹配才有 frame；同 UUID 的别对象、params 副本和晚查询均无资格。wrapper 将原 frame 保存在本次局部闭包，后续私有 leaf 使用它，不能凭 socket/flags 重建。

frame 的冻结容器包含 descriptor、batchId/callId/targetOrdinal、实际 message/roll/rollIndex/item/token/patient/contextualActor、multiplier/addend/stage、sourceBinding/sourceDigest、poolUUID/effectKey/selectedPatientUUID、原 grant、paramsSnapshot 及只读 `isCurrent()`。`isCurrent()` 保留 source、options、对象身份和接收资格，在异步 wrapper 的最终 leaf 可拒绝 Stop 或改源；正常原 HP 更新不因患者 HP 数值改变而失效。它不创建许可、写入或 terminal。Task 2 仍须一次消耗原 grant 并捕获原患者调用、原 receipt 和原 Toolbelt master Promise；Task 4 在 ledger await 后继续独立重核 pool 图、权限、source/免疫/receipt 和 recording 状态。

原生按钮允许 outcome 为 undefined。仅 paramsSnapshot 在该值为 undefined 时省略 outcome；显式 null 和其他已提供值仍保留并经过严格 JSON 校验。原 candidate.params 不改写，异步后的 outcome 值变化仍使来源失效；shieldBlockRequest 与其余字段的严格要求不变。
