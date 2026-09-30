# Task 7：激发武器绑定原次攻击

新增 `weapon-surge.mjs` provider 和 `weapon-surge-snapshot.mjs`。真实效果来源为 `Compendium.pf2e.spell-effects.Item.qlz0sJIvqc0FdUdr`，选择为 `flags.system.rulesSelections.spellEffectWeaponSurge`。复制原生完整规则，保留攻击 +1 status、spirit d6 的 1/5/9 环缩放及 sanctified；不重写数值规则。

指定武器的实际命中、暴击、未命中、严重未命中均消费原效果；取消、其他武器、同 ID 替换或刷新均保留。逐 actor 的 gate 防止并发 native Strike 捕获同一效果。等待后重新选取当前 preparedStrike 的相同武器、alternate usage 和 MAP variant；已确认攻击先消费旧效果、释放 gate，再 await 业务 callback，避免 callback 内再攻击死锁。重复 callback 只结算一次。

无 AC / target 的原生 Strike 可能实际投骰而 `context.outcome===null`。provider 仅在真实 variant callback 提供 `_evaluated===true` 且 finite `roll.total` 时，给 frame 额外本地完成证明；这类实际攻击也消费原效果。原 frame 未证明的 unresolved / cancelled 仍不消费。其已发表 context 保留 nonce option 与原（包括空的）snapshot，普通 Damage 只恢复它，不能借用后来新增的 Surge。

真实 QA 发现关闭检定窗口时，付款 `actor.update` 刷新 preparedStrike，旧 invocation 重绑当前 Strike；Knowledge 的外层 wrapper 使基于函数身份的去重失效，Surge 被再包一次并等待自身 gate。已改按 variant 对象和 strike+damage method 去重；外层 handler 不影响已有 Surge 包装的唯一性。已付款不确定旧卡保留现场，不重新投骰或付款。

原生 Check middleware 在发布攻击卡前写入 `flags.pf2e.context.weaponSurgeSnapshot`：

```js
{schema:1, actorUuid, weaponId, weaponUuid, nonce, effects:[originalEffectSourceData]}
```

同次 options 保留 `pf2e-third-party-automation:weapon-surge:<nonce>`。PF2e 8.5.1 的真实 `Check.rerollFromMessage` deepClone `flags.pf2e`，包括此 context 和 options；重掷后的 frame 优先使用原 snapshot，不能消费新效果。额外 snapshot 不放在会被原生重掷丢弃的 module message flags。

接口：

```js
createWeaponSurgeAutomation({game})
// => {wrapStrike(strike,actor), interceptCheck(native,check,context,...args), register()}

createNextStrikeEffectFrame({actor,strike,target,consumeTumble:true})
// => capture, consume, damageOptions, snapshot, transientItems, damage(currentStrike)

prepareWeaponSurgeDamageSnapshotItems(actor, transientItems) // => source items[]
```

`frame.transientItems()` 提供无规则 marker，包含原快照；`frame.damage()` 在临时 native actor 内剔除该武器后续新 Surge、展开原次规则。marker 留在临时 source，保证 owner RPC 只发送额外效果差异后仍能恢复。helper 重入只展开一次，并保留组合活动 Twin/Forceful 和其他武器的效果。原世界 Actor、Item 均不被延后伤害修改。冻结效果改为独立 slug，移除原 compendium source，避免被第三方的下一次效果清理认成后来新施加的使用。

主代理拥有接线：provider 加入已有 `prepareStrike` reduce 链；`interceptCheck` 加入原生 Check middleware；owner RPC 用 helper 返回的 items 做一次 `actor.clone({items},{keepId:true})`。本任务未改 main/owner 文件。无新 hook、计时器、全世界扫描或移动/距离/视线检测。

## RED / GREEN

- `weapon-surge-red.txt`：13 项，2 通过、11 失败。新 frame/snapshot/provider 合同未实现。
- `weapon-surge-callback-red.txt`：callback 失败后旧效果未消费，18 项中 1 失败。
- `weapon-surge-callback-once-red.txt`：重复 native callback 两次执行；19 项中 2 失败（含上述消费问题）。
- `weapon-surge-malformed-red.txt`：外来 saved context 被接受，20 项中 1 失败。
- `weapon-surge-concurrent-red.txt`：并发捕获同一效果及 callback 内再 Strike 借用旧效果，22 项中 2 失败。
- `weapon-surge-green.txt`：Weapon Surge、existing next-strike effects、activity sequence **60 项全部通过、0 失败、0 skip**。真实 PF2e 8.5.1 bundle 的 `DamageDiceRuleElement.beforePrepareData` 被提取执行，1/5/9 环分别准备 1/2/3 spirit d6。syntax check 两个新模块通过。
- `weapon-surge-no-ac-red.txt`：真实 evaluated native callback、null outcome 时，原 Surge 未消费及原空 snapshot 借入后来新效果，2 项失败。
- `weapon-surge-no-ac-green.txt`：最终 **66 项全部通过、0 失败、0 skip**，同时覆盖原次有/无 Surge、无 AC 的延后伤害，以及未 evaluated / 非 finite total / 纯 context 字串不能证明一次 Strike。
- `weapon-surge-reprepare-red.txt`：实际 `beforeNativeRoll(showDialog:false)` + 付款重prepare + 真实 Knowledge outer wrapper 复现 gate 自等 timeout；damage 外层 wrapper 也造成重复包装，2 项失败。
- `weapon-surge-reprepare-green.txt`：最新 **68 项全部通过、0 失败、0 skip**。上述付款后重绑只有一次 native attack、一次 payment 和一次原 Surge 删除；外层 damage handler 未被再次 Surge 包装。

```powershell
$env:PF2E_NATIVE_BUNDLE='C:\Users\Taka\Desktop\fvtt\tmp\fortress-gap-audit-20260925\code\systems\pf2e\pf2e.mjs'
node --test modules/pf2e-third-party-automation/tests/weapon-surge.test.mjs modules/pf2e-third-party-automation/tests/next-strike-effects.test.mjs modules/pf2e-third-party-automation/tests/activity-attack-sequence.test.mjs
```

## 隔离 Foundry 现场验收入口

使用 QA world 的原生角色卡 Strike（已由 companion `prepareStrike` wrapper 接线），再用该攻击卡的 native Damage / Critical Damage。不要通过自行构建的无 context 宏来代替此入口。

1. 给 QA 角色创建上述 native effect，已选中 weapon A；攻击 weapon B 后 effect 仍在。
2. weapon A 产生实际命中和未命中各一例；原 effect 删除一次，卡上 `flags.pf2e.context.weaponSurgeSnapshot.effects.length===1`，原攻击 modifier 为 +1 status。
3. 对原次攻击延后伤害前，重新施加 9 环 Surge；原次 1 环 Damage 仍只有 1 spirit d6、sanctified，真实 Actor 的新效果保留。再攻击的新次使用 3 spirit d6 并消费新效果。
4. 在原次攻击上实际使用英雄点重掷，核对原生付款一次、新 card `context.isReroll` 与旧 nonce/1 环 snapshot；重掷不消费之后的新效果，延后 Damage 仍用 1 环。
5. 原生检定窗口取消、不作额外确认；效果与资源按原生取消规则保留。两个点击请求分别执行，首个窗口确认才进入第二个，只有首个保存旧 Surge。
6. 双重打击/法术打击的原操作者 native attack/damage 和 GM 卡均保留同一个 snapshot，owner RPC 只传 marker/组合修正等差额，无整张物品清单，spirit dice 不重复。

离线结果证明 native 规则准备与调用时序；双 client 的真实 dialog、socket、重掷付款和 card 点击仍须现场验收。native-flow 代理已独立只读审查 gate、fresh usage/variant、消费与 callback 时序、old/empty/reroll snapshot 和 owner 差额展开，未发现阻止合入的问题。
