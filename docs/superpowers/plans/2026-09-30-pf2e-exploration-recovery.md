# PF2e Exploration Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Foundry V14 的现有自动化模组中实现真实 RNG 的探索期恢复、手动活动记录和尊重并行及冷却的共同时间账本。

**Architecture:** 活动账本保存真实来源与处理状态；原生提供者负责资格、掷骰、HP／资源应用和回执；协调器负责安排检查点、推进时间及重新规划。开始与完成使用同一活动上下文，不回填过去；任何未知的掷骰、应用或时间提交均停止自动重试。

**Tech Stack:** Foundry 14.368、PF2e 8.5.1、现有 socketlib／libWrapper、ES modules、Node 原生 `node:test`。本机验证运行时 Node v26.7.0；不增加产品依赖。

**Spec:** [已批准设计](../specs/2026-09-30-pf2e-exploration-recovery-design.md)。用户已于本会话批准该正式设计；本实现计划待审阅和选择执行方法。

## Global Constraints

- 用户使用 Foundry V14。
- 默认一键按真实 RNG 治到目标血量；也能记录玩家手动使用原生或 Workbench 治疗的结果。
- 无需等待所有玩家选择活动并反复确认。
- 每次应用后重新读取当前 HP、熟练度、免疫和资源，再选择下一步。
- 不得预先掷出未来结果择优，不得为了达标重掷同一次治疗。
- 活动时长为非负秒数。绝对时刻可以为负数，不能重置世界纪元。
- 接受治疗本身不自动占用患者的探索活动；共享 HP 与患者免疫身份分别保存。
- 不全局关闭普通卡片处理，也不新增第二套 HP 应用包装。
- 法术槽、物品次数、消耗品和每日能力默认不自动消耗；激进手术需要保存的风险策略。
- 默认两小时时间预算、一百次治疗活动上限；到限时保留已完成结果与剩余缺口。
- 集成验证使用独立测试世界或世界副本；不能在当前玩家角色上为了测试扣资源、反复掷骰或修改 HP。
- 本计划批准不包含正式世界部署或向其他会话发送消息的授权。

## Review Focus

1. 旧 helper 的共享开关关闭，但另一个模组实际同步 HP：发现当前 Toolbelt 提供者并等待真正的主人更新，不再启用一套同步；Task 2／3／8 测试。
2. 原生 `use` 已返回而子消息尚未创建，或零 HP 变化的回执 `appliedDamage=null`：仍按完整来源确认完成；Task 3 测试。
3. 当前时间看似达到目标，但提交回应丢失或被其他 GM 改过：不得第二次推进；Task 6／7 测试。
4. Workbench 的源成功度、实际治疗骰和割伤卡关联不一致：保留源事实，不靠文本／邻接猜测；Task 5 测试。
5. 只有未选中的 Party 成员获得新的快速治疗规则：也会受到时间 hook 影响，每个检查点都应检查；Task 6 测试。

---

## 基线与工作区

2026-09-30 的只读核对：远端 `/root/fvtt14-data/Data/modules/pf2e-third-party-automation` 仍为 0.9.18.6；PF2e 8.5.1。匹配快照为 `output/dependency-updates-20260929/live/pf2e-third-party-automation`。本地 `模组/pf2e-third-party-automation` 为旧版 0.8.0，禁止直接在其中叠加本功能。

隔壁“审查自动化插件需求”使用 `C:/Users/Taka/.codex/worktrees/automation-native-20260930/fvtt`、分支 `codex/automation-native-20260930`，正在制作未交付的 0.9.19，HEAD 与工作树仍在变化。不要复制其 dirty 工作树作为基线，也不要编辑该工作树。执行时若其已交付固定版本，可采用交付提交；否则从已核验 0.9.18.6 开始独立工作，在最后的集成步骤核对并合入已完成的更新。

执行时使用 using-git-worktrees 技能，先检查 attached artifacts，复用合适的独立工作树或创建一个从本计划提交起始的工作树。所有产品文件只在返回的独立工作区修改。下文 `M` 是该工作区内的 `模组/pf2e-third-party-automation`；文件表给的是 `M` 内的准确路径，命令的 cwd 为 `M`。

如果独立工作树不含现行模组，将哈希匹配的发行版完整导入 `M`，保存 `tests/exploration/baseline.json` 的版本、来源与逐文件 SHA256。先提交这份基线，再开始功能提交，以便最后只提取功能 diff。不得在用户原来的旧版目录覆盖导入。

可复用的固定源码／测试基线提交为 `0f03b2bc2ff8cca4c050f06f7d70686058909029`，其 `modules/pf2e-third-party-automation/module.json` 为 0.9.18.6，测试目录有八十八个文件。发行快照若不含 tests，从该提交导入对应 tests，不复制隔壁的未提交测试。导入前再次验证版本与产品脚本哈希，出现差异需说明具体文件；不得把不存在的测试报为通过。

## 文件职责与任务依赖

| 文件 | 职责 | 任务 |
|---|---|---|
| `scripts/exploration/schema.mjs` | 事实校验、时间和状态类型 | 1 |
| `scripts/exploration/ledger.mjs` | 持久事实、状态转换、准确证据引用 | 1 |
| `scripts/exploration/document-store.mjs` | GM 私有 Journal 与 world setting 存储 | 1 |
| `scripts/exploration/capabilities.mjs` | 准备后能力、资格、HP 池、冷却 | 2 |
| `scripts/exploration/hp-pool.mjs` | Toolbelt 共享发现和实际主人更新确认 | 2 |
| `scripts/exploration/treatment.mjs` | 治疗伤势规则、群体容量、延长治疗 | 3 |
| `scripts/exploration/native-treatment.mjs` | 原生执行和检定／结果／应用证据链 | 3 |
| `scripts/exploration/owner-operations.mjs` | 活动 GM 到原所有者的限定执行适配 | 3 |
| `scripts/exploration/refocus.mjs` | 再聚能、仙露复合活动、聚能治疗 | 4 |
| `scripts/exploration/manual-events.mjs` | 原生及 Workbench 实际调用与手动证据 | 5 |
| `scripts/exploration/timeline.mjs` | 已知顺序与并行假设下的最早完成时间 | 5 |
| `scripts/exploration/time-effects.mjs` | 被动恢复／休息来源与完成能力检查 | 6 |
| `scripts/exploration/clock.mjs` | 世界时间声明、nonce hook、恢复核对 | 6 |
| `scripts/exploration/policy.mjs` | 不预知 RNG 的下一步选择 | 7 |
| `scripts/exploration/coordinator.mjs` | 开始、检查点、预算、中断、续接 | 7 |
| `scripts/exploration/panel.mjs` | 恢复面板与易理解的实际结果 | 8 |
| `styles/exploration.css` | 限定在面板内的样式 | 8 |
| `scripts/main.mjs`、`module.json` | 唯一注册、可选 API、样式入口 | 8 |

测试按任务存放在 `tests/exploration/*.test.mjs`；真实来源 fixture 存在 `tests/exploration/fixtures/`，只保留必需能力、规则与链接，删除人物传记和无关聊天。

现有文件的修改限定为必要接口：`salubrious-kiss.mjs`、`salubrious-kiss-executor.mjs`、`salubrious-kiss-rules.mjs`、`salubrious-kiss-context.mjs`、`salubrious-kiss-check-scope.mjs`、`salubrious-message-privacy.mjs`、`salubrious-kiss-chat.mjs`、`salubrious-kiss-damage-guard.mjs`、`av-refocus-events.mjs`、`av-automation.mjs`、`amp-cast-events.mjs`、`native-context.mjs`、`native-action-events.mjs`、`medic-native.mjs`、`patreon-treatment-compat.mjs`。隔壁新 `native-owner-operations.mjs` 不存在于 0.9.18.6，只有采用已交付的 0.9.19 基线时才接入。先确认固定基线中的名称和接口；若隔壁已调整文件，沿用已交付的对应接口，不创建第二个同用途包装。

依赖：Task 1 → 2；Task 3／4／5／6 使用 1、2 的契约；Task 7 使用全部提供者；Task 8 最后统一注册。整个功能构成一个可验收的休整流程，不把未接入的空接口作为已交付功能。

## 共同接口

下列字段和签名是后续任务的固定契约。返回值均是事实，不是序列化权限凭证；执行授权保存在私有闭包，并使用现有所有者 RPC 认证。

```js
// schema.mjs 的 createActivity(input) 输出；proof 的数组可以为空，但
// 完成条件由具体提供者的 expectedStages 判定，不能只看 state。
const activity = {
  id: 'A1', sessionId: 'S1', providerId: 'treat-wounds',
  actorUUID: 'Actor.H', patientUUIDs: ['Actor.P'],
  hpPoolUUIDs: ['Actor.P'], groupId: 'A1',
  startedAt: 0, endsAt: 600, kind: 'treatment', mode: 'automatic',
  options: { skill: 'medicine', rank: 'trained', riskySurgery: false },
  source: { itemUUID: null, userId: 'U1', origin: 'coordinator' },
  state: 'planned',
  proof: { useId: null, checkIds: [], resultIds: [], receiptIds: [], immunityIds: [] }
};

// 提供者；ctx 包含私有的当次执行范围，不能从公开 flags 重新构造。
// describe(actorUUID) -> Promise<Capability[]>
// propose({session, snapshot, now}) -> Promise<Proposal[]>
// begin(activity, ctx) -> Promise<{status:'started'|'blocked', reason?:string}>
// complete(activity, ctx) -> Promise<ActivityCompletion>
// observe(event, ctx) -> Promise<ActivityCompletion|null>
// reconcile(activity, ctx) -> Promise<ActivityCompletion>

// Proposal 包含可验证的开始资格，不包含预掷的骰点。
const proposal = {
  providerId: 'treat-wounds', actorUUID: 'Actor.H',
  patientUUIDs: ['Actor.P'], hpPoolUUIDs: ['Actor.P'],
  durationSeconds: 600, earliestStart: 0,
  actorExclusive: true, patientTreatmentExclusive: true,
  expectedNetHealing: 9, resourceCost: {}, options: activity.options
};

// ActivityCompletion.status: 'confirmed'|'awaiting-evidence'|'blocked'|'uncertain'
// 必须保存来源 degree 与实际 effectiveOutcome，二者可能不同。
const completion = {
  status: 'confirmed', proof: activity.proof,
  sourceDegree: 2, effectiveOutcome: 'success',
  rolledHealing: 9, resourceReceiptIds: [], reason: null
};
```

Session 保存 `id/startedAt/budgetEndsAt/goalsByPool/activityIds/status/stopReason/assumptions`。目标结构为 `{poolUUID, targetHP, requireFullFocus, requireWoundedRemoved}`，实际 HP 与聚能从当前快照读取。ClockCommit 保存 `id/sessionId/from/to/gmId/state/evidence`，状态为 `started/confirmed/uncertain`。

### Task 1: 持久活动账本与 GM 唯一声明

**Files:** Create `scripts/exploration/schema.mjs`、`ledger.mjs`、`document-store.mjs`；Test `tests/exploration/ledger.test.mjs`。

**Interfaces:** `createActivity(input)` 验证并返回上述事实；`createLedger({read,write,isAuthority})` 返回 `createSession(input)`、`getSession(id)`、`getActivity(id)`、`insertActivity(record)`、`transitionActivity(id,{expected,patch})`、`getClockCommit(id)`、`upsertClockCommit(record)`、`transitionClockCommit(id,{expected,patch})`、`snapshot(sessionId)`。所有写操作为 Promise；`expected` 为允许的当前状态数组，不匹配时抛错。`createDocumentStore({game,JournalEntry,fromUuid})` 返回 `read()/write(state)`；非活动 GM 不写账本。

- [ ] **1. 固定基线并创建失败测试。** 在隔离工作树导入固定发行版后，编写实际测试：

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createActivity} from '../../scripts/exploration/schema.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';

test('负纪元合法，未知执行不能被重置为待执行', async () => {
  let state = {sessions: {}, activities: {}, clocks: {}};
  const store = {
    read: async () => structuredClone(state),
    write: async next => { state = structuredClone(next); },
    isAuthority: () => true
  };
  const ledger = createLedger(store);
  const a = createActivity({
    id:'A1',sessionId:'S1',providerId:'treat-wounds',actorUUID:'Actor.H',
    patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],groupId:'A1',
    startedAt:-1200,endsAt:-600,kind:'treatment',mode:'automatic',
    options:{},source:{userId:'U1',origin:'coordinator'},state:'started',
    proof:{useId:null,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]}
  });
  await ledger.createSession({id:'S1',startedAt:-1200,budgetEndsAt:6000,
    goalsByPool:[],activityIds:[],status:'running',stopReason:null,assumptions:[]});
  await ledger.insertActivity(a);
  await ledger.transitionActivity('A1',{expected:['started'],patch:{state:'uncertain'}});
  await assert.rejects(ledger.transitionActivity('A1',{
    expected:['planned'],patch:{state:'started'}
  }));
  assert.equal((await ledger.snapshot('S1')).activities[0].state,'uncertain');
});
```

- [ ] **2. Run:** `node --test tests/exploration/ledger.test.mjs`。首次应因缺少模块失败；同时加入非 GM 写入、重复 activity ID、负时长和重复确认状态的测试。
- [ ] **3. 实现持久状态转换。** 使用隐藏的 GM Journal，world setting `explorationLedgerUUID` 只保存其 UUID。写入使用最新快照，并在每次副作用前后重新确认当前活动 GM；本地队列只负责该 GM 客户端内串行，不声称跨文档原子事务。状态转换的关键操作为：

```js
async function transitionRecord(read, write, isAuthority, collection, id, expected, patch) {
  if (!isAuthority()) throw Error('active-gm-required');
  const state = await read();
  const record = state[collection][id];
  if (!record || !expected.includes(record.state)) throw Error('state-conflict');
  state[collection][id] = {...record, ...patch};
  if (!isAuthority()) throw Error('gm-changed');
  await write(state);
  if (!isAuthority()) throw Error('gm-changed-after-write');
  return structuredClone(state[collection][id]);
}
```

Journal 的创建结构明确为：

```js
async function createLedgerJournal(JournalEntry, moduleId) {
  return JournalEntry.create({name:'探索活动账本',ownership:{default:0},
    flags:{[moduleId]:{explorationLedger:{sessions:{},activities:{},clocks:{}}}}});
}
```

document-store 只有首次授权写入才创建该文档；普通读取不创建世界文档。创建后保存 UUID，后续通过 fromUuid 读取并更新该文档的 `explorationLedger` flag。

- [ ] **4. Run:** 重跑本任务测试；重新创建 ledger 实例读取相同 store，确认回执与 uncertain 状态保留。核对 Journal 默认 ownership 为 NONE，玩家通过权限过滤的 API 读取允许的结果，不能修改公开 flags 获得执行权。
- [ ] **5. Commit:** `git add scripts/exploration/schema.mjs scripts/exploration/ledger.mjs scripts/exploration/document-store.mjs tests/exploration/ledger.test.mjs tests/exploration/baseline.json`；`git commit -m 'feat: persist exploration activities and claims'`。

### Task 2: 当前能力、免疫与共享 HP 身份

**Files:** Create `scripts/exploration/capabilities.mjs`、`hp-pool.mjs`；Test `tests/exploration/capabilities.test.mjs`、`hp-pool.test.mjs`、`fixtures/roster-capabilities.json`。

**Interfaces:** `wardCapacity({wardMedic,medicineRank}) -> number`；`cooldown({startedAt,finishedAt,continualRecovery}) -> {expiresAt,remainingSeconds}`；`earliestTreatmentStart({now,existingExpiresAt}) -> number`；`createCapabilities({game,fromUuid,hpPools})` 返回 `snapshot(actorUUIDs)`、`discover(actorUUID)` 和 `activePassiveRules()`，均为 Promise。后者检查全部当前 Party，返回 `{actorUUID,providerId,key,passing}` 事实数组；选择与 predicate 使用已核验 handler 的当前规则逻辑。快照包含准备后 Statistic、源条目、实际资源、患者效果、共享池和需要确认的能力，不能使用静态存档的估算加值代替。

`createHpPools({game,actorUpdateEvents})` 返回 `discover(actor)` 和 `withNativeApplication(activity,patient,operation)`：前者返回 `{poolUUID,memberUUIDs,provider,ready}`；后者只调用原生 operation 一次，并等待准确应用及主人更新确认，返回原生结果与池回执。`actorUpdateEvents.addActorUpdateMiddleware(fn)` 是现有 amp-cast-events 的入口；不另注册全局 update 包装。

- [ ] **1. 编写失败测试。**

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {wardCapacity,cooldown,earliestTreatmentStart} from '../../scripts/exploration/capabilities.mjs';
test('看护师与免疫使用各自正确来源', () => {
  assert.equal(wardCapacity({wardMedic:true,medicineRank:2}),2);
  assert.equal(wardCapacity({wardMedic:true,medicineRank:3}),4);
  assert.equal(wardCapacity({wardMedic:true,medicineRank:4}),8);
  assert.deepEqual(cooldown({startedAt:0,finishedAt:600,continualRecovery:false}),
    {expiresAt:3600,remainingSeconds:3000});
  assert.deepEqual(cooldown({startedAt:0,finishedAt:600,continualRecovery:true}),
    {expiresAt:600,remainingSeconds:0});
  assert.equal(earliestTreatmentStart({now:600,existingExpiresAt:3600}),3600);
});
```

- [ ] **2. Run:** `node --test tests/exploration/capabilities.test.mjs`。加入“源医疗受训而 Medic grant 准备后为专家”“仅神秘驾轻就熟”“看护师持有但前置不足”“既有免疫还剩 183 秒”的 fixture 测试。
- [ ] **3. 实现规则发现。** 标准资格来自 `actor.getStatistic('medicine')`、所选替代技能、当前 owned item 的来源和准备后规则。治疗伤势免疫按患者；战地医疗按患者＋治疗者，医师免疫绕过单独保存资源。低阶源熟练度不覆盖准备后的 Medic grant。不存在已验证来源的替代技能组合标为需要人工处理。

```js
export function wardCapacity({wardMedic,medicineRank}) {
  if (!wardMedic || medicineRank < 2) return 1;
  return 2 ** (medicineRank - 1);
}
export function cooldown({startedAt,finishedAt,continualRecovery}) {
  const expiresAt = startedAt + (continualRecovery ? 600 : 3600);
  return {expiresAt,remainingSeconds:Math.max(0,expiresAt-finishedAt)};
}
export function earliestTreatmentStart({now,existingExpiresAt}) {
  return Math.max(now,existingExpiresAt ?? now);
}
```

- [ ] **4. 接入已确认的 Toolbelt Share Data。** 当前肆季的实际提供者为 Toolbelt 3.56.5，已启用 shareData，贝肯 `.flags['pf2e-toolbelt'].shareData.data` 为 `master=水长东ID, health=true`。读取 `game.toolbelt.api.shareData.getMasterInMemory(actor)` 与 `getSlavesInMemory(master,false)`，校验全局 enabled 和各成员 health；仅共享护甲等数据的成员不加入 HP 池。保存的源 HP 73／9 不代表运行时不同池。保留旧 helper `sharedHP=false`，不调用写链接的 setSummonerHP。

Toolbelt 的奴仆 `_preUpdate` 将 HP 转给主人但未 await。在现有 Actor.update middleware 内以私有当次范围捕获实际转发的主人更新 Promise 并等待其提交，核对 Actor 身份、字段、来源和单次应用；无确切来源时返回 uncertain。不得仅等到“HP 数值看着相等”就当成回执，也不得自己第二次 master.update。新增测试让奴仆 apply 先 resolve、主人更新延迟 resolve，确认下一轮不会开始；无关主人 HP 更新不被收为该治疗的证明。

发现代码的核心条件为：

```js
function discoverToolbeltPool(game, actor) {
  const api = game.toolbelt?.api?.shareData;
  const own = actor.getFlag('pf2e-toolbelt','shareData')?.data;
  const enabled = game.settings.get('pf2e-toolbelt','shareData.enabled');
  const master = enabled && own?.health && api?.getMasterInMemory(actor);
  const root = master || actor;
  const slaves = enabled && api ? api.getSlavesInMemory(root,false) : [];
  const members = [root,...slaves.filter(a=>
    a.getFlag('pf2e-toolbelt','shareData')?.data?.health)];
  return {poolUUID:root.uuid,memberUUIDs:[...new Set(members.map(a=>a.uuid))],
    provider:enabled && api ? 'pf2e-toolbelt' : 'native',ready:true};
}
```

调用前验证 Toolbelt 已启用且 setting 已注册；不存在该模块时直接返回本演员的独立池。更新确认须使用实际调用范围捕获的 Promise，不另做 HP 写入。

- [ ] **5. Run:** `node --test tests/exploration/capabilities.test.mjs tests/exploration/hp-pool.test.mjs`。验证同效果影响双方只选择较大一次、两个不同效果均保留；共享发现不可用或转发来源不能确认时显示具体限制，其他角色仍可用。Commit 本任务列出的文件，消息 `feat: discover prepared recovery capabilities and cooldowns`。

### Task 3: 原生治疗伤势与真实完成回执

**Files:** Create `scripts/exploration/treatment.mjs`、`native-treatment.mjs`、`owner-operations.mjs`；Modify `salubrious-kiss-check-scope.mjs`、`patreon-treatment-compat.mjs`、`native-context.mjs` 的必要范围。若基线已交付共享 `native-owner-operations.mjs`，使用其已有认证和结果恢复接口；Test `tests/exploration/treatment.test.mjs`、`native-treatment.test.mjs`、`owner-operations.test.mjs`。

**Interfaces:** `extension({startedAt,checkpointAt,outcome,rolledHealing}) -> {endsAt,additionalHealing}|null`；`completionState({outcome,checkId,expectedResults,persistedResultIds,applicationReceiptIds,expectedApplications}) -> string`。`createNativeTreatment({game,Hooks,fromUuid,ownerOperations,checkScope,damageGuard,hpPools})` 返回 `run(activity,ctx)`、`applySavedResult(activity,result,ctx)` 和 `reconcile(activity,ctx)`；`createTreatmentProvider({capabilities,nativeTreatment,ledger})` 实现共同提供者接口。`createExplorationOwnerOperations({game,fromUuid,ledger,sharedOwnerOperations})` 返回 `register({socket})`、`registerOperation(id,handler)`、`runActivityWithOwner(activity,operationId)` 和 `isActivityContext(ctx,activityId)`：operationId 必须是本客户端已注册的原生调用，私有范围由认证调用创建；公开数据复制不能通过验证。

- [ ] **1. 编写真实风险与异步来源测试。**

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {extension,completionState} from '../../scripts/exploration/treatment.mjs';
test('返回检定不代表治疗与应用已完成', () => {
  assert.equal(completionState({outcome:'criticalSuccess',checkId:'C1',expectedResults:2,
    persistedResultIds:['D1'],applicationReceiptIds:[],expectedApplications:2}),
    'awaiting-evidence');
});
test('延长只追加同次结果，失败不延长', () => {
  assert.deepEqual(extension({startedAt:0,checkpointAt:600,
    outcome:'success',rolledHealing:19}),{endsAt:3600,additionalHealing:19});
  assert.equal(extension({startedAt:0,checkpointAt:600,
    outcome:'failure',rolledHealing:null}),null);
});
```

- [ ] **2. Run:** `node --test tests/exploration/treatment.test.mjs tests/exploration/native-treatment.test.mjs tests/exploration/owner-operations.test.mjs`。用可控 Promise 模拟 `use()` 已返回、子消息稍后才创建，验证不会提前继续。加入激进手术失败／大失败、HP 为零、受领者抗性、健壮体魄和医师的同类加值、零有效 HP 变化、同一消息复制 flags、共享主人更新延迟和所有者断线的测试。

0.9.18.6 下沿用 `createSalubriousExecutor`／`createMedicOwner` 的认证模式，新增唯一 `exploration:execute` RPC，不重新注册它们的旧方法。原所有者只接受当前活动 GM 的请求，并从 actor UUID 和本地准备数据验证 OWNER 与能力；只能选择已注册的 operationId，不接收代码、函数或任意调用路径。以 WeakMap 保存 ctx→activity 身份。GM 返回后核验确切保存的原生消息及来源，回包丢失先恢复已有消息，不重掷。采用 0.9.19 固定基线时，薄适配委托其已交付的 sharedOwnerOperations，而不是重复实现该层。

接收端的最低认证逻辑为：

```js
async function requireOwnerRequest(game,fromUuid,payload,callerId,operations) {
  if (callerId !== game.users.activeGM?.id) throw Error('active-gm-required');
  if (!operations.has(payload.operationId)) throw Error('unknown-native-operation');
  const actor = await fromUuid(payload.activity.actorUUID);
  if (!actor?.testUserPermission(game.user,'OWNER')) throw Error('original-owner-required');
  return actor;
}
```

nonce 和 payload 只是来源数据；上面认证、actor 资格和私有范围均通过后才能执行，保存原生结果时仍需逐阶段核验。
- [ ] **3. 建立开始／完成时间范围。** begin 验证开始资格并保存真实 startedAt 与预留；complete 只能接受该已开始活动的私有上下文，实际世界时间必须到达其结束检查点。免疫使用 `startedAt+3600` 或 `+600`，不能从完成时再增加十分钟。群体治疗逐患者产生真实检定与结果，但共同 groupId、共同活动时长。
- [ ] **4. 调用真正原生路径，并先注册精确输出观察。**

```js
const results = await game.pf2e.actions.get('treat-wounds').use({
  actors:[healer], target:patient,
  selection:{skill:activity.options.skill,rank:activity.options.rank,
    modifier:0,feats:{'risky-surgery':activity.options.riskySurgery,
      'mortal-healing':false}},
  rollOptions:[`exploration-activity:${activity.id}`],message:{create:true}
});
const check = results[0];
// check 是 {actor,message?,outcome,roll}；关联后续持久消息：
const belongs = message => message.flags?.pf2e?.origin?.messageId === check.message?.id;
```

`use()` 和其 callback 不等待全部子消息，不能只 await 此调用就宣布完成。scope 在调用前创建，捕获其检定、子消息和该阶段输出，关联保存在 ledger。原生已做激进手术成功提升；不提升第二遍。纯失败没有治疗骰，激进手术失败仍有割伤骰，大失败还另有治疗失败伤害。

自动化原生检定修正面板和 Assurance 只在精确 actor／活动上下文内选择，验证保存的原生替代骰结果；没有公开 `assurance:true` 参数，不用一个 rollOption 冒充选择。未能验证的组合退回明确固定 DC／原生手动选择，面板说明该能力未自动适配。

- [ ] **5. 原生应用与免疫。** 从保存的 `message.rolls[index]` 读取 DamageRoll；原始公式从 `toJSON().formula` 读取。用已证明的 outcome／阶段分类大失败，不仅检查 `kinds`。割伤和失败伤害走原生 IWR，治疗用已确认实际骰总值的负数与明确来源。`applyDamage` 返回 Actor，不是回执；等待精确 `damage-taken` 消息，即使 `appliedDamage=null` 也可验证一次合法的零变化应用。经 Task 2 的 hpPools.withNativeApplication 等待实际共享主人更新；同一组同一效果对一个池只应用较大一次，两个独立治疗效果均可应用。需要计算受领者 IWR 的比较采用已有原生 IWR bridge，无法证明时拒绝自动重复池组合。重用现有私有应用保护和单一管线；仅自动活动的精确输出设置 scoped skip-handling，普通卡片不受影响。
- [ ] **6. 延长治疗。** 第十分钟真实检定并正常结算；成功且选择延长才登记后续五十分钟，同次治疗总值保存为追加额。终点应用该追加额一次，不生成新检定、新治疗骰或新割伤骰；每阶段独立应用 ID，同一治疗来源防止重复。失败只耗十分钟。
- [ ] **7. Run:** 本任务测试全部通过，原生消息关联和重复执行测试必须有真实来源 fixture；`node --check` 所有修改脚本。Commit 消息 `feat: execute native treatment with durable evidence`。

### Task 4: 再聚能、仙露三吻与聚能治疗

**Files:** Create `scripts/exploration/refocus.mjs`；Modify `av-refocus-events.mjs`、`av-automation.mjs`、`salubrious-kiss.mjs`、`salubrious-kiss-executor.mjs`、`salubrious-kiss-rules.mjs`、`salubrious-kiss-context.mjs`、`salubrious-message-privacy.mjs`、`salubrious-kiss-chat.mjs`、`salubrious-kiss-damage-guard.mjs`；Test `tests/exploration/refocus.test.mjs`。

**Interfaces:** `createRefocusProvider({game,ledger,capabilities,refocusEvents,salubriousKiss,ownerOperations})` 实现共同提供者接口；复合活动使用同一 `activity.id`。Refocus adapter 增加 `complete(activity,ctx) -> Promise<{id,focusBefore,focusAfter}>`，基于现有准确更新观察而非原函数返回。现有 subscriber 增加 `claimActivity(activity,ctx)` 只预留并返回 started、`completeActivity(activity,ctx)` 结算并返回 ActivityCompletion、`getActivityResult(id)` 读取既有结果；只接受 Task 3 所有者接口验证的私有范围。普通手动 Refocus 仍沿用旧入口。聚能治疗提供者通过同一接口产生完整施法、聚能消耗与 HP 应用回执。

- [ ] **1. 编写失败测试。**

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRefocusProvider} from '../../scripts/exploration/refocus.mjs';
test('仙露 completion 和 subscriber 共用一次执行', async () => {
  let treatments = 0;
  const saved = new Map();
  let stored = {id:'A1',state:'started'};
  const trustedContext = Object.freeze({});
  const salubriousKiss = {
    claimActivity:async () => ({status:'started'}),
    completeActivity:async a => {
      if (!saved.has(a.id)) saved.set(a.id,{status:'confirmed',proof:{receiptIds:['R1']}});
      else return saved.get(a.id);
      treatments++;
      return saved.get(a.id);
    },
    getActivityResult:id => saved.get(id)
  };
  const provider = createRefocusProvider({salubriousKiss,
    refocusEvents:{complete:async () => ({id:'F1'})},
    capabilities:{snapshot:async () => ({focus:{value:3,max:3}})},
    ledger:{getActivity:async () => structuredClone(stored),
      transitionActivity:async(_id,{patch}) => {
        stored={...stored,...patch};return structuredClone(stored);
      }},game:{},
    ownerOperations:{isActivityContext:ctx => ctx === trustedContext}});
  const a = {id:'A1',actorUUID:'Actor.H',startedAt:0,endsAt:600,
    options:{threePecks:true},state:'started'};
  await provider.complete(a,trustedContext);
  await provider.complete(a,trustedContext);
  assert.equal(treatments,1);
});
```

- [ ] **2. Run:** `node --test tests/exploration/refocus.test.mjs`。加入聚能已满、一次恢复一／多点、受治疗者同时再聚能、selected actor 与 controlled token 不同、最终补满聚能选项的测试。
- [ ] **3. 复用实际 Refocus 回执。** 使用既有 invocation＋准确 focus update wrapper；Workbench 原函数无可靠完成返回且依赖 controlled token，不仅传 `[actor]` 就当作正确角色。建立当次所有者、actor、token 和活动来源范围；只在该范围调用原生功能。
- [ ] **4. 合并仙露执行权。** 对已认领活动，旧 subscriber 调用本活动完成或读取已保存结果；协调器不另发第二次治疗。支持新增开始／完成上下文、神秘替代、免工具包、命能和满聚能情况。现有 provider 对 Ward／Medic／持续恢复等组合的拒绝只能在对应规则测试通过后移除，未知组合仍标明未自动适配。

完成路径首先从账本取状态，避免重复触发聚能更新：

```js
async function completeComposite(activity,ctx,ledger,refocusEvents,kiss,ownerOperations) {
  if (!ownerOperations.isActivityContext(ctx,activity.id)) throw Error('invalid-scope');
  const prior = await ledger.getActivity(activity.id);
  if (prior.state === 'confirmed') return prior.completion;
  if (prior.state !== 'started') return {status:'uncertain',reason:'activity-in-flight'};
  await ledger.transitionActivity(activity.id,{expected:['started'],patch:{state:'executing'}});
  await refocusEvents.complete(activity,ctx);
  const result = await kiss.completeActivity(activity,ctx);
  await ledger.transitionActivity(activity.id,{expected:['executing'],patch:{
    state:result.status==='confirmed'?'confirmed':'uncertain',completion:result}});
  return result;
}
```

实际 Refocus subscriber 与该函数共享同一调用任务和结果；subscriber 已在该次更新中参与完成时，外层读取结果，不能互相 await 形成循环。
- [ ] **5. 接入聚能治疗。** 从 owned spell 来源与已验证原生 cast 调用执行，保存真实聚能消耗和 DamageRoll／应用回执；默认六秒计时约定，明确活动时间的条目优先。圣疗后再聚能是两个顺序活动；仙露是一个复合活动。结束 HP 达标且未要求补满聚能时停止，不额外免费恢复。

聚能目标是否结束由当前真实数值决定：

```js
function focusFinishSatisfied(goal,actor) {
  return !goal.requireFullFocus ||
    actor.system.resources.focus.value >= actor.system.resources.focus.max;
}
```

原生 cast 接入现有 amp-cast-events 的 paid cast 与所有者回执，保留其确切资源来源；不直接减聚能再制造一张治疗卡。
- [ ] **6. Run:** 本任务与 Task 3 测试通过；对普通手动仙露回归验证仍只治疗一次。Commit 消息 `feat: coordinate refocus and renewable focus healing`。

### Task 5: 手动原生记录与最早完成时间

**Files:** Create `scripts/exploration/manual-events.mjs`、`timeline.mjs`；Modify `native-action-events.mjs`、`medic-native.mjs` 的观察接口；Test `tests/exploration/manual-events.test.mjs`、`timeline.test.mjs`。

**Interfaces:** `createManualEvents({game,Hooks,ledger,nativeActions,ownerOperations})` 返回 `start()/stop()/observe(event)`；`reconstructEarliest({startedAt,activities,assumptions}) -> {endsAt,durationSeconds,certainty,scheduled,missing}`。每项活动保存 actor 顺序和治疗者／患者顺序，只有明确来源才能合并 group。

- [ ] **1. 编写失败测试。**

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {reconstructEarliest} from '../../scripts/exploration/timeline.mjs';
test('患者冷却与各演员顺序共同决定最早时间', () => {
  const activities = [0,1,2].map(order => ({id:`T${order}`,
    actorUUID:'Actor.H',patientUUIDs:['Actor.P'],order,durationSeconds:600,
    treatmentImmunitySeconds:3600,kind:'treatment'}));
  activities.push({id:'F1',actorUUID:'Actor.B',patientUUIDs:[],order:0,
    durationSeconds:600,kind:'refocus'});
  const result = reconstructEarliest({startedAt:0,activities,
    assumptions:['different-actors-may-overlap']});
  assert.equal(result.durationSeconds,7800);
  assert.equal(result.certainty,'earliest-under-assumptions');
});
```

- [ ] **2. Run:** `node --test tests/exploration/manual-events.test.mjs tests/exploration/timeline.test.mjs`。加入三个不同患者三十分钟、持续恢复三十分钟、两治疗者二／三患者、Ward 同组只计一次、缺失分组不猜测、接受治疗时可再聚能等测试。
- [ ] **3. 记录真实原生事件。** 先捕捉 actual action use，再关联 Task 3 的检定、输出及应用证据。原生 use 返回不是唯一完成证据。普通医疗检定和孤立 HP 更新只能作为待核对记录；记录器没有应用 HP 或再次掷骰的能力。
- [ ] **4. 适配已核验 Workbench 宏。** 只在实际宏调用的词法／callback scope 内捕获相关卡；验证当前宏来源 SHA，未知版本仅保存不完整来源，不能用聊天内容补全。读取 `flags.treat_wounds_battle_medicine`：`id` 是目标 Token ID，`healerId` 是 Actor ID，`healing` 是骰总值，`dos` 是源成功度。

```js
const flag = message.flags?.treat_wounds_battle_medicine;
const observed = flag && {
  patientTokenId:flag.id, healerActorId:flag.healerId,
  sourceDegree:flag.dos, rolledHealing:flag.healing,
  bmBatonUsed:flag.bmBatonUsed
};
```

Assurance＋激进手术可能 sourceDegree=2 而实际为 4d8；保存有效效果和源 degree 两份事实。失败的 healing 可以 undefined，不能据此认定完整零治疗。割伤卡没有目标 flags／origin，只在真正调用范围绑定，禁止按相邻卡猜测。拥有原生提供者的手动卡继续由它应用 HP 与免疫。

- [ ] **5. 实现约束前推。** 每个执行者串行游标、患者最后一次治疗开始＋免疫截止点、已知依赖和组活动完成点共同给出下一次最早开始。固定已证明顺序，不重新排列历史；无来源时标记 missing。只有真实开始／结束和并行都具备时标为 `observed`；不得倒改世界时间。

每个已拓扑排序事件的开始点为：

```js
function earliestStartOf({sessionStart,actorFreeAt,patientReadyAt,dependencyReadyAt,notBefore}) {
  return Math.max(sessionStart,actorFreeAt ?? sessionStart,
    ...(patientReadyAt ?? []),...(dependencyReadyAt ?? []),notBefore ?? sessionStart);
}
```

actorFreeAt 在排他活动结束后更新；患者治疗免疫更新为此次开始＋该来源时长；同组只更新一次。加入循环依赖与观测时间违反免疫的测试，分别输出不完整／矛盾来源，不能无限循环或静默改写历史。
- [ ] **6. Run:** 验证 observe 没有 native apply／roll 调用权限，输入重复消息 ID 不新增活动。Commit 消息 `feat: record manual recovery and reconstruct feasible time`。

### Task 6: 世界时间来源、被动恢复与未知提交

**Files:** Create `scripts/exploration/clock.mjs`、`time-effects.mjs`；Test `tests/exploration/clock.test.mjs`、`time-effects.test.mjs`。

**Interfaces:** `createTimeEffects({game,Hooks,fromUuid,capabilities,completionAdapters})` 返回 `beforeAdvance(checkpoint)`、`settle(checkpoint)`、`observeRest(event)`、`diagnostic()`；completionAdapter 有 `matches(rule,checkpoint)`、`beforeAdvance(checkpoint)`、`settle(checkpoint)`，后两者返回带 status 的 Promise 并提供实际来源证明。`createClock({game,Hooks,ledger,timeEffects,isAuthority,confirmationTimeoutMs=10000})` 返回 `advanceTo({id,sessionId,from,to})`、`reconcile(commit)`、`stop()`。结果状态为 `confirmed/blocked/uncertain`；测试可注入短 timeout，不依赖真实十秒等待。

- [ ] **1. 编写失败测试。**

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createClock} from '../../scripts/exploration/clock.mjs';
test('advance 已 resolve 但缺 nonce hook，仍为未知提交且不能重试', async () => {
  let advances = 0;
  const commits = new Map();
  const clock = createClock({
    game:{user:{id:'GM'},time:{worldTime:0,advance:async delta => {
      advances++;return delta;
    }}},
    Hooks:{on:() => 1,off:() => {}},isAuthority:() => true,
    confirmationTimeoutMs:1,
    ledger:{getClockCommit:async id => commits.get(id),
      upsertClockCommit:async c => commits.set(c.id,c),
      transitionClockCommit:async(id,{patch}) => {
        const next = {...commits.get(id),...patch};commits.set(id,next);return next;
      }},
    timeEffects:{beforeAdvance:async () => ({status:'ready'}),
      settle:async () => ({status:'ready'})}
  });
  const commit = {id:'C1',sessionId:'S1',from:0,to:600};
  assert.equal((await clock.advanceTo(commit)).status,'uncertain');
  await clock.reconcile({...commit,state:'uncertain'});
  assert.equal(advances,1);
});
```

- [ ] **2. Run:** `node --test tests/exploration/clock.test.mjs tests/exploration/time-effects.test.mjs`。测试外部时间变化、旧 GM awaiting 时交接、同时间值不同 nonce、休息关联歧义、未选中 Party 成员获得新的 FastHealing 规则。
- [ ] **3. 使用核验过的 V14 时间接口。** `game.time.advance(delta,options)` 返回捕获的当前时间＋delta，不具备事务 ID／CAS。`updateWorldTime(worldTime,dt,options,userId)` 接收同次自定义 options。提交前先注册精确监听，保存 started 声明，再调用：

```js
await game.time.advance(commit.to-commit.from,{
  pf2eThirdPartyAutomation:{exploration:{
    sessionId:commit.sessionId,checkpointId:commit.id,
    expectedFrom:commit.from,expectedTo:commit.to,gmId:game.user.id
  }}
});
```

自己的监听严格校验活动 GM、checkpoint、from/to、dt 和 userId，并持久保存确认回执。原生 promise resolve、当前时间达到目标或数值相等均不能替代来源确认。缺少确认或有并发外部变化时进入 uncertain，不回退、不再 advance。监听设置有有界超时，只表示未知而非失败重试；超时不能冒充副作用已完成。

- [ ] **4. 每检查点核对被动恢复。** Patreon fastHealingTime 对全部 Party 成员而非选中患者生效；其 handler 及 actor.update 未统一 await。先检查当时规则与 predicate，要求可等待的 completionAdapter；没有时在推进前阻止自动检查点，提示具体能力需要人工核对。若执行中出现未知写入也阻止治疗完成和下一轮；不靠固定 sleep 判断稳定，不静默关闭世界设置。没有生效规则时无需等待不存在的写入。

适配能力不足的判断为：

```js
async function requirePassiveCompletion(capabilities,adapters,checkpoint) {
  const rules = await capabilities.activePassiveRules();
  for (const rule of rules.filter(r=>r.passing)) {
    let supported = false;
    for (const adapter of adapters) if (await adapter.matches(rule,checkpoint)) supported=true;
    if (!supported) return {status:'blocked',reason:'passive-completion-unavailable',rule};
  }
  return {status:'ready'};
}
```

此次二十一名 PC 物品快照没有 FastHealing 规则候选，但运行时 synthetics、Token delta 和新授予效果仍必须检查。
- [ ] **5. 整夜休息归因。** 观察真实 `pf2e.restForTheNight` 入口和其后的 Calendaria 时间事件；记录既有时间权威，不再加一次八小时。Calendaria 的内部 source=rest 未进入最终 advance options，关联有歧义时保留待核对，不能仅根据 delta 很大猜测。
- [ ] **6. Run:** 精确 nonce 事件可确认，丢失来源和第三方完成能力不足必须暂停；clock.stop 释放本任务安装的监听。Commit 消息 `feat: confirm exploration clock checkpoints without replay`。

### Task 7: 真实结果驱动的恢复协调与选择策略

**Files:** Create `scripts/exploration/policy.mjs`、`coordinator.mjs`；Test `tests/exploration/policy.test.mjs`、`coordinator.test.mjs`。

**Interfaces:** `chooseNext({snapshot,proposals,session,now}) -> {activities,checkpointAt,reason}`，activities 是 Proposal 加 `startedAt/endsAt` 的 ScheduledProposal 数组；coordinator 为其创建持久唯一 ID 后才成为 Activity。`createCoordinator({ledger,capabilities,providers,clock,policy,isAuthority})` 返回 `start(config)`、`step(sessionId)`、`stop(sessionId,reason)`、`resume(sessionId)`、`addActivity(sessionId,input)`、`snapshot(sessionId)`。所有持久／执行方法为 Promise。没有可执行活动时 chooseNext 可选择最早合法冷却检查点；无未来合法动作时 reason 为 blocked，不无限循环。

- [ ] **1. 编写失败测试。**

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chooseNext} from '../../scripts/exploration/policy.mjs';
test('双治疗者选择不同患者且共同十分钟完成', () => {
  const proposals = ['H1','H2'].flatMap(actor => ['P1','P2','P3'].map(patient => ({
    providerId:'treat-wounds',actorUUID:`Actor.${actor}`,
    patientUUIDs:[`Actor.${patient}`],hpPoolUUIDs:[`Actor.${patient}`],
    durationSeconds:600,earliestStart:0,actorExclusive:true,
    patientTreatmentExclusive:true,expectedNetHealing:9,resourceCost:{},options:{}
  })));
  const next = chooseNext({snapshot:{},proposals,now:0,
    session:{budgetEndsAt:7200,goalsByPool:[],assumptions:[]}});
  assert.equal(next.activities.length,2);
  assert.equal(new Set(next.activities.flatMap(a=>a.patientUUIDs)).size,2);
  assert.equal(next.checkpointAt,600);
});
```

- [ ] **2. Run:** `node --test tests/exploration/policy.test.mjs tests/exploration/coordinator.test.mjs`。用注入的已确认骰序列验证失败→成功→达标；产品不用注入序列。测试预算两小时、一百活动、同池目标、后续冷却、外部变化中断、HP 达标但聚能未满两种结束条件、部分结果完成后 owner 掉线。
- [ ] **3. 实现无未来 RNG 的策略。** proposal 的期望值仅排序；优先可行、单位时间恢复收益、少资源与少风险，同分用稳定 UUID 排序。组活动计算所有患者收益且只有一个时长。自动 DC 对每个验证过的上下文枚举二十个 d20 面的成功度与治疗期望，不能真正掷二十次骰或用未解析的条件加值；未知程度调整将该上下文标为固定 DC。使用当前受领者条件估算，实际结果永远交给原生。

估算的纯函数接受已验证的程度与均值函数，不调用骰子 API：

```js
function expectedHealingPerMinute({outcomeForFace,meanForOutcome,expectedDamage,durationSeconds}) {
  let sum = 0;
  for (let face=1;face<=20;face++) sum += meanForOutcome(outcomeForFace(face));
  return (sum/20-expectedDamage)/(durationSeconds/60);
}
```

meanForOutcome 纳入所选患者缺血与该上下文的规则；Assurance 直接使用已验证的确定成功度，不能当作普通二十面分布。
- [ ] **4. 实现检查点循环。**

```js
async function runCheckpoint(session, activities, checkpointAt, services) {
  for (const a of activities) await services.begin(a);
  const time = await services.advance({sessionId:session.id,
    from:services.now(),to:checkpointAt,id:services.checkpointId()});
  if (time.status !== 'confirmed') return {status:'uncertain',reason:'clock-unconfirmed'};
  for (const a of activities.filter(a=>a.endsAt===checkpointAt)) {
    const result = await services.complete(a);
    if (result.status !== 'confirmed') return result;
  }
  return {status:'confirmed',snapshot:await services.refresh()};
}
```

`services` 由 coordinator 的已定义依赖组成：begin/complete 绑定到具体 provider，advance 为 clock.advanceTo，now 读取实际世界时间，checkpointId 创建并保存唯一 ID，refresh 为 capabilities.snapshot。长活动跨多个检查点保持已有 started 状态，不重新 begin；一小时延长只有成功后才加入五十分钟后续。

- [ ] **5. 开始与恢复验证。** 每次 begin、时间副作用、原生 complete 前确认当前活动 GM 和当前资格；late join 从当前点开始。resume 只继续尚未执行的后续，已存在 check／application 或 clock 来源不明时提示核对，不重试该步骤。普通手动记录不调用该循环。停止后保留历史，释放已确认可释放的预留，不退款或回退真实结果。
- [ ] **6. Run:** 单治疗者三患者、两治疗者两／三患者、Ward、仙露、焦点循环、失败和未知状态用组合测试通过。Commit 消息 `feat: schedule recovery from real results and goals`。

### Task 8: 恢复面板、现有更新集成与独立世界验收

**Files:** Create `scripts/exploration/panel.mjs`、`styles/exploration.css`；Modify `scripts/main.mjs`、`module.json`、`README.md`；Test `tests/exploration/panel.test.mjs`、`tests/exploration/integration.test.mjs`；Create `tests/exploration/acceptance.md`。

**Interfaces:** `createRecoveryPanel({game,coordinator,manualEvents,capabilities})` 返回 `open(actorUUIDs)`；注册 API 为 `game.modules.get(MODULE_ID).api.exploration = {version:1,open,start,stop,snapshot,record,diagnostic}`。start 等方法只向已有认证的活动 GM 请求；玩家只能操作自己拥有的演员和被允许的对象，不能传任意私有上下文。

- [ ] **1. 编写面板契约失败测试。** 测试默认目标最大 HP、资源策略关闭、激进手术风险开关、保存策略后一次启动、未支持能力说明、实际 elapsed 与 earliest-under-assumptions 分别展示，重复打开不安装第二套 hook。核心渲染断言为：

```js
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {formatSessionResult} from '../../scripts/exploration/panel.mjs';
test('时间证据不足时不能显示为已发生的准确时间', () => {
  const text = formatSessionResult({durationSeconds:1800,
    certainty:'earliest-under-assumptions',remainingHP:12,status:'stopped'});
  assert.match(text,/最早|假设/);
  assert.match(text,/30/);
  assert.match(text,/12/);
});
```

`formatSessionResult(record)` 在本任务实现为导出纯函数；DOM 部分使用 V14 ApplicationV2／既有 DialogV2 模式，不引入前端框架。Application 类在 panel factory 内创建，纯函数导入不读取尚未存在的 Foundry 全局对象。

- [ ] **2. Run:** `node --test tests/exploration/panel.test.mjs tests/exploration/integration.test.mjs`。增加非 GM／非 owner 请求拒绝和 actor 名称中的 HTML 字符转义测试。
- [ ] **3. 接入唯一注册点。** main 构造 ledger、capabilities、native providers、manualEvents、clock 和 coordinator 后开放 panel/API。把共用 Refocus subscriber 和 Check／damage middleware 放在已有管线中，避免额外全局包装。module.json 只新增本功能样式入口，已有依赖保持。

```js
game.modules.get(MODULE_ID).api.exploration = Object.freeze({
  version:1,
  open:actorUUIDs => panel.open(actorUUIDs),
  start:config => coordinator.start(config),
  stop:(id,reason) => coordinator.stop(id,reason),
  snapshot:id => coordinator.snapshot(id),
  record:event => manualEvents.observe(event),
  diagnostic:() => ({clock:timeEffects.diagnostic()})
});
```

start／stop 由 coordinator 核验活动 GM；玩家实际手动调用通过已经认证的所有者事件链上报，public record 不能执行治疗。main 只开放一次 API，panel 的关闭清理 UI 监听而不销毁正在运行的持久会话。
- [ ] **4. 核对隔壁最终基线。** 若其已交付固定提交，在本独立工作树集成其必要更新，按功能 diff 合并，不复制覆盖。重点核对 main、av-automation、native-owner-operations、usage-events、metapower/provider 的来源回执与唯一注册；产品接口由一个负责人最后合并。未经用户授权不向隔壁发送消息；可通过只读 thread／commit 确认已交付状态。
- [ ] **5. 跑全部新测试。**

```powershell
node --test tests/exploration/ledger.test.mjs tests/exploration/capabilities.test.mjs tests/exploration/hp-pool.test.mjs tests/exploration/treatment.test.mjs tests/exploration/native-treatment.test.mjs tests/exploration/owner-operations.test.mjs tests/exploration/refocus.test.mjs tests/exploration/manual-events.test.mjs tests/exploration/timeline.test.mjs tests/exploration/clock.test.mjs tests/exploration/time-effects.test.mjs tests/exploration/policy.test.mjs tests/exploration/coordinator.test.mjs tests/exploration/panel.test.mjs tests/exploration/integration.test.mjs
```

此外运行固定基线已有真实来源测试命令，并对所有新增／修改 `.mjs` 执行 `node --check`，`git diff --check`。只在发生新修改、失败或未解疑点时扩展／重复检查。

从 M 运行既有测试前，配置此次已核验的实际源码路径；检查缺失时先取得匹配的源码，不能允许相关测试因缺少来源而跳过：

```powershell
$nativeSourcePaths = @{
  PF2E_NATIVE_BUNDLE = 'C:/Users/Taka/Desktop/fvtt/tmp/fortress-gap-audit-20260925/code/systems/pf2e/pf2e.mjs'
  FVTT_PF2E_BUNDLE = 'C:/Users/Taka/Desktop/fvtt/tmp/fortress-gap-audit-20260925/code/systems/pf2e/pf2e.mjs'
  FVTT_NATIVE_APP = 'C:/Program Files/Foundry Virtual Tabletop/resources/app'
  FVTT_COUNTERACT_MAIN = 'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/qa/runtime/Data/modules/pf2e-counteract/scripts/main.js'
  FVTT_REACTION_BUNDLE = 'C:/Users/Taka/Desktop/fvtt/output/dependency-updates-20260929/live/pf2e-reaction/pf2e-reaction.js'
  FVTT_FORCE_BARRAGE_MACRO = 'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/force-barrage-actual-command.txt'
  FVTT_WORKBENCH_MACRO = 'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/treat-wounds-actual-command.txt'
  FVTT_WORKBENCH_RECALL_MACRO = 'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/recall-actual-command.txt'
}
foreach ($sourceEntry in $nativeSourcePaths.GetEnumerator()) {
  if (!(Test-Path -LiteralPath $sourceEntry.Value)) { throw ('Missing native source: ' + $sourceEntry.Key) }
  [Environment]::SetEnvironmentVariable($sourceEntry.Key,$sourceEntry.Value,'Process')
}
node --test tests/*.test.mjs
```

重点保留 `shared-refocus-update.test.mjs`、`salubrious-kiss-spatial.test.mjs`、`medic-native.test.mjs`、`native-action-events.test.mjs` 的回归。若采用隔壁最终交付基线，同时运行其已交付的 `native-owner-operations.test.mjs` 和 `activity-result-lifecycle.test.mjs`；不能导入仍在编辑的版本。完成后的真实来源通过数以本次日志为准，不使用隔壁早于最终 HEAD 的 1876 条通过记录作验收。

- [ ] **6. 创建测试世界并验收。** 使用独立世界或复制的角色 fixture，GM 与普通玩家同在线。复制数据库只在独立副本进行，不打开实时数据库；若线上新建测试世界需选择不与正在运行测试世界冲突的 ID。按 approved spec 的十二项场景执行，记录 check/result/receipt/effect ID、from/to 与时间 nonce，而不仅观察通知。

验收需包括：盾哥组合；极雷满聚能仙露；水长东／贝肯的 Toolbelt 共享池和主人更新等待；科索斯 Assurance=14；普莱德持有但前置待确认；猩红剩余免疫183秒；Patreon 未选中 Party 成员；整夜休息。用户已确认 shared HP 玩法；若实际同步来源适配未验证，清楚保留该项限制，不能报全场景通过。

- [ ] **7. 做一次新鲜整体审查并修复实质问题。** 检查源码、测试、真实运行证据和 spec 覆盖；特别检查没有以宏返回、当前 HP 或当前时间值替代因果回执。审查后仅重跑受修正影响的检查和必要全套。
- [ ] **8. 文档与本地交付。** README 说明两种模式、未知来源暂停、能力支持表、六秒短动作约定与有限资源策略。候选包版本使用 `0.9.20-exploration.1`；若已被其他固定发行版占用，递增最后的 exploration 数字，不覆盖已有包。保存独立 zip、SHA256、基线版本、变更和验收结果；不在此步骤上传发行或覆盖实时模组／世界。
- [ ] **9. Commit:** 本任务列出文件及验收记录，消息 `feat: expose exploration recovery and verify native integration`。最终向用户交付候选包和可审阅差异，注明自动适配与手动记录能力，以及剩余需要确认的玩法。

## 已核验的原生与兼容细节

- PF2e 8.5.1 Treat Wounds 返回 `CheckResultCallback[]`；治疗 callback 返回 void，子消息创建未 await。必须捕捉 `flags.pf2e.origin.messageId` 与真实应用回执。
- 大失败原始公式无明确 damage/healing 注释；分类用已证明的阶段。治疗 DamageRoll 总值为正数，原生治疗应用取负。
- `DamageRoll.formula` 是显示文本，原始公式用 `toJSON().formula`。
- Workbench Assurance＋Risky 的 source dos 与有效骰可能不同；割伤卡缺目标来源，不按聊天邻接猜测。
- V14.368 `client/helpers/time.mjs:146-150` 的 advance 捕获当前值并调用 settings.set；`:212-218` 把 options/userId 传给 updateWorldTime。Setting 的 onChange 未 await，原生返回也不等于第三方已稳定。
- 现有仙露有独立的资格拒绝清单，没有通用时间上下文；复用机制不等于所有组合自动支持。
- Calendaria 1.4.2 的休息 source 未传入 advance，需真实休息入口归因。
- 已确认 Toolbelt 3.56.5 Share Data 才是当前 HP 共享提供者；旧 helper 开关保持。Toolbelt 的主人 update 未 await，应用需要额外准确完成证明。

详细来源见 approved spec 的规则链接和 `output/exploration-healing-research-20260930/automation-integration-audit-b52ce6f7.md`，该报告也记录了固定快照及源码哈希。

共享 HP 的精确来源为 `output/exploration-healing-research-20260930/toolbelt-sharedata-facts.json` 与 `toolbelt-sharedata-code-evidence.json`：`share-data/tool.ts` 89–92 是公共发现 API，536–603 是更新转发，659–686 是准备后的 HP 合并。用户已经确认当前共用血池；这些事实替代仅查旧 helper 得出的不完整配置推断。

## 计划自审与执行交接

本会话完成计划自审：每项 spec 要求有对应 task；所有 consumed 接口在本计划定义；五项 Review Focus 分别由 Task 2、3、5、6、8 覆盖。步骤包含实际接口、验证输入、命令和交付条件。

推荐 **Native**：本会话按计划逐项实施，最后由新鲜审阅者整体审查。这八项任务共享原生来源和时间上下文，统一实现可减少接口反复交接的成本。也可选择 **Subagent-driven**：每项任务由新 agent 实现、另一新 agent 审查，再进入下一项，最后整体审查；检查更细，但增加每项上下文与审阅成本。

用户需审阅本计划并选择执行方法。可同时授权把已批准设计及具体接口归属发送给隔壁“审查自动化插件需求”，仅协调固定基线和提供者修改范围；没有该授权就保持本会话独立执行和只读了解其交付。
