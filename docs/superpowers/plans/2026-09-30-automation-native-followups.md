# Automation Native Followups Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development with disjoint file ownership; root integrates shared entry points. Run meaningful regression tests before claiming completion.

**Goal:** 落实已批准的审查方案及最新无移动检测、统一回忆知识要求。

**Architecture:** 以生产 0.9.18.6 为基线。权限安全的名称和原操作者执行共享小接口；独立 provider 处理机械后果。维护采用相关文档索引和 dirty 合并；真实使用的凭据绑定当前原始消息。

**Tech Stack:** ESM JavaScript、node:test、Foundry v14/PF2e、socketlib、libWrapper、Workbench 7.7.5。

**Spec:** `docs/superpowers/specs/2026-09-30-automation-native-followups.md`

## Global Constraints

- 不检测移动、路线、距离、视线或触碰事实，不增加重复确认。
- 不改变原生动作经济、资源规则、反应决定或秘骰权限。
- 所有回忆知识一次骰点；固定技能及 Assurance 保留。
- 不重新引入第三方插件版本和源码指纹门槛。
- 隐藏目标名称按接收者而非执行 GM 权限计算。
- 不截断仍有效的付款、伤害或不确定结果凭据。
- 不修改其他工作区；各 agent 仅编辑分配文件，root 编辑 main 和共享接线。

## Review Focus

1. 原操作者断线、GM 切换或文档慢同步：不代投、不重复扣资源。
2. 隐藏 NPC、多个 T 目标、无 T 目标、Lore：玩家端不泄露，GM 知识裁定沿用原骰。
3. 原生取消、英雄点重掷、已应用伤害、延迟投伤害：旧结果不可再被应用。
4. 长时间开关角色卡和无相关效果的大场景：引用清理，相关维护不遗漏。
5. 能力固定技能、Assurance、熟练度和次数限制：统一入口不改规则。

### Task 1: Capture and establish the actual baseline

Files: module runtime files, this spec/plan, `docs/fortress-automation/native-followups-2026-09-30.md`.

- [x] 从 CN 只读捕获 137 个运行文件并校验版本 0.9.18.6。
- [x] 创建隔离工作区，覆盖运行代码为生产捕获，保留 canonical 回归测试。
- [x] 运行 `node --test modules/pf2e-third-party-automation/tests/*.test.mjs`，保存完整基线结果。
- [ ] 更新因取消版本锁而过时的测试，补齐热修新增契约；真实失败不能改断言掩盖。
- [ ] Commit verified baseline and test reconciliation.

### Task 2: Target privacy and remove redundant spatial workflow (native_flow_audit)

Files: `native-context.mjs`, `eldamon-basic-settlement.mjs`, `fear-automation.mjs`, `eat-fortune.mjs`, `roaring-sustain.mjs`, `medic-actions.mjs`, `defensive-advance.mjs`, `salubrious-kiss*.mjs`, `bard-familiar.mjs`; matching tests. Root owns `usage-events.mjs` and `main.mjs`.

Interfaces: shared `publicTargetName(token,{game,user})` returns recipient-visible text; existing provider interfaces remain unchanged; provider usage result may return `{status,result}` for waiting/cancelled/done.

- [ ] Add tests proving hidden names never reach player choices/titles and ordinary native action requires no movement confirmation.
- [ ] Run focused tests RED, implement recipient-aware labels and remove spatial gates/confirmation without moving Tokens or inventing rule results.
- [ ] Keep reaction choices/true branching; hide technical lifecycle diagnostics from players.
- [ ] Release familiar sheet listener on re-render/close, with detached-root regression test.
- [ ] Run all affected provider tests GREEN and report exact files/evidence.

### Task 3: Indexed maintenance and coalescing (performance_audit)

Files: `spiritual-scar-expiry.mjs`, `party-automation.mjs`, `av-automation.mjs`, `glimpse-configuration-events.mjs`, focused helper(s) and tests. Root handles main chat early filter and usage listener lifecycle.

Interfaces: retain provider `register({Hooks})` and cleanup signatures; indexes recognize world and synthetic actors and rebuild on relevant scene lifecycle.

- [ ] Add tests: zero related effects reads no inventories on unrelated chat; one relevant effect still expires from correct source; bursts coalesce without dropping dirty work.
- [ ] Observe RED, add relevant-effect/source indexes and field filters; remove movement-dependent automation per spec.
- [ ] Run scar/party/configuration/load tests GREEN; provide operation counts and exact scope.

### Task 4: One-roll Recall Knowledge and Automatic Knowledge (history_gaps)

Files: `knowledge-automation.mjs`, new `knowledge-workbench.mjs`/focused helper(s), RK tests. Root installs provider wrapper entry points in main if needed.

Interfaces: `createKnowledgeAutomation` retains provider contract; new RK bridge exposes wrapper/register functions through provider, authorizes original owner and GM result, uses actual native same-roll data rather than trusting remote totals.

- [ ] Read installed Workbench macro UUID `Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros.Macro.xcFr7PWwG5OVALNJ`; inspect exact roll contexts and target skill/DC selection.
- [ ] Add tests for one shared d20, target-based applicable skills, secret results, no duplicate incidental RK, fixed skill, Assurance and validated GM result driving benefits.
- [ ] Observe RED, implement bridge using runtime interfaces rather than version/source lock; route all normal RK actions and existing incidental requests.
- [ ] Add Automatic Knowledge entry and preserve fixed choice/frequency; GM adjudication of lore/information remains.
- [ ] Run focused tests GREEN and provide integration instructions for main.

### Task 5: Usage lifecycle and operation receipts (root)

Files: `usage-events.mjs`, focused DOM/receipt helper(s), `main.mjs`, usage and lifecycle tests.

Interfaces: normalized `{status:'done'|'waiting'|'cancelled',result?:string}`; legacy provider string means completed only when no persisted waiting provider state. Frequency `claim(id,{itemUuid,userId,messageId})` binds observed receipt to one real message; single claim remains enforced.

- [ ] Write RED tests for normal delay beyond 5s, receipt replay/different item/user, pending chat latency, waiting/cancelled render and detached sheet roots.
- [ ] Claim validated receipt before waiting on pending chat write; bind proof, prune completed receipt bookkeeping without allowing an old observed decrement to replay.
- [ ] Track application listeners by application identity; remove previous root on render and all on close/unregister.
- [ ] Filter non-damage chat before enumerating cycle actors.
- [ ] Run usage/bard/main/load tests GREEN.

### Task 6: Original-owner combination operations and final-result lifecycle (root)

Files: `spell-combination.mjs`, `dual-strike-automation.mjs`, focused owner/attack helper(s), related tests; main register through provider.

Interfaces: GM sends bounded original-card operation nonce; original actor owner invokes native roll APIs; GM validates persisted native message identity/context before extra settlement. Definite cancellation is distinct from uncertain execution; payment is not repeated.

- [ ] Add RED tests for original owner UI settings, cancelled attack before native result, lost reply, duplicate request and hero-point replacement.
- [ ] Move native attack/damage interaction to original owner RPC; preserve original PF2e check flags/targets and source linkage.
- [ ] Make definite cancellation recoverable without duplicated payment; preserve uncertain state for maintenance.
- [ ] Invalidate stale unapplied combined results on a kept reroll; subsequent damage uses latest native outcome and same paid operation. Already applied damage uses native undo rather than compensating HP.
- [ ] Run combination/activity/payment tests GREEN.

### Task 7: Weapon Surge and bounded followup evaluation

Files: `next-strike-effects.mjs`, related rule-compat provider and tests, release notes.

- [ ] Add RED test: next attack with bound weapon consumes Surge on hit or miss, another weapon does not; delayed damage from that attack retains original extra die.
- [ ] Implement using existing next-strike snapshot mechanism; verify native consumption and no duplicate added dice.
- [ ] Evaluate external merged IWR, one-result reactions and summoner ownership boundaries against installed source; record reproducible limits and avoid unsupported broad hooks.
- [ ] Run next-strike and fortress tests GREEN.

### Task 8: Integration, review and release artifact

- [ ] Run full suite, syntax checks, source-file integrity and focused load probes; reconcile all baseline failures with evidence.
- [ ] Fresh independent whole-change review covers Review Focus above; fix important findings with tests.
- [ ] Create versioned release artifact preserving 0.9.18.6 hotfixes; inspect manifest/package contents and rollback instructions.
- [ ] Document verified versus actual multi-client UI coverage precisely; only deploy a verified candidate, retaining remote module backup and checking hashes afterwards.
