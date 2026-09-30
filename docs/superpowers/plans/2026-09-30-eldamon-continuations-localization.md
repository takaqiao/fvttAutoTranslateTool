# Eldamon continuations and scoped localization implementation plan

> For agentic workers: use the existing native-flow design approved by the user on 2026-09-30; execute independent file groups in parallel, then review and verify the integrated result.

**Goal:** Continue Eldamon activities from current native cards and actor actions without searching old cards, remove geometry gates, and complete scoped visible-text localization under 标准PF2汉化流程-v3.md.

**Architecture:** Existing durable ledgers remain authoritative. New card/sheet entrances project and continue the same activity; original Use, native rolls, ownership, payment, IWR and receipts remain in their existing providers. Localization changes visible strings only, with a version-bound Workbench adapter when its macro has no translation carrier.

**Tech stack:** Foundry 14.368 / PF2e 8.5.1, ES modules, Node test runner, socketlib/libWrapper, Workbench 7.7.5, Eldamon 1.5.1.

**Approved design:** The preceding chat table and the user's “那就继续”; existing flows are being extended, no new authoritative subsystem or floating panel.

## Global constraints

- Work in `C:/Users/Taka/.codex/worktrees/automation-native-20260930/fvtt`, baseline `f45338970afec79e956f3685175758c784f9e098`.
- Do not modify original dirty checkout, other worktrees, production worlds, third-party source packages or unrelated running QA.
- Same nonce/message/phase across entrances; no repeated Use, payment, effects or speculative recovery. Uncertain results stay blocked for GM review.
- Never auto-roll damage immediately after a save; the original native damage click retains the hero-reroll window.
- Remove distance/adjacency gates and redundant confirmations; retain real target, actual native outcome, owner/resource/source validation and optional reaction choices.
- No coordinate/movement hooks, polling or repeated whole-chat scans; recover indexes once and update from relevant documents.
- Follow v3: freeze machine identifiers/rules, visible headings/labels/body Chinese, document names Chinese plus their exact English name; native translated labels take precedence over new hardcoded terms.
- Source version/fingerprint, before/after, independent review and covered/unverified fields must be recorded. Existing unaffected QA is reusable, not proof for changed UI.

## Tasks

### Task 1: High Voltage continuation entrances

Owner: native_flow_audit. Files: `eldamon-voltage.mjs`, `eldamon-voltage-executor.mjs`, new local continuation helper if needed, voltage tests. No edits to main, electricity or knowledge files.

- [x] Add and observe failing behavior tests for current hit-card trigger, target save/caster damage continuation, actor action continuation after activeNonce is cleared, permission filtering, duplicate clicks and uncertainty.
- [x] Project original activity onto current native attack/save cards and existing actor action area; provide a public continuation method for HUD/hotbar macros if direct supported integration is practical. Preserve the original card as a record and fallback.
- [x] Remove voltage geometry gate; select actual attack/touch branch once. Native-proven hit requires no repeated confirmation. Permission-aware labels, no hidden token names.
- [x] Translate visible voltage strings/errors; use pure Chinese refresh labels. Preserve existing ledger errors/phase semantics and native rolls.
- [x] Run affected tests and report red/green evidence, interfaces and diff. No commit or release; root integrates.

### Task 2: Electricity direct-use, trigger shortcuts and distance removal

Owner: performance_audit. Files: `eldamon-basic-settlement.mjs`, `eldamon-electricity.mjs`, `eldamon-electricity-provider.mjs`, their tests, relevant pure electricity eligibility helper if required. No main, voltage or knowledge edits.

- [x] Add failing tests: actual manipulation Use settles captured target, display does not settle, replay/cross-entrance duplicate does not refresh effect, shield only on real trigger, current damage receipt chooses exact chain source without bypassing checks, no geometry access.
- [x] Expose actual-use resolveAction/requiresActualUse/executeUsage through the existing usage observer. Validate original definition/UUID and captured target before mutation; shield activation never shocks an enemy.
- [x] Add current attack-card shield shortcut with original activity identity, and current damage-taken chain shortcut that enters genuine owned native Use with bound receipt and preserves branch/target/reaction/resource selection. Provide actor fallback for unrecorded actual trigger.
- [x] Remove chain geometry requirements and text; retain ownership, native actual damage/IWR attribution, target/effect and reaction conditions.
- [x] Localize visible electricity errors, buttons and captions. Report public interfaces for root main wiring and native QA.
- [x] Run affected tests and report red/green evidence. No commit or release.

### Task 3: Scoped RK localization and v3 inventory

Owner: history_gaps. Files: knowledge scripts and new version-bound Workbench display adapter/tests; localization evidence docs. No Eldamon runtime/main/module manifest edits.

- [x] Snapshot exact inputs and establish EN/CN field inventory/terminology with consumer and source fingerprints; names derive their exact own original text, never invented English.
- [x] Translate RK headings, Assurance notices, Workbench tables/ranks/degrees/tooltips/dynamic notices in the scoped bridge. Verify Workbench 7.7.5 producer/fingerprint/current values; preserve HTML attributes, native runtime objects, rolls and source/permissions. No global replacement.
- [x] Reuse native labels for Society/proficiency/degrees. New adapter errors remain actionable Chinese; unchanged third-party macro stays unchanged.
- [x] Test both target/no-target display, GM secret/player visibility and mechanical invariants with actual macro source.
- [x] Report field counts, independent-review needs, raw/source issues and native QA cases; do not label static coverage as native proof.

### Task 4: Integration, review and delivery

Owner: root. Files: main wiring/API, remaining metapower visible strings/activity-result label, scoped docs and release metadata as authorized.

- [x] Review each task diff and independently inspect rule/permission/payment/translation invariants; send findings back to owner and re-review fixes.
- [x] Wire direct-use provider and continuation API only after exact public interfaces are known. Translate remaining scoped visible strings with field record and native labels.
- [x] Run actual-source full Node suite, syntax and diff checks. Start only this task's own own 30426 QA instance after checking exact paths/port; validate changed real sheet/chat/GM/player flows and Chinese target/no-target RK.
- [x] Complete v3 field/coverage and input/candidate hashes. Reuse unchanged dependency evidence; explicitly record any unresolved coverage rather than pass by test count.
- [ ] Produce verified candidate/release under existing release workflow and installation authorization from this ongoing task. Never cover world documents. Validate actual ZIP/manifest/runtime inventory and only deploy when existing production guard proves safe.

## Review focus

1. Simultaneous old/current card clicks or reconnect must not produce a second payment/effect/roll.
2. A claimed voltage activity with cleared activeNonce remains discoverable to the target owner and caster without leaking hidden creature names.
3. Hero-rerolled save must be adopted at damage click; no automatic damage publication after initial save.
4. Display-only cards, expired reactions, wrong source/target and mixed unconfirmed electricity cannot bypass the real native activity.
5. Workbench unknown source/version and translated names must preserve source HTML/rules and private roll visibility; no heuristic text replacement.

## Progress

- Baseline worktree clean; last release 0.9.19.
- Shared-file check: Task 1/2 only share consumer contracts, not runtime files; Task 3 isolated knowledge scripts; main and release metadata owned by root. Existing implementation agents can resume with their audit context; higher-level parallel delegation is used only for these disjoint file groups.
- No production/world writes during implementation; delivery checks remain sequential.
