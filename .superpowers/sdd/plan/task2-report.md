# Task 2 integration report — 2026-09-18

## Implemented checkpoints

- f23a28dd: initial native admission/ledger, original-card snapshots and damage/Region wiring.
- c9579061: actual startup fixes (compose existing libWrapper registrations), source-specific native Use controls, normal-channel snapshots, policy updates, target capture, bounded replay archive, interrupted-use recovery and provider composition.
- e316b0ab: Electric Shot already-Shocked failure outcome, same-roll native arithmetic and recipient binding; non-Chain reaction confirmation preserved after Electricity selection hook.

No push, release, production edits, QA server or browser operation was performed by this implementer. Root owns actual native QA and final acceptance. Electricity files are another agent's current independent work and must be committed with the importing main checkpoint before branch delivery.

## Runtime behavior

Exact owned source identities authorize Widen/Siphoning and known power profiles. Widen is repaired to one-action metadata. Original actor rows get native `data-action=use-action` controls only for exact supported metapowers/powers and Medic provider routes when no native/Toolbelt Use already exists. This works without enabling optional Toolbelt settings. The public PF sheet handler awaits admission, invokes the unchanged private native builder and commits the original persisted card. Native frequency stays under the native builder. Hotbar item usage composes the existing usage wrapper; spell consumption composes the existing cast middleware; Check processing composes the existing reaction/Salubrious wrapper rather than double-registering libWrapper paths.

Frozen Toolbelt API descriptors are not replaced. Its actual DOM entry invokes the captured native helper under admission. Reviewed HUD2.55.2 controller methods use pinned method hashes and the same awaited helper contract; unknown versions do not silently claim support. Native cached and future action variants include subclass completion and multi-actor use. Deprecated callback-only aliases are explicitly refused while an activation/lease is live, since their cancellation/side effects cannot be verified. Description sends and drafts do not arm a metapower.

The active GM serializes actor state mutations with a durable reservation/start/finish lease and nonce/client sequence. Original card, author, actor, embedded item and source are verified. Snapshots freeze prepared source, level, branch, area, policy and traits at admission. Ordinary channel uses also carry a normal snapshot for Voltage/Electricity. Cancelled pre-native or verified no-result native checks preserve activation; other successful/uncertain actions consume the captured old activation. Turn changes before native start reject execution. Later completions cannot clear newer activation. Original activation cards expose explicit clear for oral or otherwise unobservable actions.

Ordinary action receipt detail is bounded to64, with client high-water marks rejecting pruned replay. Source/channel/uncertain/live proofs are retained. A GM sheet recovery button archives an interrupted outstanding lease as uncertain after explicit confirmation; no refund, retry, invented effects or duplicate charge occurs. Charged discharge uses one payer for normal/meta channels. Payment intent is persisted before native counter update; PF2e deletes Charged at zero and cascades GrantItem children. A lost deletion response cannot charge again or silently assume payment: it requires explicit/manual reconciliation. Existing nonzero payment proof permits idempotent completion.

Current user policy is persisted granularly: dischargeArea/range/saveDowngrade retain; coarse dischargeNonDamage remove retains the previous restriction on Reactive Chain target relaxation; highVoltage convert. Siphoning suppresses added Charged/Shocked/persistent effects. Voltage executor handles future HV trigger and suppresses only Siphoning Refresh. Normal Voltage/standalone Refresh remain its provider's responsibility.

Original-card area anchors carry the actual selected `data-distance`, so PF2e's original Region builder and canvas.regions.placeRegion receive Widen's dimension. Damage links retain native augmentation/modifiers/evaluation and bind to original snapshot via roll options. Same evaluated native DamageRoll is converted before publication, with true untyped instances. Material/metadata partitions unsupported for lossless general conversion visibly reject before mutation. Per-recipient Siphon .5/1 coefficient runs after native outcome and before IWR. Proof persists through native alter.

Electric Shot fixed failure remains level damage. A separate original native link selects already-Shocked failure: one selected recipient must own the exact Shocked source; evaluated native DamageRoll.alter(.5,0) is adopted in place before Siphon conversion, preserving dice/modifiers and original object identity. Applying this alternate roll to another recipient is rejected. Its card already contains failure half; apply the normal full amount. As with native attack outcome controls, choosing the failure link is an explicit outcome selection, not a new inferred hit event.

## Provider composition in main

- Voltage comes first in beforeDamage for delayed-roll private application grants.
- Electricity receives beforeChannel, GM validateSelection, onCommittedChannel, Check middleware, composed DamageRoll.toMessage, alter preservation, before/after damage and actual post-IWR observation. Native PF8.5.1 lines32148–32149 explicitly await the nativeDamageIWR adapter result. Shield adapter decision is awaited first; Electricity observes the same result, no additional applyDamage call.
- Voltage onRefresh callback is passed to Electricity; root is completing its Voltage-side callback implementation.
- P0 createEldamonDataRepair and Medic providers are wired into existing maintenance/usage/register loops.
- Initial target UUIDs are captured before asynchronous selection/admission and carried into original usage input; Toolbelt capture cannot silently replace them with later user targets.

## Verification

Latest full command in the shared worktree:

```powershell
$env:FVTT_NATIVE_APP='C:/Program Files/Foundry Virtual Tabletop/resources/app'
$env:PF2E_NATIVE_BUNDLE='C:/Users/Taka/Desktop/fvtt/output/bob-transfer-audit-20260917/resources/systems/pf2e/pf2e.mjs'
node --test modules/pf2e-third-party-automation/tests/*.test.mjs
```

207 tests passed,0 failed,0 skipped (latest shared working tree, including the independent Electricity tests). Native dice tests load installed Foundry14/PF2e8.5.1 classes; no licensed upstream source is copied into repository. TDD failing→passing checks cover new entrance injection, normal snapshots, archived replay, turn change, abandoned lease recovery, lost zero-counter payment response, Electric Shot9→4→2 native arithmetic and bound recipient. `node --check main.mjs` and scoped `git diff --check` passed.

Root reported actual native UI Widen→Surge original-card success with Toolbelt enabled, snapshot30→40 and zero page errors. Root's default Toolbelt-disabled run armed Widen but Surge click was dropped: inspected native sheet handler removes itself for50ms after any action (including toggle-summary), whereas QA immediately clicked Use after toggling. Root reran with100ms after summary: default native Use now passes and actual native Region preview has40ft. Persistent Region placement is still under root investigation; Siphon IWR and full provider acceptance remain root-owned and must not be represented as passed by this report.

## Explicit boundaries / remaining acceptance

- Callback-only legacy macros, direct frozen Toolbelt API calls outside owned DOM, custom macros and unrelated external automation that do not enter an observed native action remain manual boundaries. Clear activation explicitly before unobservable actions.
- Uncertain external side effects require GM reconciliation; no automatic refunds or dangerous retries.
- Retained source receipts can grow over long campaigns; only provably unconsumed ordinary details are pruned. Live source-proof pruning is intentionally not guessed.
- Mixed material/metadata native damage partitions use visible manual resolution rather than lossy conversion.
- Non-Chain reactive trigger confirmation remains explicit; Electricity provides actual Reactive Chain receipt evidence. Nonattack physical touch for HV remains the Voltage provider's explicit GM factual confirmation entry.
- Root must complete native default-UI, Region, Siphon/IWR, Voltage/Electricity/Medic, multiple-client/GM-handover and cancellation QA before final completion. This report is an implementation handoff, not a claim that all Claudius scope has passed native QA.

Actor-specific encounter identity followup: use the actor's single started encounter from game.combats rather than the GM's viewed tracker. Regression confirms merely viewing another encounter preserves Widen. An actor in no started encounter gets a null turn; multiple simultaneous actor encounters reject rather than guessing. 12/12 lifecycle tests passed after this change.
