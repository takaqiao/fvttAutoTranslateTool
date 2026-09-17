# Claudius Medic provider checkpoint

Owned files only: `scripts/medic-actions.mjs`, `scripts/medic-rules.mjs`, `scripts/medic-native.mjs`, corresponding `tests/medic-*.test.mjs`, and this report, under `modules/pf2e-third-party-automation` unless otherwise stated. No production, runtime, main, existing provider, or dependency source edits.

## Integration

Import `createMedicActions` from `scripts/medic-actions.mjs`; call with `{game,fromUuid,choose,onError}`. Existing provider interface: `resolveAction(item)`, `captureUsage(item)` (returns fresh `medicInput:{nonce}`), `executeUsage({actor,item,message,user,action})`, `register({Hooks,socket})`, `maintain()`. Register owns its chat render hook. Routes are `medic:treat-condition` and `medic:doctors-visitation`. `register` must run on all clients; authority mutation is active-GM-only. Existing original-Use usage pipeline supplies `usageInput.actualUse` and recorded target UUIDs. Do not route ordinary description cards as Uses.

Original feat Use is the only entrance. The actor's original speaker token and one original targeted token are required. Visitation prompts its original owner for one of four branches: Battle Medicine / Treat Poison (1 total action); Administer First Aid / Treat Condition (2 total actions). Treat Condition and Battle Medicine require their exact feat source. The original card renders the chosen total action glyph/status, then continuation/cancel controls for its original user.

`continueUsage(message,user,{cancel})` and `recordMovement(token,movement,operation,user)` are exposed for integration QA. Movement uses Foundry's actual `moveToken` callback, `movement.id`, `passed.cost`, and `passed.waypoints[].action==='walk'`; it never moves a token. All recorded segments must fit the actor's ground Speed, come from the original owner in the same turn, and cannot contain teleport/other methods. Continuation rechecks token identity, adjacency, tools, branch entitlement and the actor turn. Cancel keeps the committed activity/flourish. Combat turn changes expire continuation. Native action callbacks carry `[MODULE_ID].metapowerContinuation:{actorUuid,cardId,nonce}`.

Optional boundaries for tests/other integrations:

- `distance(healerToken,targetToken) -> feet|null`; default native `Token.distanceTo` on one scene.
- `requestFacts({actor,target,condition,user,message}) -> {dc,restricted,continuous}|null`; default private active-GM DialogV2 asks source DC, artifact/effect above20, and ongoing circumstance. It never asks for degree. Facts are not saved in the original public message. Unknown source facts remain a GM decision.
- `rollCheck({...context,item,target,nonce,dc}) -> {roll,message}|null`; default native PF2e `Check.roll(new CheckModifier(label,medicineStatistic), context, null, callback)`. Domains include native Medicine domains and check/counteract-check; type counteract; exact target; option `pf2e-third-party-automation:medic:<nonce>`; DC visible:false; messageMode blind; normal native modifiers dialog. The callback supplies actual degree. Counteract0.3.0's global helper is deliberately not called: it has mutable global DC/rank fields, finds messages globally, and returns no receipt. The existing Counteract dialog fields are removed only from this provider's marked dialog to prevent global rank/DC contamination. The native Medicine roll remains unchanged.
- `delegateTreatment({...context,healer,target,branch,continuation,validate})`: default `createMedicNative` below. The validation closure must run immediately before native submission.
- `commitActivity({...context,cost,flourish,nonce})`: optional external action-pool adapter. PF2e itself has no general action pool; current campaign's Auto Action Tracker is disabled. Only one provider activity receipt is committed. There is no claim of general third-party tracker compatibility.

## Source/dependency facts

PF2e **8.5.1** official tag verified, not translated-name matching:

- https://raw.githubusercontent.com/foundryvtt/pf2e/pf2e-8.5.1/packs/pf2e/feats/archetype/medic/treat-condition.json
- https://raw.githubusercontent.com/foundryvtt/pf2e/pf2e-8.5.1/packs/pf2e/feats/skill/level-15/legendary-medic.json

The English source requires **Legendary Medic feat**, UUID `Compendium.pf2e.feats-srd.Item.Kk4AMZtpQnLEgN0b`, for artifact/above20 sources, then DC+10. It does not say merely legendary Medicine rank; the actor's Chinese translation differs. Ordinary and expanded healer's toolkit exact sources are supported. Parent-granted/locked/in-memory conditions are rejected without deleting their parent; continuous circumstances are explicitly ineffective.

Workbench **7.7.5** adapter loads the existing macro `XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine` from `xdy-pf2e-workbench.asymonous-benefactor-macros-internal`. Foundry `Macro.execute` supports named lexical parameters. The adapter supplies pinned `actor/token`, read-through `game` with fixed original target set, `canvas` with fixed healer selection, `ChatMessage.getSpeaker`, and a local Dialog subclass. This changes no globals, macro source, healing formula, cooldown, or immunity. The subclass forces Battle Medicine, retains native choices, validates before submission and awaits its existing async callback. Native Treat Poison and First Aid use `game.pf2e.actions.get(slug).use` with exact actor/target; First Aid asks the original owner to choose stabilize or stop bleeding. Neither branch sends a second feat Use.

## Verification

Observed RED then GREEN for rules, execution, replay hardening and scoped delegates. Command:

```powershell
node --test modules/pf2e-third-party-automation/tests/medic-*.test.mjs
```

Checkpoint: **46 passed, zero failed/skipped**. Assertions exercise native-result consequences, four degrees and removal floor, tool/source/adjacency/continuous boundaries, all four activity costs, movement/cancel/turn validation, owner/GM checks, double provider delivery, copied nonce/full-card/continuation rejection, stale/rerolled/inconsistent receipt rejection, same-turn flourish and exact native/Workbench actor/target scope. External Foundry effects are modeled by synthetic Documents; these are not a substitute for real browser/native multiple-client QA. Root owns that QA.

## Honest limits for runtime QA

- Check/condition application is at-most-once and fails closed on uncertain writes. GM reload/handoff during rolling/applying does not resume automatically. The activity remains committed; no blind replay/duplicate healing.
- The supported active-GM queues serialize actors and target conditions in that client. Independent edits during the check are rejected by exact embedded condition/source snapshot. Foundry has no atomic multi-document compare-and-swap; check claim precedes application to avoid duplicate side effects.
- Other flourish cards are observed when they carry the native actual-Use marker while this provider is registered. Uninstrumented manual macros and actions used before registration cannot be reconstructed reliably.
- Visitation requires an actual nonzero native walk movement receipt. A zero-distance Stride or older Foundry movement interface without this callback is rejected rather than fabricated. Movement legality beyond native path cost/walk type (e.g. unusual terrain/forced movement) remains the native movement provider's responsibility.
- The Workbench dialog currently runs on authority, bound to the original actor/target; branch/condition/First Aid choices use the existing original-owner chooser. Native Workbench result cards retain their existing apply-healing/immunity interaction. This provider does not claim direct automatic HP/immunity application, and does not add a second Patreon immunity writer. Root must verify current Workbench+Patreon behavior in native QA.
- First Aid and Treat Poison preserve the system/provider's native effects coverage; delegated checks are not advertised as newly implemented automatic downstream effects.
- Locked/granted conditions, alternate magical kit types, a general action-tracker pool, GM handoff recovery, and zero-distance Stride are explicit unsupported boundaries, not secretly emulated outcomes.

## Root native QA entry points

Use the original feat on the character sheet, not `toMessage` without actualUse and not the provider API as evidence of native entry. Exact source IDs above route renamed features. The original card should contain `flags.pf2e.origin.uuid`, `flags[MODULE_ID].usageInput.actualUse===true`, captured target UUID, fresh `medicInput.nonce`; active GM writes `medic.status`. Select exactly one patient token before the original Use. The sheet click boundary already captured by the shared usage events is `[data-action="use-action"],button.use-action` inside `[data-item-id]`.

Treat Condition GM form inputs: `input[name=dc]`, `input[name=restricted]`, `input[name=continuous]`; confirm button action `confirm`. Supply actual missing source facts, then roll the native check normally. Its native message must have `context.type=counteract`, hidden DC, blind mode, exact target actor/token and `pf2e-third-party-automation:medic:<nonce>` option. Verify selected embedded condition changes according to resulting degree and other conditions/parents remain intact. Source DC is never written to `medic` activity flags. A changed condition while the native dialog is open should reject application.

Visitation: choose among the four branch labels via the existing native choice UI; after native owner walk movement, use `.medic-activity [data-medic="continue"]` on the original card. Cancel selector `[data-medic="cancel"]`. Expected durable state sequence: `committed → movement → treatment → done` (or Treat Condition `rolling → applying → done`). Inspect `cost` 1/2, `movementIds`, `movementCost`, `turn`, actor `medicUses[nonce]` and `medicFlourish`. Wrong token/owner or teleport movement does not authorize continuation. A second same-turn Visitation rejects before another commit. Movement may be delivered asynchronously; wait for the original card's movement flags before clicking.

Workbench Battle Medicine dialog is the existing `Treat Wounds / Battle Medicine`; its `[name="useBattleMedicine"]` is fixed at 1. It must use original actor and single recorded patient even if GM currently controls/targets unrelated tokens. Native Workbench controls for DC/assurance/etc remain its own. No provider code rolls healing dice or creates immunity. Validate those native cards with Patreon enabled rather than expecting this checkpoint to replace them.
