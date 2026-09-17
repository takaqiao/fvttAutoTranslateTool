# 0.9.5 verification

Candidate: PF2e Third Party Automation 0.9.5, Foundry 14.368, PF2e 8.5.1, Node 24.19.0. This batch adds the current fortress Desperate Prayer, Defensive Advance and Glimpse of Redemption workflows. It does not complete every current fortress ability. Publication and production installation are tracked separately.

## Automated checks

Run from the repository root with locally installed dependencies:

```powershell
$env:FVTT_NATIVE_APP = '<Foundry installation>/resources/app'
$env:PF2E_NATIVE_BUNDLE = '<PF2e installation>/pf2e.mjs'
$env:FVTT_COUNTERACT_MAIN = '<Counteract installation>/scripts/main.js'
$env:FVTT_WORKBENCH_MACRO = '<private export>/Treat Wounds and Battle Medicine.mjs'
$env:FVTT_REACTION_BUNDLE = '<Reaction installation>/pf2e-reaction.js'
node --test modules/pf2e-third-party-automation/tests/*.test.mjs
```

Result at source checkpoint `f2c46c93`: **531 tests, 531 passed, 0 failed, 0 skipped**, including the electricity no-op expiry correction discovered in final acceptance. Native dependency tests deliberately skip without those private paths; that reduced run is not equivalent. Native findings received focused failing-then-passing regressions and scoped independent review.

## Native checks

Checks ran in a loopback-only world with independent clones of the current party records. The profile enables **23 relevant modules**, including Trigger Engine 1.35.0, Trove 2.3.5, Patreon 3.2.28, Reaction 1.4.3, Toolbelt 3.56.2 and Workbench 7.7.5. This is not a fresh full-profile test of the production world's 121 modules. The older 122-module smoke belongs to 0.9.3. No production document, setting, module or service was changed during these checks.

| Area | Observed behavior |
| --- | --- |
| Prayer start and original Use | A real native turn start prompted the player. The original Use consumed daily frequency 1→0 and granted focus 0→1 with matching opportunity, message and payment records. |
| Prayer native devotion payment | Original Lay on Hands cast paid focus 1→0 once and committed its matching temporary-credit receipt in the same native actor update. Original Weapon Surge after ordinary focus recovery paid temporary credit first, leaving the ordinary point in the pool. |
| Prayer expiry and invalid opportunities | Unused temporary credit plus one recovered ordinary point expired from pool 2→1. Declining the start choice prevented a later original hotbar use from paying daily frequency. Starting with ordinary focus 1 offered no usable opportunity and left daily frequency unchanged. |
| Defensive Advance display and startup | A display card caused no shield effect or movement plan. A client whose initial Patreon rule cache predated the repaired Use predicate reported that a page reload was needed; reloading produced the ready state. The test did not manually change settings to bypass that check. |
| Defensive Advance GM | Original sheet Use raised one native shield effect. Real map dragging produced a matching native plan and server movement of 10 feet. After both movement completion and animation, one native melee Strike used MAP 0 and the original weapon and target. The activity recorded cost 2; the included Strike carried its free-action marker. |
| Defensive Advance player and GM | A separate player client performed original Use, native map movement, target/weapon/MAP selection and native Strike. The active GM bound the player movement to its exact plan and use. The player selected MAP 1, and the native roll retained the −5 penalty. Repeating Use in the same turn created no further plan, attack or shield effect. |
| Glimpse positive resistance | Two authenticated clients used a real NPC Strike and its original DamageRoll. One attack containing 24 slashing and 20 fire damage applied 30 after native resistance 7 to each instance. One reaction was paid and one Enfeebled 2 effect was created through the real Trigger Engine path. |
| Glimpse zero resistance result | A real 6-damage Strike became zero after resistance and still created Enfeebled 2 once. A native negated-damage result was accepted as the actual application receipt. |
| Glimpse prevent-damage branch | A real 44-damage source applied as native zero. Ally HP and shield HP stayed unchanged, and the ally's shield reaction was not consumed; the Champion's reaction was paid once. No Enfeebled effect was created. |
| Glimpse payment and decline | The original native Use merged with the unique waiting source and paid once. Applying an already completed source again was rejected before another damage application. Declining the reaction retained the reaction resource and applied the original full 44 damage. |
| External rule integrity | The Glimpse core run confirmed that the original actor and the Patreon/Trove rule configurations were unchanged by its transactions. The module uses an owned condition path and scoped compatibility behavior. |

The Prayer, Defensive Advance GM/player and Glimpse core runs recorded no page errors in their final successful runs. Missing private image assets remained baseline console errors. This statement does not apply to the separate expiry run below.

Native tests used original Use controls, actual native checks and damage, real map movement, source documents and installed module callbacks. They did not fabricate movement completions, resource payments, damage receipts or condition outcomes. Fixtures and licensed evidence remain private and are not release assets.

## Additional native checks and release smoke

**Actual enemy-turn expiry:** Both dedicated mechanical cases passed: damage before an upcoming enemy turn expired that round; damage after the enemy had acted expired next round at that enemy's real end. All initiatives were tied at 20, and one case had the GM viewing a different genuinely started encounter. However, the script's final page-error gate failed with **five EncounterTracker `_onRender` errors** (`turn` checked in an undefined value), whose stacks name Foundry core and PF2e 8.5.1. A separate two-NPC reproduction, without Glimpse damage, use, payment or owned effects, reproduced the same error when viewing one encounter and advancing another. The core tracker selects render data using the viewed encounter ID; when that entry is absent, its `turn in data` check receives undefined. This independently reproduces the UI limitation without the Glimpse workflow. The original expiry run remains failed at its page-error gate, with both mechanical assertions intact; it is not an all-green or zero-error acceptance result.

**Reaction fallback:** Eight native assertions passed. Disabling the owned Trigger graph restored the original reminder; a real NPC Strike and damage card then produced one correctly bound Reaction reminder while retaining the available reaction. Stored discovery-cache values stayed unchanged, while the scoped getter supplied the missing member. Re-enabling the graph suppressed the old reminder again. Other reminder edits survived, and unsupported world and unlinked token actors caused fallback until removed. Original party records, existing test actors, graph and builtin reminder settings, and stored caches were unchanged after cleanup; configuration audit entries were retained and no owned fixtures remained. The final run had zero page errors and five missing-image/resource console errors, with no module/backend error.

The earlier fallback runs remain recorded: a retained test actor changed and its prior Raise Shield effect appeared in an Item-not-found error; incomplete before/after snapshots prevent attributing every field change. Separately, deleting test actors before asynchronous encounter-end handlers settled produced a backend error. The final harness awaited those actual handler promises before deleting its actors. This did not hide or relabel earlier errors. It also exposed an electricity expiry no-op write issue, corrected in this release.

**Electricity no-op expiry:** Three native mechanical assertions passed. An NPC and a Champion clone without electricity state completed actual turn transitions and encounter deletion without creating module flags or empty charge operations; an actual Charged 1 effect still cleared at encounter end with its exact completed operation. The clone omitted Prayer to isolate electricity writes from Prayer's valid turn receipts. Original actors and relevant settings were restored. The script retained a failed page-error gate: three errors occurred during cleanup, two in native/Pixi floating-text destruction and one in Patreon's start-turn callback. Additional animation console errors were retained. These stacks and their timing do not alone prove a separately reproduced upstream defect; this check is not labelled entirely clean. No recurrence of the earlier module/backend `_id` error was observed.

**Final release smoke:** A fresh client after the isolated server restart reported manifest and API **0.9.5**, Foundry 14.368 and PF2e 8.5.1. The complete source and installed runtime file sets both contained exactly **106 files**, and every source, runtime-disk and fresh-HTTP SHA-256 matched. The smoke passed with zero page errors, zero module console errors and two retained missing-resource console errors. The first immediate restart attempt hit Foundry's short-lived process lock after the owned Windows process stopped; retry followed verification that both prior processes were absent, the port was free and Foundry's own lock check returned false. No lock or database was manually deleted. Publication and downloadable asset verification are recorded separately from this smoke; it does not prove production installation.

## Observed limitations

Prayer's unrelated focus-spell payment fixture could not import the test spell through either supported import route, so **native non-devotion payment was not verified**. Focused tests cover preserving the restricted temporary point while ordinary focus pays unrelated spells. Deferred or otherwise unprovable native payments remain uncertain; the provider does not infer success from nearby pool changes. Synthetic unlinked actors and arbitrary third-party action code are outside this party scope.

Defensive Advance currently uses the prepared **20-foot land Stride**. Zero-cost or cancelled movement does not automatically grant the included Strike. Other movement modes, unknown Patreon startup configuration and future AAT-enabled accounting require manual continuation. The successful cost-2 assertion concerns the original activity and this module's records, not an AAT action-pool test. Only the chosen melee Strike's own reach is required; enemy reverse reach is not a rule gate.

Glimpse requires **PF2e 8.5.1** and is limited to the current **level-5, no-Weight-of-Guilt** configuration with proven aura, relationships, actual encounter and primary native attack/save source. The native damage matrix exercised attack sources. Persistent-only workflows, Toolbelt merged multiple attacks, unknown provenance and unsupported variants retain a manual boundary. Single-source mixed damage was tested; that does not extend coverage to merged attacks. Unknown dependencies do not authorize a guessed payment or replay. Existing Patreon/Trove rules are preserved. The native fallback cases above cover graph availability and unsupported holders, not every possible dependency upgrade.

The [0.9.4 verification](verification-0.9.4.md) remains version-specific history. No older test result is relabelled as new 0.9.5 native coverage.
