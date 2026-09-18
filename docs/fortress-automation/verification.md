# 0.9.9 verification

This release adds the current fortress Spiritual Scar damage reaction. The original owner chooses and pays through the original action; the elected GM binds one native damage application, reaction expense and daily use. Original resistance rules, native IWR, a non-basic Will against the actual class DC and a source-local Slowed grant complete the supported flow. Foundry 14.368 / PF2e 8.5.1 are the tested versions. Publication and production installation are separate.

## Automated checks and provenance

The complete suite passed **1436 tests, zero failures and zero skips** at mechanical commit '4e6bdcdcc45d73ee8fe5132a189049c217178e8b'. All 200 recursive script/test inputs were recorded and reverified before release-only edits. Installed Foundry, PF2e, Counteract, Workbench and Reaction sources were supplied to source-dependent tests. No independent agent review was available in this batch; no such review is claimed.

Checks include authenticated source and original private invocation; daily Use and original-card accounting; one reaction/Reaction Checker reservation; refund only before a provably unattempted Use; preservation of intervening turn refresh; exact native resistance preparation and cleanup; IWR contribution attribution; private native check entry; durable no-repeat save/effect receipts; source-local Slowed, undo, reload and creator-turn expiry. Saved flags alone do not authorize replay of uncertain payment or dice.

The current production inventory has 125 enabled modules. Native acceptance used 23 relevant enabled modules, including Patreon 3.2.28, Toolbelt 3.56.2, Workbench 7.7.5, Dice So Nice 6.3.1 and Reaction Checker 1.4.3. This is not a full125-module compatibility claim. Current source review includes the newer inventory and coverage changes; source review alone does not replace native runtime acceptance.

## Full native damage acceptance

Owned clones of the latest character card and native fiend fixtures ran with real GM and player clients. Each run verified the entire 129-file candidate on source disk, installed disk and fresh HTTP. Provider, payment, original Use, damage resistance and IWR attribution were not replaced with fixture results.

| Case | Observed result |
| --- | --- |
| GM declines | Original damage applies; daily use and reaction remain available. |
| GM and player accept | Exactly one original daily use and one generic/Reaction Checker expense; one native resistance application; no persistent toggle or resistance. |
| Fully prevented spirit damage | One original native Will against class DC 21, then source-local Slowed 1 on failure. |
| Spirit damage exceeds resistance | Remaining HP loss applies; no followup Will or Slowed. |
| Spirit plus fire | Spirit resistance does not prevent the remaining fire damage; no followup Will or Slowed. |
| Actual Toolbelt damage buttons, GM and player | Original source-to-target options, original Use and the complete provider finish through real UI controls. |
| Blind GM-only and private GM/player sources | Original Use and final Will retain the exact blind/whisper scope. |

Public API, private UI and partial/mixed runs passed at 'dda53a22'. The only subsequent runtime change before the final commit passed source privacy into native check middleware before rolling. Private UI was then rerun at the final mechanical commit, with all three GM-decline, GM-use and player-use cases passing.

## Native private check and duration

A separate native followup probe used a current bard clone with its original available Halfling Luck, a real Will modifier fixture, an actual class DC and an existing independent Slowed 3. The blind Will completed without a result-dependent Luck prompt, retained Luck's daily use, created one native in-memory Slowed 1 grant and kept Slowed 3 unchanged. The installed main lifecycle removed only its own grant on the actual creator's next turn start. Exact native context 'messageMode: blind' and final GM-only recipients were checked.

This targeted probe used controlled followup authorization and a controlled damage receipt. It verifies native privacy, condition preparation and lifecycle; full provider authority is evidenced separately by the integrated damage runs above.

The duration follows the general numbered-round rule in [Player Core p426](https://2e.aonprd.com/Rules.aspx?ID=2378). Applying creator-turn start to this ability's one-round duration is an explicit rules inference; it does not invent a fiend-turn-end clause. Changed initiative, reordered/skipped turns or rollback use a finite duration from the original start and a visible warning rather than extending the effect indefinitely.

All five accepted integrated/final probes preserved original actors, scenes, combats, messages, settings and world time after fixture cleanup. Each had zero page and cleanup errors and eight known missing-token-image console errors. No zero-console-error claim is made.

Earlier evidence is retained: an initial API harness omitted native source-to-origin option mapping; another asserted the obsolete card user field rather than Core 14 author and did not await a fixture floaty. Earlier followup fixtures had an incomplete damage receipt and an unsupported substitution assumption. Finally, an old blind-save probe allowed a Luck prompt: that privacy behavior was diagnosed, covered by a failing regression, fixed before the die and superseded by the passing final private probe. None of those earlier failures is relabeled as passing final coverage.

## Scope and manual boundaries

The actual original action, current daily charge, available general reaction, one actual encounter and authenticated original native damage are required. A prepared original rule supplies the resistance value; the module does not leave a permanent actor toggle or substitute its own damage formula. Confirmed consumption and original Use are merged into one expense. A manual unbound Use does not retrospectively attach itself to an old damage result.

No arbitrary range, enemy relation or attack-only rule was added to the source resolver. However, native end-to-end evidence here covers ordinary Strikes. Synthetic actor and non-attack source paths are unit-covered but have no complete native UI acceptance in this batch. Merged attacks, ambiguous same-label or competing resistance contributions, missing source proof, unsupported compatibility, active AAT and outside-encounter cases remain manual. These are automation limits, not declarations of game-rule illegality.

Only an attributable full prevention triggers the followup. Uncertain GM handoff, payment, save or effect creation remains recorded and cannot automatically repeat. The final native save degree is respected; immunity and independent conditions remain native. Original damage undo/removal cleans only this source effect and does not guess a resource refund. Existing module behavior remains responsible for other abilities.

Counter Performance, Primary Threat, Reflexive Cover, Goblin Scuttle, Fane Escape, Elemental Shield, Terrifying Resistance and other actual remaining gaps are subsequent work. The old serrated-blade/Double Slice report is record-only. Historical evidence is preserved in [0.9.8](verification-0.9.8.md), [0.9.7](verification-0.9.7.md), [0.9.6](verification-0.9.6.md), [0.9.5](verification-0.9.5.md) and [0.9.4](verification-0.9.4.md).

## Release and installation

At preparation, the last independently verified production installation was 0.9.7. Production has not been modified by this work. The user-selected default remains Setup ('options.world=null'). Final candidate startup, package/download hashes and any coordinated production installation have separate receipts; publication does not itself imply installation.
