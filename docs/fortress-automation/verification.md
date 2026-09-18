# 0.9.8 verification

This release adds the current fortress bard's original rank-3, single-target Roaring Applause flow: one native Cast payment, the first original Toolbelt Will result, source-owned effects, actual caster-turn expiry, native Sustain with GM confirmation, and reaction restrictions. It targets Foundry 14.368 and PF2e 8.5.1. Publication and production installation are separate.

## Automated checks

The complete module suite passed **1280 tests, zero failures and zero skips** at mechanical commit `2167f81ddd5c3faa4d593f8fd61b3ac7cabfcb61`. Installed Foundry, PF2e, Counteract, Workbench and Reaction sources were supplied to tests requiring them. All 188 recorded source/test/runtime hashes were reverified before release-only documentation and manifest edits.

Coverage includes native invocation/payment binding; first-save authentication and both hook/row arrival orders; source expiry, native parent/grant identity and uncertain writes; manual deletion and independent stronger conditions; original-card Sustain; readonly reaction queries and actual commit gates; and the exact Reaction Checker bundle/callback adapter. Independent reviews covered the provider, effect/save adapters, consumers, original-card wiring and compatibility layer. Focused regressions prove confirmed parent deletion clears before a manual/GM barrier, and an older asynchronous write cannot consume a newer GM-continuity barrier.

## Native spell and Sustain flows

The isolated profile enabled 23 relevant modules, including Patreon 3.2.28, Toolbelt 3.56.2, Workbench 7.7.5, Dice So Nice 6.3.1 and Reaction Checker 1.4.3. Fixtures were owned clones of current actors. This is not a rerun of all 121 production modules.

The accepted base run at core commit `0eb624a8` used the player's actual sheet Cast and original Toolbelt save buttons. Cancel-before-payment and four native adjusted Will outcomes passed. Each paid Cast consumed one rank-3 slot and retained its exact original spell card. Critical success created no effect; success established the source restriction; failure granted Slowed 1; critical failure additionally granted Fascinated. Existing independent Slowed 2 retained its value and ownership.

Actual caster-turn progression ended the source at the correct next turn-end. A genuine reroll moved the source to manual review without another payment. Manual removal of its own Fascinated child did not cause recreation. Native Sustain Use followed by actual GM completion or disruption passed, with no additional spell cost. The provider does not infer undisrupted completion from the native Use return.

The base run had zero page and cleanup errors. All original actors, scenes, combats, messages and settings were preserved after precise fixture cleanup and restoration of only observed native clock advances. Native core.time modification audit metadata was retained. Earlier Toolbelt sparse-default and native GrantItem alteration-default failures remain in evidence; both were narrowly corrected and verified against actual installed schemas.

## Native reaction restrictions

The integrated run at `2167f81d` checked all 122 source, installed and fresh HTTP files, then exercised two real Roaring success sources on a current champion clone.

| Case | Observed result |
| --- | --- |
| Existing original Reaction reminder, GM and player | Trusted clicks were cancelled while the source was active. The exact original card, raw resources, expense records, frequency and HP remained unchanged. |
| Eight native resource getters | General reaction and seven special keys were masked without modifying their raw fields or creating missing counters. The fixture does not claim positive spending of every special reaction. |
| Actual source expiry | Getter values returned to current raw values. The same original reminder survived, its native click succeeded, and subsequent native Shield Block reused one expense. |
| Direct Shield Block | Native damage application was refused before a new payment or HP/shield change. |
| Original-card prepaid Shield Block | Refusal preserved the original paid card and entry; it neither paid again nor refunded that earlier expense. |

This run had zero page and cleanup errors. The original document baselines were preserved except native clock audit metadata; observed time 282→312 was restored to 282 after actual cleanup callback completion.

Two further fresh-session native scheduling cases passed. These were controlled delays of genuine results, not forged outcomes or real network outages:

- Shield Block awaited its first actual audited resource-bundle fetch. The harness retained the real unread Response, completed a real player Roaring Cast/save while it waited, then released that identical Response. The final payment gate rejected the native damage application; no reaction, HP, shield, claim or expense was changed.
- Glimpse traversed the genuine player Use and GM Resist choices. The harness delegated six real source-authentication RPCs, held only the sixth successful response after the resource snapshot, completed real Roaring, then released the identical response. The final reserve boundary refused payment and damage application. The ally's HP, raw reaction, claims and expense records remained unchanged. An additional genuine Reactive Shield targeting reminder was retained and checked by its exact observed document and source identity.

Both scheduling runs had zero page and cleanup errors, restored their own wrappers, and preserved original documents/settings including unchanged world time. Each retained eight known missing-token-image console errors and one PF chevron-down image 404. The earlier base and integrated runs also retained missing-image and aborted Dice So Nice audio records. These are not zero console/network-error claims. Earlier harness failures—role order, inactive encounter, partial teardown, version spelling, hook argument shape and additional native reminder—remain recorded separately; none were relabeled as passing native cases.

## Scope and manual boundaries

Only the current eligible original rank-3 single-target Cast is admitted. The player confirms that the target can see, hear or otherwise understand the caster. Unsupported geometry, private or unverified variants, missing source/payment proof and ambiguous target identity require manual handling. This does not declare those game options illegal.

Only the first authenticated original Toolbelt save settles automatically. Reroll revisions remain manual in this delivery. Reload, GM/ownership changes, changed rows or uncertain effect writes require renewed proof; persisted raw flags alone do not justify automatic enforcement. Confirmed source termination or confirmed deletion of its own parent clears that source without restoring spent reaction resources. Current caster death does not itself invent an earlier spell expiry.

Native Sustain still needs the GM's actual completion/disruption fact on the original card. The once-per-target-turn clap is a prompt; no automatic action-pool payment, reactive attack, undisrupted clap receipt or unsupported animation is claimed. Existing module/provider behavior remains responsible for other effects.

The reaction query is readonly. Supported consumers check it before new payment/reservation, including prepaid and post-await paths, while already-paid bookkeeping and cleanup remain available. Reaction Checker compatibility requires the audited 1.4.3 bundle and live original callback. Unknown dependencies, external manual macros and previously selected external modifiers remain outside this claim. The original card displays confirmed, manual and compatibility states separately.

Counter Performance, Spiritual Scar and other current-party gaps remain subsequent work. The deferred serrated-blade/Double Slice report remains record-only. Historical evidence is preserved in [0.9.7](verification-0.9.7.md), [0.9.6](verification-0.9.6.md), [0.9.5](verification-0.9.5.md) and [0.9.4](verification-0.9.4.md).

## Release and installation

Release metadata, final startup smoke, downloaded asset verification and production installation receive separate evidence. At this preparation stage production retains the separately verified 0.9.7 module. The user-selected saved default remains Setup (`options.world=null`); an actively launched fortress world is compatible with that default. A coordinated fresh maintenance window is required before any production restart while players are online.
