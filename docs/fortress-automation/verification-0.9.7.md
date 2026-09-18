# 0.9.7 verification

This candidate adds the original Force Barrage Cast bridge for the current fortress character's rank 1–3 occult spontaneous signature spell. It reuses the verified Workbench 7.7.5 damage construction, with one native payment before target damage delivery. It does not add another damage formula or apply hit point changes automatically. Foundry 14.368 and PF2e 8.5.1 were tested; publication and production installation are separate.

## Automated checks

The complete module suite passed **885 tests, zero failures and zero skips** at mechanical commit `54a1de0d`. The installed Foundry, PF2e, Counteract, Workbench and Reaction sources were supplied to tests that require them. Tests cover exact invocation enrollment without adding an actor-wide spell matcher, private native variants, payment cancellation and uncertain replies, same-update slot witnesses, original-card binding, target allocation, source/owner/GM changes, and refusal to replay uncertain work.

Two native findings have focused regression coverage. Foundry's later `updateItem` events contain only changed marker fields, so each payment requires its new nonce and exact changed fields together with the full options marker, full committed marker, actual slot update and returned document. Dice So Nice adds `dsnRole` and `dsnRoleManaged` to a native `Die`; comparisons omit only those two confirmed Die presentation fields in addition to the previously verified display annotations. Roll type, damage-instance flavor, formula, totals, dice and results remain checked, and the complete original roll JSON is retained.

## Native normal flows

The owned loopback QA world enabled 23 relevant modules, including Patreon 3.2.28, Toolbelt 3.56.2, Workbench 7.7.5 and Dice So Nice 6.3.1. It used linked clones of the current character and targets. This is not a rerun of all 121 production modules.

The player's real sheet Cast button opened the action and allocation dialogs. The GM observed native slot writes, original spell cards, real target DamageRolls and persistent delivery receipts. All 113 runtime HTTP hashes matched the frozen mechanical candidate, whose manifest still reported 0.9.6 during this development run.

| Case | Observed result |
| --- | --- |
| Cancel allocation | No slot change, payment receipt, spell card or damage card. |
| Rank 3, three actions, allocation 4/2 | One slot payment, one original card and one combined damage card per target. Each target's resistance applied once through PF2e's original damage consumer. |
| Rank 2, two actions, allocation 2/0 | One payment and one target damage card; the zero allocation produced none. |
| Rank 2 again, one action, allocation 0/1 | A new payment nonce and one further slot payment, despite unchanged rank and marker fields. The second target alone received a damage card. |
| Rank 1, one action | One missile against force resistance 6 produced zero actual damage through native IWR. |

The final five-case run passed with zero page errors and zero cleanup errors. Console output retained ten missing-private-image errors and one generic HTTP 404 without a recorded URL. Preexisting actors, messages, world settings and the active scene matched their baseline after cleanup. The bridge's exact original card disabled its separate generic damage button and pointed to the allocated cards; unrelated spell cards were not changed.

Earlier failed attempts remain in private evidence: DialogV2 converted a null cancel callback into its button action; DSN added live Die role metadata; and a second slot update omitted unchanged receipt fields. Each issue was reproduced, fixed narrowly and rechecked. Earlier fixture hover/shape errors are also retained; the final five-case run had none.

## Native failure checks

Two additional native cases passed on the same frozen mechanical candidate. An exact QA-only `preUpdateItem` veto rejected the native slot write: slots stayed unchanged, no original or damage card was created, and the attempt was recorded uncertain without a retry. The second case used a scoped libWrapper wrapper that first awaited the real second target ChatMessage and then threw a simulated lost reply. The slot decreased once, the original card and both damage cards remained, the first target stayed published and the second stayed publishing under an uncertain bridge. The saved native rolls matched both real cards, and no fee, roll or card was repeated. This was controlled fault injection after native creation, not an actual network outage.

Both injected hooks/wrappers were restored, and all preexisting actors, messages, settings and the active scene matched their baselines. The final run had zero page and cleanup errors. Its console retained ten missing-image errors, one generic HTTP 404 and the two expected injected-failure reports. A prior harness attempt could not install its second injection because an existing method descriptor was protected; that failed attempt was preserved, and the passing run used the supported libWrapper registration/unregistration API.

## Scope and manual boundaries

Only an eligible original Cast starts this bridge. The independent old Workbench macro remains available and is not globally rewritten. Only the audited internal macro UUID, version and full command hash are admitted; its original damage construction is retained. Unknown versions or changed macro contents require manual handling.

The bridge requires public message mode, the current spell/entry/rank, a unique linked source token, a supported same-elevation scene, valid target range and sight, and the caster's explicit visibility confirmation. Hidden/invisible targets, unsupported geometry, private casts, consumables, other spell sources, overlays and unverified damage bonuses remain outside the automatic path. This does not declare those game options illegal.

The player chooses 1–3 actions and a nonnegative integer allocation before payment. Existing native cast policies can still disrupt the spell. Automatic action-pool consumption and Sequencer/JB2A animations are not newly implemented. Damage cards retain their exact native source and Toolbelt target; applying damage remains an explicit normal operation.

An uncertain payment or delivery remains recorded and is not automatically retried, refunded or resumed after reload. An already created target card is not recreated to compensate for a missing reply. GM/ownership changes stop subsequent work when their next required proof cannot be established.

The previous [0.9.6 verification](verification-0.9.6.md), [0.9.5 verification](verification-0.9.5.md) and [0.9.4 verification](verification-0.9.4.md) remain historical evidence. This batch does not implement Roaring Applause, Counter Performance, Spiritual Scar or the deferred serrated blade/Double Slice report.

## Release and installation

The final 0.9.7 candidate passed an owned QA restart. Manifest and API both reported 0.9.7, and all **113 source, installed and fresh-HTTP file hashes matched**. Startup smoke had zero page and module-console errors, with two existing missing-image errors. The restart waited 8.4 seconds for Foundry's native lock check; no lock or database was removed.

The release ZIP was independently enumerated locally against that full runtime manifest; every entry and the separately supplied module manifest matched. ZIP SHA-256: `59942efed9156197860a9ab5b37092ec6674f12ecaa8a39ba20b0f46752bc91c`. Manifest SHA-256: `731dffae1b158ad871aab43d82dcc442dfa22f6501fe465f38a25c4b26bdbdaa`. Downloaded-release verification and production installation are separate receipts.

At release preparation on 2026-09-18, production had the separately verified 0.9.6 release and the user-selected Setup default. Its exact native administrator login gate was verified without logging in or launching a world. The first deployment checker had omitted this normal authentication branch; its original failure journal was retained alongside a successful independent supplemental installation verification, without another restart.
