# 0.9.4 verification

Candidate: PF2e Third Party Automation 0.9.4, Foundry 14.368, PF2e 8.5.1, Node 24.19.0. This batch covers actual-use gating for Overwhelming Combination, the fortress Shield Block / Disarming Block chain, Familiar Focus and Accompanist. It does not complete every current fortress ability. Production installation is tracked separately from publication.

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

Result at implementation commit `bc234e81`: **358 tests, 358 passed, 0 failed, 0 skipped**. Native dependency tests deliberately skip without those private paths; that reduced run is not equivalent. The release changes only version metadata and documentation after this implementation commit. Scoped independent reviews and regressions preceded the native checks.

## Native checks

Checks ran in a fresh, loopback-only world using the latest captured four characters and familiar. The profile enables 23 relevant modules, including current Trigger Engine 1.35.0, Trove 2.3.5, Patreon 3.2.28, Reaction 1.4.3, Toolbelt 3.56.2 and Workbench 7.7.5. This is not a new full-profile compatibility matrix; the older 122-module smoke belongs to 0.9.3. No production document, setting, module or service was changed during these checks.

| Area | Observed behavior |
| --- | --- |
| Overwhelming Combination | Sending its original display card caused no choices, attacks or use. Real player sheet Use produced two native attacks with the selected order and MAP. Native combined damage of 75 applied 70 against slashing resistance 5, applying resistance once. |
| Hotbar | Dragging the original action to the hotbar and using it started one activity, produced two attacks at MAP 1/2, and respected fist-first selection. |
| Direct Shield Block | Native damage 9 against hardness 8 removed 1 character HP and 1 shield HP, used one Reaction resource, and offered one free Disarm. While viewing a different encounter, the event and MAP 1 still belonged to the actor/token's actual encounter. |
| Native prepayment | The original Shield Block card was bound to one budget entry. The real Reaction reminder Yes button also paid once; subsequent native damage application reused that payment without consuming another reaction. |
| Already-spent reaction | After original Glimpse of Redemption use, another Shield Block was rejected before native HP or shield changes. The existing payment stayed intact. |
| Familiar Focus | A display card changed nothing. Original familiar Use at a full master pool stopped before daily payment. Player Use at 1/2 focus increased it to 2/2, consumed the original daily use, and saved one transfer receipt. Reload preserved that single transfer. |
| Accompanist | A native Performance check received the confirmed +1 circumstance modifier. A separate check with existing +2 circumstance and +1 status kept those modifiers and disabled the smaller Accompanist modifier. Declining on the following check added nothing. |

The master-rank Accompanist +2 branch, full-pool races and ownership/nonce failure boundaries have automated coverage; they were not each repeated in the browser. Native tests retained the original sheet buttons, native damage and checks, source documents and actual module callbacks. Fixtures and licensed evidence remain private and are not release assets.

After the metadata update and an owned QA-server restart, both manifest and runtime API reported 0.9.4. All **93 allowlisted runtime files** matched the source, installed copy and fresh HTTP responses by SHA-256. This final smoke captured no page error or module startup error; four baseline private-asset errors remained. It did not repeat the earlier mechanical matrix.

## Observed limitations

The successful shield run also recorded two EncounterTracker render errors and one effect-item deletion error during encounter switching/cleanup. Their stacks did not contain this module, but responsibility was not independently isolated. They did not invalidate the recorded mechanical assertions, and this report does not claim an error-free browser. Missing private image assets also caused baseline errors. Familiar Focus's full-pool rejection produced its expected validation message.

Reaction synchronization is restricted to the verified Reaction 1.4.3 source and supported settings. Unknown active versions/source, uninitialized resources and ambiguous actor/token encounters stop the automated block before applying damage and require manual resolution. The existing real-Shield-Wall upstream error remains outside this batch; only the audited empty-candidate compatibility path is covered.

Glimpse of Redemption, Roaring Applause, Counter Performance and other audited current abilities remain in subsequent work. Existing module coverage is retained when calculating those gaps. The reported serrated-blade / Double Slice +1 issue is recorded only and was neither investigated nor fixed here.
