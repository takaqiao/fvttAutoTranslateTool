# Magus Spellstrike investigation — 2026-09-25

The reported character is 伊格内修斯, level 15, in 赤凰斗士 (`pnvfcgjbf2cjp7gz`). The user identified Blazing Bolt as the action-dependent spell, clarified that the force component stayed at 3, and recalled a two-Strike activity and possibly Spell Swipe.

## Confirmed fix: distinguish spell action variants

The actor's Blazing Bolt has two unnamed native override variants. One uses one action; the other uses two or three actions. Both previously appeared in the Spellstrike picker as `灼热火矢 Blazing Bolt`. Their underlying damage and heightening differ.

The picker now includes each variant's native action time and uses native variant ordering. It still passes the selected overlay and cast rank to PF2e. Damage arithmetic, payment and targeting are unchanged. No hooks, monitors, movement checks or persistent state were added.

The actor's stored spell rank is 3 and it is a signature spell. Actual native Foundry casts verified:

| Selected slot | One-action version | Two-action version |
| --- | --- | --- |
| 3 | 3d6 fire | 6d6 fire |
| 7 | 7d6 fire | 14d6 fire |

Each full Spellstrike used the selected rank, consumed one corresponding slot, retained the original target even while the GM targeted the attacker, and retained that number of fire dice after Toolbelt merging. Canceling the variant picker spent no slot, did not discharge Spellstrike, and produced no attack or spell card.

This confirms and fixes an ambiguous selection UI. The original failed damage card was no longer present in the captured world, so it does not prove which variant the player selected that day. No independent cast-rank arithmetic bug was demonstrated.

## Confirmed fix: preserve the individual attack's off-guard context

The actor has Laughing Shadow, Greater Weapon Specialization and Cutting Heaven, Crushing Earth. The latter grants off-guard for a specific next attack. Its temporary effect applies to attack checks and is consumed after that check. Spellstrike, Spell Swipe and Overwhelming Combination calculate their damage later. PF2e's damage context does not automatically reuse the saved attack's roll options.

This was reproduced with the actual character copy and native campaign callback: Overwhelming Combination's second attack correctly saw off-guard, but its damage still used 3 force; Spell Swipe likewise lost a target-specific, one-attack off-guard condition before damage. Final damage-card flags could still mention off-guard even though the damage calculation had not received it.

The shared weapon-damage path now freezes that single fact from each native attack, verifies that its target Token UUID matches, and supplies it to native damage. It is captured before post-attack effects are consumed. Native rules still decide eligibility, damage type, amount and critical doubling. No persistent condition, unconditional +4, or unrelated attack roll options are copied.

| Native conditions in the test | Verified force damage |
| --- | --- |
| Ordinary Spellstrike: no off-guard, one free hand | 3 |
| Ordinary Spellstrike: persistent off-guard, one free hand | 7; 14 on a critical hit |
| Persistent off-guard, both hands occupied | 3 |
| Combination: only the second attack gains Cutting Heaven off-guard | 3 + 7; previously 3 + 3 |
| Same Combination with fist first | 3 + 7 |
| Same Combination, second attack critical | 3 + 14 |
| Combination against persistent off-guard | 7 + 7 |
| Swipe: only the first target has persistent off-guard | 7 / 3 |
| Swipe: only the second target has persistent off-guard | 3 / 7 |
| Swipe: both targets have persistent off-guard | 7 / 7 |
| Swipe: first target has only Cutting Heaven's one-attack off-guard | 7 / 3; previously 3 / 3 |

Ordinary persistent off-guard worked before this fix. The reproduced defect is specifically the loss of attack-time context during deferred settlement. The original failed card was no longer in the world snapshot; the reproduction matches the clarified symptom and the character's actual feat combination.

## Verification and scope

- The variant regression first failed because the choices had identical labels. It checks that a rank-7 prepared slot keeps its slot index and selected native overlay through payment and damage.
- Off-guard regressions failed for all three activities before the damage fix. The final focused suite passed 109 tests across `spell-combination.test.mjs` and `activity-attack-sequence.test.mjs`; tests cover independent per-attack facts, consumed effects, unrelated options and mismatched/missing target contexts.
- Full native-fixture module suite: 81 files, 1,682 passed, zero failed or skipped.
- Native activity verification uses Foundry 14.368, PF2e 8.5.1 and Toolbelt 3.56.3, with automation 0.9.16 plus these candidate fixes. All 17 cases passed with zero browser errors: nine baseline/spell-selection cases and eight additional combination/off-guard cases check actual native rolls and merged components.
- A preliminary native run with Toolbelt 3.56.2 also retained the expected force damage and heightening.
- Isolated local QA only. Production actor/world data were copied read-only while the main service was in Setup. No production rolls, actor updates, configuration changes or module deployment occurred.
- Independent review found no actionable issues in either fix. The local QA copy exposes the existing campaign provider solely to reuse the exact registered callback and avoid creating a duplicate provider during tests.
- Local evidence is under `C:/Users/Taka/Desktop/fvtt/tmp/magus-spellstrike-20260925`: pre-fix reproduction `magus-combo-1790267505846.json`; final verification `magus-combo-1790267860026.json` and `magus-verify-1790267957179.json`. Source snapshots and raw actor records remain local and are not included in this repository.

Released and deployed as [0.9.17](deployment-0.9.17.md) after public-package and production-file verification.
