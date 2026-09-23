# 0.9.16 verification

This release shares activity-local weapon, target and native outcome facts across the module's combination attacks. Spell Swipe now supplies Forceful damage and Backswing attack eligibility from its own preceding Strike. Double Slice and Twin Takedown retain the verified Twin bonus through the same helper. Overwhelming Combination keeps weapon and fist identities separate.

Only completed native checks enter an invocation's short-lived history. Damage frames retain their own preceding facts even when all attacks finish before damage is rolled. MAP is not used as weapon attack count. Activity-specific MAP, precision-once rules, original recipients, spell payment and Toolbelt merging remain in their providers. Native typed stacking, weapon dice, striking and critical multiplication remain authoritative.

There are no new hooks, timers, movement checks, chat queries or persistent actor history. Earlier-turn conditions remain manual. This does not automatically enroll combination macros owned by other modules. [Spell Swipe](https://2e.aonprd.com/Feats.aspx?ID=9069), [Twin](https://2e.aonprd.com/Traits.aspx?ID=717), [Forceful](https://2e.aonprd.com/Traits.aspx?ID=611), [Backswing](https://2e.aonprd.com/Traits.aspx?ID=545).

## Regression and review

- Full native-fixture module suite: 81 files, 1,676 passed, zero failed or skipped.
- Twenty new regressions cover first miss/critical failure, deferred damage, same-weapon identity, Token identity, canceled or unknown outcomes, repeated record attempts, separate activity isolation, frozen weapon facts and alternate usage.
- Before implementation, the new provider tests exposed four failures for missing Backswing and Forceful. The final focused helper/provider suite passes 103 tests.
- Independent code review found no actionable findings and separately reran the focused suite.

## Native Foundry validation

Foundry 14.368, PF2e 8.5.1, Toolbelt 3.56.2. Native compendium weapons and a prepared cantrip were used. Deterministic native SubstituteRoll rules control attack outcomes; native damage dice, modifiers, merge and IWR are real.

| Case | Verified behavior |
| --- | --- |
| Twin normal pair / first miss | Only the second qualifying weapon damage gets +1 |
| Twin striking / critical | Prepared weapon dice and native critical multiplication apply |
| Twin higher circumstance / mixed weapon type | No stacking with a larger bonus; no false pairing |
| Twin Takedown | Increasing MAP and second-weapon Twin remain correct |
| Scimitar Spell Swipe | First damage `1d6+3`, second `1d6+4`; Sweep applies to both attacks |
| Swipe first miss | Second Forceful damage still receives its bonus |
| Striking critical second Swipe | Native formula `4d6 + 2 * 5 slashing` |
| Swipe higher circumstance bonus | Existing +4 wins; Forceful is not added on top |
| Greatclub Swipe first miss / first hit | Backswing applies to the second attack only after the miss |
| Sweep plus Backswing | Both conditions are recognized, but native circumstance stacking gives only +1 |

All fourteen cases retained original recipients despite selecting the attacker, verified actual native damage application, and left ordinary subsequent damage without a stored automatic bonus. Test actors, messages and scenes were removed; the owned QA server was stopped. Raw local reports and file hashes are recorded under `output/activity-sequence-20260923` and `tmp/activity-sequence-20260923` in the Desktop workspace.

This is rule and entry verification. Full 60-foot cone placement/microadjust UI acceptance and long-duration fortress performance measurement remain separate work. Publication and production deployment are recorded only after checking the public assets and served files.
