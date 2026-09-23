# 0.9.15 verification

The release combines native settlement and GM-adjudication changes with the Twin weapon damage fix. Deployment is recorded separately after validating the published artifact.

## Twin weapon damage

PF2e 8.5.1 creates the native `twin-second` circumstance modifier disabled, expecting manual selection. Double Slice and Twin Takedown previously called native damage without enabling the equivalent bonus. The missing damage occurs before Toolbelt merges the rolls.

The activity now supplies a temporary native FlatModifier only for the second attack's damage, when it used another Twin weapon of the same base type. Exact shared compendium identity is a fallback for missing base types. The preceding attack need not hit. The modifier resolves the actual weapon damage dice, receives native critical multiplication and circumstance stacking, and does not persist on the actor. No additional hook, movement listener, combat-history scan or background transaction was introduced. Ordinary attacks keep PF2e's manual Twin option. This implements the locally known sequence; it does not infer attacks earlier in the turn. [Twin rules](https://2e.aonprd.com/Traits.aspx?ID=717).

Real native QA used Foundry 14.368, PF2e 8.5.1, Toolbelt 3.56.2 and an imported Sawtooth Saber. Controlled attack outcomes used PF2e's SubstituteRoll rule; damage dice, native formulas, Toolbelt merge and resistance application were real.

| Case | Result |
| --- | --- |
| Previous implementation | Both normal hits remained `1d6+3`; Twin was disabled |
| Fixed normal pair | First `1d6+3`, second `1d6+4` |
| First attack misses | Second hit still `1d6+4` |
| Striking weapons | First `2d6+3`, second `2d6+5` |
| Second critical | The +1 participates in native critical multiplication |
| Existing +3 circumstance bonus | Twin does not stack with the higher bonus |
| Different base weapon types | No automatic Twin bonus |
| Twin Takedown | Correct second-attack bonus with native increasing MAP |

Each case retained the recorded recipient despite selecting the attacking token, produced one combined damage card, applied slashing resistance 2 once, and left no stored bonus effect. All test actors, messages and scenes were removed. This does not constitute long-duration production performance measurement.

## Regression and review

- Full native-fixture module suite: 80 files, 1,656 passed, zero failed or skipped.
- Thirteen new regressions cover second-hit ownership, first miss, critical dispatch, mismatched weapons, source identity fallback, native rule scoping, no persistent effects and duplicate activity protection. The initial implementation failed the six new eligible-sequence tests; the repair passes them.
- Independent review found no blocking issues; its requested source-fallback and injected-rule assertions were added.
- Earlier native two-client checks cover medical owner execution and fixed patients, final area-card targets, manual Voltage settlement, rank-three Force Barrage allocation/payment/cancel paths and Defensive Advance after ordinary token movement. The full 60-foot cone preview/microadjust UI and long-running production sessions remain outside those checks.

## Release contents

The module README describes the complete user-facing changes. Sight, distance and ordinary movement legality are adjudicated at the table; native resource payment, actual check results, original recipients, permissions, source identity and duplicate protection remain. Only this module is included in the deployment. The default Setup state, world data, other modules and prior core patches are preserved and checked separately against the fresh deployment baseline.
