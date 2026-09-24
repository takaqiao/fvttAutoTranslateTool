# 0.9.17 verification

This release fixes deferred weapon damage in Spellstrike, Spell Swipe and Overwhelming Combination when a target's attack-only off-guard effect has already been consumed. Each native attack freezes only its own, UUID-matched off-guard fact before post-attack effects run. Native damage rules still determine eligibility, amount, damage type and critical multiplication. No unconditional bonus, additional hooks, movement checks, chat scans or persistent history are added.

The spell picker now labels native action variants and follows their native ordering. Blazing Bolt's formerly identical option names now distinguish one action from two or three actions. The selected native overlay, cast rank, spell-slot payment and original recipients remain authoritative. The original report did not establish a separate heightening arithmetic defect.

## Verification

- Full native-fixture module suite: 81 files, 1,682 passed, zero failed or skipped.
- Six new regressions first exposed ambiguous variant names and missing per-attack off-guard options; they also cover wrong or missing target context. The final focused suite passed 109 tests.
- Independent implementation review found no actionable issues.
- Seventeen native Foundry 14.368 / PF2e 8.5.1 / Toolbelt 3.56.3 cases passed with zero browser errors. They check baseline force damage, occupied hands, critical multiplication, target-specific and consumed conditions, both combination orders, rank-3 and rank-7 action variants, original recipients, exactly-once spell payment and cancellation.
- Native examples: temporary second-attack off-guard produces force damage 3 + 7, or 3 + 14 on a second critical hit; first-target-only Spell Swipe produces 7 / 3. Rank-7 Blazing Bolt produces 7d6 with one action and 14d6 with two actions.
- The QA actors and scenes were removed and the owned QA service was stopped. Original actor snapshots and raw reports remain local.

See [the investigation](magus-spellstrike-2026-09-25.md) for reproduction details and scope. Public-package verification and the production deployment receipt are recorded separately after release.
