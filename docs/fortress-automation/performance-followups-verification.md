# 2026-09-30 performance follow-ups

Scope: Task 3 of `docs/superpowers/plans/2026-09-30-automation-native-followups.md`, using the production 0.9.18.6 baseline. No remote changes or live movement/distance/vision checks were performed.

## Changes

- Added `createDirtyMaintenance`: one microtask burst becomes one maintenance pass; changes arriving during an awaited pass produce one fresh pass. Actor requests merge, a global request covers them, and disposal or active GM loss cancels queued work. Real action queues are unchanged.
- Spiritual Scar recovery scans once, then indexes exact effect documents and their original damage-message IDs. Unrelated chat performs zero inventory scans. Receipt settlement resolves its exact scene/token directly. Actor imports, rebuilt synthetic documents, lifecycle removal, stale awaited documents and registration teardown are covered.
- Party and AV world/combat maintenance uses the coalescer. Cosmetic combat changes skip it. Party effect creation schedules its affected actor. Party no longer subscribes to Token movement; Guardian retains shield/source/feat checks without geometry. AV ally selection and Mental Balm omit range/vision/area-fact confirmation gates; ally choices use the requesting user's public target name.
- Glimpse configuration skips HP, focus, cosmetic actor fields and item frequency/description fields. Token coverage refreshes only for actor rebinding and actor delta imports.

## Evidence

`performance-followups-red.txt`: 33 tests, 21 pass, 12 fail before implementation. New failures identified repeated maintenance, unrelated scar-chat rescans and unrelated reminder refreshes.

`performance-scope-red.txt`: 11 tests, 0 pass, 11 fail before Party/AV implementation, including absence of movement monitoring and recorded-ally settlement without canvas geometry.

`performance-scope-merge-red.txt`: actor-scope merge failed before its implementation. `performance-dispose-red.txt`: queued receipt settlement failed the teardown contract before the generation guard.

Final command:

```powershell
node --test modules/pf2e-third-party-automation/tests/maintenance-scheduling.test.mjs modules/pf2e-third-party-automation/tests/maintenance-load.test.mjs modules/pf2e-third-party-automation/tests/spiritual-scar-expiry.test.mjs modules/pf2e-third-party-automation/tests/glimpse-configuration-events.test.mjs
```

`performance-followups-green.txt`: **53 tests, 53 pass, 0 fail, 0 skip**, exit 0. Assertions exercise the real registered provider hooks: an eight-event maintenance burst reads inventory once; requests during a blocked write read twice; ordinary scar chat reads inventory zero times after recovery; an eight-event scar encounter burst settles once. These are operation counts, not browser timing measurements.

Whole-suite baseline reconciliation and live Foundry validation remain with the integrating agent. Profile normal chat, state updates and real owner actions after deployment; verify one original native action/payment, one consequence and correct source-bound expiry. No manual movement, distance, vision or touch-fact validation is required.
