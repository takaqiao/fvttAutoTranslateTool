# 0.9.3 verification

Candidate: PF2e Third Party Automation 0.9.3, Foundry 14.368, PF2e 8.5.1, Node 24.19.0. Native checks used isolated loopback worlds and private copies of the required licensed dependencies. No production actor, world, system or service was changed. Licensed source text, actors, browser evidence and QA database files are not release assets.

## Automated checks

Run from the repository root in PowerShell, supplying local licensed dependency locations:

```powershell
$env:FVTT_NATIVE_APP = '<Foundry installation>/resources/app'
$env:PF2E_NATIVE_BUNDLE = '<PF2e installation>/pf2e.mjs'
$env:FVTT_COUNTERACT_MAIN = '<Counteract installation>/scripts/main.js'
$env:FVTT_WORKBENCH_MACRO = '<private source export>/Treat Wounds and Battle Medicine.mjs'
node --test modules/pf2e-third-party-automation/tests/*.test.mjs
```

Final candidate result: **231 tests, 231 passed, 0 failed, 0 skipped**. Without the local dependency paths, the corresponding native tests deliberately skip; such a run is not the complete result. Tests cover source identity, same-roll conversion, immutable channel policy, actual-use ordering, owner/GM recovery, condition handling, healing continuation, Refresh, voltage windows, authenticated electricity receipts and real-encounter reaction accounting. Independent scoped reviews were followed by fixes and regression tests. Native findings added regressions for PF2e's actual check/damage link handlers, Foundry's dotted flag expansion and persisted deletion semantics.

## Observed native behavior

The minimal profile enabled 21 relevant mechanical modules, including Workbench 7.7.5, Counteract 0.3.0 and the verified PF2e native IWR bridge.

| Area | Observed result |
| --- | --- |
| Source repair | Existing power attack potency and resistance restored; known Surge discharge dice corrected; traits restored; preparation/custom rules preserved; second repair made no changes. |
| Widen | Original sheet Use, then original Surge Use, transformed the selected 30-foot line to 40 feet. Actual native Region placement produced a 40-foot line. |
| Siphon | Original Use and native damage produced one untyped roll. Matching-element and ordinary targets received full and half damage respectively, before native IWR; no Charged gain. |
| Multiple users | Player-origin Widen/Surge completed under the active GM with original-author receipts. A second GM took over after the first disconnected; subsequent activation succeeded. |
| High Voltage | Normal original Use immediately refreshed only prepared powers. Siphoned Use did not refresh. Original-card touch confirmation led to native Reflex and damage; duplicate damage application was rejected; the window expired at the source's next turn start. |
| Treat Condition | Native Medicine modifiers and critical success removed the selected Clumsy condition. The source DC remained private. Counteract's generic dialog did not replace the supplied DC. |
| Doctor's Visitation | Native TokenDocument movement and original-card continuation reached Workbench Battle Medicine. Changing current selected/targeted tokens preserved the original healer and patient. Continuation awaited both native check and Workbench result. |
| Electricity damage | Native fully resisted damage preserved Charged. Positive actual damage absorbed by temporary HP still removed it. These checks used native damage-taken receipts, not HP-delta estimates. |
| Anvil | Actual original-card Fortitude clicks retained DC21 and the channel marker. Failed saves followed by native damage applied Shocked; it expired at the source's next turn end. |
| Reactive Chain | A real 13-point hit against resistance5 produced an authenticated 8-point receipt. Original prepared Use spent its use/reaction and generated a native 4-point damage link. A native critical-failure Reflex save resulted in 8 applied damage. |
| Static Shock | Original Use, actual native power attack and native damage applied Shocked; the source-based duration expired correctly. |
| Persisted history | After seeding 66 synthetic ordinary completed entries on a disposable actor, real Widen Use retained 64 ordinary entries plus its activation. A full browser reload confirmed the two pruned entries stayed deleted and sibling flags were preserved. |

Workbench's existing Apply Healing/Immunity buttons were present and retained; this test did not click those buttons or claim automatic HP/immunity application. Visitation movement used the native TokenDocument API rather than a pointer-drag test. High Voltage's confirmed-touch path was exercised natively; its automatic melee-hit observer has source/unit coverage but was not separately exercised in the browser. Every discharge permutation and Static Shock's fixed-failure branch were not separately exercised in the browser; they have automated coverage. HUD loaded in the full profile but was not operated separately.

## Full installed-profile smoke

A separate profile enabled **122 modules**. All 91 allowlisted product files matched the candidate on disk and over HTTP; manifest and runtime API both reported 0.9.3. Real Toolbelt-enabled sheet Use armed Widen and committed Surge. After the asynchronous card renderer completed, both native area links displayed the transformed 40-foot distance. No error from this module was captured.

The full snapshot lacks some media/presets. Existing TokenRing, Dice So Nice and Hero Deck resource errors remain, alongside baseline errors from other modules. This is a bounded sheet/Use/card compatibility check; it does not establish that every installed module or the full canvas is error-free. Actual Region and damage tests belong to the separate minimal profile. The full-profile test restored its captured actor/settings state and stopped only its owned loopback server.

## Scope and remaining boundaries

The adopted table policy retains discharge damage, range, area and save-degree benefits under Siphon, while removing added conditions and Refresh. Reactive Chain's discharge target relaxation remains excluded under Siphon. Movement-dependent powers remain unaffected by Siphon. These policies describe the user's chosen interpretation, not a unique official RAW ruling.

Seven reviewed electricity source profiles support metapower conversion. Event completion targets the five currently prepared powers; unprepared Retributive Shock's complete status workflow and arbitrary future powers are not promised. Mixed electricity attribution requires GM confirmation on the original receipt. Unknown custom macro action entrances, ambiguous source/encounter identity and unobservable narrative events remain explicit manual boundaries. The native IWR bridge is required for actual-electricity lifecycle and chain receipts; a module-only installation does not install that bridge.

Release publication does not install or enable this candidate in the live campaigns. Cotct has no player actors yet; its remaining work is world integration followed by actor-specific coverage after import. The broader five-world research backlog is separate from this bounded Claudius implementation.
