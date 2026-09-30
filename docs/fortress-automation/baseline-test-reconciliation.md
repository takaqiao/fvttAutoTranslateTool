# Production baseline test reconciliation

The captured 0.9.18.6 runtime baseline had 67 failures in the older canonical suite. The changes below update tests to current production contracts; no runtime version/source locks were restored.

- `eldamon-voltage.test.mjs`: implement the native `Actor.updateEmbeddedDocuments('Item', updates)` fixture API; assert one batch for the original Refresh and exact per-item receipt recovery after a partially committed batch. The reused spent power is never refilled by recovery.
- `defensive-advance-compat`, `fortress-rule-compat`, `glimpse-compat`, `glimpse-reaction-cache`, `reaction-budget-shield`, `roaring-reaction-compat`, `roaring-save-evidence`, `shield-world-gates`, `startup-load`: replace obsolete third-party semver/hash rejection with functional installed interfaces and verified business results. Negative active-GM, owner, exact source, native template, graph override, module identity, native handler/descriptor, malformed resource and unsupported system/world checks remain.
- `metapower-hud.test.mjs`: mark the installed mock modules active, preserve one original Toolbelt use/await contract, accept functional changed controller methods, and reject unavailable native use descriptors.

Focused verification: `baseline-refresh-compat-green.txt` records 70 tests / 69 pass / 1 existing optional integration skip; `baseline-interface-green.txt` records 51 / 50 / 1 existing optional integration skip; `baseline-adapters-green.txt` records 106 / 104 / 2 existing optional integration skips. All three commands exited 0.

Final verification of all 11 owned test files after integration set the real `PF2E_NATIVE_BUNDLE`: `baseline-reconciled-final.txt` records **227 tests / 224 pass / 0 fail / 3 optional integration skips**, exit 0. Those skips require separate Foundry reaction bridge and Workbench macro fixtures, rather than the PF2e source bundle.

A fresh whole-suite run set `PF2E_NATIVE_BUNDLE` to the captured real PF2e 8.5.1 bundle and used `node --test modules/pf2e-third-party-automation/tests/*.test.mjs`. `baseline-full-native-verification.txt` records **1832 tests, 1781 pass, 4 fail, 47 skip**, exit 1. All original 67 baseline failures were eliminated. The four remaining failures belong to work then in progress: three old Eldamon Basic confirmation-card assertions and the new Metapower no-extra-confirmation regression. This was an intermediate integration run, not a completion claim. Remaining environment-dependent integration checks and final whole-suite evidence are owned by the integrating agent.
