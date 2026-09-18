# 0.9.13 verification

The optional Glimpse of Redemption provider treated an unsupported initial damage source as a reason to reject ordinary native damage whenever the recipient was within a champion's aura. Risky Surgery's inline `1d8[slashing]` card has no attack-item origin, so the provider threw `not-native-attack-damage` before asking for or paying any reaction.

The initial unsupported-source path now returns the original parameters unchanged and reports that no automatic reaction was used. It does not create a reaction scope, spend a resource, or add resistance or a condition. Diagnostic failures cannot veto damage. A private cycle source key still blocks an existing live scope or any non-refunded claim for the same message, roll index, and recipient. Source changes after entering the reaction workflow, payment, native execution, and uncertain receipts retain the previous strict checks. The resolver's proof requirements are unchanged.

## Verification

- Full suite with the native Foundry/PF2e/module fixtures: **1,481 passed, zero failed or skipped**.
- New regressions cover inline and macro surgery, missing source, throwing/rejecting diagnostics, existing paid/native/followup/done/uncertain records, refunded or other-target records, and a source changed while a choice is pending.
- Independent review and provider/source run: **71 passed, zero skipped**.
- In the local fortress mirror, the actual historical Toolbelt target-damage button reproduced the 0.9.12 error and left HP unchanged at 36. After loading the new provider, the same button applied the existing 8-point roll once: HP **36 → 28**, one HP update, one native receipt, and Toolbelt marked the target applied.
- Champion reaction/combat flags, recipient flags, and item IDs remained unchanged. The patched application produced no page errors. HP, the original message's target-helper data, and the temporary receipt were restored after QA.

The browser test used Foundry 14.368, PF2e 8.5.1, Toolbelt 3.56.2, and the final provider SHA256 `a04ca1fdb10a1f5d14e149705fa390af09a725fdb2cb6a35486f8f7989c9d88e`. The mirror's package metadata still identified 0.9.12 during this deliberate before/after source reload; the tested provider bytes match the 0.9.13 release candidate.

The 130-file runtime package changes only README.md, module.json, and scripts/glimpse-of-redemption.mjs relative to deployed 0.9.12. Sequencer is handled in another task; the deferred Wayfinder candidate is excluded. Publication and production installation are separately recorded.
