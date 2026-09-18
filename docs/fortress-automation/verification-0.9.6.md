# 0.9.6 verification

Candidate: PF2e Third Party Automation 0.9.6, Foundry 14.368 and PF2e 8.5.1. This batch adds Halfling Luck for verified public native skill checks and saving throws in the fortress world. The current Patreon configuration makes skills private, and Toolbelt target-row saves remain a manual Luck boundary. Publication and production installation are separate.

## Automated checks

The complete module suite passed **705 tests, zero failures and zero skips** at mechanical commit `49aeeb3b`. It uses the actual installed Foundry/PF2e/Counteract/Workbench/Reaction dependencies where those tests require them. The dependency environment and command are documented in the [0.9.5 verification](verification-0.9.5.md).

Tests cover exact source and native result eligibility, ordinary versus occupied fortune/misfortune, adjusted failure, public and unpublished result contracts, contextual actor routing, the player's real local choice adapter, original native frequency payment, ownership and GM changes, concurrency, stale/recharged payments, duplicate callbacks, and interrupted delivery. These checks do not claim every interruption was reproduced in a browser.

The implementation keeps one existing Check wrapper. Native unpublished drafts are returned to their original callback without creating a separate ChatMessage. The Halfling provider records an exact original invocation and requires the original Use, same-update daily payment marker, observed frequency receipt and matching source card before a second die. A newer payment or a reloaded client without its live witness cannot resume an old operation. Evaluated results and completed callbacks may be recorded if the actor subsequently cannot act; beginning another action still requires the appropriate live checks.

## Native checks

The isolated, loopback-only world enabled **23 relevant modules**, including Patreon 3.2.28, Toolbelt 3.56.2, Workbench 7.7.5 and Dice So Nice 6.3.1. It used tagged clones of the current character, the original feat and native statistics. All **109 runtime file hashes** matched source, installed files and fresh HTTP responses for mechanical candidate `49aeeb3b`, whose API and manifest still reported 0.9.5 during these checks. Final 0.9.6 metadata integrity and startup verification are recorded separately below. This is not a rerun of the complete production module profile.

| Case | Observed result |
| --- | --- |
| Player public Will save | Real native failure, local player choice, original feat Use, one daily 1→0 update, one reroll and one final check card/callback. Original Use, payment receipt, ledger and final result agreed on their exact nonces, source, author and outcome. |
| Depleted frequency | A further failed native check completed normally without another Use or payment. |
| Decline | The original result was retained; daily frequency stayed 1. |
| Unpublished native save | The original caller received one native draft callback. No check message, automatic Luck offer or payment was added. This directly verifies the Check contract, rather than a Toolbelt target-row persistence claim. |
| Explicit blind save | The result remained blind and no Luck offer or payment occurred. |
| Patreon skill privacy | A requested-public Performance check became blind under the existing Patreon policy. Luck respected its actual privacy without changing that setting. |
| GM public Will save | The active GM's own roll completed the same actual Use, payment, native reroll and final callback path. |

The final seven-case run passed with **zero page errors and zero cleanup errors**. All preexisting actor documents and the active scene were unchanged after cleanup. Eight console errors referred only to missing private Pride/Huncheng token images. Actor flags, including reaction state, did not change through either paid Luck use.

The actual player die changed from 3 to 16 and the GM die from 14 to 20. Both confirm delivery of the new native result. Keeping a *worse* result was verified by focused tests, not by those two browser rolls. Public skill admission was checked with the full native Check function and actual native classes in an offline probe; the world policy prevented a public skill browser positive case.

Dice So Nice adds `term.options.type` and `result.indexThrow` to persisted dice for display/throw grouping. Native comparisons excluded only those two observed display annotations while preserving all mechanical fields. The recorded ledger result and callback roll matched after JSON serialization. The original harness failure caused by comparing those decorations remains in private evidence; no product relaxation was made for it. The earlier Performance run also remains recorded as a failed public-dialog expectation, correctly explained by the actual Patreon policy rather than a fee or prompt defect.

## Toolbelt original target rows

An additional native probe used original Befuddle source cards and the actual Toolbelt target-row save button. Both a public player row and a private GM row passed. The latter selected GM privacy in the real native check dialog. Each original card persisted the exact native callback roll and outcome on its intended target row; the private row stayed private. There were no Luck dialogs, feat updates, daily payments or separate check cards. Both clients observed actual row persistence, rather than treating Toolbelt's detached callback return as proof.

That final probe had zero page and cleanup errors. Preexisting actors, messages and world settings matched their saved hashes. Missing private token images remained console errors. Earlier harness attempts are retained: duplicate chat/notification renderings and an offscreen control prevented clicks; one pre-roll attempt recorded Core notification `hidden`-on-null errors. A later Ctrl-click attempt saved a public row because the installed helper returned legacy `rollMode` while PF2e 8.5.1 expects `messageMode`. The passing private case used the actual native dialog; this release does not claim to repair that existing shortcut mismatch.

## Boundaries and release status

Toolbelt's original `createMessage:false` save flow does not carry its complete target-row privacy into Check. This release does not treat a caller-supplied boolean or a recent chat card as proof. Those Luck decisions stay manual. The direct draft and original target-row checks above verify preserved native behavior, not automatic Luck on those rows.

No old-card reroll, automatic uncertain-operation retry, unknown fortune combination, unlinked synthetic actor, offline GM or competing Clock/Squawk holder is claimed as newly covered. A returned callback does not by itself prove that an external module's detached persistence queue completed.

The [0.9.5 verification](verification-0.9.5.md) and [0.9.4 verification](verification-0.9.4.md) remain historical records. Their native successes and recorded UI/cleanup limitations are not relabelled as new 0.9.6 testing. This batch does not implement Force Barrage's original Cast bridge, Roaring Applause, Counter Performance or Spiritual Scar, and does not investigate the recorded old-build serrated blade/Double Slice issue.

The final 0.9.6 candidate started successfully after an owned QA server restart. Its manifest and API both reported **0.9.6**, and all **109 source, installed and fresh-HTTP SHA-256 hashes matched**. Fresh startup smoke recorded zero page errors, zero module console errors and two missing-image baseline console errors. The restart waited 6.6 seconds for Foundry's own lock check before starting the replacement; no lock/database was deleted.

The ZIP was independently enumerated locally against the complete 109-file runtime manifest; every entry and the separately supplied manifest matched the frozen source hashes. Downloaded-release verification remains a separate publication receipt.

Production installation is still separate and pending the actual startup-world preference: the latest read-only observation has the fortress running while `options.json` selects CotCT for the next restart. No production setting, actor, service or module was changed by this batch.
