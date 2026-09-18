# 0.9.11 movement performance verification

The automation expansion remains paused. This release changes only configuration maintenance on token motion; it does not change damage, saving throws, reaction payment, or resource accounting.

## Code and checks

Pure native motion fields (including movement history and scene level) bypass the team configuration reconciliation. Mixed or unknown changes still take the original conservative route. Empty repair lists return after other configuration maintenance, before reading or cloning Patreon rulesV3.

The final source suite passed 1,447 tests with zero failures and zero skips using the existing audited native fixtures. BoB 0.1.1 independently passed 49 tests, including environment transitions, disabling and cleanup, actor identity, unrelated module flags, and Foundry deletion forms. Independent reviews found and fixed the mixed unrelated flag refresh issue before freezing the BoB runtime.

## Native mirror

A consistent offline copy of the current Bastion world was used in a loopback-only Foundry 14.368 / PF2e 8.5.1 server. It retained all world actors/documents and the original module configuration, with BoB enabled for the comparison. The active scene contained 607 tokens, 13,425 walls and 1,230 tiles. The existing GM identity and ownership were preserved. External A/V was disabled in the mirror. Relevant assets were cached; this was not a full copy of every media file. Chrome used the NVIDIA hardware renderer at 60 FPS.

The same real pointer drag and restoration, ten token updates, and three sheet open/close operations were run before and after the two module fixes. Both pointer drags moved the token by one grid square and restored it. No page errors occurred.

| Measure | Original modules | Candidates |
| --- | ---: | ---: |
| Actor.reset calls across the route | 7,308 | 0 |
| rulesV3 reads on drag/moves | 12 | 0 |
| Long-task total | 42,336 ms | 5,361 ms |
| Mean token update promise latency (10 updates) | 546.1 ms | 115.1 ms |

This is one controlled pair, not a guarantee of an identical improvement on every client. Long-task totals cover the whole measured route, and phase labels are assigned when the observer receives records. Update promises do not capture all subsequent module work. Baseline inspection independently showed each ordinary movement scheduling a full-scene BoB actor refresh.

The measured BoB candidate had the same motion optimization as the final release; the later flags refinement retained that path and passed the full regression suite. A later mirror run with the final BoB runtime also confirmed zero actor resets and zero rulesV3 reads. It included a separate disposable wall-initialization experiment; that experiment is not part of either released module.

## Boundaries

Remaining cost includes native scene initialization and rendering. A generic wall fast path remains experimental pending compatibility and cost evaluation. Claudius's single occurrence of saving successfully and then not accepting damage remains recorded separately; this release does not claim to resolve it. Publication and actual production installation are tracked separately.
