# 0.9.18 verification

This release consumes the witnessed Patreon Tumble Behind effect between completed attacks in this module's combination activities, while preserving the first attack's off-guard fact for deferred native damage. It does not consume a newly refreshed effect or let a delayed damage card consume it. Independent off-guard conditions and each activity's precision-damage rules remain native.

Fortress compatibility repairs identify exact known rule shapes: Albatross Curse's critical failure uses the native one-hour Will keep-lower effect; nonlegendary Uplifting Overture upgrades failure to success; Lay on Hands includes ordinary failure against undead and uses the current unified native effect. The normal native save path did not reproduce wrong-recipient behavior; explicit saver binding is hardening. Repairs preserve custom settings, disabled groups and backups, and are scoped to the audited world and versions.

Electric Elemental Manipulation and Elemental Shield gain explicit settlement buttons on their native cards. The original player confirms the trigger, with one T target fixed before confirmation. Existing GM settlement applies native Shocked and expires it on the source's appropriate turn. Pending duplicate clicks and uncertain replies retain the original operation identity; confirmed pre-mutation rejection permits a new attempt. No movement, distance or line-of-sight monitoring is added.

## Verification

- Full native-fixture regression: 85 files, 1,758 passed, zero failures or skips.
- Local Foundry 14.368 / PF2e 8.5.1 verification: 27 result assertions passed, without browser errors in the valid runs. These cover actual Patreon rules, ordinary-player-to-GM settlement without recipient ownership, native source-turn expiry, and native Tumble Through followed by first-attack-only off-guard and deferred Sneak Attack damage.
- The local QA world enabled 24 relevant modules rather than all 78 production modules. Dailies, Trigger Engine and Skill Issue were one version older than the production snapshot. This is not a production performance benchmark.
- Card QA began from native Item.toMessage, and combination QA used the provider with an actual-use test message. Original sheet Use, all activity variants, browser-level network loss and final HP/IWR application were not revalidated in those native cases. Behavior tests cover additional branches, including uncertain RPC results.
- Final review findings concerning effect expiry, exact rule matching, cached Patreon configuration and uncertain retries were resolved before the final regression.
- Original QA actor snapshots were unchanged; created documents were removed, world time restored and the owned loopback service stopped.

On the first successful Patreon configuration repair, the GM receives a notice asking already-online clients to refresh once. Later clients load the persisted rules directly. A Setup deployment installs the module; it does not start the world or claim its runtime maintenance has already run.

See [implementation details and deferred items](settlement-followups-2026-09-25.md). Complex independent macros, generic mixed-damage merging and post-roll reaction recalculation remain outside this release. Public asset verification and production deployment are recorded separately.
