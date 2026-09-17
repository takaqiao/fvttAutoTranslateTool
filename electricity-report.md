# Electricity lifecycle provider — 2026-09-18

Owned files are the new `eldamon-electricity*.mjs` source/tests and this report. No production/browser/server operations were performed by this agent. The private book, current actor evidence, current rulings and native PF2e 8.5.1 entrypoints informed the implementation; private source text is not copied into tests.

## Integration contract

`scripts/eldamon-electricity-provider.mjs` exports `createEldamonElectricityProvider({game,fromUuid,onError,selectChoice,refreshOutsideEncounter})` and `preserveElectricityOnAlter(original,result)`.

- `onCommittedChannel({receipt,message,user})`: authenticated original committed use; the active GM retains the original receipt author even when that player disconnected. Grants the original Charged source once for the four reviewed gain/discharge active powers, caps at 3, preserves GrantItem and Resistant Shell. Siphon/discharge branches never gain. Metapower alone pays discharge.
- `beforeChannel({item,selection,kind})`, `validateSelection({actor,item,selection,kind,user})`: source-bound Reactive Chain choice and active-GM revalidation. Evidence includes exact application/receipt/source-effect/target/caster-token IDs. Both 30-foot distances, enemy/Shocked status, original target exclusions, current turn and generic reaction availability must still hold. Normal discharge relaxation requires adjacency to the caster; Siphon never relaxes it. No additional reaction payment.
- `interceptDamageMessage(roll,data,options,native)` and `preserveElectricityOnAlter`: stamps original damage publication and native alterations with durable source identity. Same committed channel yields the same effect key across all targets.
- `beforeDamage` / `afterDamage`: enters the existing application pipeline; adds a unique application nonce, records original source and target, and accepts only one current native `damage-taken` receipt with matching author/token/item/nonce. Never replaces native application/IWR.
- `observeNativeIWR(actor,params,iwr,options,context,handled)`: main calls this after awaiting the existing shield adapter. Captures actual post-IWR, post-hardness actor damage; a handled/suppressed application is zero. A synchronous `preCreateChatMessage` hook attaches that exact fact to its receipt. Overkill, temp HP and stamina do not collapse to HP loss. No `pf2e.damageRoll.total` or isolated HP update establishes damage taken.
- `interceptCheck(native,check,context,...args)`: recognizes electricity base weapon damage, native Shock/Greater Shock runes, and the reviewed source-bound Static Shock/Electric Shot branches. Applies only the additional native AC circumstance penalty needed for off-guard and the per-check target option; electricity trait alone and Siphon do not qualify. Existing published Fort/Ref rules stay unchanged.
- `onRefresh({actor,nonce})`: outside-encounter charge removal. The voltage writer invokes it once after successful normal Refresh. Siphoned High Voltage never invokes it. Encounter ending calls voltage's existing `refreshOutsideEncounter` writer; no second frequency writer.
- `register({Hooks,socket})`, `maintain(actor)`: original native checks queue Anvil/Static Shock consequences until their matching damage application finishes. Expiry tracks the source combatant and phase. Recovery verifies a unique receipt and resumes unapplied lifecycle mutations without applying damage again.
- Explicit GM mixed attribution: provider `confirmMixed({actorUuid,nonce,receiptUuid,amount,confirmed:true})`; socket method `electricity:confirmMixed`. This requires active GM, original unchanged mixed source/receipt, and an amount between zero and the actual total. UI button is on the receipt. An explicit completed-Interact API is `interact({actorUuid,nonce,confirmed:true})`.

## Validation and native QA selectors

`node --test modules/pf2e-third-party-automation/tests/eldamon-electricity*.test.mjs`: 25 tests pass. The fixtures validate actual exported behavior and source/state transitions, including RED/GREEN for missing providers, source-turn expiry, mixed attribution, copied/wrong-author receipts, same-effect target exclusion, and interrupted cleanup. Additional RED/GREEN regressions cover unrelated viewed encounters, outside/ambiguous membership and offline original-user delivery; native reaction availability receives the actual encounter in a bounded facade. They are not claims of live Foundry validation.

Full module suite at this checkpoint: 207 tests, 196 passed, 11 skipped, zero failed (`node --test modules/pf2e-third-party-automation/tests/*.test.mjs`).

Durable state: `actor.flags['pf2e-third-party-automation'].electricity`. New receipt UI: `[data-electricity-receipt]`; source flag `message.flags['pf2e-third-party-automation'].electricitySource`; exact IWR fact `message.flags['pf2e-third-party-automation'].electricityApplied`; native nonce roll option starts `pf2e-third-party-automation:electricity-apply:`. Actor damage records store source-message fingerprints and receipt fingerprints; pending/ambiguous transactions retain their evidence rather than replaying damage.

Suggested native acceptance: normal gain vs Siphon vs discharge; Charged 3; shell + own charge; native Anvil failed Fort then damage with differing source/target initiatives; Static failure/critical failure; immunity-zero receipt; pure-electricity overkill/temp HP; mixed physical+electricity (including electricity immunity); two targets of one area effect; chain target in/out of each distance; source-turn expiry; normal/outside Refresh and encounter-end cleanup. Select target(s) before the original Use entry so the immutable use binds them.

## Deliberate boundaries

- Mixed damage cannot derive an electricity share from the combined native total. It stays unproven until the GM attributes the exact share on that receipt. No split applications or invented component IWR.
- Preexisting untagged damage cards do not receive fabricated source identity. Generate a fresh native damage card.
- Arbitrary custom attack damage synthetics are not inferred from an electricity trait; automatic off-guard is limited to the recognized exact damage sources above. Other attacks require the native GM adjudication.
- The original source target set is conservatively excluded while a multi-target effect may still be applying. A target originally included but actually immune is not automatically offered as a chain destination.
- Anvil/Static status requires an authenticated source-bound native check and application. Outside-encounter timed consequences need GM duration adjudication; no invented initiative clock.
- Interrupted destructive Charged updates whose original parent disappeared without a completed ledger marker stop for GM reconciliation. The provider does not guess or replay the payment.
- This provider observes actual completed uses/application events; it cannot prove a narrative touch, an unobserved action, or an arbitrary macro's extra damage component.
