# Native manual shared-pool static receivers

This private source adapter adds a bounded constant receiving model to the frozen native batch component. It does not install a system, grant an application, create a receipt, or register another patient.

The patcher accepts only the fixed PF2e 8.5.1 bundle SHA `d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157`. It verifies all five batch component files and reconstructs its exact `f7f80dcbbf951b840f6472abf76ada9a0a7e4f24370f296801fbb5d795a6d196` output before composing the receiver seam. Altered inputs or component files are rejected. The output directory must be new and outside this checkout; keep the complete bundle in private evidence storage, never in Git or a release ZIP.

```powershell
node tools/native-manual-pool-static-receiver/patch.mjs <fixed-pf2e.mjs> <new-private-directory>
$env:PF2E_MANUAL_POOL_BATCH_SOURCE = '<fixed-pf2e.mjs>'
node --test tools/native-manual-pool-static-receiver/seam.test.mjs tests/exploration/manual-pool-static-receiver.test.mjs
```

Tests also accept `PF2E_NATIVE_BUNDLE` when the specific variable is absent. They extract the fixed native FlatModifier constructor/beforePrepareData, Predicate, extractModifiers and stacking paths into an offline fixture. Native document/schema and button invocation boundaries use fixtures; this is not actual-world acceptance. The original empty-reception component remains a negative control.

## Model boundary

Only an original FlatModifier construct pushed by native preparation can be registered in the private WeakMap. The receiving selector must be exactly `healing-received`. Its raw and prepared value must be the same finite number, with optional finite min/max. Supported types are untyped, status, circumstance, potency and proficiency. Unknown callbacks, receiving dice/adjustments, item/ability types, formulas, injections, battle forms and damage categories remain unsupported.

The only supported raw predicates are an empty array (including an omitted predicate) and exactly `[{or:['action:battle-medicine','action:treat-wounds']}]`, in that order. Prepared predicates must match the raw source. This is a predicate-shape restriction, not a feat slug allowlist: original callback/rule/item provenance is still required. Other native predicates, including comparisons and other boolean expressions, remain unsupported for participating shared batches; ordinary unregistered applications retain their native behavior. The restriction keeps source and private receipt markers added later from changing the selected maximum.

The model copies the actual options and prepared predicate. It does not invoke construct, RuleElement.test, Modifier.test or StatisticModifier. It uses the fixed native Predicate on a deep copy, and the exact stacking implementation on fresh scalar records; native slug duplicates are retained. It preserves native healing-only critical normalization and clamp semantics. Prediction neither mutates the prepared source nor consumes RNG. A native callback is invoked only by the eventual original system application, outside the predictor.

Rule/item/callback identities, ordered arrays, raw/prepared snapshots, original methods and exact options are checked again after selection and at the captured native leaf. Normal HP changes do not invalidate a receiver model. A changed source makes the pending invocation fail; it never causes a replacement application.

## Integration contract

The combined public batch API still contains only descriptor, subscribe and currentCall. Its descriptor reports model `numeric-static-reception.v1`, `staticReceiverModelVersion: 1` and `receiverPredicateModelVersion: 1`. The last field identifies the exact empty/Godless predicate boundary. It exposes no model registration or execution endpoint. The application integration must still validate that final options preserve the original options and add only its exact allowed source/receipt markers; this adapter does not authorize arbitrary option changes.

Each select-phase candidate adds a frozen `receiver: {qualified:true, amount, flatTotal, entries}`. The authoritative gate must choose the largest amount in each genuine effect/pool; equal values use original target order. The source adapter verifies this selection. The exact private grant object and existing source/call frame semantics remain unchanged. Internal model current checks remain part of `callFrame.isCurrent`.

These summaries cannot replace authenticated source admission or the original applyDamage and master.update Promise halves. The runtime accepts this exact model version and checks the final original options before the native application. A single-patient card cannot authorize an added patient; Workbench results rolled separately remain independent effects. Full multi-patient 5/10 acceptance requires a real common-source and per-patient immunity contract.

The distributed tool includes the two pinned, module-owned test files required by its component byte check. Complete PF2e source remains private. Use the whole distributed directory when generating the patch; copying only this patcher loses its qualified dependencies.
