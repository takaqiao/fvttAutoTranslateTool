# Eldamon metapower implementation and release

Date: 2026-09-18. Authorized scope: implement Siphoning Element and Widen Element in the existing module, test, commit, push, and publish a formal release. Five-world research is context, not a request to implement the other backlog. Production enablement and migration are separate from publication.

## Baseline and constraints

Product baseline is the 76 hash-verified files from `tmp/team-automation-foundry14-20260917/source` (0.9.2). Worktree starts at origin/main `61e39e754df35cffb4cbdedfac45695f2e22d494`; only `modules/pf2e-third-party-automation` and this documentation belong to the release. No paid compendium text, actors, private fixtures, license keys, or unrelated shared changes may be published.

Use exact compendium UUIDs, including old actors with missing traits; never infer identity from translated names. Siphoning action is actions.Item.4w72ljp4eLBqeZB2; Widen feat is feats.Item.3YasBiZw3N96rdUW; Disruptive Siphon is feats.Item.kG0HSsDc6eHYjTU9, all under battlezoo-eldamon-pf2e.

Siphoning makes all eligible direct damage untyped, excludes persistent damage and all non-damage added effects, and halves total damage (including modifiers). Disruptive Siphon substitutes full damage for each target with a matching associated elemental trait. Original save/attack outcomes still apply. Do not halve dice counts or apply electricity resistance before conversion. Powers whose damage depends on removed additional effects retain their original behavior because Siphoning has no effect on them.

Widen changes actual native Region geometry for instantaneous burst/cone/line: burst >=10 +5, smaller unchanged; cone/line <=15 +5, larger +10. Work from the selected legal base variant, once. Spell-only area RE cannot modify feat inline templates.

Either metapower applies to the immediately following channel only. Any intervening action, free action, reaction, or end of the actor's turn expires it. A second metapower replaces the first. A qualifying reaction power can itself be that next channel. UI previews/cancellations are not actions. Committed channel card substeps use their immutable snapshot and cannot claim a later activation. Require actual use, actor ownership, active-GM serialization, durable nonce/card receipts and recoverable failure states. Include explicit cancellation for actions declared outside VTT, which software cannot observe.

Current scope exercises all seven known electricity powers (five prepared) without bypassing preparation or reaction prerequisites. Precisely sourced profiles describe supplemental damage/effect behavior; unsupported profiles must not silently receive a guessed conversion or be advertised as fully automated. Full two-feature release requires resolving the rules and source coverage boundaries, not only changing labels.

## Design

Native PF2e keeps damage modifiers/dialog, dice evaluation, save/critical arithmetic, IWR and Region placement. A metapower provider adds source profiles, actual-use lifecycle, immutable channel snapshots, and targeted effect suppression. No second plugin is created. Do not duplicate Patreon/Trigger/Workbench behavior; verify their installed source and affected paths.

Prefer a source/card option passed through native inline damage. At the native Roll-to-message boundary convert the same Roll object (and preserve evaluated dice), so message, return value and pf2e.damageRoll subscribers agree. Keep a shared untyped base roll when Disruptive Siphon can make different targets full/half; apply the one per-target coefficient before native IWR and show that distinction on the card. Preserve original power traits. Validate native object conversion rather than relying on textual assertions.

Activation, next-action ordering and channel finalization share the actor GM queue. Hook awaitable real-use entry points, not asynchronous chat hooks alone. Existing base action costs remain native. Ordinary description cards, drafts and reposts never activate. Snapshot transformations are idempotent on rerender and remain associated with the original channel.

Original-source cross-check on 2026-09-18 resolved Widen's cost: the Pathfinder 2e book, printed page 60 (PDF page 61), has a one-action glyph, and the Pathbuilder distribution's Widen feat has action: 1. Foundry's passive/missing cost is a data omission. Widen must use one action rather than an unresolved GM policy.

The book's Charged definition (printed page 97) classifies discharge benefits as additional effects. Read together with Siphoning (printed page 57), remove discharge's non-damage benefits, including extra area/range, save downgrade, and Reactive Chain's relaxed target eligibility; retain legal damage improvements and their discharge cost. This is a direct reading of the combined rules, not a separately published author clarification. Normal level-based range/area growth remains. Avoid automatically spending a charge for a branch whose only benefit has been removed.

High Voltage remains unresolved: its future touch/hit damage may or may not fall under Siphoning's dependent-effect exclusion. Normal High Voltage refreshes immediately; if Siphoning is allowed to alter it, the treatment of its included Refresh activity still needs clarification. Do not infer that delayed damage alone establishes the exclusion, or silently decide the Refresh interaction. Relevant private source research is outside the release worktree.

## Execution

- [x] Isolated worktree, authoritative baseline and 37 relevant baseline tests.
- [ ] Task 1: Domain source profiles, native damage transformation and geometry policy; meaningful RED/GREEN tests and scoped review.
- [ ] Task 2: Provider, entry/lifecycle integration, snapshot UI and side-effect integration; race/cancellation/ownership tests and scoped review.
- [ ] Task 3: Fresh local Foundry 14.368/PF2e 8.5.1 QA, normal use/real rolls/actual Regions/multiple clients and installed-module interaction checks.
- [ ] Task 4: Whole-diff review, fixes, version/artifact validation, bounded commits, push and formal GitHub release.

## Validation and delivery

Record exact commands, observed test counts and native evidence. Preserve frozen 0.9.2 evidence. Native QA uses a new loopback runtime owned by this task and copies private dependencies read-only. No production actor edits, restarts or world enablement are part of these tests.

Release uses the existing repository with a module-specific tag and manifest/ZIP. ZIP contains only runtime files; public tests use synthetic data. Report version, commit, release link, validated boundary and whether CN has actually been installed/enabled.
