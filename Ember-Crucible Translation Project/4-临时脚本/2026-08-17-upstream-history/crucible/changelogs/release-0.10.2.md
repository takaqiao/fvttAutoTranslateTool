![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.10.2 - Version 14 Alpha
Hi friends, I've been away for the last couple weeks on a bit of a "vacation" (of sorts), but I'm back with a new small Crucible release that precedes an Ember release that will be dropping tomorrow.

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @XinSysVTT
- @roth-michael
- @Streyder
- @Lioheart

## Highlights
- Added spell effects visualization for the **Death Rune** with corresponding particle artwork and sound effects.
- Redesign the way that **Flanking** works to be computed from the attacker's point of view and respect perception. Different attackers benefit from the presence of enemies and allies according to whether they can see them.
- A new **Overrun** condition lets a GM or an effect impose a stage of Flanking directly, opening up some interesting gameplay possibilities for talents and features to apply that status effect. The prior "Flanked" active effect is gone.
- A number of helpful **bug fixes**.

## New Features

### Combat Mechanics
- Redesigned the way that engagement and flanking are computed, we compute engagement on token movement (as before), but now wait to compute flanking stage until an attack is actually being made, so that every attack can benefit from different circumstances that affect how flanking is computed. #1424
- Added the **Overrun** condition, which imposes one additional stage of Flanking against every attacker regardless of engagement. #1424
- Revised the rules text for Engagement and Flanking to describe the per-attacker model, the role of perception, and the new Overrun condition. #1424
- Rendered hazard enrichers in a more terse format, reserving the defense, damage, and resource detail for the expanded tooltip. #1385

### Interface
- Added support for Death rune particles with the touch, influence, ray, fan, arrow, and blast gestures, along with accompanying audio effects. #1411
- Enhanced skill tooltips to display the computed formula annotated with the value each term contributes. #1135
- Updated Polish localization strings.

## API Changes
- Exposed the action event-stream resolver internally so that action rewrites, for example Interpose, resolve their resource deltas through the same simulation as ordinary actions.
- Made Action message confirmation safe against repeated confirmation attempts.
- Called `receiveAttack` for misses and dodges in addition to glances and hits, so that talents such as Swashbuckler can respond to any attack outcome. #1152
- Improve the approach to active effect updates which remove or replace managed Region or Token documents.
- Standardized the way a dice roll is rebuilt during the Action Use Dialog, preserving manually applied boons and banes across a rebuild.

## Content Updates
- Allowed the Berserker signature to spend Focus while Enraged on any action which does not have the `spell` tag, via a general mechanism for per-action focus blockers and non-blockers. #1394
- Automated the **Eye of the Storm** talent by recording the original combat turn from which a delay was performed. #1395
- Added skill progression to the Occultist archetype.
- Removed playtest notes from the Counterspell descriptions now that its automation has been implemented.
- Changed the damage type of Noxious Spray from electricity to poison, matching its description.

## Bug Fixes
- Fixed character creation failing on completion, and errors on world startup, for any actor holding an item with a Rune Spellcraft hook. #1423
- Fixed the Physical Defense dodge bonus failing to increase when an affix was added before the enchantment tier was modified, matching the equivalent fix previously made for weapon attack bonuses. #1396
- Fixed a skill check which allowed a choice between multiple skills rolling the wrong skill, by adopting the check returned from the roll dialog after it may have been reconfigured. #1398
- Fixed the featured equipment display showing a maximum of two natural weapons even where there was room to show three. #1388
- Fixed currency zeroing out when the character sheet is popped out into a separate window. #1379
- Fixed Alchemist's Fire applying a Burning effect whose description referred to the condition rather than to the originating action. #1310
- Fixed attack rolls displaying "DC" rather than the targeted defense name for players, where only the numeric difficulty should have been withheld. #1402
- Fixed the Private Biography section rendering for users holding only limited permission on an actor. #1406
- Fixed manually granted knowledge areas being treated as absent by passive knowledge tooltips, and allowed Adversaries to hold knowledge areas. #1412
- Fixed action tag effects such as Empowered, Deadly, Keen, and Weakened applying twice when an action was re-prepared, for example after planning movement. #1404
- Fixed Interpose failing to deduct any damage from the interposing creature. Multi-strike attacks redirected through Interpose now resolve against a single simulation of the interposer rather than one per strike.
- Fixed a token with no configured vision perceiving nothing at all, which prevented it from ever flanking. Such a token now receives the detection modes it would otherwise have been assigned, while remaining unable to perceive invisible creatures. #1424
- Recomputed action usage flags such as `isMelee` and `isRanged` when changing weapons in the Action Use Dialog, so that a re-prepared action no longer retains an assertion which no longer holds.
- Fixed the flanking visualization persisting after being toggled off, and failing to update as tokens moved. #1311
- Fixed Backstab being usable against a target flanked only by other creatures while the attacker stood alone. #1424
- Fixed an error when Pack Hunter was applied to a ranged attack against a flanked target. #1424