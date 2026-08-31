![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.10.0 - Version 14 Alpha
Happy weekend everyone. This is a huge release moving Crucible forward into version `0.10.x`. The headline of this patch is a sweeping expansion of the signature talent system - every pairing of ability scores now has a full menu of high-impact options to build toward, and the talent tree has been reorganized around them. Alongside that, this release redesigns how critical hit effects flow through the action lifecycle, a meaningful action economy improvement for 2h heavy weapon users, improvements to the Active Effect system, deeper rules for Grapple, Escape, and movement tactics, more spellcraft animations, and way more fixes, balance corrections, and quality-of-life improvements throughout. Thank you all for the continued playtesting and feedback!

## Highlights
- **Signature talent expansion.** Every pairing of ability scores now offers at least three signature talents to choose from, closing all remaining gaps in the signature matrix. This release adds a wave of new signatures including Wizard, Sage, Spiritbreaker, Primalist, Juggernaut, Duelist, Acrobat, Bastion, Telekinetic, Peltast, Gambit, Warchanter, Warden, and Champion, each accompanied by supporting passives and upgrade talents. Signature talents are now paced on a clean three-level rhythm with a cap on how many can be taken per character level, making high-impact build decisions feel more deliberate.
- **Talent tree reorganization** relocating and rebalancing many existing talents to make room for the new signatures and improve the flow of build progression.
- **Critical hit pipeline.** Critical hit effects are now identified earlier, during the action usage phase rather than at confirmation, enabling richer on-critical behavior. This is paired with the new `keen` action tag for improved critical thresholds, the Vicious weapon affix (bleeding or staggered on a critical hit), and an expansion of Runic on-critical talents to apply to non-spell actions that channel a Rune.
- **Improved Active Effect system.** A new Crucible-specific Active Effect sheet presents effects in a consistent visual style with new system fields, including a removal DC that drives tests to end an effect. Actions can now snapshot deleted effects so reversal can restore them, and a new Scale Resource change mode supports percentage scaling of resource pools.
- **Expanded Grapple mechanics** that track mutual effect origins, end the grapple on forced movement, and prevent a grappled target from being forcibly relocated.
- **Improved token collision model** that compares Movement Strength against Blocker Strength, giving real tactical texture to the question of who can push past whom.
- **Expanded Spellcraft animations** adding the Influence and Touch gestures for the Flame, Frost, and Life runes, building on the animation framework introduced in 0.9.5.
- **New movement options** including a Climb action that can cling to an adjacent surface without falling, a contextual Escape action when restrained, and automated gliding for characters equipped with a Glider.
- **Rune: Lightning becomes Rune: Storm**, reflecting a broader elemental command over wind, storm, tempest, and lightning, alongside new elemental scaling conversion talents for Earth, Frost, and Storm.
- **Two-Handed Gameplay Improvements**. Every two-handed weapon automatically has access to one spell hand (two for a two-handed talisman) and the strike action cost for two-handed heavy weapons is reduced from 4 to 3, dramatically improving gameplay.

A full accounting of all the additions and fixes in this patch is presented below, along with thanks to the community contributors who helped make it happen.

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @danielrab
- @Lioheart
- @salamander-git

## New Features
### Signature Talents and Talent Tree
- Implemented a full slate of new signature talents so that every ability score pairing now offers at least three options to choose from. #1224
- Simplified signature talent acquisition to a steady three-level cadence and added a pacing restriction on the number of signatures allowed per character level. #1246
- Added three new Wisdom + Intellect signatures: Sage, Wizard, and Spiritbreaker. #1187
- Added the Duelist signature along with a Disarming Riposte upgrade talent designed to double down on duelist gameplay loops. #1227
- Added the Primalist signature, which channels elemental runes into Stances with added on-critical effects to weapon strikes. #1206
- Added the Juggernaut signature alongside the corresponding Grapple improvements. #1229
- Added the Champion signature plus the Sentinel and Harrier support passives.
- Added the Acrobat signature.
- Added the Bastion signature, built on the new Movement Strength vs. Blocker Strength comparison. #1241
- Added the Telekinetic signature.
- Added the Peltast signature.
- Added the Gambit signature.
- Added the Warchanter signature.
- Added the Warden signature.
- Fixed the signature purchase ineligibility badge in the talent tree HUD.

### Combat and Critical Effects
- Redesigned and relocated the applyCriticalEffects workflow out of the confirm phase and into the action usage phase so critical hit effects can be identified earlier in the action lifecycle. #1212
- Added a `keen` action tag that can improve the critical success threshold for a specific action. #1226
- Expanded the behavior of Runic on-critical talents to apply to non-spell actions which specifically incorporate usage of that Rune. #1207
- Added elemental scaling conversion talents for Earth, Frost, and Storm to shift their ability score modifier. #1211
- Improved the Champion mechanics using an active effect that increments on turn end.
- Significantly improved Grapple mechanics: mutual effect origins are tracked, the grapple ends on forced movement, and a grappled target can no longer be forcibly moved. #1229
- Added a generally available Escape action which is contextually available when a character is restrained by a temporary effect that allows escape. #1052
- Added a custom Active Effect change mode for "Scale Resource" used in the final application phase to enact percentage scaling of a resource pool. #1225

### Active Effects
- Added a Crucible-specific Active Effect sheet which presents AEs with a consistent Crucible visual style and adds system-specific fields. #1233
- Extended the Active Effect system model to track a removal DC that can be used for tests to achieve the effect's removal. #643
- Record a snapshot of active effects being deleted in the CrucibleAction#events pipeline so that an action which deletes an effect can restore it via reversal. #1228

### Movement and Positioning
- Improved token-to-token movement collision by comparing Movement Strength against Blocker Strength for actions that can bypass tokens. #1241
- Added a Climb movement action that can cling to an adjacent surface without falling. #1253
- Implemented a custom Fall action override for the Glider which automates gliding instead of falling.
- Avoid assigning the Flying or Burrowing statuses at the conclusion of a movement that lands exactly on a supported surface. #1219

### Spellcraft Animation
- Added spell animations for the Influence gesture, covering the Flame, Frost, and Life runes. #1247
- Added Touch gesture animation support for the Flame, Frost, and Life runes. #1234
- Modernized the arrow projectile VFX animation for scrolling combat text and impact effects. #1249
- Prevent unconfigured spells from using a fallback arrow attack animation.

### Items and Equipment
- Added 10 new affixes that grant inflection knowledge as item suffixes. #1203
- Added the Vicious prefix affix that allows a weapon to cause bleeding or staggered on a critical hit. #1205
- Redefined two-handed weapon spellcasting eligibility so all two-handed weapons have at least one spell hand available (two for a two-handed talisman). #1185
- Allowed the user to set an affix's identifier when creating their own affix. #1208
- Annotated weapons that would experience a skill penalty during character creation and when equipped on the actor sheet. #1132

### User Interface
- Group chat cards per target and add a per-target summary addendum documenting additional resource or effect changes that apply to each target. #1218
- Display the Action description in the ActionUseDialog with a constrained max-height for long-scrolling action descriptions. #1222
- Surface an invalid target designation result in the ActionUseDialog as a UI notification warning for visibility. #1221
- Mark a plan as incomplete when a placed Region or planned Movement results in invalid targets, requiring it to be redefined. #1220
- Close the talent tree application HUD instantaneously rather than on a delayed animation. #1200
- Added better spacing and a separator between multiple Actions defined on a single Talent. #1183
- Added a means for certain default actions to be suppressed from the character sheet entirely. #1223

## API Changes
- Allow the `summon` schema of CrucibleAction to be nullable for the sake of data frugality. #1257
- Improve the determination of whether a Region is ephemeral or persistent based on whether an Active Effect added by the action retains it as part of its ongoing effect. #1255
- Allow select talent actor hooks to be async and awaited; the `confirmAction` actor hook is now async as part of the async confirmation flow. #1231
- Allow select actor hooks to throw, surfacing their error to the caller for handling. #1145
- Allow each element of the `resources` array in the CrucibleAction events stream to be annotated with an optional `damageType` so multiple hits from different damage sources can be represented. #1204
- Added a `rest` action tag which requires that the action may only be used immediately following a successful and confirmed Rest. #1197
- Factored out common-use action hook dispatch methods. #1117
- Ensure target-based boons and banes are available in the `prepareAttack` actor hook. #1109
- Removed schema-enforced caps on `resource.value` fields in favor of dynamic clamping against the maximum field. #1178

## Content Updates

### New Adversaries
- Added Stone Elemental and Mud Elemental taxonomies as subsets of the Earth Elemental category. #1230
- Added the Constrictor creature archetype.
- Added the Telekinetic creature archetype.
- Refactored adversary limb balance and adversary parts.
- Synced playtest actors and pregenerated characters with updated talent data.

### Rune: Storm
- Renamed and migrated "Rune: Lightning" to "Rune: Storm" to reflect a more general elemental command over wind, storm, tempest, and lightning. #1250
- Tidied up missed lightning migrations for the Storm rune rename.

### Talent Tree Reorganization
Reorganized many talents as part of the signature expansion. #1192

Moved talents:
- Strong Grip moves to `str1b`
- Deft Grip moves to `dex2a`
- Adrenaline Surge moves to `str2b`
- Last Stand moves to `tou2b`
- Slipstep moves to `dex2b`
- Ruthless Momentum moves to `toustr1`
- Cleave moves to `str1a`
- Berserk Strike moves to `str3a`
- Cadence moves to `str2a`
- Tumble Through moves to `dextou1`
- Decisive Action moves to `int2b`
- Rallying Cry moves to `pre2b`
- War Mage downgraded from a signature talent to a regular talent at the `preint4` node
- Operator signature moved to Toughness + Presence with balance changes
- Arcane Archer signature moved to Dexterity + Wisdom

New talents added: Wizard, Sage, Artificer, Spiritbreaker, Primalist (STR+WIS), Juggernaut (STR+TOU), Duelist (STR+DEX), Acrobat (STR+DEX), Bastion (STR+TOU), Telekinetic (PRE+INT), Peltast (STR+INT), Gambit (STR+INT), Warchanter (STR+PRE), and Warden (STR+WIS), along with Wrestler (`str2c`), Hulking Physique (`str4a`), Disarming Riposte (`str4b`), Gathering Storm (`wis4d`), Flash Frost (`int4a`), and Acrid Earth (`int4a`).

### Localization
- Localized cost tags for spellcraft components. #1209
- Talent and rules text now use condition enrichers with rules tooltips rather than plain bold text. #1172
- Reverted an erroneous usage of the Condition enricher.
- Updated the Enraged rules text to reflect that focus can still be spent so long as the spending action is a strike. #1215
- Updated Polish localization strings.
- Updated Russian localization strings.

### Balance Changes
- Heavy two-handed weapons are reduced to action cost 3, the same as other two-handed weaponry, allowing substantially more effective action economy. Includes minor talent rebalances for Uppercut and Swipe. #1254
- Rebalanced the rules for Strong Grip and Deft Grip to reflect the changes to two-handed weapon spell hands. #1190
- Allowed the Javelin to have an expanded thrown range of 60 feet. #1053

## Bug Fixes
- Fixed a critical bug with the allocation of damage to reserve pools.
- Ensured that status conditions like weakened or incapacitated which reduce resource pools are applied before clamping the current resource value to the allowed range. #1180
- Added conservative re-rendering of the actor sheet after an input field change in cases where no database update occurred because of no diff. #1182
- Fixed the character sheet not returning to its original size after minimizing the browser window and returning to full screen. #1147
- Properly suspend talent tree rendering and animation while it is closed to improve poor rendering performance reported on certain hardware. #1137
- Fixed the request panel rendering incorrectly in Firefox by switching to CSS grid. #1116
- Fixed ranged natural weapons being unusable for basic strike actions. #1199
- Fixed `acquireSingleTarget` not rejecting enemies when a target must be an ally and vice versa. #1240
- Fixed the Crucible Hazard region behavior rolling against the GM's targeted token instead of the token that triggered its event. #1194
- Fixed the Staggered applied effect using a different icon than its canonical status icon. #1174
- Fixed Spellband's affix being misconfigured with the `affixFocus` identifier instead of `affixFocusing`. #1213
- Fixed miscellaneous non-localized texts, including the reset talents dialog. #1216
- Localized the Actions fieldset label.
- Use localized resource costs for spellcraft tags.