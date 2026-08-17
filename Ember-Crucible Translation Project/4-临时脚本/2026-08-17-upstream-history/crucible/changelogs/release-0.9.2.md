![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.9.2 - V14 Alpha 3
Hello friends, and welcome to Crucible `0.9.2`! I've been really busy since the last release, but finally had time to focus on Crucible this weekend and I accomplished *a lot*!

This update focuses somewhat on exploration of the world outside of combat encounters: a brand new Hazard system gives Gamemasters a flexible way to throw environmental challenges at the party, from extreme heat and cold to suffocation and falling, all backed by a draft Exploration rules chapter that I am eager to expand in releases to come. 

On the tactical Combat side of things, you'll find automated Swarm mechanics, a much friendlier Action Use Dialog, the ability to safely undo movement with Ctrl+Z, and a long-requested Forced Movement toggle for moments where a creature is required to move as a side effect of some other action.

This update also introduces the first built-in foreign language localizations for the Crucible system (UI translations only, not compendium content) with Polish, Russian, and German being contributed by community members fluent in those languages.

Thank you all for your continued support, feedback, and bug reports - they help make every Crucible release better!

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @JiDW
- @danielrab
- @Aioros
- @Lioheart
- @salamander-git
- @MrSoris

## Highlights
The main highlights of the `0.9.2` release include:

- A new **Hazard** system with a powerful `HazardDialog` and a **Configure Hazard** macro lets Gamemasters quickly set up environmental challenges that affect the whole party.
- A draft **Exploration** rules chapter introduces formal mechanics for environmental hazards, body temperature tiers, and the new `Suffocating` condition.
- **Thermal Vision** is now a first-class detection mode, automatically applied by talents and ancestries that grant it.
- **Torch and Lantern automation** finally make light sources behave the way you'd expect, with Lanterns consuming Lantern Oil and Torches as self-consuming consumables.
- Automated **Swarm mechanics** change the way Swarm adversaries fight, making them feel meaningfully different from a single creature.
- A new **Forced Movement** toggle and Ctrl+Z movement undo give Gamemasters and players much better control over the flow of combat.
- Three **Languages** with new built-in localization support: Polish, Russian, and German.
- Numerous bugfixes and quality-of-life improvements throughout the system.

## New Features
### Environmental Hazards and Exploration
- Added a powerful `HazardDialog` that allows the GM to configure and manage environmental hazards affecting the party, similar to a Group Skill Check. #896
- Added an Exploration rules page in the system documentation featuring draft rules for Environmental Hazards. #934
- Added a `Configure Hazard` macro to the System Macros compendium pack so a GM can quickly set up a hazard from the hotbar. #982
- Added a new `Suffocating` condition that deals damage over time to both Wounds and Madness. #971
- Formalized `TEMPERATURE_TIERS` for use by both environmental challenges and creature body temperatures. #963
- Allowed each Taxonomy and Ancestry to configure the body temperature of that creature type. #964

### Vision, Lighting, and Tool Automation
- Added a Thermal Vision detection mode for use in Crucible core talents and Ember's Ashka Lineage. #940
- Automated the application of Thermal Vision so it activates the corresponding detection mode when appropriate. #967
- Added support for Tool actor hooks, allowing equipped Tools (such as a Torch or Climbing Spikes) to modify actor attributes when carried. #969
- Automated the new Torch consumable as a tool with an active effect that produces a token light source. #780
- The Lantern now consumes Lantern Oil charges, providing a meaningful tradeoff with the self-consuming Torch.
- Added automation for several new Equipment items: Gem of Illumination, Focusing Pearl, Electrocharge Ampoule, Frostdrop Phial, Gem of Conjured Flame, and Omniglot Decoction. #910

### Movement and Action Use
- Added a Gamemaster-only Forced Movement toggle which allows moves that incur no action cost (e.g. tidal wave or Aqueous Transmission), with corresponding System Rules documentation.
#897
- Undoing (Ctrl+Z) movement now restores Action Points and Free Movement allowance, properly reverting and deleting the originating movement action. #898
- Comprehensive Action Use Dialog improvements: #918
- Self-anchored, non-rotatable regions (pulse, aura) now auto-place their preview without requiring a Place Region click. #951
- The Use Action button is now disabled when selected targets are invalid. #952
- Clicking Use Action without first placing a region or planning movement now prompts for that placement on use. #953
- Changing the spellcraft components (rune, gesture, damage type, inflection) no longer deselects targets that remain valid. #950
- Improved the mechanics of the Delay action: a delaying combatant now acts before other combatants at the chosen delayed Initiative, with ties resolved by original pre-delay
Initiative. #937
- Added the new `Investiture` default action, which allows actors to activate the Investment state of their equipped items. #954
- Added a framework for item Investiture mechanics and corresponding rules content explaining Investment and Investiture. #894
- Added an `aura` target type for self-anchored persistent areas. #903
- Treat certain default actions (cast, strike, reactiveStrike, investiture, recover, reload, rest) as automatic favorites, displaying them on the actor sheet by default if available.
#956

## API Changes
- Added a new `prepareToken` actor hook that allows talent automation to alter properties of Token data. #966
- Added a new `prepareMessage` action hook that allows specific actions to customize the presentation of their chat message content. #958
- Refactored the Rest and Recover action flows to use the full Action event stream, allowing hooks to modify resource and effect changes that accrue during a rest or recovery. #867
- Improved `CrucibleActor##updateSize` to be more frugal for performance and to assign `{depth: size}` for better V14 support. #973
- HUD-toggled status effects now run through the full generator function, supporting related statuses, damage-over-time effects, and other generator features that were previously
unavailable when toggled from the Token HUD. #980
- Removed the Ember 0.5.0 compatibility shims; added a small set of talent shims for Ember 0.5.1 forward-compat. #968
- Added a dedicated method for recursively allocating and constraining resource changes to an actor.
- Added integration with Dice So Nice for damage type customization, exposing `DiceTerm.options.type` and `Roll.options.type` so users can configure dice appearances per damage type and
per skill. #919
- Improved Initiative Report message integration so modules like Dice So Nice (Force Actor Dice Appearance, Disabled for Initiative Roll) can react to initiative rolls. #932
- Improved Group Skill Check rendering with Dice So Nice, smoothing the experience of temporarily hiding group check results until the check has been fulfilled. #931

## Content Updates
- Added automated Swarm mechanics that tactically change how Swarm adversaries behave in combat. #909
- Added the missing Fiend creature type to the Taxonomy Physiology Category selector. #908
- Imported adversary talents and equipment from Ember content conversion into the core system compendiums.
- Additional system compendium updates pulled in from Ember content conversion.
- Added `data-no-toc` to headers in embedded talent cards so they no longer contribute additional structure to the Table of Contents of Journal Entry Pages within which they are
embedded. #972
- Ensured item, action, affix, and talent descriptions are HTML-enriched everywhere they may appear (including image captions and inline previews). #886
- Clarified the Throw Weapon description: the +2 banes only apply when the weapon being thrown does not have the `Thrown` tag. #915
- Clarified that Backstab does not require a mainhand weapon and may be performed with the offhand. #935
- Clarified in System Rules that flanking bonuses apply only to melee attacks. #936
- Clarified the wording of the Bard's Laughing Matter action to specify that the triggering attack must deal no damage and cause no effects. #938

## User Interface
- Enabled initiating an Actor drag operation from the Party sheet to transfer actors or place them into an area map. #902

### Localization
- Added built-in support for Polish localization. #930
- Added built-in support for Russian localization. #929
- Added built-in support for German localization. #928
- Cleaned up tech debt around system constant pre-localization, standardizing on `defineEnum()` and `preLocalizeConfig()` syntax; the legacy `Enum` class is now deprecated. #965
- Improved localization keys for plural forms across the system, including clearer wounds/madness defense labels.

## Bug Fixes
- Fixed chat messages for skill checks and attack rolls displaying full result details to non-recipients when sent in blind, GM, or self-whisper modes. #942
- Fixed Penetrating Shot's scope to affect all targets, not just enemies. #944
- Fixed buying two of a non-stacking equipment item during character creation causing one of the items to disappear when equipping. #948
- Fixed Refocus action no longer working when the Talisman item was located in a different inventory section. #947
- Fixed reversing an attack against a creature already at 0 Health incorrectly healing that creature. #945
- Fixed migration of physical items containing deprecated properties (keen, reach, returning, reliable) leaving invalid entries on the item. #885
- Fixed the `talisman` tag's `canUse` hook throwing an error by propagating the `strike` tag instead of assuming it. #881
- Fixed status effects added via the Token HUD not appearing on the token. #882
- Fixed `ActionUseDialog` throwing an error when there is no viewed Scene and target designation, region placement, or movement planning is required. #879
- Fixed localization issues with `ACTOR.GROUP.FIELDS.members.many`, `CostHand.few`, and other plural keys in places where `Intl.NumberFormat`/`Intl.PluralRules` was used. #880
- Fixed the Returning affix incorrectly causing returning weapons to be unequipped after a throw. #924
- Fixed `#equipItemAction` failing to properly update the equipped slot. #939
- Fixed Crucible development mode ID generation incorrectly allowing collision with or reuse of an existing Document ID. #907
- Fixed creating a randomized item from an explicit base that already contained affixes erroring if the same affix was redrawn. #906
- Increased the height of the requested-actors drop area in the roll dialogue to make it easier to drag a second actor below the first. #893
- Ensured the Roll button in the roll dialogue always does something when an actor is selected. #892
- Registered the party sheet as an active Application of each Actor member so it automatically re-renders (e.g. Party HP now updates after resting without requiring a sheet reopen).
#891
- Stripped ownership data from owned Items configured during character creation so a player continuing creation of a Gamemaster-created Actor no longer encounters an ownership error.
#923
- Display the unconstrained grid when working on canvas layers other than the Tokens Layer. #889
- Use the actual distance rather than the formatted string distance in the Token Ruler so waypoint labeling no longer fails in certain locales. #926
- Display in the Actor inventory explicitly when an item has `quantity=0` and remains only as a placeholder. #970
- Clean up stale `flags.crucible.delay` flags on actors when initiative is rolled for a new round or new combat. #974
- Fixed clicking on a group roll from a journal entry not initiating the roll. #949
- Fixed scoping issue with the `_hadTokenHooks` cache.
- Fixed typos, targeting, and cost on the Aqueous Transmission spell.
- Associate non-ephemeral regions with their originating effects so they are properly cleaned up.
- Removed a legacy `toggleStatusEffect` override that conflicted with the new HUD-toggled effect generator path.