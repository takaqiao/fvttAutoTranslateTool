![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.9.5 - V14 Alpha 5
Hey there Crucible nerds, I'm really excited about today's release which adds some HUGE new features to the system that I've been working on for a long time. The headline items in version 0.9.5 are listed below, but this is a really big release adding a ton of small features, bugfixes, balance corrections, and documentation improvements across the board. The system is really coming along and I'm very proud of this release as a big step towards the long-term vision!

## Highlights
- A new Spellcraft Animation framework with built-in animations and accompanying sound effects supporting the Life, Flame, and Frost runes with support for the Arrow, Fan, Ray, and Blast gestures. More runes and gestures will be added in future patch cycles.
- Big improvements for tactical movement and positioning with the addition of token-to-token movement collision during combat with specific Actions and movement modes which can bypass that collision.
- Support for forced movement mechanics where an Action can indirectly cause other tokens to be automatically moved, for example the Push and Pull spellcraft inflections.
- Support for Flying and Burrowing as special movement modes which leave tokens with a state-tracking status effect.
- Automation support for Falling mechanics which automatically triggers when a Token ends its movement in the Falling status.
- The addition of optional "Important" adversaries which track Wounds and Madness for long-term persistence of resource tracking for recurring characters, whether they be companions or antagonists.
- An opt-in choice for flexible Ancestry ability scores which allows for a rules alteration to the character creation process.
- The addition of content to an "Adversary Equipment" compendium pack providing a variety of easy-to-include options for building adversaries with natural weapons and armor.
- Support for Taxonomies to include automatic equipment grants, and added support for Archetype-granted and Taxonomy-granted equipment to automatically scale in quality with creature level.
- An overhaul of item stacking rules for equipped and backpack items to streamline the experience of having items which stack in the backpack but are singularly equipped to toolbelt slots.

A full accounting of all the additions and fixes in this patch is presented below, along with thanks to the community contributors who helped make it happen.

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @Lioheart
- @Fyorl
- @Feu-Secret
- @danielrab
- @MrSoris
- Drewzelle and Ian (TabletopRPGMusic) from the Ember team for their work on spell particle sprites and sound effects.

## New Features
### Spellcraft Animation and VFX
- Added a built-in architecture for spellcraft animation using the V14 core VFXEffect API and an original set of particle sprites and sound effects that are unique to Crucible and its spellcraft system. #1121
- Added spellcraft animation support for the Flame, Frost, and Life runes with initial gesture support for Arrow, Fan, Ray, and Blast. More runes and gestures coming soon! #1122
- Added a client setting that allows fully disabling the VFX animation system on a per-client basis. #1123
- Adapted the intensity of particle animation by decreasing particle counts on lower performance modes. LOW performance mode uses 25% particles, MEDIUM performance mode uses 50% particles. Photosensitivity Mode also forces 25% particles. #1124
- VFX animation configurations are now defined in grid distance units instead of pixel units to adapt to different grid sizes (microgrid required). #1125
- Fully integrated token forced movement with VFXEffect animation playback so that secondary movements (knockback, pull) are synchronized with animation impacts. #1143
- Replaced the particle blend curve with an exposure curve and custom shader.

### Movement and Token Automation
- Added token-to-token movement collision while in Combat. The Jump, Blink, and Burrow movement types bypass creature obstacles, and specific actions like Tumble Through, Shield Charge, Bull Rush, and Trampling Charge bypass token collision as part of their effect. #1098
- Added Burrow as an available token movement action. #1092
- Added support for Flying and Burrowing statuses that are automatically toggled if a target ends a movement using a fly or burrow movement action. #1119
- Implemented automation for Falling as an action that is automatically prompted if a movement is concluded without an underlying supporting surface. Falling damage and status effects are automated. #1120
- Implemented movement automation for the Push and Pull spellcraft Inflections. #1062
- Added automatic token scale adjustment for distance away from the currently viewed level base. #1107
- Improved movement-based targeting to attack enemies which are base-to-base with the final movement position. (no linked issue)

### Combat and Gameplay
- Added the ability to create "Important" NPCs that track Wounds and Madness without being a Hero (Player Character). #771
- Added support for Flexible Ancestry Ability Scores as an optional setting that allows changing the Primary and Secondary ability scores during the Ancestry step of character creation. #1084
- Added a horizontal band to visualize the level of Wounds or Madness that an Actor has on their character sheet. #1087
- Added core system automation for when an effect providing the Invisibility status is ended, applied automatically as part of action finalization. #1065
- Improved the intentionality behind whether a requested skill check is specifically a Group Check or a set of individual check requests. #1050
- Display scrolling damage text results for zero damage outcomes like "0 Health" when a target cannot receive any more healing or "0 Dodge" when an attack is dodged. #1118
- Automatically include the text content of Condition rules pages in the template effects that are generated for that Condition. #1148
- Support dragging an Action icon from the ActionUseDialog to create a macro for that action, specifically useful for creating macros for Composed Spells that retain their name and icon. #1115

### Items and Equipment
- Improved handling of item stacking: items no longer stack when equipped, but stack freely in the unequipped backpack. #685
- Enabled Taxonomy items to grant equipment, restricted to only natural weapons and armor if the Taxonomy cannot use equipment. #1001
- Taxonomy and Archetype granted equipment now auto-scales in quality at appointed level thresholds. #1162
- Redesigned the approach to valid weapon choices to fix bugs and provide a better UX for actions that support weapon choice. #1043
- Item names now automatically update when applying affixes if the previous name was identical to the previous auto-generated name. #1090

### Documentation
- Added a Creature Collision section to the Combat rules describing how creatures block movement while in Combat, which movement types and Actions bypass obstructing creatures, and how Forced Movement interacts with creature obstacles. #1171
- Added Burrow to the documented Movement Types, with cross-references from the Fly and Burrow movement types to their automatically applied conditions. #1171
- Added a Flying condition page to the Conditions rules and documented the automatic application and removal of the Flying and Burrowing conditions based on how a movement ends. #1171
- Documented the automated Falling workflow in the Exploration rules: the falling condition, the automatically prompted Fall action presented to the Gamemaster, and its automated damage and Prone application. #1171
- Revised the Forced Movement rules to describe action-driven automation of pushes, pulls, and knockbacks, repositioning the Forced Movement toggle as the manual fallback. #1171
- Added Item Stacking rules to the Equipment documentation covering the Stackable property, the equipped-items-never-stack rule, and automatic stack splitting and recombination. #1171
- Expanded the Adversaries rules with new Talent Progression, Skill Progression, and Equipment Progression sections documenting level-gated talent grants, automatic skill rank advancement, and granted equipment with automatic Quality scaling. #1171

## API Changes
- Improved the `CrucibleActor#rollSkill` method to have more flexible options for how a skill check is rolled. #1088
- Refactored and reorganized shared code in the GroupCheck class to be shared with StandardCheck. #1076
- Added a system-level framework for managing movement plans and forced movement. (no linked issue)
- Enabled CrucibleAction to handle a singleton movement event for every actor in its events stream. #1067
- Allowed `CrucibleActor#getAbilityBonus` to select a best bonus instead of an average. #1066
- Added a `CrucibleTokenObject#canDetect` method for simulating whether a token is able to detect another. #1064
- Added an intentional difference between DC=0 and DC=undefined for checks with unknown difficulty. #1031
- Simplified to an explicit `_id` approach for CONFIG.statusEffects, removing some method overrides. #1063
- Aligned owned item preparation with only the initial active effect application phase in `CrucibleActor#applyActiveEffects`. #729
- Made it easier to provide enricher support for default actions using `@Action[default actionId]` instead of requiring a valid Actor UUID. #1081
- Allowed the duration units of an effect configured for a CrucibleAction to be blank rather than requiring an infinite duration declared in a specific unit. #1091
- Shimmed hierarchical VFXEffect reference resolution until core 14.364 is released. (no linked issue)

## Content Updates
- Added a preliminary selection of provided adversary equipment featuring prefab natural weapon and armor options. #1164
- Renamed "Disengagement Strike" to "Reactive Strike" throughout system content. (no linked issue)
- Marked talents and actions that are known not to be implemented as such. (PR #1157)
- Added adversary options from Ember content conversion, including the Polybrachial adversary talent. (no linked issue)
- Corrected errors in the descriptions of Bloodletter and Concussive Blows. #1080
- Talent descriptions now consistently reference conditions using condition enrichers with rules tooltips, rather than plain bold text. #1169
- Added a complete set of German translation strings. #1078
- Updated Polish localization strings for new keys and changes. #1079

### Balance Changes
- The Longbow is now an Oversized weapon, differentiating it from the Shortbow in a meaningful way. #1153
- Adjusted balance and action economy of the Monk Double Punch action (2A 1F). #1146
- Reduced the carryover benefit provided by zero Toughness or zero Presence adversaries to half of the unused pool instead of the full unused resource pool. #1082
- Decay damage-over-time now scales off Presence, aligned with the Necromancer. (no linked issue)
- Marked most consumables and tools as Stackable; rations are now stackable and blankets are lighter. (no linked issue)

## Bug Fixes
- Corrected a large number of talents that were incorrectly or incompletely implemented. #1038
- Added missing HTML enrichment of Action cards used in tooltips. #1168
- Ensured `generic`-tagged actions are capable of healing. #1161
- Ensured effect events are recorded for relevant targets, even if those targets are not affected by a resource change. #1160
- Ensured effect IDs generated from existing camelCase identifiers don't strip letter casing. #1144
- Ensured summoned tokens are controllable by players, and ownership is not cleared when importing a summon from a compendium. #1099
- Ensured action-created effects have their icon always shown when they convey a status effect or otherwise represent an indefinite but terminable condition. #1089
- Enabled bar changes via the TokenHUD to properly flow into secondary Wounds or Madness pools. #1085
- Improved reliability of scrolling damage numbers by trusting the resource changes in the action events stream. #1077
- Added missing HTML sanitization declaration for `actions.description`. #1075
- Fixed creation/conjuration gestures not removing the summon from targets when switching to another rune. #1068
- Ensured that enricher UUID strings can contain hyphen characters. #1061
- Fixed cancelling an attack clearing targets. #1056
- Fixed scrollbar overlapping text in ancestry descriptions. #1047
- Fixed Taxonomy or Archetype items with an Invalid talent not being removable via the [Remove Talent] button. #1041
- Tooltips which extend beyond the screen edge are now scrollable. #841
- Improved the text labeling of attack chat card results by shortening healing defense labels and explicitly displaying "Critical Hit" rather than just "Hit". #1111 #1108
- Improved the presentation of the equipped slot tag for weapon items. #1159
- Improved the appearance of disabled item card title H1s. #1154
- Natural weapons and armor are no longer procedurally prefixed with the quality tier label. #1163
- Fixed missing or broken localizations: un-equip weapon, party sheet actor tag, macro level tag, adversary active effects, talent removal during character creation, and various others. #1072 #1073 #1074 #1110 #1112 #1113
- Updated various action hooks to properly iterate over action targets. (no linked issue)
- Fixed action use dialog clearing targets on close and initializing empty summons usage. (no linked issue)
- Cleaned up references to ability bonuses and fixed defense formulae. (no linked issue)
- Added a guard for effects without declared system data. (no linked issue)
- Improved the unknown DC prompt. (no linked issue)