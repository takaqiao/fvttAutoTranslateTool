![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.9.1 - V14 Alpha 2
Version `0.9.1` is a major update to the Crucible game system featuring some really exciting and significant changes. The biggest highlights include:
- The **Action lifecycle** has been rebuilt around a new **event stream** architecture that provides a unified and chronological record of everything that occurs during an action.
- A comprehensive **item affix and enchantment system** introduces prefix and suffix modifiers for weapons, armor, and accessories, with name generation, rarity scaling, loot randomization, and over 80 initial affixes to play with. 
- **Group skill checks** add support for queried multi-actor checks as a foundation for social and exploration challenges. 
- **Eleven new Toughness talents** fill gaps in the talent tree with some exciting new options for gameplay.
- A new **system macros compendium** provides convenience macros for common operations. 
- Numerous bugfixes

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @Plushtoast
- @danielrab

## Features
### Actions and the new Event Stream

The `CrucibleAction` usage and confirmation lifecycle has been comprehensively refactored to use a chronologically ordered event stream (`action.events`) instead of the previous outcome structure (`action.outcomes`). This architectural change provides a unified, serializable record of everything that occurs during an action, enabling robust reversal, clearer chat message rendering, and a better hook API.

> [!CAUTION]
> This refactor is unfortunately disruptive, but necessary to empower the full vision for Crucible action automation. For those of you who have been making custom 3rd party content using action or actor hooks, you will have to make some changes to your hooks to account for both new hook signatures as well new internal logic using the action events stream.

- Introduced the `CrucibleActionEvent` class providing a comprehensive event stream attached to every `CrucibleAction`. #822
- Migrated all hooks and workflows throughout the action lifecycle to use this event stream as the source of truth for what the Action does. #825
- Comprehensively refactored all action and actor hooks to utilize new hook signatures and work with the event stream. #829
- Added `action.recordEvent()` helper for registering events to the stream. #823
- Event streams are serialized and deserialized through `ChatMessage` flags. #824
- Added `CrucibleActionWeaponState` and per-event weapon snapshots to capture stateful weapon properties at the time of use. #826
- Recorded `CrucibleAction#targets` as a first-class `Map<CrucibleActor, ActionUseTarget>` property. #827
- Refactored actor attack methods (`weaponAttack`, `spellAttack`, `skillAttack`, `receiveAttack`) with unified `AttackRollData` passed as options. #828
- Introduced `_configureAttackerRollData` and `_configureTargetRollData` as standardized workflows for modifying roll data prior to an attack roll. #830
- Added `CrucibleActor#eventsByActor`, `eventsByTarget`, and `selfEvents` convenience accessors. #831
- Eliminated `CrucibleActionOutcome`, `configureOutcomes`, and all outcome-based infrastructure. #832
- Expanded documentation of the `CrucibleAction` class covering purpose, lifecycle, hook events, and related concepts. #834
- Generalized item snapshot and reversal workflow across all actions and tags, removing the need for tag-specific reversal logic. #843
- Tracked consumable item snapshots so that reversal of consumable usage correctly restores equipped state, uses, and quantity. #844
- Added the `talisman` action tag that requires a talisman to be used. #860
- Show the built-in implementations of hooks on actions and talents. #766

> [!CAUTION]
> The following change deserves special mention because it may be particularly disruptive (for which I apologize). There were some unresolvable problems with offering inline text editing of action and talent hooks, flaws with that approach that could not be properly secured against risk of cross-client manipulation. This design needed to change for security reasons. What this means is that now all talent and actor hooks need to be defined via a Javascript module, which ensures such code is authored by a trustable source (you chose to install the module, or you wrote the script yourself). In order to better support homebrew creation of automated talents and actions in the future, I'll be researching and investing towards a macro-based solution that could restore a more user friendly interface for writing custom hook logic but without the unresolvable problems that this prior approach entailed.

- **Breaking:** Eliminated support for database-persisted string hooks. Only module-defined hooks are now supported. #837

### Item Affixes and Enchantment

A new affix system allows weapons, armor, and accessories to be enchanted with prefix and suffix modifiers that alter item stats, provide hooks, grant actions, and compose dynamic item names and rarity. This release also introduces loot text enrichers and an item randomization API built on top of the affix system.

- Added the core affix system for weapon, armor, and accessory items. #672
- Added the affix active effect subtype with a dedicated data model, sheet, and item sheet integration. #786
- Affixes are validated against quality capacity, type restrictions, and identifier uniqueness. #787
- Enchantment tier and item names are automatically derived and composed from equipped affixes. #788
- Affixes can provide hooks and actions that attach to items and their owning actors. #789
- Added the Unique item property for one-of-a-kind items that prevents affix enchantment. #790
- Replaced discrete enchantment-based rarity with continuous per-affix-tier rarity scaling. #805
- Added static helpers for computing item rarity and price. #806
- Enforced non-stacking enchantment bonuses across weapon potency, rune potency, and skill affixes. #803
- Added skill enchantment bonus with a finalization step separating base and derived skill values. #804
- Supported user-defined compendium packs as sources for affixes and equipment. #812
- Extracted Reach, Returning, Reliable, and Keen from weapon properties into affixes. #785
- Deprecated keen, reliable, reach, and returning weapon properties with version-gated culling. #791
- Added tradeoffs to the parrying, blocking, and engaging weapon properties. #792
- Prepared critical thresholds as weapon damage data and scaled the oversized bonus by hand count. #793
- Added weapon suffix affixes: potency, deflection, guarding, keen, reliable, reach, and returning. #794
- Added 12 damage type weapon suffix affixes and 12 rune potency weapon suffix affixes. #795
- Added 9 damage conversion weapon prefix affixes for non-physical damage types. #796
- Added 12 damage resistance prefix affixes and 12 skill enchantment prefix affixes for armor and accessories. #797
- Added accessory prefixes for health, morale, action restoration, and disguise. #798
- Added armor prefixes for healing threshold reduction and critical hit resistance. #799
- Created base unenchanted accessory templates for each common type. #800
- Added 12 damage resistance prefix affixes and 12 skill enchantment prefix affixes for armor and accessories. #797
- Added accessory prefixes for health, morale, action restoration, and disguise. #798
- Added armor prefixes for healing threshold reduction and critical hit resistance. #799
- Created base unenchanted accessory templates for each common type. #800
- Migrated deprecated weapon properties and accessory hooks to affix equivalents. #801
- Replaced Kindly Visage with generic diplomacy skill prefix and disguise suffix. #802
- Removed enchantment from consumables and scaled scroll capacity by quality tier. #807
- Converted clothing items from accessories to armor in the `unarmored` category. #734
- Added Spellcraft Rune knowledge prefix affixes (with training). #870
- Added Spellcraft Gesture knowledge suffix affixes. #871
- Added Creature Type Bane weapon prefix affixes. #872
- Added `@Loot` text enricher for defining ephemeral composed items with drag-and-drop and click-to-preview. #808
- Overrode `CrucibleItem.fromDropData` to compose loot items from enricher data. #809
- Extended the `@Action` enricher to support item-owned actions. #810
- Added `CrucibleItem.randomize` API and a GM dialog for generating random items within a price budget. #811

### Group Skill Checks

The new group skill checks system is really exciting and allows for checks to be requested from multiple actors, with results coalesced and presented in an summary group skill check chat card.

- Implemented support for queried group checks as a prerequisite for social and exploration challenge play. #97
- Add support for the group skill check to designate specific actors or easily add the whole party as targets for the roll.
- Add support for group skill checks to include multiple skills with, optionally, different DCs each. 
- When requested to make a roll, players can choose which skill to perform. 
- When the GM rolls automatically on the players' behalf, it automatically chooses the skill for which the actor has the highest passive score.
- There is a convenient "Configure Group Check" macro included in the new built-in "System Macros" compendium pack that allows a Gamemaster to, from their hotbar, quickly configure a new group skill check for the party to undertake.
- Updated the `[[/skillCheck {skillId} {dc} group]]` check enricher to automatically initiate a group check with pre-filled actors and skill configuration. #773

### Talent Tree

- Added 11 new talents to the Toughness sextant of the talent tree to significantly round out . #868
- **Adrenaline** (`tou4a`): Gain +1 Focus once per round when you sustain Health damage from an attack.
- **Armor Crusher** (`toustr4`): Mastery of striking armored opponents so the armor itself works against them.
- **Armored Instinct** (`dextou2`): Gain +1 Resistance against damage dealt when an attack results in a Glancing Blow.
- **Battle-Worn** (`tou1b`): Recover additional Wounds equal to half your Toughness score when performing the Rest action.
- **Defiant Will** (`tou4d`): Gain bonus Willpower defense equal to Toughness scaled by the proportion of Health lost.
- **Evasive Armor** (`dextou4`): Excess Reflex defense above Physical defense is added as bonus Armor.
- **Headbutt** (`toustr1`): Deliver a brutal close-range strike with your head that catches opponents off-guard.
- **Impenetrable Guard** (`tou4d`): When an enemy's Strike results in a Block or Glancing Blow, that enemy suffers 1 Bane on their next Strike against you.
- **Interpose** (`dextou4`): Throw yourself into harm's way to protect allies, taking danger in their stead.
- **Iron Resolve** (`tou1b`): The critical success threshold for Strikes against you is increased by +1.
- **Never Yield** (`tou4a`): While Weakened, your action point penalty is reduced from -2 to -1.

### System Macros

- Added a system macros compendium pack with convenience macros for Rest, Recover, Loot Randomization, and Group Skill Check. #562, #813

### Other Improvements

- Added the missing "Plant" creature type. #869
- Added an Ember compatibility layer directly in the Crucible system to provide a temporary mechanism for releasing Crucible updates that would otherwise break Ember. #833
- Optimized repeated `CrucibleActor` embedded document calls using the new `modifyBatch` API. #769
- Improved computation of token move speed animation based on an assumption of 6 strides per round. #777
- Improved the presentation of Crucible Token Ruler waypoints, fixing several bugs with incorrect labeling. #725

## Documentation

- Rewrote Equipment rules documentation with updated weapon properties, new Accessories, Consumables, and Affixes pages. #814
- System Rules documentation improvements across multiple areas to fix out-of-date information and clarify missing mechanics. #784

## Bug Fixes
- Fixed `CrucibleActionConfig` causing an actual form submission/refresh when adding an effect. #878
- Fixed heroism bar not visually filling despite correct percentage. #877
- Fixed token bar pips not being real-time accurate. #876
- Fixed GMs no longer seeing the default DC selector on a basic skill check dialog. #875
- Fixed character creation being tolerant of not defining a character name, which caused a failure upon completion. #874
- Fixed NaN ability scaling for actions without any ability scaling rule. #845
- Fixed tooltips not working in detached windows. #842
- Fixed region/movement planning minimizing detached windows. #864
- Fixed setting the level of a character with ability increases to 0 bricking the character. #782
- Fixed certain damage/restoration over time turn summaries throwing an error when getting pip color. #761
- Fixed `action.usage.weapon` always using main hand. #758
- Fixed talent ID migration not being accounted for with existing actors' detail items. #746
- Fixed Control Run scaling with Intelligence instead of Wisdom. #745
- Fixed having nothing (unarmed) or a shield in offhand breaking non-mainhand-tagged strikes. #743
- Hamstring throws an error if using a two-handed weapon #767
- Vampiric Bite does not account for natural weapons training #767
- Armored Shell throws an error if using a two-handed weapon #767
- Adrenaline Surge does not recognize the effect #767
- Vow of Animus does not recognize the effect #767