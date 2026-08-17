![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.9.3 - V14 Alpha 4
Hello friends, and welcome to Crucible `0.9.3`! This is a medium-sized release that adds a mixture of exciting new features and important bug fixes to facilitate ongoing playtests. Thank you all for your continued support, feedback, and bug reports - they help make every Crucible release better!

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @Lioheart

## Highlights
- Improved group skill checks with rules for collective success or failure
- Thermal Vision and Creature Sense detection modes
- Improve group Rest dialog UX
- A new `@Scroll[]` enricher for creating spell scrolls on the fly
- High priority bug fixes

## New Features
### Exploration Gameplay
- Implemented display of Group Skill Check aggregate results and corresponding rules for determining the collective success or failure of a group skill check.
- Placed the group Rest workflow behind a custom dialog that allows the GM to confirm which party members are resting (all by default) and the duration of the rest (default 10h). #994
- Improved the ergonomics of skill checks and hazards when a party token is controlled. When a party token is selected, it is now used as the target for checks instead of the party
group actor itself. #1023
- Created a Crucible Hazard region behavior that implements the effects of a Hazard via a Scene Region. #1006
- Added an automated Detection Mode used to detect creatures of a specific type (as with Gesture: Sense). #1002

### Game Content
- Added a `@Scroll[component1 component2 component3]` enricher that allows dynamically composing a Scroll for distribution via Journal text. #1000
- Added a "Limitless" status (alongside Invulnerable and Resolute) for system testing, which causes actions to not consume any Action/Focus/Heroism cost. #1018
- Included the full set of Item Properties as header tags when viewing a physical item, with discoverable tooltips to learn their behaviors. #999
- Added an Adversary Equipment compendium pack, collecting existing one-off adversary equipment into a centralized pack (not yet populated with content). #1005

### API Changes
- Refactored `CrucibleScene#_preUpdate` to instead focus on `CrucibleScene.fromImport`, specifically limiting scope to JSON import operations. #1017
- Added a `crucible.preGroupRest` hook which allows interrupting, modifying, or cancelling a rest.
- Actor sheet sidebars (favorites) now re-render when an action is used, confirmed, reversed, or its chat message is deleted. #1019

### Documentation
- Corrected outdated information about level-up rules and talent progression in the Character Creation rules. #998
- Improved the clarity of rules text around Skill Checks, including expanded rules for Passive Checks and Group Skill Checks. #1010
- Adjusted the Life and Soul rune descriptions to avoid confusion regarding damage versus healing. #989

### Balance Changes
- Added the missing performance skill to Horrific Critical.
- Rebalanced Twinned Shot to be a non-Difficult focus spender, differentiating it more clearly from Quick Draw. #1011
- Swapped Deep Behemoth from a talent to an archetype. #985

### Localization
- Updated Polish localization with new translation strings.

## Bug Fixes
- Fixed Counterspell not working out of combat and not conditionally showing in favorites based on the last action. #1020
- Fixed healing and rallying tags in hazard enrichers incorrectly dealing damage instead of providing restoration. #914
- Fixed a bug with incorrect application of `prepareToken` actor hooks for group actors. #992
- Fixed a non-GM attempting to finalize character creation throwing an error and blocking the update. #990
- Fixed a bug with ray template construction in the `ActionUseDialog`. #988
- Fixed a bug which prevented weapon attacks from using an explicit override damage type configured via action tags. #987
- Fixed initial actor creation not setting prototype token width, height, and depth (which are now only assigned on concrete updates). #995
- Renamed Movement to Traits in character creation and placed the section before ability scores. #996
- Improved feedback when invalid values are entered in scaling fields on Taxonomy and Archetype sheets, including resistance/immunity persistence on save. #986
- Improved handling of microgrid identification, providing an informative warning if the Scene is not microgrid-compatible and requiring scene dimensions to be a multiple of 5. #1007
- Fixed hint paragraphs being misaligned in Firefox inside flex column fieldsets. #997
- Added localization support for composed item names (replacing the hardcoded "prefix1 ... itemName of suffix1 ..." format). #981
- Removed unnecessary clamping rules from `StandardCheck##configureData`.
- Removed a redundant unique tag in physical item handling.
- Ensured that lazy-cloned actions have a clone of action usage data, rather than sharing references.