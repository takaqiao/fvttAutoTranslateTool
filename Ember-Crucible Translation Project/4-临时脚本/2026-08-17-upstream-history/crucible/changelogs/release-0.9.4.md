![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)
# Crucible Version 0.9.4 - V14 Alpha 4 Hotfix
A small bugfix patch to address some urgent issues in yesterday's 0.9.3 release.

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael

## New Features
- Added an automated Echolocation Detection Mode for the Echolocation adversary talent. #1003

## Bug Fixes
- Fixed an urgent bug which prevented actor hooks from applying at all following the removal of persisted `actorHooks` from the actor system schema. #1039
- Fixed item sheets failing to minimize correctly (content removed but sheet not minimized). #1033
- Fixed Loot enrichers generated during item randomization including a duplicate affix identifier when the base item already had an affix. #1032
- Fixed adversary threat not being shown on the sheet, and corrected missing localization for "Level" on the adversary sheet and for hazards. #1029
- Allowed module hook expand buttons in locked compendiums to remain functional so implementation code can still be viewed, while hiding the "+" add button. #1025
- Used `crucibleTags` in character creation equipment so property tags are formatted correctly. #1040