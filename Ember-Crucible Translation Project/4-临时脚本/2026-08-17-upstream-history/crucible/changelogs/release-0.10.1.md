![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.10.1 - Version 14 Alpha
Hello friends, I'm excited to start the week with a Crucible system update that include a bunch of quality of life improvements and bug fixes. Nothing too major here, but laying some foundation for big things to come in the 0.10.x release cycle.

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @danielrab
- @XinSysVTT
- @Lioheart

## Highlights
- A general system-wide (heroes and adversaries) rule that allows substituting Heroism for Focus
- Improved mechanics for ongoing active effects which require use of one or more hands
- Improved mechanics for actions like Counterspell which partially negate a prior Action's event stream
- Simplified the rule for whether "Fall" action prompts are presented or not depending on whether the Scene has surface regions defined.
- Improve the automation of actions or features which change a creature's Size.
- Numerous bug fixes and tidying

## New Features

### Combat and Rules
- Added a general system rule allowing any actor, both heroes and adversaries, to substitute Heroism for Focus when spending. #1369
- Allowed all Rune critical hit talents to scale using whatever ability score the associated Rune scales with. #1272
- Added a `resources.focus.block` prepared record documenting the reasons an actor may be unable to spend Focus, which individual talents can opt into or out of. #1368
- Introduced a concept of "continued hand use" on Active Effects so that maintaining an effect (such as a Grapple) reduces an actor's number of free hands. #1359
- Redesigned action negation to occur on a per-event basis, and updated Counterspell and the Inquisitor to counteract spells using event-stream negation.

### Movement and Summoning
- Improved the mechanics for fall surface resolution using a simple razor: whether the scene has intentionally configured surface regions or not. #1262
- Improved the mechanics of terminal waypoint target range for movement-type actions. #1327
- Allowed actions to specify `range.maximum = 0` for actions which require base-to-base touch. #1366
- Ensured that a summoned token inherits its summoner's `{elevation, level}`. #1329
- Implemented a bespoke range clamp mechanism for summon target placement. #1367
- Allowed summons to be explicitly sized by provided token data. #1365
- Improved ruler labeling of forced movement and implemented a mechanism for overriding the calculated movement cost. #1351 #1325

### Character Creation and Interface
- Added a prioritized heuristic to suggest auto-equipping weapons and armor purchased during character creation. #1312
- Enabled locking a previewed talent's description in the talent tree via middle-mouse so its nested action tags can be hovered. #750
- Added a Rule enricher that embeds a tooltipped inline element describing a specific game mechanic. #1166
- Added conditional-only autoFavorite behavior for the Recover action. #1263

## API Changes
- Further improved the CrucibleAction event-stream confirmation flow with sequential simulation of events for point-in-time resource and effect modeling. #820
- Ensured that skill and spell attacks route through `AttackRoll#resolveDamage` to compute damage totals. #1315
- Added an `AttackRoll#hasDamage` convenience getter that replaces inconsistently used checks. #1316
- Made the `damage` object present on every attack roll's data, not only rolls that dealt damage. #1317
- Updated the Crucible sheets' `TABS` property to align with the core `ApplicationV2` specification. #1331
- Inferred the initial `migrationVersion` using the adventure quickstart import version. #1296
- Added CrucibleActor disposition helpers.
- Raised the priority with which an action is sorted into the reaction section.

## Content Updates

### Adversaries
- Added the Prankster creature archetype.
- Improved the automation and rules of the adversary Swallow and Regurgitate talents. #1268
- Added automation for the Iramancer's bonus damage. #1326
- Updated the taxonomies and archetypes used for Earth elementals and modified the associated elemental talent components.

### Text and Localization
- Added the missing "DC" abbreviation translation, localizing the default label for unknown defense types. #1270
- Converted additional talent, rules, and item text to use inline enrichers. #1336
- Clarified the rules on the timing of when the Unaware status drops (start of turn). #1259
- Clarified that Intent applies to "actions made within the next Round" but expires on turn start. #1373
- Updated the outdated "within 6 spaces" wording in the Extoll Deeds description. #1374
- Corrected the Gem of Conjured Flame chat description, which used `@ref` incorrectly. #1299
- Clarified the Disarming Strike description. #1339
- Updated the description of the Pinning Shot action.
- Updated Polish localization strings. #1258

### Balance Changes
- Fixed and rebalanced Ferocious Leap, converting it to movement-based targeting and improving its action economy to `W-1A 1F`. #1274
- Improved the action economy of Ruthless Momentum to `W-2A 1F`. #1328
- Reversed the progression order for adversary auto-scaling equipment so that weapon quality increases arrive before armor quality improvements, prioritizing offensive challenge. #1264

## Bug Fixes
- Socketed the maintenance dialog to the designated owner of the actor. #1242
- Stopped displaying an unmet training requirement for adversary-equipped natural weapons. #1261
- Fixed Provocateur incorrectly making yourself enraged instead of your target. #1269
- Fixed spells using Blood Magic being reversed incorrectly when wounds are applied. #1271
- Fixed physical defense being calculated incorrectly when a character has both Block and Parry. #1273
- Corrected height flexing for the purchased equipment container in character creation so it scrolls vertically. #1275
- Updated token sizes when an actor's talents change, fixing a bug where the Hulking Physique talent failed to resize tokens. #1277
- Fixed an erroneous scrollbar on the action description in chat and the action use dialog at certain UI scalings. #1278
- Avoided removing detail-granted items that are provided by some other active source. #1279
- Fixed the Creation Sheet never showing a step banner label. #1280
- Prevented a strike with a warning, rather than allowing a no-weapon strike, when the only equipped weapon needs to be reloaded (e.g. an unloaded two-handed Mechanical weapon). #1282
- Fixed bomb effects not applying to the thrower when thrown (e.g. Electrocharge Ampule, Alchemist's Fire). #1285
- Fixed Warchanter not extending chants on successful hits. #1289
- Fixed Alchemist's Fire and Caustic Phial dealing typeless damage. #1293
- Corrected assorted bomb consumable misconfigurations: clarified Blast Flask wording to "all enemies who fail to avoid the attack", retagged Choking Ampoule from Reflex to Fortitude, aligned Electrocharge Ampoule and Frostdrop Phial to a Reflex defense, and renamed Frostdrop Phial's action and effect that were mislabeled "Frostdrop Flask". #1294
- Unified tagging across bomb-type consumables, removing the redundant `athletics`/`dexterity` tags that clobbered the usage ability bonus. #1291
- Fixed Gambit charging on a glance rather than only on a hit, and All In applying to every roll of an action rather than just the first. #1295
- Fixed Gambit's re-resolve not applying properly to spell attacks, skill attacks, or unarmed attacks. #1297
- Fixed the Invisibility iconic not actually making the character invisible. #1298
- Fixed newly summoned "sprite"-level adversaries assuming a 4x4 size when they should be 3x3. #1305
- Fixed dragging a consumable-granted action to the hotbar producing a nonfunctional macro. #1306
- Fixed versatile weapon grip switching by using sibling categories, so Versatile on a one-handed heavy weapon no longer increases its cost. #1308
- Wrapped arbitrary action scrolling status text in a subtracted parentheses format upon reversal. #1319
- Fixed incorrect armor designation for creatures without an Armor defense, resolving NaN defense percentages and enforcing a minimum defense of 0. #1322
- Fixed the Poison Blades interaction with other actions. #1334
- Fixed spell scroll Touch gestures leaking across scrolls by removing "Touch" as a scroll-grantable gesture, since it is inherited automatically on acquiring any Rune. #1335
- Ensured Crucible preCreate token logic does not clobber token sizing in Ember vistas. #1355
- Verified and fixed removal of the Unaware status at the start of an actor's turn. #1361
- Removed the Falling status when a GM closes a Fall dialog without performing the fall. #1362
- Fixed an error when an action involving time advancement (such as Rest or Recovery) and the action itself both try to expire the same elapsing effect. #1363
- Fixed animation desync issues between the size animation and the hitbox shader. #1364
- Repaired the framework that grants a threat-rank focus bonus to adversaries, so Elite (+1) and Boss (+2) once again receive expanded focus pools. #1370
- Fixed the Kinesis physical damage dropdown continuing to appear and affect spell damage after the actor loses access to the Kinesis rune. #1372
- Fixed War Machine trainings.
- Enabled Patient Deflection and Unarmed Blocking to work when only natural weapons are present.
- Swapped the mainhand requirement for mechanical on the Salvo action.