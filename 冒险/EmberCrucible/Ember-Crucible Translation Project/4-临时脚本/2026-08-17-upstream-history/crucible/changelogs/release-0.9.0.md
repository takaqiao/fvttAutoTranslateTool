![crucible-text](https://github.com/user-attachments/assets/a7b698dc-328c-4919-b758-1567ace8d206)

# Crucible Version 0.9.0 - V14 Alpha 1
Hello everyone, welcome to a **major** Crucible update that brings the system into the world of Foundry Virtual Tabletop version 14 (now in User Testing, not yet Stable). 

> [!IMPORTANT]
> This release is usable ONLY with Version 14, to continue using Crucible with Version 13 remain on system version `0.8.9`.

## Contributors
Many thanks to the following contributors who added features or improvements as part of this release!
- @roth-michael
- @Plushtoast

## Highlights
- Comprehensive V14 compatibility and updates
- New core V14 managed approach to Active Effect duration and expiration automation
- Core v14 migration of Measured Templates to Scene Regions with a brand new region placement workflow as part of Action usage.
- Core v14 updates to use Planned Movement so that actions that involve movement can be planned as part of Action usage.
- New targeting UX and workflows for AoE and movement-based target models.
- A new informative turn start or turn end confirmation message which communicates resource and active effect changes that happen during combat.
- Use the new core V14 VFX framework to start enabling built-in crucible system visual effects attached to actions. A prototype of this system is now added for strike attacks made with a projectile weapon.

## New Features
- Replaced all `MeasuredTemplate` usage with `RegionDocument`, using the new `canvas.regions.placeRegion` API for the region placement workflow. `CrucibleAction#region` is now a reference to a `RegionDocument` instance. Persisted regions are eagerly created on action use, visible only to the acting user and GMs, and revealed to all players when the action is confirmed. Region placement now enforces elevation ranges and wall constraints https://github.com/foundryvtt/crucible/issues/609
- Replaced Crucible's custom round/turn ActiveEffect duration tracking and expiry logic with the V14 native `ActiveEffectRegistry` expiry system. All action effects now track duration in rounds using `turnStart` and `turnEnd` expiry events. The `CrucibleActionConfig` sheet provides an updated UI for defining effect durations https://github.com/foundryvtt/crucible/issues/607
- Implemented planned movement for actions that use the movement tag. `ActionUseDialog` gains a "Plan Movement" button; submission is blocked until a path is committed. The planned path is automatically constrained to the action's range. The action cost display updates after a path is chosen. Actions involving planned movement can be reversed to restore prior movement history in active combat https://github.com/foundryvtt/crucible/issues/612
- Implemented a framework for actions to designate whether their associated region is persistent or ephemeral https://github.com/foundryvtt/crucible/issues/717
- Placed regions are now kept visible on the canvas for the duration of the `ActionUseDialog`, giving players a clear visual of the AOE while reviewing targets before committing https://github.com/foundryvtt/crucible/issues/718
- Planned movement is now restricted to the action's maximum range budget, preventing a token from pathing beyond what the action cost allows https://github.com/foundryvtt/crucible/issues/719
- Added a new movement action target type that identifies targets based on tokens whose space intersects the planned movement path. Converted Gesture: Step and related movement actions to use this target type https://github.com/foundryvtt/crucible/issues/726
- action.usage.movement can now be used to customize the movement requirements of an action. New action tags allow actions to modify the movement budget cost per foot, enabling abilities like hasted movement https://github.com/foundryvtt/crucible/issues/727
- Targets are now acquired and highlighted in real-time during both the region placement drag and the movement planning workflow, providing immediate feedback on which tokens will be affected https://github.com/foundryvtt/crucible/issues/739
- Updated the `CrucibleHitBoxShader` visual to use distinct colors for tokens that are visible but not targeted versus tokens that are actively targeted, making the distinction unambiguous at a glance https://github.com/foundryvtt/crucible/issues/740
- Added a client-side setting to enable (default) or disable built-in VFX animations when actions are confirmed https://github.com/foundryvtt/crucible/issues/741
- The Action Use dialog now enforces that a region must actually be placed before the action can be submitted, preventing incomplete region-targeted actions https://github.com/foundryvtt/crucible/issues/712
- Converted Initiative summary and Turn Change summary cards to Handlebars templates https://github.com/foundryvtt/crucible/issues/738
- Added Embed support for physical item types (weapon, armor, accessory, consumable, tool, loot) and iconic spells, allowing them to be embedded in journal pages https://github.com/foundryvtt/crucible/issues/431
- The "Request Roll" button on the standard check dialog is now hidden from players, as it is a GM-only workflow https://github.com/foundryvtt/crucible/pull/723

## API Changes
- Replaced all usages of `game.i18n.localize` and `game.i18n.format` throughout the system with the new `_loc()` global shorthand, and localized all remaining user-facing error and warning strings https://github.com/foundryvtt/crucible/issues/693
- Completed a system-wide compatibility pass for Foundry V14 syntax changes, API deprecations, and other breaking changes, including YAML source data migrations https://github.com/foundryvtt/crucible/issues/690

## Bug Fixes
- Fix an issue where actor sheets were not re-rendered when an active effect's duration was updated, and restore the previously active canvas layer after the region placement workflow
ends https://github.com/foundryvtt/crucible/issues/711
- Fix region anchoring and maximum distance snapping during the region placement drag so that constrained shapes correctly stay within the allowed range
https://github.com/foundryvtt/crucible/pull/737
- Fix a bug where several targeting workflows incorrectly called user.targets.clear() instead of canvas.tokens.setTargets([]), causing inconsistent targeting state
- Fix an issue where the upstream V14 parameter order in placeRegion callbacks was not accounted for, breaking the onMove constraint handlers
- Fix an issue where combatants with a turnNumber of 0 were not handled correctly in turn-change workflows
- Fix an issue where weapons that need to be reloaded could still be selected for strike actions