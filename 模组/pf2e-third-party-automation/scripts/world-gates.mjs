// Which worlds each feature runs in. World ids are written here and nowhere
// else: to open or merge a campaign, edit this file instead of forking the module.
const AV3='-',SOG='sog',FOTRP='pnvfcgjbf2cjp7gz',FORTRESS='ujx5r8oipw7ercdr',ECNU='avecnu';
// QA worlds; they do not exist on the live server.
const TEAM_QA='team-automation-qa2',PILGRIM_QA='sog-pilgrim-qa';

// The shield chain, audited campaign by campaign.
const SHIELD_TEAMS=[AV3,SOG,FOTRP,ECNU,TEAM_QA];
export const DESTRUCTIVE_BLOCK_WORLDS=new Set(SHIELD_TEAMS);
export const RUNE_TRANSFER_WORLDS=new Set(SHIELD_TEAMS);
export const NATIVE_IWR_STATUS_WORLDS=new Set(SHIELD_TEAMS);
export const SHIELD_BLOCK_WORLDS=new Set([...SHIELD_TEAMS,FORTRESS]);
export const SHIELD_WALL_COMPAT_WORLDS=new Set([...SHIELD_TEAMS,FORTRESS]);
export const REACTION_CHECKER_SETTING_WORLDS=new Set([...SHIELD_TEAMS,FORTRESS]);

// The fortress campaign only.
export const DEFENSIVE_ADVANCE_WORLDS=new Set([FORTRESS]);
export const DESPERATE_PRAYER_WORLDS=new Set([FORTRESS]);
export const ELECTRICITY_SETTLEMENT_WORLDS=new Set([FORTRESS]);
export const FORCE_BARRAGE_WORLDS=new Set([FORTRESS]);
export const FORTRESS_RULE_COMPAT_WORLDS=new Set([FORTRESS]);
export const HALFLING_LUCK_WORLDS=new Set([FORTRESS]);
export const ROARING_APPLAUSE_WORLDS=new Set([FORTRESS]);
export const SPIRITUAL_SCAR_WORLDS=new Set([FORTRESS]);
export const GLIMPSE_WORLDS=new Set([FORTRESS,TEAM_QA]);

// SoG pilgrim rewards.
export const SOG_PILGRIM_WORLDS=new Set([SOG,PILGRIM_QA]);
