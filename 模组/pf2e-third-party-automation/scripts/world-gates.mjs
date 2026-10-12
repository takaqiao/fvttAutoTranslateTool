// 各功能在哪些世界启用。全模组只在这里写世界 ID：
// 新开一个团、或把一个团并进来，改这一个文件，不要复制模组。
const AV3='-',SOG='sog',FOTRP='pnvfcgjbf2cjp7gz',FORTRESS='ujx5r8oipw7ercdr',ECNU='avecnu';
// 测试世界，主服上没有。
const TEAM_QA='team-automation-qa2',PILGRIM_QA='sog-pilgrim-qa';

// 憎恶三团、肆季、赤凰、ECNU：盾挡链这一组按团验收过的功能。
const SHIELD_TEAMS=[AV3,SOG,FOTRP,ECNU,TEAM_QA];
export const DESTRUCTIVE_BLOCK_WORLDS=new Set(SHIELD_TEAMS);
export const RUNE_TRANSFER_WORLDS=new Set(SHIELD_TEAMS);
export const DISRUPT_PREY_WORLDS=new Set(SHIELD_TEAMS);
export const NATIVE_IWR_STATUS_WORLDS=new Set(SHIELD_TEAMS);
export const SHIELD_BLOCK_WORLDS=new Set([...SHIELD_TEAMS,FORTRESS]);
export const SHIELD_WALL_COMPAT_WORLDS=new Set([...SHIELD_TEAMS,FORTRESS]);
export const REACTION_CHECKER_SETTING_WORLDS=new Set([...SHIELD_TEAMS,FORTRESS]);

// 亵渎堡垒。
export const DEFENSIVE_ADVANCE_WORLDS=new Set([FORTRESS]);
export const DESPERATE_PRAYER_WORLDS=new Set([FORTRESS]);
export const ELECTRICITY_SETTLEMENT_WORLDS=new Set([FORTRESS]);
export const FORCE_BARRAGE_WORLDS=new Set([FORTRESS]);
export const FORTRESS_RULE_COMPAT_WORLDS=new Set([FORTRESS]);
export const HALFLING_LUCK_WORLDS=new Set([FORTRESS]);
export const ROARING_APPLAUSE_WORLDS=new Set([FORTRESS]);
export const SPIRITUAL_SCAR_WORLDS=new Set([FORTRESS]);
export const GLIMPSE_WORLDS=new Set([FORTRESS,TEAM_QA]);

// 肆季朝圣奖励。
export const SOG_PILGRIM_WORLDS=new Set([SOG,PILGRIM_QA]);
