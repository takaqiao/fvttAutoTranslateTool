/**
 * Sun Halo 复原全场 —— 演出跑砸/中断时的救场宏
 *   把「Sun Halo 居合连段」隐掉的 token 全部恢复原来的 alpha，
 *   并清掉尸体半身、居合、主动画留下的持续特效和电影黑边。
 */

const FLAG_SCOPE = "world";
const FLAG_KEY   = "sunHaloAlpha";

// 本宏和官方两套宏会留下的持续特效
const NAMES = [
  "SunHaloCorpse|*",
  "IaijutsuStrike*",
  "IaijutsuText",
  "StormDash Crosshair",
  "Rage",
  "Casting *",
];

for (const name of NAMES) {
  await Sequencer.EffectManager.endEffects({ name });
}

// 官方 deathEffect 的半身是按 token 名字命名的（同名 token 会共用一个名字）
for (const t of canvas.tokens.placeables) {
  await Sequencer.EffectManager.endEffects({ name: `${t.document.name}Top` });
  await Sequencer.EffectManager.endEffects({ name: `${t.document.name}Bottom` });
}

let restored = 0;
for (const t of canvas.tokens.placeables) {
  const orig = t.document.getFlag(FLAG_SCOPE, FLAG_KEY);
  if (orig === undefined) continue;
  await t.document.update({ alpha: orig, hidden: false });
  await t.document.unsetFlag(FLAG_SCOPE, FLAG_KEY);
  restored++;
}

await globalThis.eskie?.overlay?.cinemaBars?.stop?.();

ui.notifications.info(`Sun Halo：已复原 ${restored} 个 token，清掉残留特效`);

// 还有东西没清干净就把下面这行取消注释（会清掉场景上所有 Sequencer 持续特效，包括别的宏的）
// await Sequencer.EffectManager.endAllEffects();
