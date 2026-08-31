/**
 * 冒烟：ember-hardcoded-cn.mjs 的 patchRegionBehaviorSchemas。
 *
 * 用最小桩件模拟 CONFIG.RegionBehavior.dataModels，跑 ready 钩子，
 * 然后断言三件事：① 三个子类型的 label/hint/choices 都被改写；
 * ② 上游改过 schema（字段缺失 / 选项多出来）时**跳过并告警**，不炸也不凭空造；
 * ③ 不该动的 `EFFECT.Image` 这类真 i18n 键没被碰。
 *
 * 跑法：node "<本文件>"
 */
const MOD = new URL(
  "../../../1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs",
  import.meta.url
).href;

/* ---------- 桩件 ---------- */

const hooks = {};
globalThis.Hooks = {
  once(n, fn) { (hooks[n] ??= []).push(fn); },
  on(n, fn) { (hooks[n] ??= []).push(fn); }
};

const field = (o) => ({ ...o });
const schema = (fields) => ({
  fields,
  getField(path) {
    let f = fields;
    for (const part of path.split(".")) f = (f.fields ?? f)?.[part];
    return f;
  }
});

const trap = schema({
  once: field({ label: "Once", hint: "Does the trigger automatically disable after firing once?" }),
  locked: field({ label: "Locked", hint: "..." }),
  discovered: field({ label: "Discovered", hint: "..." }),
  behaviors: field({ label: "Triggered Behaviors", hint: "..." }),
  script: field({ label: "Script", hint: "..." }),
  message: field({ label: "Trigger Text", hint: "..." }),
  pause: field({ label: "Pause Game", hint: "..." })
});
const area = schema({
  img: field({ label: "EFFECT.Image" }),                     // 真 i18n 键，不该被动
  description: field({ label: "Chat Message Description", hint: "..." }),
  save: { ...field({ hint: "..." }), fields: { ability: field({ label: "Ability Score" }) } },
  // dc 故意缺失 —— 模拟上游改过 schema
  damage: field({ label: "Damage Formula", hint: "..." }),
  effects: field({ label: "Effect Data", hint: "..." })
});
const foot = schema({
  material: field({
    label: "Material", hint: "...",
    // water 故意缺失 —— 模拟上游换了枚举
    choices: { grass: "Grass", metal: "Metal", stone: "Stone", wood: "Wood" }
  })
});

globalThis.CONFIG = {
  RegionBehavior: {
    dataModels: {
      "ember.trapTrigger": { schema: trap },
      "ember.areaEffect": { schema: area },
      "ember.footstepSurface": { schema: foot }
    }
  }
};
globalThis.game = {
  modules: { get: () => ({ active: true }) },
  i18n: { translations: {}, localize: (k) => k, format: (k) => k },
  settings: { get: () => false, register() {} },
  system: { id: "crucible" }
};
globalThis.ui = { windows: {} };
globalThis.foundry = { utils: {}, canvas: {}, applications: {} };
globalThis.document = undefined;

const warns = [];
const origWarn = console.warn;
console.warn = (...a) => warns.push(a.join(" "));

await import(MOD);
for (const fn of hooks.ready ?? []) { try { await fn(); } catch { /* 别的补丁缺桩件，无所谓 */ } }
console.warn = origWarn;

/* ---------- 断言 ---------- */

let pass = 0, fail = 0;
const check = (label, cond, extra = "") => {
  if (cond) { pass++; console.log(`  PASS  ${label}`); }
  else { fail++; console.log(`  FAIL  ${label}  ${extra}`); }
};

console.log("① 三个子类型都被改写");
check("trapTrigger.once.label", trap.fields.once.label === "仅一次", trap.fields.once.label);
check("trapTrigger.once.hint 是中文", trap.fields.once.hint === "触发一次后，触发器是否自动停用？", trap.fields.once.hint);
check("trapTrigger.pause.label", trap.fields.pause.label === "暂停游戏", trap.fields.pause.label);
check("areaEffect.description.label", area.fields.description.label === "聊天消息描述", area.fields.description.label);
check("areaEffect.save 的嵌套子字段", area.fields.save.fields.ability.label === "属性值", area.fields.save.fields.ability.label);
check("footstepSurface.material.label", foot.fields.material.label === "材质", foot.fields.material.label);
check("footstepSurface 的 choices", foot.fields.material.choices.grass === "草地" && foot.fields.material.choices.wood === "木头",
  JSON.stringify(foot.fields.material.choices));

console.log("② 上游改过 schema 时跳过并告警，不凭空造");
check("缺失的 save.dc 没有被创建", area.fields.save.fields.dc === undefined);
check("缺失的 save.dc 有告警", warns.some(w => w.includes("save.dc")), warns.join(" | ").slice(0, 200));
check("choices 里不存在的 water 没被凭空加上", !("water" in foot.fields.material.choices),
  JSON.stringify(foot.fields.material.choices));
check("多出来的 water 有告警", warns.some(w => w.includes("water")));

console.log("③ 真 i18n 键没被碰");
check("areaEffect.img.label 仍是 EFFECT.Image", area.fields.img.label === "EFFECT.Image", area.fields.img.label);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
