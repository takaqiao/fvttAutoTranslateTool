/**
 * 冒烟：patchWeatherLabels 是否真的改到了 `slices[*].config.weather`。
 *
 * 桩件复刻上游真实形状（ember.mjs:119637 起的 weather 配置 + :21825 的 getConfig 取值链），
 * 并**同时**放一个假的 `slice.weather`（只有 elevation、没有 label）来复现旧路径的空转。
 *
 * 跑法：node "<本文件>"
 */
const MOD = new URL("../../../1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs", import.meta.url).href;

const hooks = {};
globalThis.Hooks = { once(n, f) { (hooks[n] ??= []).push(f); }, on(n, f) { (hooks[n] ??= []).push(f); } };

// 上游真实形状：label 在 slice.config.weather[type]，档位在 .strengths[N].label
const surface = {
  config: {
    weather: {
      clear: { id: "clear", label: "Clear" },
      rain: {
        id: "rain", label: "Rain",
        strengths: {
          1: { label: "Drizzle" }, 2: { label: "Rain" }, 3: { label: "Storm" }, 4: { label: "Tempest" }
        }
      },
      fog: {
        id: "fog", label: "Fog",
        strengths: { 1: { label: "Mist" }, 2: { label: "Fog" }, 3: { label: "Dense Fog" } }
      },
      wind: {
        id: "wind", label: "Wind",
        strengths: {
          0: { label: "Calm" }, 1: { label: "Breeze" }, 2: { label: "Windy" },
          3: { label: "Gale" }, 4: { label: "Squall" }
        }
      }
    }
  },
  // 旧路径指向的东西：Vista 场景的 weather，只有 elevation，没有 label
  weather: { elevation: -50 }
};

globalThis.ember = { scenes: { region: { slices: { surface } } } };
globalThis.game = { modules: { get: () => ({ active: true }) }, i18n: { translations: {} }, settings: { get: () => false, register() {} }, system: { id: "crucible" } };
globalThis.CONFIG = {};
globalThis.ui = { windows: {} };
globalThis.foundry = { utils: {}, canvas: {}, applications: {} };

const warns = [];
const origWarn = console.warn;
console.warn = (...a) => warns.push(a.join(" "));
await import(MOD);
for (const fn of hooks.ready ?? []) { try { await fn(); } catch { /* 别的补丁缺桩件 */ } }
console.warn = origWarn;

let pass = 0, fail = 0;
const check = (l, c, e = "") => { if (c) { pass++; console.log(`  PASS  ${l}`); } else { fail++; console.log(`  FAIL  ${l}  ${e}`); } };

const w = surface.config.weather;
console.log("① 顶层天气名");
check("clear.label", w.clear.label === "晴朗", w.clear.label);
check("rain.label", w.rain.label === "雨", w.rain.label);
check("wind.label", w.wind.label === "风", w.wind.label);
console.log("② 档位名（悬浮提示上真正显示的那一层）");
check("rain.strengths[1] Drizzle", w.rain.strengths[1].label === "细雨", w.rain.strengths[1].label);
check("rain.strengths[4] Tempest", w.rain.strengths[4].label === "狂风暴雨", w.rain.strengths[4].label);
check("fog.strengths[3] Dense Fog", w.fog.strengths[3].label === "浓雾", w.fog.strengths[3].label);
console.log("③ 风力档位 —— 风向箭头的悬浮提示就是这一档拼出来的");
for (const [k, want] of [[0, "无风"], [1, "微风"], [2, "有风"], [3, "疾风"], [4, "狂风"]])
  check(`wind.strengths[${k}]`, w.wind.strengths[k].label === want, w.wind.strengths[k].label);
console.log("④ 不该炸也不该乱动");
check("旧路径 slice.weather 没被误伤", surface.weather.elevation === -50 && !("label" in surface.weather));
check("没有发出「一条都没改到」的告警", !warns.some(x => x.includes("一条都没改到")), warns.join(" | ").slice(0, 160));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
