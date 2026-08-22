/**
 * `crucibleDescription` 的形状回测 —— 补的正是 2026-08-22 那次导入崩溃暴露的盲区。
 *
 * ⚠ 旧的 2769 条全包回测**没覆盖到「源字段不存在」这一支**：采集器只收了
 *   `system` 里已经有 `description` 的条目，`value === undefined` 一次都没跑到。
 *   而那一支恰恰是唯一能由**我们自己**造出「对象位置上放了个裸字符串」的路径。
 */
import { pathToFileURL } from "node:url";
const F = "C:/Program Files/Foundry Virtual Tabletop/resources/app";
globalThis.foundry = { utils: await import(pathToFileURL(`${F}/common/utils/helpers.mjs`).href) };
const { crucibleDescription: f } = await import(pathToFileURL(
  "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/babele-mappings.js").href);

let pass = 0, fail = 0;
const shape = (v) => v === undefined ? "undefined" : v === null ? "null"
  : typeof v === "string" ? "string" : "object{" + Object.keys(v).sort().join(",") + "}";
const ck = (label, got, want) => {
  const ok = shape(got) === want;
  ok ? pass++ : fail++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label.padEnd(52)} → ${shape(got)}${ok ? "" : "  期望 " + want}`);
};

console.log("① 源是字符串（crucible 多数 item 类型）—— 形状必须还字符串");
ck("源 string · 译 string", f("EN", "中文"), "string");
ck("源 string · 译 {public}", f("EN", { public: "中文" }), "string");
ck("源 string · 译 undefined（不译）", f("EN", undefined), "string");

console.log("\n② 源是 {public,private}（装备类）—— 形状必须还对象");
ck("源 {public,private} · 译 {public}", f({ public: "EN", private: "" }, { public: "中文" }), "object{private,public}");
ck("源 {public,private} · 译 string", f({ public: "EN", private: "" }, "中文"), "object{private,public}");
ck("源 {public,private} · 译 {public,private}", f({ public: "EN", private: "P" }, { public: "中", private: "私" }), "object{private,public}");

console.log("\n③ 源是 {value,chat}（dnd5e / 别人的包）—— 只能碰 .value，不许塞 public");
ck("源 {value,chat} · 译 string", f({ value: "EN", chat: "" }, "中文"), "object{chat,value}");
ck("源 {value,chat} · 译 {public}（不该误入）", f({ value: "EN", chat: "" }, { public: "中文" }), "object{chat,value}");
const r5 = f({ value: "EN", chat: "" }, { public: "中文" });
ck("  ↑ 且不许长出 public 键", ("public" in r5) ? {} : r5, "object{chat,value}");

console.log("\n④ **源不存在** —— 本轮收紧的那一支：不许猜成裸字符串");
ck("源 undefined · 译 {public}", f(undefined, { public: "中文" }), "object{public}");
ck("源 undefined · 译 {public,private}", f(undefined, { public: "中", private: "私" }), "object{private,public}");
ck("源 null · 译 {public}", f(null, { public: "中文" }), "object{public}");
ck("源 undefined · 译 string（译文自己就是字符串）", f(undefined, "中文"), "string");
ck("源 undefined · 译 undefined", f(undefined, undefined), "undefined");
ck("源 undefined · 译 {}（没有可用字段）", f(undefined, {}), "undefined");

console.log("\n⑤ 回归对照：旧写法在 ④ 的第一格会吐字符串（这一验若在旧版上跑必红）");
const oldWay = (value, translation) => {
  const isStr = (v) => typeof v === "string" && v.trim().length > 0;
  if (typeof value === "string" || value === undefined || value === null) {
    if (isStr(translation)) return translation;
    if (isStr(translation?.public)) return translation.public;
    return value;
  }
};
ck("旧写法 源 undefined · 译 {public} → 裸字符串", oldWay(undefined, { public: "中文" }), "string");

console.log(`\n══════ ${pass} 通过 / ${fail} 失败 ══════`);
process.exit(fail ? 1 : 0);
