/**
 * 抢回器的**端到端**离线验 —— 补的正是 0.9.16 漏掉的那一段。
 *
 * ⚠⚠ 为什么非有它不可：第三轮的离线复刻器验的是 `reclaimTranslations`（纯函数），
 *   表是**直接从磁盘读**的，于是 `ownLanguagePath()` / `loadOwnTranslationsSync()`
 *   —— 真正会坏、而且真的坏了的那一段 —— **一次都没被执行过**。
 *   判据那一侧同样只钉了 `xhr.open('GET', url, false);` 这一行，没钉 `url` 是怎么拼的。
 *   这是本项目登记的空转形态 (h) 的又一变体：**验的对象不是会坏的那个对象**。
 *   ⇒ 本文件从 `registerLangReclaim()` 这个真入口进，走完钩子 → 取路径 → 取值 → 回写全链。
 *
 * ⚠ 语料形状取自**出问题的那台机器**（2026-08-22 VPS 控制台实测），不是本机的 foundry_chn：
 *     顶层 TOKEN   = object，内容是**别的模块**的 `TABS`（我们的 LABELS/MOVEMENT 已被冲掉）
 *     顶层 WARNING = string "警告"
 *   这一点是本轮的教训之一：拿本机语料推 VPS 的病因，第二次栽了。
 */
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const FOUNDRY = "C:/Program Files/Foundry Virtual Tabletop/resources/app";
const PLUGIN = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件";
const U = await import(pathToFileURL(`${FOUNDRY}/common/utils/helpers.mjs`).href);

let pass = 0, fail = 0;
const bad = [];
const ck = (cond, label, detail) => {
  if (cond) { pass++; console.log("  PASS  " + label); }
  else { fail++; bad.push({ label, detail }); console.log("  FAIL  " + label + (detail === undefined ? "" : "  " + JSON.stringify(detail))); }
};

const CN_TEXT = fs.readFileSync(`${PLUGIN}/lang/cn.json`, "utf8");
const CN = JSON.parse(CN_TEXT);
const VICTIMS = Object.keys(CN).filter(k => ["TOKEN", "WARNING"].includes(k.split(".")[0]));
if (VICTIMS.length !== 42) throw new Error(`前置自证挂：受害键应为 42，实得 ${VICTIMS.length}`);
console.log(`前置 OK  自家 cn.json ${Object.keys(CN).length} 键，其中 TOKEN/WARNING 命名空间 ${VICTIMS.length} 条`);

// ── 假服务器：**只有**正确的那个 URL 存在 ────────────────────────────────
const SERVER = { "/modules/crucible-cn/lang/cn.json": CN_TEXT };
function makeXhrClass(log) {
  return class {
    open(method, url) { this._url = url; log.push(url); }
    send() {
      const hit = SERVER[this._url];
      if (hit === undefined) { this.status = 404; this.responseText = "<!DOCTYPE html>"; }
      else { this.status = 200; this.responseText = hit; }
    }
  };
}

// ── VPS 上实测到的翻译表形状 ────────────────────────────────────────────
function buildVpsTranslations() {
  const t = {};
  U.mergeObject(t, U.expandObject(CN), { inplace: true });           // crucible-cn 先并
  U.mergeObject(t, U.expandObject({ WARNING: "警告", TOKEN: "指示物" }), { inplace: true }); // 裸串顶掉
  U.mergeObject(t, U.expandObject({                                   // 后面的模块又把 TOKEN 建成对象
    "TOKEN.TABS.deathEffects": "死亡特效",
    "TOKEN.TABS.gridAwareAuras": "光环",
  }), { inplace: true });
  return t;
}

function run({ declaredPath, moduleId = "crucible-cn" }) {
  const log = [];
  const translations = buildVpsTranslations();
  const pkg = { languages: new Set([{ lang: "cn", name: "中文", path: declaredPath }]) };
  globalThis.XMLHttpRequest = makeXhrClass(log);
  globalThis.foundry = { utils: U };
  globalThis.game = { modules: { get: (id) => (id === moduleId ? pkg : undefined) }, i18n: { translations } };
  const hooks = {};
  globalThis.Hooks = { once(name, fn) { hooks[name] = fn; }, on() {} };
  return { translations, pkg, log, hooks };
}

const MOD = await import(pathToFileURL(`${PLUGIN}/lang-reclaim.js`).href);

console.log("\n══════ ① 已安装形态（VPS 上的真实形状）══════");
{
  const { translations, pkg, log, hooks } = run({ declaredPath: "modules/crucible-cn/lang/cn.json" });
  MOD.registerLangReclaim();
  hooks.i18nInit();
  const st = MOD.getReclaimState();
  ck(log.length === 1 && log[0] === "/modules/crucible-cn/lang/cn.json",
    "① 请求的 URL 正好是 /modules/crucible-cn/lang/cn.json（不重复前缀）", log);
  ck(st.phase === "done", "① phase = done", st.phase + " / " + st.error);
  ck(st.report?.reclaimed.length === 42, "① 抢回 42 条", st.report?.reclaimed.length);
  ck(U.getProperty(translations, "TOKEN.MOVEMENT.COST.Forced") === "强制",
    "① 复验 TOKEN.MOVEMENT.COST.Forced = 强制", U.getProperty(translations, "TOKEN.MOVEMENT.COST.Forced"));
  ck(U.getProperty(translations, "TOKEN.LABELS.Allies") === "{allies} 盟友",
    "① 复验 TOKEN.LABELS.Allies", U.getProperty(translations, "TOKEN.LABELS.Allies"));
  ck(typeof U.getProperty(translations, "WARNING.NoParty") === "string" &&
     U.getProperty(translations, "WARNING.NoParty").includes("主要队伍"), "① 复验 WARNING.NoParty");
  ck(U.getProperty(translations, "TOKEN.TABS.deathEffects") === "死亡特效",
    "① **没有误伤**别的模块建在 TOKEN 下的键");
  ck(typeof pkg.api?.getReclaimState === "function", "① 账本挂上了 module.api");
}

console.log("\n══════ ② 清单形态（未安装包 / 上游哪天改回去）══════");
{
  const { log, hooks } = run({ declaredPath: "lang/cn.json" });
  MOD.registerLangReclaim();
  hooks.i18nInit();
  ck(log[0] === "/modules/crucible-cn/lang/cn.json", "② 归一到同一个 URL", log);
  ck(MOD.getReclaimState().report?.reclaimed.length === 42, "② 同样抢回 42 条");
}

console.log("\n══════ ③ 回归对照：0.9.16 的**旧**拼法必须打不中 ══════");
{
  const oldUrl = U.getRoute(`modules/crucible-cn/${"modules/crucible-cn/lang/cn.json"}`);
  ck(oldUrl === "/modules/crucible-cn/modules/crucible-cn/lang/cn.json",
    "③ 旧拼法复现出用户报的那个 URL", oldUrl);
  ck(SERVER[oldUrl] === undefined, "③ 该 URL 在假服务器上 404 ⇒ 本验若在 0.9.16 上跑必红");
}

console.log("\n══════ ④ 取不到时必须如实记账（不许静默当成没被顶）══════");
{
  const { hooks } = run({ declaredPath: "lang/WRONG.json" });
  MOD.registerLangReclaim();
  hooks.i18nInit();
  const st = MOD.getReclaimState();
  ck(st.phase === "error", "④ phase = error", st.phase);
  ck(/HTTP 404/.test(st.error) && /WRONG\.json/.test(st.error),
    "④ 报错同时带上最终 URL 与 languages[].path 的声明值", st.error);
}

console.log("\n══════ ⑤ 没被顶时是 no-op ══════");
{
  const log = [];
  const translations = {};
  U.mergeObject(translations, U.expandObject(CN), { inplace: true });   // 没有肇事者
  const pkg = { languages: new Set([{ lang: "cn", path: "modules/crucible-cn/lang/cn.json" }]) };
  globalThis.XMLHttpRequest = makeXhrClass(log);
  globalThis.foundry = { utils: U };
  globalThis.game = { modules: { get: () => pkg }, i18n: { translations } };
  const hooks = {};
  globalThis.Hooks = { once(n, f) { hooks[n] = f; }, on() {} };
  const before = JSON.stringify(translations);
  MOD.registerLangReclaim();
  hooks.i18nInit();
  const st = MOD.getReclaimState();
  ck(st.report?.reclaimed.length === 0, "⑤ 写入 0 次", st.report?.reclaimed.length);
  ck(JSON.stringify(translations) === before, "⑤ 逐键快照零差异");
}

console.log("\n══════ 合计 " + pass + " 通过 / " + fail + " 失败 ══════");
if (fail) { console.log(JSON.stringify(bad, null, 1)); process.exit(1); }
