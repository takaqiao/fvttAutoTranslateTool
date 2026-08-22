/**
 * 「上游硬编码串」× 「我们盖住了多少」。
 *
 * 覆盖判据用的是**发布中的真身** `translateText` / `translateNotification`，
 * 并且把所有作用域表并成一张喂进去 ⇒ 这是**覆盖率的上界**
 * （真实运行时每棵子树只拿到其中一张，所以实际生效面只会更小、不会更大）。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs";
const s = fs.readFileSync(SRC, "utf8");
const h = path.join(process.cwd(), "_cov.mjs");
// `translateText` 文件里已经 export 过，再列一次是 SyntaxError
const NAMES = ["translateNotification", "DIALOG_UI", "EMBER_WINDOW_UI",
  "TOKEN_MAKER_UI", "TOKEN_MAKER_PARTS", "MOOD_PANEL", "ATTUNEMENTS", "DIALOG_TITLES", "EXACT"];
fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once(){}, on(){} };\n"
  + s.replace("import * as SELFCHECK from './ember-cn-selfcheck.mjs';",
    "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck(){}, keyLiveness(){} };")
  + "\nexport { " + NAMES.join(", ") + " };\n", "utf8");
const M = await import(pathToFileURL(h).href);
fs.unlinkSync(h);

const SCOPED = {...M.DIALOG_UI, ...M.EMBER_WINDOW_UI, ...M.TOKEN_MAKER_UI,
                ...M.TOKEN_MAKER_PARTS, ...M.MOOD_PANEL, ...M.ATTUNEMENTS, ...M.DIALOG_TITLES};
console.log(`并表后作用域表 ${Object.keys(SCOPED).length} 键 · 全局 EXACT ${Object.keys(M.EXACT).length} 键`);

const covered = (str) => M.translateText(str, SCOPED) !== str
  || M.translateNotification(str) !== str;

const hbs = JSON.parse(fs.readFileSync("hbs_scan.json", "utf8"));
const js = JSON.parse(fs.readFileSync("js_scan.json", "utf8"));

const groups = {
  "ember 模板·文本节点": Object.keys(hbs.ember.texts),
  "ember 模板·属性": Object.keys(hbs.ember.attrs),
  "ember JS·展示字段": Object.keys(js.ember.literal),
  "ember JS·通知": Object.keys(js.ember.notif),
  "crucible 模板·全部": [...Object.keys(hbs.crucible.texts), ...Object.keys(hbs.crucible.attrs)],
  "crucible JS·展示字段": Object.keys(js.crucible.literal),
};
const out = {};
console.log("");
for (const [name, list] of Object.entries(groups)) {
  const uniq = [...new Set(list)];
  const yes = uniq.filter(covered);
  const no = uniq.filter((x) => !covered(x));
  out[name] = { total: uniq.length, covered: yes.length, missing: no };
  const pct = uniq.length ? Math.round(yes.length / uniq.length * 100) : 0;
  console.log(`${name.padEnd(22)} 唯一 ${String(uniq.length).padStart(5)}  已盖 ${String(yes.length).padStart(5)} (${String(pct).padStart(3)}%)  仍缺 ${String(no.length).padStart(5)}`);
}
fs.writeFileSync("coverage.json", JSON.stringify(out, null, 1), "utf8");
console.log("\n→ coverage.json");
