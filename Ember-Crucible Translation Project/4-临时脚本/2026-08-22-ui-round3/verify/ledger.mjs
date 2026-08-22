/**
 * 覆盖账：1512 条「玩家真会看到的部件显示名」在**改前 / 改后**各被盖住多少。
 * 改前那一侧走的是发布中的 `translateText(name, TOKEN_MAKER_WINDOW_UI)`（v1.1.28 原样）；
 * 改后那一侧再叠上 `TOKEN_MAKER_PARTS` 这张只在 `.choice.layer .part` 上跑的表。
 *
 * ⚠ 前置自证：全集 1512 条从 recon/table.json 的同一份枚举来（图集 5649 帧 → 1535 末段
 *   → 扣 23 条上游未使用的残留），条数对不上当场失败。
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const STUB = "import * as SELFCHECK from './ember-cn-selfcheck.mjs';";
const die = (m) => { console.error("ledger: " + m); process.exit(2); };
async function load(src, tag, names) {
  const s = fs.readFileSync(src, "utf8");
  const add = [];
  for (const n of names) {
    const own = new RegExp("\\n(?:const|function) " + n + "[ (=]").test(s);
    const exp = new RegExp("\\nexport (?:const|function) " + n + "[ (=]").test(s);
    if (!own && !exp) die(`${tag}: 找不到 ${n}`);
    if (!exp) add.push(n);
  }
  const h = path.join(process.cwd(), `_l_${tag}.mjs`);
  fs.writeFileSync(h, "globalThis.Hooks = globalThis.Hooks ?? { once() {}, on() {} };\n"
    + s.replace(STUB, "const SELFCHECK = { SUBTREE_SELECTORS: [], registerSelfCheck() {}, keyLiveness() {} };")
    + `\nexport { ${add.join(", ")} };\n`, "utf8");
  return import(pathToFileURL(h).href);
}

const [, , F_OLD, F_NEW, F_UNIV, F_OUT] = process.argv;
const OLD = await load(F_OLD, "old", ["TOKEN_MAKER_WINDOW_UI", "translateText"]);
const NEW = await load(F_NEW, "new", ["TOKEN_MAKER_WINDOW_UI", "translateText", "TOKEN_MAKER_PARTS"]);

const univ = JSON.parse(fs.readFileSync(F_UNIV, "utf8"));
const shown = univ.shown_display;
if (!Array.isArray(shown) || shown.length !== 1512) die(`全集条数 ${shown?.length} ≠ 1512`);
console.log(`前置 OK  全集 ${shown.length} 条`);

const oldHit = shown.filter(k => OLD.translateText(k, OLD.TOKEN_MAKER_WINDOW_UI) !== k);
const newHit = shown.filter(k => (k in NEW.TOKEN_MAKER_PARTS)
  || NEW.translateText(k, NEW.TOKEN_MAKER_WINDOW_UI) !== k);
const still = shown.filter(k => !newHit.includes(k));
const wrongBefore = oldHit.filter(k => (k in NEW.TOKEN_MAKER_PARTS)
  && NEW.TOKEN_MAKER_PARTS[k] !== OLD.translateText(k, OLD.TOKEN_MAKER_WINDOW_UI));

console.log(`改前盖住 ${oldHit.length} 条（其中 ${wrongBefore.length} 条**译错**：${wrongBefore.join(" · ")}）`);
console.log(`改后盖住 ${newHit.length} 条 ⇒ 缺口 ${shown.length - oldHit.length} → 补 ${newHit.length - oldHit.length} → 仍缺 ${still.length}`);
fs.writeFileSync(F_OUT, JSON.stringify({
  total: shown.length, oldHit: oldHit.length, newHit: newHit.length,
  gapBefore: shown.length - oldHit.length, added: newHit.length - oldHit.length,
  stillMissing: still.length, wrongBefore, oldHitList: oldHit, stillList: still
}, null, 1), "utf8");
