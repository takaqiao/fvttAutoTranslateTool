/**
 * 离线复现面板 D 档「键活性」，并核对第二十四轮三份定性报告给出的豁免名单。
 *
 * 复现的是 ember-cn-selfcheck.mjs:567-666 的 keyLiveness：
 *   语料 = modules/ember/scripts/ember.mjs + systems/<sys>/<sys>-compiled.mjs
 *   判据 = 裸 `corpus.includes(key)`（无词边界、无空白折叠）
 * 只跑 crucible 世界这一侧（onlyOn:'dnd5e' 的表在 crucible 世界本来就 skip）。
 *
 * 用法：node fix_probe_corpus.mjs <harness.mjs>
 */
import fs from "node:fs";

const EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs";
const CRUCIBLE = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/crucible-compiled.mjs";

const corpus = fs.readFileSync(EMBER, "utf8") + "\n" + fs.readFileSync(CRUCIBLE, "utf8");
const has = (s) => corpus.includes(s);

const mod = await import(process.argv[2]);
const T = mod.__SELFCHECK_TABLES;

let grand = 0, grandMiss = 0;
const rows = [];
for (const [name, spec] of Object.entries(T)) {
  const table = spec?.table ?? spec;
  const kind = spec?.kind ?? "literal";
  const onlyOn = spec?.onlyOn ?? null;
  if (onlyOn && onlyOn !== "crucible") { rows.push([name, "skip(onlyOn)", 0, []]); continue; }
  if (Array.isArray(table)) { rows.push([name, "skip(正则表)", 0, []]); continue; }
  const keys = Object.keys(table ?? {});
  if (!keys.length) { rows.push([name, "skip(空表)", 0, []]); continue; }
  if (kind === "composed" || kind === "data") { rows.push([name, `skip(${kind})`, keys.length, []]); continue; }
  if (kind === "absent-by-design") {
    const found = keys.filter(has);
    grand += keys.length;
    rows.push([name, found.length ? `warn(反向:找到${found.length})` : "ok(反向)", keys.length, found]);
    continue;
  }
  const miss = keys.filter((k) => !has(k));
  grand += keys.length; grandMiss += miss.length;
  rows.push([name, miss.length ? `warn(缺${miss.length})` : "ok", keys.length, miss]);
}

for (const [name, status, n, list] of rows) {
  console.log(`${status.padEnd(22)} ${String(n).padStart(4)}  ${name}`);
  for (const k of list) console.log(`      · ${JSON.stringify(k)}`);
}
console.log(`\n合计核 ${grand} 键，正向缺 ${grandMiss}，反向命中 ${rows.filter(r => r[1].startsWith("warn(反向")).reduce((a, r) => a + r[3].length, 0)}`);
console.log(`面板口径「上游查无此串」报文条数 = ${grandMiss}`);
