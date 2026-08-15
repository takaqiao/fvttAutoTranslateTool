/**
 * probe_global_mapping_blast.mjs  —— 只读探针，不写库
 *
 * 判据（机械化）：
 *   `babele.registerMapping()` 注册的是 **全局** 文档 mapping 层，
 *   babele 2.9.1 把它按 `documentType` 名字合进 DocumentMappings，
 *   优先级 registered(10+) > built-in(0)，且对世界里**所有**被翻译的合集生效
 *   （MappedCompendium 构造时统一取 documentMappings.mappingFor(metadata.type)）。
 *
 *   所以：我们注册的层里，凡是与 babele 内置 defaultMappings **同名同键**
 *   但定义不同的字段，就是「越界改写了别人包的翻译行为」。
 *   这跟种子缺陷同形：闸门看的是 documentType 这个**表面特征**，
 *   而不是「这个包是不是我拥有的」。
 *
 * 输出：
 *   1) 被我们的注册层覆盖掉的 babele 内置字段清单（= 全局爆炸半径）
 *   2) 用真实的 crucibleDescription 跑一次 dnd5e 形状的 Item.system.description，
 *      给出实际后果。
 *
 * 用法： node probe_global_mapping_blast.mjs
 */

import { pathToFileURL } from "node:url";

const FOUNDRY_UTILS = "C:/Program Files/Foundry Virtual Tabletop/resources/app/common/utils/helpers.mjs";
const BABELE = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele/script";
const PROJ = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project";

const utils = await import(pathToFileURL(FOUNDRY_UTILS).href);
globalThis.foundry = { utils };

const { defaultMappings } = await import(pathToFileURL(`${BABELE}/mapping/default-mappings.js`).href);
const { DocumentMappings } = await import(pathToFileURL(`${BABELE}/mapping/document-mappings.js`).href);
const { ConverterRegistry } = await import(pathToFileURL(`${BABELE}/converter/converter-registry.js`).href);
const { IdentityExtractorRegistry } = await import(pathToFileURL(`${BABELE}/identity/identity-extractor-registry.js`).href);

const ember = await import(pathToFileURL(`${PROJ}/1-Ember汉化插件/babele-mappings.js`).href);
const crucible = await import(pathToFileURL(`${PROJ}/2-Crucible汉化插件/babele-mappings.js`).href);

const converterRegistry = new ConverterRegistry();
// babele 内置转换器名（只需占位，探针不执行它们）
for (const n of ["name", "nameCollection", "textCollection", "document", "structured",
                 "referencedDocumentField", "lookup", "tokenizedLookup"]) {
  converterRegistry.register(n, (v) => v);
}
converterRegistry.registerAll(ember.PROJECT_CONVERTERS);

const identityExtractors = new IdentityExtractorRegistry(IdentityExtractorRegistry.defaultExtractors());

function effectiveFor(layers) {
  return new DocumentMappings(defaultMappings, {
    registeredMappings: layers,
    loadedMappings: [],
    identityExtractors,
    converterRegistry,
  });
}

const S = (v) => JSON.stringify(v);

for (const [label, layer] of [["ember_cn_unofficial", ember.DOCUMENT_MAPPINGS],
                              ["crucible-cn", crucible.DOCUMENT_MAPPINGS]]) {
  const eff = effectiveFor([layer]).current();
  console.log(`\n### 注册层 = ${label}`);
  for (const [docType, def] of Object.entries(defaultMappings)) {
    const after = eff[docType] ?? {};
    for (const [field, before] of Object.entries(def)) {
      if (field.startsWith("_")) continue;
      const now = after[field];
      if (S(before) !== S(now)) {
        console.log(`  [覆盖] ${docType}.${field}`);
        console.log(`         babele 内置 : ${S(before)}`);
        console.log(`         被改成      : ${S(now)}`);
      }
    }
  }
}

/* ---- 后果实测：dnd5e 形状的 Item 走 crucibleDescription ---- */
console.log("\n### 后果实测 —— dnd5e 形状 Item.system.description");
const dnd5eItem = { system: { description: { value: "<p>English text</p>", chat: "" } } };
const zh = "<p>中文译文</p>";
const out = ember.crucibleDescription(dnd5eItem.system.description, zh);
console.log("  babele 内置映射 (system.description.value) 本应得到 :",
  S({ value: zh, chat: "" }));
console.log("  被覆盖后 crucibleDescription 实际返回              :", S(out));
console.log("  → .value 是否被翻译 :", out.value === zh);
