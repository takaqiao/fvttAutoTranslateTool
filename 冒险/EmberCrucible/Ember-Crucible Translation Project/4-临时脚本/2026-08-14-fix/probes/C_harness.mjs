/**
 * Drive babele 2.9.1's REAL mapping pipeline with the PATCHED project layer
 * (probes/cand/) on documents lifted verbatim out of the real LevelDB packs,
 * and print what the runtime actually writes back.
 *
 * Shims only the foundry globals babele touches. mergeObject reproduces
 * Foundry's defaults (recursive / insertKeys / overwrite, arrays are leaves).
 */
const B = 'file:///C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele/script';
const CAND = 'file:///C:/Users/Taka/Desktop/fvtt/Ember-Crucible%20Translation%20Project/4-%E4%B8%B4%E6%97%B6%E8%84%9A%E6%9C%AC/2026-08-14-fix/probes/cand';

const isPlain = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
function mergeObject(original, other = {}, { inplace = true } = {}) {
  const target = inplace ? original : structuredClone(original);
  for (const [k, v] of Object.entries(other ?? {})) {
    if (isPlain(target[k]) && isPlain(v)) mergeObject(target[k], v, { inplace: true });
    else target[k] = isPlain(v) || Array.isArray(v) ? structuredClone(v) : v;
  }
  return target;
}
const getProperty = (o, p) => String(p).split('.').reduce((a, k) => (a == null ? a : a[k]), o);
globalThis.foundry = {
  utils: { mergeObject, getProperty, deepClone: (v) => (v == null ? v : structuredClone(v)), parseUuid: () => null, Collection: Map },
};
globalThis.Hooks = { callAll() {}, on() {}, once() {} };
globalThis.game = { settings: { get: () => false } };
globalThis.CONFIG = { debug: {} };

const { DocumentMappings } = await import(`${B}/mapping/document-mappings.js`);
const { ConverterRegistry } = await import(`${B}/converter/converter-registry.js`);
const { IdentityExtractorRegistry } = await import(`${B}/identity/identity-extractor-registry.js`);
const { Converters } = await import(`${B}/converter/converters.js`);
const { DocumentConverter } = await import(`${B}/converter/document-converter.js`);
const { StructuredDataConverter } = await import(`${B}/converter/structured-data-converter.js`);
const { ReferencedDocumentFieldConverter } = await import(`${B}/converter/referenced-document-field-converter.js`);
const { MappedCompendium } = await import(`${B}/compendium/mapped-compendium.js`);

const { EMBER_LAYER } = await import(`${CAND}/mappings.mjs`);
const { PROJECT_CONVERTERS } = await import(`${CAND}/runtime-converters.js`);

const converterRegistry = new ConverterRegistry({
  ...Converters.legacyRegistrations(),
  document: new DocumentConverter(),
  structured: new StructuredDataConverter(),
  referencedDocumentField: new ReferencedDocumentFieldConverter(),
  ...PROJECT_CONVERTERS,
});
const identityExtractors = new IdentityExtractorRegistry({
  range: (d) => { const [s, e] = d?.range ?? []; return Number.isInteger(s) && Number.isInteger(e) ? `${s}-${e}` : null; },
});
// EMBER_LAYER is a plain data object -> exactly what generate_runtime.mjs bakes
// into babele-mappings.js and what babele-register.js feeds registerMapping().
const documentMappings = new DocumentMappings(undefined, {
  registeredMappings: [JSON.parse(JSON.stringify(EMBER_LAYER))],
  identityExtractors,
  converterRegistry,
});

const eff = documentMappings.current();
console.log('effective Actor keys         :', Object.keys(eff.Actor).join(', '));
console.log('effective Scene keys         :', Object.keys(eff.Scene).join(', '));
console.log('effective RegionBehavior     :', JSON.stringify(eff.RegionBehavior));
console.log('effective JournalEntryPage   :', Object.keys(eff.JournalEntryPage).join(', '));

const pack = (type, entries) => new MappedCompendium(
  { id: `x.${type}`, name: type, type, packageName: 'x', packageType: 'module' },
  { label: 'X', entries },
  { documentMappings, converterRegistry, translationStrategies: [] },
);

let fails = 0;
const check = (label, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) fails += 1;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${label}\n        got  ${JSON.stringify(got)}\n        want ${JSON.stringify(want)}`);
};

/* ---------- 1. tokenName is now honoured, and does not fall back to name ---------- */
console.log('\n[1] Actor.tokenName');
{
  const p = pack('Actor', {
    'Pallid Ultra Drake': { name: '苍白极巨龙 Pallid Ultra Drake', tokenName: '苍白龙兽' },
    'Untranslated Token': { name: '未译 Untranslated Token' },
  });
  const a = { _id: 'a'.repeat(16), name: 'Pallid Ultra Drake', type: 'adversary', prototypeToken: { name: 'Pallid Drake' } };
  const out = p.translate(structuredClone(a));
  check('name translated', out.name, '苍白极巨龙 Pallid Ultra Drake');
  check('prototypeToken.name uses tokenName (NOT the actor name)', out.prototypeToken.name, '苍白龙兽');

  const b = { _id: 'b'.repeat(16), name: 'Untranslated Token', type: 'adversary', prototypeToken: { name: 'Keep Me English' } };
  const out2 = p.translate(structuredClone(b));
  check('missing tokenName key leaves the English source alone', out2.prototypeToken.name, 'Keep Me English');
}

/* ---------- 2. encounter token override names ---------- */
console.log('\n[2] JournalEntryPage.ember.questEvent -> system.encounter.tokens');
{
  const p = pack('JournalEntryPage', {
    'A Miner Matter': {
      name: '矿工之事 A Miner Matter',
      encounterTokens: { 'Brevin Villager': '布雷文村民', 'Friendly Ooze': '友善软泥' },
    },
  });
  const page = {
    _id: 'c'.repeat(16),
    name: 'A Miner Matter',
    type: 'ember.questEvent',
    system: {
      overview: 'ov',
      encounter: {
        tokens: [
          { levelId: 'L1', actors: [
            { actor: 'Actor.aaa', number: 1, tokenData: { name: 'Brevin Villager', x: 10, texture: { scaleX: -1 } } },
            { actor: 'Actor.bbb', number: 2, tokenData: { name: 'Friendly Ooze', elevation: -9 } },
            { actor: 'Actor.ccc', number: 1, tokenData: { name: 'Not In Translation', x: 5 } },
            { actor: 'Actor.ddd', number: 1, tokenData: { x: 7 } },
          ] },
          { levelId: 'L2', actors: [] },
        ],
      },
    },
  };
  const src = structuredClone(page);
  const out = p.translate(structuredClone(page));
  const t = out.system.encounter.tokens;
  check('group 0 actor 0 name', t[0].actors[0].tokenData.name, '布雷文村民');
  check('group 0 actor 0 keeps siblings', [t[0].actors[0].tokenData.x, t[0].actors[0].tokenData.texture], [10, { scaleX: -1 }]);
  check('group 0 actor 0 keeps entry fields', [t[0].actors[0].actor, t[0].actors[0].number], ['Actor.aaa', 1]);
  check('group 0 actor 1 name', t[0].actors[1].tokenData.name, '友善软泥');
  check('untranslated override stays English', t[0].actors[2].tokenData.name, 'Not In Translation');
  check('tokenData without a name is untouched', t[0].actors[3].tokenData, { x: 7 });
  check('empty group survives', t[1], { levelId: 'L2', actors: [] });
  check('group count unchanged', t.length, 2);
  check('SOURCE document not mutated', src.system.encounter.tokens[0].actors[0].tokenData.name, 'Brevin Villager');
  check('page name still translated', out.name, '矿工之事 A Miner Matter');

  const raw = converterRegistry.named('emberEncounterTokens');
  check('extract direction', raw.extract({ value: page.system.encounter.tokens, params: {} }), {
    'Brevin Villager': 'Brevin Villager',
    'Friendly Ooze': 'Friendly Ooze',
    'Not In Translation': 'Not In Translation',
  });
}

/* ---------- 3. region behaviors ---------- */
console.log('\n[3] RegionBehavior subtype variants');
{
  const p = pack('RegionBehavior', {
    'Searing Light': { name: '灼光', text: '灼光！' },
    'Pressure Plate': { name: '压力板', message: '压力板已触发！' },
    'Whirling Blades': {
      name: '旋刃',
      description: '<p>数把旋转的刀刃从近旁墙壁中弹出。</p>',
      effects: { Bleeding: '流血' },
    },
  });
  const st = p.translate({ _id: 'd'.repeat(16), name: 'Searing Light', type: 'displayScrollingText', system: { text: 'Searing Light!', color: '#ffe4a8' } });
  check('displayScrollingText name', st.name, '灼光');
  check('displayScrollingText system.text', st.system.text, '灼光！');
  check('displayScrollingText system.color untouched', st.system.color, '#ffe4a8');

  const tt = p.translate({ _id: 'e'.repeat(16), name: 'Pressure Plate', type: 'ember.trapTrigger', system: { message: 'Pressure Plate Triggered!', once: true } });
  check('ember.trapTrigger name', tt.name, '压力板');
  check('ember.trapTrigger system.message', tt.system.message, '压力板已触发！');

  const ae = p.translate({
    _id: 'f'.repeat(16),
    name: 'Whirling Blades',
    type: 'ember.areaEffect',
    system: {
      description: '<p>Several whirling blades erupt from the nearby wall, scything through the hallway ahead of you.</p>',
      img: 'icons/skills/melee/strikes-sword-scimitar.webp',
      effects: [{ name: 'Bleeding', changes: [], _id: 'g'.repeat(16) }],
    },
  });
  check('ember.areaEffect name', ae.name, '旋刃');
  check('ember.areaEffect system.description', ae.system.description, '<p>数把旋转的刀刃从近旁墙壁中弹出。</p>');
  check('ember.areaEffect system.effects[0].name', ae.system.effects[0].name, '流血');
  check('ember.areaEffect system.effects[0]._id kept', ae.system.effects[0]._id, 'g'.repeat(16));
  check('ember.areaEffect img untouched', ae.system.img, 'icons/skills/melee/strikes-sword-scimitar.webp');

  // a behavior type with no variant must still get its name
  const other = p.translate({ _id: 'h'.repeat(16), name: 'Searing Light', type: 'adjustDarknessLevel', system: { darknessLevel: 0.5 } });
  check('unvaried behavior type still gets base name', other.name, '灼光');
}

/* ---------- 4. scene sounds ---------- */
console.log('\n[4] Scene.sounds');
{
  const p = pack('Scene', { 'Bastion Apex': { name: '堡垒之巅 Bastion Apex', sounds: { 'Tar Pit Bubbles': '焦油坑气泡', Waterfall: '瀑布' } } });
  const s = p.translate({
    _id: 'i'.repeat(16),
    name: 'Bastion Apex',
    sounds: [
      { _id: 'j'.repeat(16), name: 'Tar Pit Bubbles', radius: 20 },
      { _id: 'k'.repeat(16), name: 'Waterfall', radius: 15 },
      { _id: 'l'.repeat(16), name: 'Not Translated', radius: 1 },
    ],
  });
  check('sound 0', s.sounds[0].name, '焦油坑气泡');
  check('sound 0 radius kept', s.sounds[0].radius, 20);
  check('sound 1', s.sounds[1].name, '瀑布');
  check('untranslated sound stays English', s.sounds[2].name, 'Not Translated');
}

console.log(`\n${fails === 0 ? 'ALL CHECKS PASSED' : `${fails} CHECK(S) FAILED`}`);
process.exit(fails === 0 ? 0 : 1);
