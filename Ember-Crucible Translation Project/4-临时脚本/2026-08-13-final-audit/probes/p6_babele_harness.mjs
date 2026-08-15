/**
 * Headless harness: drive babele 2.9.1's REAL mapping pipeline with this
 * project's registered layer, on synthetic documents, and print what the
 * runtime actually writes back.
 *
 * Only foundry globals babele touches are shimmed. mergeObject reproduces
 * Foundry's defaults (recursive, insertKeys, overwrite).
 */
const B = 'file:///C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele/script';

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
  utils: {
    mergeObject,
    getProperty,
    deepClone: (v) => (v === undefined || v === null ? v : structuredClone(v)),
    parseUuid: () => null,
    Collection: Map,
  },
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

const projectMappings = await import('file:///C:/Users/Taka/Desktop/fvtt/Ember-Crucible%20Translation%20Project/2-Crucible%E6%B1%89%E5%8C%96%E6%8F%92%E4%BB%B6/babele-mappings.js');

const converterRegistry = new ConverterRegistry({
  ...Converters.legacyRegistrations(),
  document: new DocumentConverter(),
  structured: new StructuredDataConverter(),
  referencedDocumentField: new ReferencedDocumentFieldConverter(),
});
const identityExtractors = new IdentityExtractorRegistry({
  range: (d) => { const [s, e] = d?.range ?? []; return Number.isInteger(s) && Number.isInteger(e) ? `${s}-${e}` : null; },
});
const documentMappings = new DocumentMappings(undefined, {
  registeredMappings: [projectMappings.DOCUMENT_MAPPINGS],
  identityExtractors,
  converterRegistry,
});

console.log('effective Actor mapping keys :', Object.keys(documentMappings.current().Actor));
console.log('effective Item  mapping keys :', Object.keys(documentMappings.current().Item));
console.log('effective Scene mapping keys :', Object.keys(documentMappings.current().Scene));
console.log('effective RegionBehavior     :', JSON.stringify(documentMappings.current().RegionBehavior));
console.log('effective TableResult        :', JSON.stringify(documentMappings.current().TableResult));

/* ---- experiment 1: tokenName ---- */
const actorPack = new MappedCompendium(
  { id: 'x.actors', name: 'actors', type: 'Actor', packageName: 'x', packageType: 'module' },
  {
    label: 'X',
    entries: {
      'Corvana Vortest': { name: '科尔瓦娜·沃泰斯特 Corvana Vortest', tokenName: '法师' },
    },
  },
  { documentMappings, converterRegistry, translationStrategies: [] },
);

const actor = {
  _id: 'aaaaaaaaaaaaaaaa',
  name: 'Corvana Vortest',
  type: 'adversary',
  prototypeToken: { name: 'Mage', displayName: 0 },
  system: { description: 'x' },
};
const out = actorPack.translate(structuredClone(actor));
console.log('\n[tokenName experiment]');
console.log('  source  name / prototypeToken.name :', JSON.stringify(actor.name), '/', JSON.stringify(actor.prototypeToken.name));
console.log('  entry   name / tokenName           :', '"科尔瓦娜·沃泰斯特 Corvana Vortest"', '/', '"法师"');
console.log('  RESULT  name                       :', JSON.stringify(out.name));
console.log('  RESULT  prototypeToken.name        :', JSON.stringify(out.prototypeToken.name));
console.log('  -> tokenName key was', out.prototypeToken.name === '法师' ? 'USED' : 'IGNORED');

/* ---- experiment 2: does an unmapped RegionBehavior variant still resolve? ---- */
const rbPack = new MappedCompendium(
  { id: 'x.rb', name: 'rb', type: 'RegionBehavior', packageName: 'x' },
  { entries: { 'Scrolling Text': { name: '滚动文字', text: '灼光！' } } },
  { documentMappings, converterRegistry, translationStrategies: [] },
);
const rb = { _id: 'b'.repeat(16), name: 'Scrolling Text', type: 'displayScrollingText', system: { text: 'Searing Light!' } };
const rbOut = rbPack.translate(structuredClone(rb));
console.log('\n[RegionBehavior displayScrollingText experiment]');
console.log('  RESULT name        :', JSON.stringify(rbOut.name));
console.log('  RESULT system.text :', JSON.stringify(rbOut.system.text), '<- babele DOES look up a `text` key here');
