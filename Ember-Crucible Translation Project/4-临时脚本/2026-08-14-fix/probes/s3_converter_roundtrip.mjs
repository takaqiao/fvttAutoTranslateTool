/* Round-trip check for the three proposed converter changes, run against REAL
   pack data with a faithful re-implementation of foundry.utils.mergeObject. */
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
import path from 'path';

/* --- faithful-enough mergeObject (recursive, insertKeys, overwrite; arrays are leaves) --- */
const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
function mergeObject(original, other = {}, { inplace = true } = {}) {
  const target = inplace ? original : structuredClone(original);
  for (const [k, v] of Object.entries(other)) {
    if (isObj(v) && isObj(target[k])) mergeObject(target[k], v, { inplace: true });
    else target[k] = v;
  }
  return target;
}
globalThis.foundry = { utils: { mergeObject } };

const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

/* ---------- proposed: crucibleDescription with the {value, chat} branch ---------- */
function crucibleDescription(value, translation) {
  if (translation === undefined || translation === null) return value;
  if (typeof value === 'string' || value === undefined || value === null) {
    if (isStr(translation)) return translation;
    if (isStr(translation?.public)) return translation.public;
    return value;
  }
  // NEW branch
  if (typeof value.value === 'string' && !('public' in value) && !('private' in value)) {
    const text = isStr(translation) ? translation : (isStr(translation.value) ? translation.value : null);
    return text === null ? value : foundry.utils.mergeObject(value, { value: text }, { inplace: false });
  }
  if (isStr(translation)) return foundry.utils.mergeObject(value, { public: translation }, { inplace: false });
  const patch = {};
  if (isStr(translation.public)) patch.public = translation.public;
  if (isStr(translation.private)) patch.private = translation.private;
  return Object.keys(patch).length ? foundry.utils.mergeObject(value, patch, { inplace: false }) : value;
}

/* ---------- proposed: emberEncounterTokenNames ---------- */
function emberEncounterTokenNames(tokens, translation) {
  if (!Array.isArray(tokens) || !translation || typeof translation !== 'object') return tokens;
  let changed = false;
  const out = tokens.map((token) => {
    if (!token || typeof token !== 'object' || !Array.isArray(token.actors)) return token;
    let hit = false;
    const actors = token.actors.map((actor) => {
      const name = actor?.tokenData?.name;
      if (!isStr(name)) return actor;
      const cn = translation[name];
      if (!isStr(cn) || cn === name) return actor;
      hit = true;
      return foundry.utils.mergeObject(actor, { tokenData: { name: cn } }, { inplace: false });
    });
    if (!hit) return token;
    changed = true;
    return foundry.utils.mergeObject(token, { actors }, { inplace: false });
  });
  return changed ? out : tokens;
}
emberEncounterTokenNames.extract = (tokens) => {
  if (!Array.isArray(tokens)) return undefined;
  const out = {};
  for (const token of tokens) for (const actor of (token?.actors ?? [])) {
    const n = actor?.tokenData?.name;
    if (isStr(n) && !(n in out)) out[n] = n;
  }
  return Object.keys(out).length ? out : undefined;
};

/* ---------- 1. description regression tests ---------- */
console.log('== crucibleDescription ==');
const cases = [
  ['dnd5e {value,chat} + string translation', { value: '<p>English</p>', chat: '' }, '<p>中文</p>'],
  ['crucible {public,private} + string',      { public: '<p>EN</p>', private: '' }, '<p>中文</p>'],
  ['crucible {public,private} + object',      { public: '<p>EN</p>', private: '<p>gm</p>' }, { public: '<p>公</p>', private: '<p>私</p>' }],
  ['crucible plain string + string',          '<p>EN</p>', '<p>中文</p>'],
  ['dnd5e {value,chat} + no translation',     { value: '<p>English</p>', chat: '' }, undefined],
];
for (const [label, v, t] of cases) console.log(' ', label, '->', JSON.stringify(crucibleDescription(v, t)));

/* ---------- 2. encounter token names against real data ---------- */
const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
for (const p of ['crucible-adventure', 'adventure']) {
  const db = new ClassicLevel(path.join(EMBER, p), { createIfMissing: false });
  const advs = [];
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!/);
    if (m && m[1] === 'adventures') advs.push(JSON.parse(v.toString()));
  }
  await db.close();

  let pages = 0, extractedKeys = 0, landed = 0, preserved = true, mutatedSource = false;
  for (const a of advs) for (const j of (a.journal ?? [])) for (const pg of (j.pages ?? [])) {
    const tokens = pg?.system?.encounter?.tokens;
    if (!Array.isArray(tokens)) continue;
    pages++;
    const before = JSON.stringify(tokens);
    const ext = emberEncounterTokenNames.extract(tokens);
    if (!ext) continue;
    extractedKeys += Object.keys(ext).length;
    const fake = Object.fromEntries(Object.keys(ext).map((k) => [k, `【${k}】`]));
    const res = emberEncounterTokenNames(tokens, fake);
    if (JSON.stringify(tokens) !== before) mutatedSource = true;
    for (const t of res) for (const ac of (t?.actors ?? [])) {
      const n = ac?.tokenData?.name;
      if (typeof n === 'string' && n.startsWith('【')) landed++;
      // every non-name key must survive
      const td = ac?.tokenData;
      if (td && n && !('x' in td) && !('texture' in td) && !('elevation' in td) && !('delta' in td)
          && !('_id' in td) && !('rotation' in td) && !('flags' in td) && !('disposition' in td)
          && !('hidden' in td) && !('actorLink' in td) && !('sort' in td) && !('alpha' in td)
          && !('depth' in td) && !('shape' in td) && !('y' in td) && Object.keys(td).length !== 1) preserved = false;
    }
  }
  console.log(`== ${p}: pages ${pages} | extracted keys ${extractedKeys} | landed leaves ${landed} | source mutated: ${mutatedSource} | sibling keys preserved: ${preserved}`);
}
