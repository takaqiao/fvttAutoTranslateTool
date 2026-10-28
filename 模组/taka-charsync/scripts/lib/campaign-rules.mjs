import {OWN_FLAG} from './rules.mjs';

export const CAMPAIGN_RULE_KEYS = [
  'pf2e.campaignFeats', 'pf2e.campaignFeatSections', 'pf2e.campaignType', 'pf2e.mythic',
  'pf2e.automaticBonusVariant', 'pf2e.freeArchetypeVariant', 'pf2e.gradualBoostsVariant',
  'pf2e.proficiencyVariant', 'pf2e.proficiencyUntrainedModifier', 'pf2e.proficiencyTrainedModifier',
  'pf2e.proficiencyExpertModifier', 'pf2e.proficiencyMasterModifier', 'pf2e.proficiencyLegendaryModifier',
  'pf2e.staminaVariant',
];

const record = value => !!value && typeof value === 'object' && !Array.isArray(value);
const identifier = value => typeof value === 'string' && value.trim().length > 0;
const source = document => document?._source ?? document;
const documentId = document => document?.id ?? document?._id;
const lookup = (collection, id) => collection?.get?.(id) ?? collection?.find?.(row => documentId(row) === id);
const unavailable = reason => ({ok: false, world: null, reason});
const booleanRules = new Set([
  'pf2e.campaignFeats', 'pf2e.freeArchetypeVariant', 'pf2e.gradualBoostsVariant',
  'pf2e.proficiencyVariant', 'pf2e.staminaVariant',
]);

function validRuleValue(key, value) {
  if (value === undefined) return false;
  if (record(value) && Object.hasOwn(value, '$charsyncDefault')) {
    return value.$charsyncDefault === true && Object.keys(value).length === 1;
  }
  if (booleanRules.has(key)) return typeof value === 'boolean';
  if (/^pf2e\.proficiency\w+Modifier$/.test(key)) return Number.isFinite(value);
  if (key === 'pf2e.campaignFeatSections') return Array.isArray(value);
  if (['pf2e.campaignType', 'pf2e.mythic', 'pf2e.automaticBonusVariant'].includes(key)) return identifier(value);
  return true;
}

/** Return a structural error without retaining an older, apparently usable profile. */
export function validateRuleSnapshot(snapshot) {
  if (!record(snapshot) || snapshot.version !== 2) return 'invalid-snapshot-version';
  if (!Number.isFinite(snapshot.generatedAt) || !Number.isFinite(snapshot.enforcedSince)
    || snapshot.enforcedSince <= 0 || snapshot.generatedAt < snapshot.enforcedSince) return 'invalid-snapshot-time';
  if (!['rules', 'hashes', 'cardWorld', 'folderWorld'].every(key => record(snapshot[key]))) return 'invalid-snapshot';
  for (const mapping of [snapshot.cardWorld, snapshot.folderWorld]) {
    if (Object.entries(mapping).some(([id, world]) => !identifier(id) || !identifier(world))) return 'invalid-assignment';
  }
  const prefixKeys = new Set(Object.values(snapshot.rules).flatMap(table => record(table)
    ? Object.keys(table).filter(key => key.startsWith('pf2e-team-plus-feats.') || key.startsWith('pf2e-sf2e-extra-feat-slots.'))
    : []));
  for (const [world, table] of Object.entries(snapshot.rules)) {
    if (!identifier(world) || !record(table)) return 'invalid-profile';
    if (!/^[a-f0-9]{64}$/.test(snapshot.hashes[world] ?? '')) return `invalid-rule-hash:${world}`;
    if ([...CAMPAIGN_RULE_KEYS, ...prefixKeys].some(key => !Object.hasOwn(table, key))) return `incomplete-profile:${world}`;
    for (const [key, value] of Object.entries(table)) {
      if (!validRuleValue(key, value)) return `invalid-rule-value:${world}:${key}`;
    }
  }
  return null;
}

/** Resolve explicit provenance only. Frozen copies retain the campaign chosen at submission. */
export function resolveActorCampaign(actor, {snapshot, actors, folders}) {
  const walk = (document, visiting) => {
    const row = source(document);
    if (!row) return unavailable('missing-actor');
    const id = documentId(document) ?? row._id;
    if (!id) return unavailable('missing-actor-id');
    if (visiting.has(id)) return unavailable('assignment-cycle');
    const next = new Set([...visiting, id]);
    const flags = row.flags?.[OWN_FLAG] ?? {};
    if (flags.copy) {
      return identifier(flags.copy.targetWorld)
        ? {ok: true, world: flags.copy.targetWorld, reason: null}
        : unavailable('missing-copy-target');
    }
    if (flags.backupOf) return walk(lookup(actors, flags.backupOf), next);
    const worlds = new Set();
    if (snapshot?.cardWorld?.[id]) worlds.add(snapshot.cardWorld[id]);
    if ((row.type ?? document.type) === 'familiar') {
      const masterId = row.system?.master?.id;
      if (!masterId) return unavailable('missing-master');
      const master = walk(lookup(actors, masterId), next);
      if (!master.ok) return master;
      worlds.add(master.world);
    }
    let folder = row.folder ?? document.folder;
    const seenFolders = new Set();
    while (folder) {
      const folderId = typeof folder === 'string' ? folder : documentId(folder);
      if (!folderId) return unavailable('missing-folder');
      if (seenFolders.has(folderId)) return unavailable('folder-cycle');
      seenFolders.add(folderId);
      const mapped = snapshot?.folderWorld?.[folderId];
      if (mapped) worlds.add(mapped);
      const folderDoc = typeof folder === 'string' ? lookup(folders, folderId) : folder;
      if (!folderDoc) return unavailable('missing-folder');
      folder = source(folderDoc)?.folder ?? null;
    }
    if (worlds.size > 1) return unavailable('assignment-conflict');
    if (worlds.size === 0) return unavailable('missing-assignment');
    return {ok: true, world: [...worlds][0], reason: null};
  };
  return walk(actor, new Set());
}
