import {OWN_FLAG} from '../../scripts/lib/rules.mjs';

const clone = value => structuredClone(value);
export const refOf = copy => ({kind: 'copy', copyId: copy.copyId, rev: copy.rev});

export function actorDoc(id, extra = {}) {
  return {_id: id, name: id, type: 'character', ownership: {default: 0, player: 3}, folder: null,
    flags: {}, system: {details: {level: {value: 2}, xp: {value: 450, max: 1000}}, attributes: {hp: {value: 12, max: 20}}},
    items: [], effects: [], ...extra};
}

export function makeCopy(cardId, kind = 'level', extra = {}) {
  return {copyId: `${cardId}-copy`, cardId, rev: 200, kind, level: 2, name: cardId,
    targetWorld: 'campaign', rulesHash: 'a'.repeat(64), pred: {kind: 'main', hash: `${cardId}-base`},
    submittedBy: 'char-player', withdrawn: false, companions: [], companionOf: null, ...extra};
}

export function applyEnvironment() {
  const files = new Map();
  const requests = [], writes = [], messages = [];
  const failures = {create: null, update: null, delete: null};
  const actors = new Map();
  actors.invalidDocumentIds = new Set();
  Object.defineProperty(actors, 'contents', {get: () => [...actors.values()]});
  actors.find = predicate => actors.contents.find(predicate);
  actors.filter = predicate => actors.contents.filter(predicate);
  const meta = {
    cards: {}, working: {}, users: {'char-player': {campaign: 'player'}}, published: [],
    char: {schema: 'schema'}, worlds: {campaign: {schema: 'schema'}},
    ruleSnapshot: {version: 2, enforcedSince: 100, rules: {campaign: {}}, hashes: {campaign: 'a'.repeat(64)}},
  };
  files.set('meta.json', meta);

  class Document {
    constructor(data) { this._source = clone(data); this.refresh(); }
    refresh() {
      this.id = this._source._id; this.name = this._source.name; this.type = this._source.type;
      this.system = this._source.system; this.ownership = this._source.ownership;
      this.folder = this._source.folder ? {id: this._source.folder} : null;
      for (const key of ['items', 'effects']) {
        this[key] = new Map((this._source[key] ?? []).map(entry => [entry._id, entry]));
        this[key].invalidDocumentIds = new Set();
      }
    }
    toObject() { return clone(this._source); }
    updateSource() {}
    async update(data, options = {}) {
      writes.push({operation: 'update', id: this.id, data: clone(data)});
      if (failures.update?.(this, data)) throw new Error(`update failed: ${this.id}`);
      for (const [key, value] of Object.entries(data)) {
        const path = key.split('.');
        const field = path.pop();
        let target = this._source;
        for (const segment of path) target = target[segment] ??= {};
        if (field.startsWith('-=')) delete target[field.slice(2)];
        else target[field] = clone(value);
      }
      this.refresh();
      return this;
    }
    async delete() {
      writes.push({operation: 'delete', id: this.id});
      if (failures.delete?.(this)) throw new Error(`delete failed: ${this.id}`);
      actors.delete(this.id);
      return this;
    }
    static async create(data) {
      writes.push({operation: 'create', id: data._id});
      if (failures.create?.(data)) throw new Error(`create failed: ${data._id}`);
      if (actors.has(data._id)) throw new Error(`duplicate actor: ${data._id}`);
      const actor = new Document(data);
      actors.set(actor.id, actor);
      return actor;
    }
    static async deleteDocuments(ids) {
      return Promise.all(ids.map(id => actors.get(id)?.delete()));
    }
  }
  globalThis.Actor = {implementation: Document};
  globalThis.CONFIG = {Item: {documentClass: Document}, ActiveEffect: {documentClass: Document}};
  globalThis.CONST = {DOCUMENT_OWNERSHIP_LEVELS: {NONE: 0, OWNER: 3}};
  globalThis.game = {
    world: {id: 'campaign'}, settings: {get: () => 'published'}, actors,
    users: {activeGM: {isSelf: true}, filter: () => []}, combats: [],
    folders: {find: () => ({id: 'archive', type: 'Actor', getFlag: () => 'archive'})},
  };
  globalThis.foundry = {
    utils: {deepClone: clone, getRoute: value => value, randomID: () => 'free-master',
      escapeHTML: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')},
    applications: {apps: {FilePicker: {implementation: {
      async upload(_storage, dir, file) { files.set(`${dir}/${file.name}`.slice('published/'.length), JSON.parse(await file.text())); return {path: `${dir}/${file.name}`}; },
      async createDirectory() {},
    }}}},
    data: {validation: {DataModelValidationError: class extends Error {}}},
  };
  globalThis.ChatMessage = {implementation: {getWhisperRecipients: () => [{id: 'gm'}],
    async create(message) { messages.push(message); return message; }}};
  globalThis.ui = {notifications: {warn() {}, error() {}, info() {}}};
  Math.clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  globalThis.fetch = async url => {
    requests.push(url);
    const path = url.slice('published/'.length);
    return {ok: files.has(path), status: files.has(path) ? 200 : 404, json: async () => clone(files.get(path))};
  };
  const addActor = doc => { const actor = new Document(doc); actors.set(actor.id, actor); return actor; };
  function publishCopy(copy, doc = actorDoc(copy.cardId)) {
    const path = `copies/${copy.cardId}.json`;
    files.set(path, [...(files.get(path) ?? []).filter(other => other.copyId !== copy.copyId), clone(copy)]);
    files.set(`history/copy-${copy.copyId}-${copy.rev}.json`, {...clone(doc), _id: copy.copyId,
      name: `${copy.name}〔已提交〕`, flags: {...doc.flags, [OWN_FLAG]: {copy: clone(copy)}}});
    meta.published.push(`copy:${copy.copyId}:${copy.rev}`);
    return copy;
  }
  function mapped(cardId, id = `main-${cardId}`, extra = {}) {
    meta.cards[cardId] = {world: 'campaign', mainId: id, charOwner: 'char-player', ...extra};
    const doc = actorDoc(id);
    files.set(`history/main-${cardId}-base.json`, clone(doc));
    return addActor(doc);
  }
  return {files, requests, writes, messages, failures, actors, meta, addActor, publishCopy, mapped};
}
