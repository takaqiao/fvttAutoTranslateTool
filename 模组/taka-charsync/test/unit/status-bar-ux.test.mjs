import test from 'node:test';
import assert from 'node:assert/strict';
import {cardStatus, renderStatusBar, endStatusSession} from '../../scripts/char/status-bar.mjs';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

const FLAG = 'taka-charsync';
const escapeHTML = text => String(text).replace(/[&<>"']/g, ch => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[ch]));

function fixture() {
  globalThis.CONST = {DOCUMENT_OWNERSHIP_LEVELS: {OWNER: 3}};
  globalThis.foundry = {utils: {deepClone: structuredClone, escapeHTML}};
  globalThis.Actor = {implementation: class {
    constructor(doc) { this.doc = doc; }
    toObject() { return structuredClone(this.doc); }
  }};
  globalThis.game = {user: {id: 'player', isGM: false}, actors: new Map()};
  const doc = (id, type = 'character') => ({
    _id: id, name: id, type, ownership: {player: 3}, items: [], effects: [],
    system: {details: {level: {value: 6}}, skills: {}},
    flags: {[FLAG]: {base: {kind: 'main', hash: 'main-hash'}}},
  });
  const actor = source => {
    const result = {id: source._id, name: source.name, type: source.type, _source: source,
      toObject: () => structuredClone(source)};
    game.actors.set(result.id, result);
    return result;
  };
  const hero = actor(doc('hero'));
  const meta = {cards: {hero: {world: 'camp', mainId: 'mainHero', charOwner: 'player'}},
    users: {player: {camp: 'mainPlayer'}}, worlds: {camp: {schema: 'schema'}}, char: {schema: 'schema'}};
  const mainLatest = {hash: 'main-hash', level: 6, doc: hero.toObject(), applied: null};
  const copy = (kind, changes = {}) => ({
    kind, copyId: `${kind}-copy`, rev: 100, cardId: 'hero', level: 6, targetWorld: 'camp',
    submittedBy: 'player', companionOf: null, companions: [], withdrawn: false, ...changes,
  });
  const status = (changes = {}) => cardStatus(hero, {meta, mainLatest, copies: [], ...changes});
  return {doc, actor, hero, meta, mainLatest, copy, status};
}

for (const kind of ['level', 'force', 'new']) {
  test(`${kind} submissions at the current level wait for GM approval`, () => {
    const {meta, copy, status} = fixture();
    if (kind === 'new') delete meta.cards.hero;
    const result = status({copies: [copy(kind)]});
    assert.equal(result.state, 'submitted');
    assert.match(result.text, /待 GM/);
    assert.equal(result.pending.length, 1);
  });
}

test('pending copies exclude withdrawn, applied and undone revisions without hiding a newer revision', () => {
  const {mainLatest, copy, status} = fixture();
  mainLatest.applied = {kind: 'copy', copyId: 'applied-copy', rev: 100};
  mainLatest.doc.flags[FLAG].undone = ['copy:ignored-copy:100'];
  const result = status({copies: [
    copy('level', {copyId: 'withdrawn-copy', withdrawn: true}),
    copy('level', {copyId: 'applied-copy'}),
    copy('force', {copyId: 'ignored-copy'}),
    copy('force', {copyId: 'ignored-copy', rev: 101}),
  ]});
  assert.equal(result.state, 'submitted');
  assert.deepEqual(result.pending.map(row => [row.copyId, row.rev]), [['ignored-copy', 101]]);
});

test('editing only a companion enables the owner current-level submission and names its changes', () => {
  const {doc, actor, meta, status} = fixture();
  const companion = actor(doc('pet', 'familiar'));
  companion._source.system.master = {id: 'hero'};
  meta.cards.pet = {world: 'camp', mainId: 'mainPet', charOwner: 'player', companionOf: 'hero'};
  const companionMain = {hash: 'main-hash', doc: companion.toObject()};
  companion._source.items.push({_id: 'ability', type: 'action', name: '飞行', system: {}});
  const result = status({companions: [{actor: companion, mainLatest: companionMain}]});
  assert.equal(result.state, 'draft');
  assert.ok(result.actions.includes('提交本级修改'));
  assert.match(result.details, /pet/);
  assert.match(result.details, /飞行/);
  assert.equal(result.relations[0].actorId, 'pet');
});

test('an unmapped familiar links to its owner and cannot be submitted on its own', () => {
  const {doc, actor, meta} = fixture();
  const pet = actor(doc('pet', 'familiar'));
  pet._source.system.master = {id: 'hero'};
  const result = cardStatus(pet, {meta, copies: [], mainLatest: null});
  assert.deepEqual(result.actions, []);
  assert.match(result.text, /随主人/);
  assert.deepEqual(result.relations.map(row => [row.actorId, row.role, row.automatic]), [['hero', 'master', true]]);
});

test('a card without draft changes has no normal submission action', () => {
  const {status} = fixture();
  const result = status();
  assert.equal(result.state, 'synced');
  assert.ok(!result.actions.some(label => label.startsWith('提交')));
});

test('inventory-only changes do not appear as ordinary pending build changes', () => {
  const {hero, status} = fixture();
  hero._source.items.push({_id: 'test-gear', type: 'equipment', name: '测试装备', system: {quantity: 99}});
  hero._source.system.attributes = {hp: {value: 99}};
  const result = status();
  assert.equal(result.draft, false);
  assert.doesNotMatch(result.details, /测试装备|当前血量/);
  assert.ok(!result.actions.some(label => label.startsWith('提交')));
  assert.ok(result.actions.includes('整卡同步到主服'));
});

for (const kind of ['level', 'force']) {
  test(`${kind} post-submission notices follow its inventory scope`, () => {
    const {hero, actor, copy, status} = fixture();
    const submitted = copy(kind);
    const frozen = hero.toObject();
    frozen._id = submitted.copyId;
    frozen.flags[FLAG] = {copy: {...submitted, name: 'hero'}};
    actor(frozen);
    hero._source.items.push({_id: 'test-gear', type: 'equipment', name: '测试装备', system: {quantity: 99}});
    assert.equal(status({copies: [submitted]}).editedAfterSubmission, kind === 'force');
  });
}

for (const change of [
  {img: 'tokenizer/new-avatar.webp'},
  {prototypeToken: {texture: {src: 'tokenizer/new-token.webp'}}},
  {prototypeToken: {ring: {subject: {texture: 'tokenizer/new-subject.webp', scale: 0.8}}}},
]) {
  test(`appearance-only changes enable current-level submission: ${JSON.stringify(change)}`, () => {
    const {hero, status} = fixture();
    Object.assign(hero._source, change);
    const result = status();
    assert.equal(result.state, 'draft');
    assert.ok(result.actions.includes('提交本级修改'));
    assert.match(result.details, /头像|棋子/);
  });
}

test('a higher-level draft labels its submission as an upgrade', () => {
  const {hero, status} = fixture();
  hero._source.system.details.level.value = 7;
  assert.ok(status().actions.includes('提交 7 级升级'));
});

test('an unresolved conflict keeps the recovery action even when no other draft difference remains', () => {
  const {hero, status} = fixture();
  hero._source.flags[FLAG].conflicts = {list: [{scope: 'actor', path: 'system.details.biography', winner: 'char'}]};
  const result = status();
  assert.equal(result.state, 'conflict');
  assert.ok(result.actions.includes('放弃草稿，恢复成主服版'));
});

test('pending submissions retain the current working-card changes after further edits', () => {
  const {hero, copy, status} = fixture();
  hero._source.system.details.biography = {backstory: '修改后的背景'};
  const result = status({copies: [copy('level')]});
  assert.equal(result.state, 'submitted');
  assert.equal(result.draft, true);
  assert.match(result.details, /biography|背景|传记/);
  assert.ok(result.actions.includes('提交本级修改'));
});

for (const kind of ['level', 'force', 'new']) {
  test(`${kind} submissions distinguish the frozen version from later working-card edits`, () => {
    const {hero, actor, meta, copy, status} = fixture();
    if (kind === 'new') delete meta.cards.hero;
    const submitted = copy(kind);
    const frozen = hero.toObject();
    frozen._id = submitted.copyId;
    frozen.name = 'hero〔已提交 6 级〕';
    frozen.flags[FLAG] = {copy: {...submitted, name: 'hero'}};
    actor(frozen);
    assert.equal(status({copies: [submitted]}).editedAfterSubmission, false);
    hero._source.system.details.biography = {backstory: '提交后的修改'};
    assert.equal(status({copies: [submitted]}).editedAfterSubmission, true);
  });
}

for (const [field, value] of [['name', '修改后的名字'], ['img', 'icons/new-portrait.webp']]) {
  test(`a level submission notices later ${field} changes outside the build-field filter`, () => {
    const {hero, actor, copy, status} = fixture();
    const submitted = copy('level');
    const frozen = hero.toObject();
    frozen._id = submitted.copyId;
    frozen.flags[FLAG] = {copy: {...submitted, name: 'hero'}};
    actor(frozen);
    assert.equal(status({copies: [submitted]}).editedAfterSubmission, false);
    hero._source[field] = value;
    assert.equal(status({copies: [submitted]}).editedAfterSubmission, true);
  });
}

test('post-submission edits to a companion are visible on the owner card', () => {
  const {doc, actor, hero, copy, status} = fixture();
  const pet = actor(doc('pet', 'familiar'));
  pet._source.system.master = {id: 'hero'};
  const submitted = copy('level', {companions: [{copyId: 'pet-copy', rev: 100, sourceId: 'pet'}]});
  const frozenHero = hero.toObject();
  frozenHero._id = submitted.copyId;
  frozenHero.flags[FLAG] = {copy: {...submitted, name: 'hero'}};
  actor(frozenHero);
  const frozenPet = pet.toObject();
  frozenPet._id = 'pet-copy';
  frozenPet.system.master.id = submitted.copyId;
  frozenPet.flags[FLAG] = {copy: {...submitted, name: 'pet', copyId: 'pet-copy', cardId: 'pet', companionOf: submitted.copyId}};
  actor(frozenPet);
  pet._source.items.push({_id: 'ability', type: 'action', name: '飞行', system: {}});
  assert.equal(status({copies: [submitted], companions: [{actor: pet, mainLatest: null}]}).editedAfterSubmission, true);
});

for (const [kind, label] of [['force', /整卡同步/], ['new', /新卡/], ['level', /级/]]) {
  test(`a frozen ${kind} copy identifies its submission type`, () => {
    const {hero, copy, status} = fixture();
    hero._source.flags[FLAG].copy = copy(kind);
    const result = status();
    assert.equal(result.state, 'copy');
    assert.match(result.text, label);
  });
}

function dom() {
  class Element {
    constructor(tagName) {
      this.tagName = tagName;
      this.children = [];
      this.dataset = {};
      this.listeners = new Map();
      this.className = '';
      this.textContent = '';
      this.innerHTML = '';
      this.open = false;
    }
    append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
    prepend(child) { child.parent = this; this.children.unshift(child); }
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
    insertAdjacentHTML(_where, html) { this.innerHTML += html; }
    addEventListener(type, callback) {
      const callbacks = this.listeners.get(type) ?? [];
      callbacks.push(callback);
      this.listeners.set(type, callbacks);
    }
    async dispatch(type) { for (const callback of this.listeners.get(type) ?? []) await callback({target: this}); }
    querySelector(selector) {
      const className = selector.replace(':scope > ', '').replace(/^\./, '');
      return descendants(this).find(element => element.className.split(' ').includes(className)) ?? null;
    }
  }
  globalThis.document = {createElement: tag => new Element(tag)};
  const content = new Element('div');
  content.className = 'window-content';
  const root = new Element('div');
  root.append(content);
  return {content, root};
}

function descendants(root) {
  return root.children.flatMap(child => [child, ...descendants(child)]);
}

const settle = () => new Promise(resolve => setImmediate(resolve));

async function sheetFixture() {
  const {hero, mainLatest, meta, copy} = fixture();
  const {a} = await makeCampaignEnvironment({isolated: false});
  a._source = hero.toObject();
  a._source._id = a.id;
  a.name = '角色';
  a.isOwner = true;
  a.toObject = () => structuredClone(a._source);
  a._source.system.details.biography = {backstory: '草稿背景'};
  meta.cards = {A: {...meta.cards.hero, world: 'a'}};
  meta.worlds = {a: {schema: 'schema', title: '测试团'}};
  meta.ruleSnapshot = ruleSnapshot();
  const actors = [a];
  actors.get = id => actors.find(actor => actor.id === id);
  actors.contents = actors;
  game.actors = actors;
  game.user = {id: 'player', isGM: true};
  const get = game.settings.get.bind(game.settings);
  game.settings.get = (namespace, key, options) => namespace === 'taka-charsync'
    ? 'test-published' : get(namespace, key, options);
  foundry.utils.escapeHTML = escapeHTML;
  foundry.utils.getRoute = value => value;
  const state = {mainLatest, requests: 0, writes: 0, dialogs: [], opened: []};
  a.update = async () => { state.writes++; };
  a.updateSource = () => { state.writes++; };
  foundry.applications.api.DialogV2 = {confirm: async data => { state.dialogs.push(data); return false; }};
  globalThis.fetch = async url => {
    state.requests++;
    return {ok: true, status: 200, json: async () => url.includes('meta.json') ? meta : state.mainLatest};
  };
  const {root, content} = dom();
  const app = {actor: a, options: {classes: []}, element: [root], rendered: true};
  renderStatusBar(app);
  await settle();
  return {app, content, state, actors, copy};
}

test('ordinary changes start collapsed and keep their expansion through sheet rerenders', async () => {
  const {app, content} = await sheetFixture();
  try {
    let details = content.querySelector('.taka-charsync-details');
    assert.ok(details, 'draft changes render inside a details section');
    assert.equal(details.open, false);
    details.open = true;
    await details.dispatch('toggle');
    renderStatusBar(app);
    details = content.querySelector('.taka-charsync-details');
    assert.equal(details.open, true);
  } finally { endStatusSession(app); }
});

test('conflicts remain visible while force and discard stay under secondary actions', async () => {
  const {app, content} = await sheetFixture();
  try {
    game.user.isGM = false;
    app.actor._source.flags[FLAG].conflicts = {list: [{scope: 'actor', path: 'system.details.biography', winner: 'char'}]};
    renderStatusBar(app);
    assert.equal(content.querySelector('.taka-charsync-details'), null);
    assert.match(content.children[0].innerHTML, /冲突/);
    const secondary = content.querySelector('.taka-charsync-secondary');
    assert.equal(secondary.open, false);
    assert.deepEqual(descendants(secondary).filter(element => element.tagName === 'button').map(element => element.textContent),
      ['放弃草稿，恢复成主服版', '整卡同步到主服']);
  } finally { endStatusSession(app); }
});

test('refreshing an open sheet reads current status without catching up or writing the actor', async () => {
  const {app, content, state} = await sheetFixture();
  try {
    game.user.isGM = false;
    state.mainLatest = {...state.mainLatest, hash: 'new-main-hash'};
    const refresh = descendants(content).find(element => element.tagName === 'button' && element.textContent === '刷新状态');
    assert.ok(refresh, 'the open sheet provides a read-only refresh');
    const requestsBefore = state.requests;
    await refresh.dispatch('click');
    await settle();
    assert.ok(state.requests > requestsBefore);
    assert.equal(state.writes, 0);
    assert.equal(content.children[0].dataset.state, 'paused');
  } finally { endStatusSession(app); }
});

test('pending rows open frozen copies and ask before withdrawing them from the original card', async () => {
  const {app, content, actors, copy, state} = await sheetFixture();
  try {
    game.user.isGM = false;
    const submitted = copy('level', {cardId: app.actor.id, name: app.actor.name});
    actors.push({id: submitted.copyId, _source: {...app.actor.toObject(), name: '角色〔已提交 6 级〕', flags: {[FLAG]: {copy: submitted}}},
      sheet: {render: () => state.opened.push(submitted.copyId)}});
    renderStatusBar(app);
    const open = descendants(content).find(element => element.tagName === 'button' && element.textContent === '查看提交');
    const withdraw = descendants(content).find(element => element.tagName === 'button' && element.textContent === '撤回');
    assert.ok(open);
    assert.ok(withdraw);
    await open.dispatch('click');
    assert.deepEqual(state.opened, [submitted.copyId]);
    await withdraw.dispatch('click');
    assert.equal(state.dialogs.length, 1);
    assert.match(state.dialogs[0].content, /引擎|发布/);
    assert.equal(state.writes, 0, 'declining confirmation preserves the copy');
  } finally { endStatusSession(app); }
});
