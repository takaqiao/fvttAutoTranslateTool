import test from 'node:test';
import assert from 'node:assert/strict';
import {createScareToDeath} from '../scripts/scare-to-death.mjs';
import {SCARE_SOURCE} from '../scripts/scare-to-death-rules.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

function assign(document, changes) {
  for (const [path, value] of Object.entries(changes)) {
    const keys = path.split('.'); let current = document;
    for (const key of keys.slice(0, -1)) current = current[key] ??= {};
    current[keys.at(-1)] = structuredClone(value);
  }
}
function fixture({change, phase = 'claim', duplicates = false} = {}) {
  const gm = {id: 'gm', isGM: true, active: true}, user = {id: 'owner', active: true}, saver = {id: 'saver', active: true};
  const users = Object.assign(new Map([gm, user, saver].map(user => [user.id, user])), {activeGM: gm});
  const writes = []; let changed = false, rolls = 0;
  const actor = {id: 'source', uuid: 'Actor.source', type: 'character', flags: {}, items: new Map(), system: {details: {languages: {value: ['common']}}}, testUserPermission: owner => owner === user, getStatistic: () => ({rank: 4}), hasCondition: () => false};
  const patient = id => ({id, uuid: `Actor.${id}`, type: 'character', modeOfBeing: 'living', items: new Map(), flags: {}, system: {details: {languages: {value: ['common']}}}, frightened: 0, testUserPermission: owner => owner === saver, isImmuneTo: () => false, getCondition(name) {return name === 'frightened' && this.frightened ? {value: this.frightened} : null;}, async update(changes) {writes.push({actorUUID: this.uuid, kind: 'actor-update'}); assign(this, changes);}, async createEmbeddedDocuments(_kind, data) {writes.push({actorUUID: this.uuid, kind: 'create'}); const docs = data.map((data, index) => ({...structuredClone(data), id: `effect${index}`, actor: this})); for (const doc of docs) this.items.set(doc.id, doc); await mutate('effect'); return docs;}, async deleteEmbeddedDocuments(_kind, ids) {writes.push({actorUUID: this.uuid, kind: 'delete'}); for (const id of ids) this.items.delete(id);}, async increaseCondition(_condition, {value}) {writes.push({actorUUID: this.uuid, kind: 'frightened'}); this.frightened = value;}});
  const original = patient('original'), replacement = patient('replacement');
  const scene = {id: 'scene', tokens: new Map()};
  const origin = {id: 'source', uuid: 'Scene.scene.Token.source', documentName: 'Token', parent: scene, actor, object: {distanceTo: () => 10}};
  const target = {id: 'target', uuid: 'Scene.scene.Token.target', documentName: 'Token', parent: scene, actor: original, object: {}};
  scene.tokens.set(origin.id, origin); scene.tokens.set(target.id, target);
  actor.getActiveTokens = () => [origin]; original.getActiveTokens = () => [target]; replacement.getActiveTokens = () => [target];
  const item = {id: 'feat', uuid: `${actor.uuid}.Item.feat`, actor, type: 'feat', sourceId: SCARE_SOURCE}; actor.items.set(item.id, item);
  const game = {user: gm, users, actors: new Map([actor, original, replacement].map(actor => [actor.id, actor])), scenes: new Map([[scene.id, scene]]), messages: new Map(), time: {worldTime: 100}};
  const context = {game, actor, original, replacement, origin, target, scene, user, gm, item};
  async function mutate(at) {if (!changed && phase === at && change) {changed = true; await change(context);}}
  if (duplicates) for (const id of ['first', 'duplicate']) original.items.set(id, {id, actor: original, type: 'effect', flags: {[ID]: {nativeEffectKey: 'scare-to-death:immunity'}}, async update(data) {writes.push({actorUUID: original.uuid, kind: 'effect-update'}); Object.assign(this, structuredClone(data)); await mutate('effect');}});
  const usage = {id: 'usage', actor, author: user, speaker: {actor: actor.id, scene: scene.id, token: origin.id}, flags: {[ID]: {usageInput: {targetUuids: [target.uuid]}}}, async update(changes) {assign(this, changes); if (this.flags[ID].scare?.claim?.state === 'intimidation-ready') await mutate('claim');}};
  game.messages.set(usage.id, usage);
  const fromUuid = async uuid => uuid === target.uuid ? target : null;
  const executor = {async roll(claim) {
    rolls++;
    const card = {id: 'original-check', author: user, actor, isCheckRoll: true, speaker: {actor: actor.id, scene: scene.id, token: origin.id}, rolls: [{_evaluated: true, total: 25, options: {degreeOfSuccess: 2}, toJSON: () => ({evaluated: true})}], flags: {pf2e: {origin: {actor: actor.uuid, uuid: item.uuid}, context: {origin: {actor: actor.uuid, token: origin.uuid}, target: {actor: claim.targetActorUuid, token: target.uuid}, type: 'skill-check', action: 'scare-to-death', dc: {slug: 'will', value: 20}, options: [`${ID}:scare:${claim.nonce}:intimidation`, 'item:trait:incapacitation'], outcome: 'success'}}}};
    game.messages.set(card.id, card); return card;
  }};
  const provider = createScareToDeath({game, fromUuid, executor, sense: () => ({originSensesTarget: true, targetSensesOrigin: true, targetHearsOrigin: true}), makeEffect: data => data});
  const run = () => provider.executeUsage({actor, item, message: usage, user, action: 'scare-to-death'});
  return {...context, usage, provider, run, writes, rolls: () => rolls};
}

const changes = {
  relink: ({target, replacement}) => {target.actor = replacement;},
  deletedToken: ({scene, target}) => {scene.tokens.delete(target.id);},
  replacedToken: ({scene, target}) => {scene.tokens.set(target.id, {...target});},
  deletedScene: ({game, scene}) => {game.scenes.delete(scene.id);},
  changedGM: ({game}) => {game.users.activeGM = {id: 'new-gm', isGM: true, active: true};},
  lostOwner: ({actor}) => {actor.testUserPermission = () => false;},
  disconnectedOwner: ({user}) => {user.active = false;},
  replacedSourceItem: ({actor, item}) => {actor.items.set(item.id, {...item});},
  replacedTargetActor: ({game, original}) => {game.actors.set(original.id, {...original});},
};
for (const [name, change] of Object.entries(changes)) test(`${name} during the saved claim cannot write a target reservation or replay`, async () => {
  const f = fixture({change});
  await assert.rejects(f.run);
  assert.equal(f.writes.length, 0);
  assert.equal(f.rolls(), 0);
  assert.ok(f.usage.flags[ID].scare.claim, 'the initial claim remains persisted');
  await assert.rejects(f.run);
  assert.equal(f.writes.length, 0);
  assert.equal(f.rolls(), 0);
});
test('a legitimate native result completes on the original actor once', async () => {
  const f = fixture();
  await f.run();
  assert.equal(f.original.frightened, 2);
  assert.equal(f.replacement.frightened, 0);
  assert.ok(f.writes.every(write => write.actorUUID === f.original.uuid));
  assert.equal(f.usage.flags[ID].scare.claim.state, 'done');
  assert.equal(f.original.flags[ID].scare.pending, null);
  const writes = f.writes.length;
  await f.run();
  assert.equal(f.rolls(), 1);
  assert.equal(f.writes.length, writes);
});
test('relink during an immunity write stops the following condition writes', async () => {
  const f = fixture({change: changes.relink, phase: 'effect'});
  await assert.rejects(f.run);
  assert.equal(f.original.frightened, 0);
  assert.equal(f.replacement.frightened, 0);
  assert.equal(f.writes.some(write => write.actorUUID === f.replacement.uuid), false);
  assert.equal(f.usage.flags[ID].scare.claim.state, 'uncertain');
});
test('relink during an existing immunity update preserves its duplicate for GM review', async () => {
  const f = fixture({change: changes.relink, phase: 'effect', duplicates: true});
  await assert.rejects(f.run);
  assert.equal(f.original.items.has('duplicate'), true);
  assert.equal(f.writes.some(write => write.kind === 'delete'), false);
  assert.equal(f.writes.some(write => write.actorUUID === f.replacement.uuid), false);
});
