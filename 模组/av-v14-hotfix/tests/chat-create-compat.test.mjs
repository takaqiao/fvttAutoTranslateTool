import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {registerLegacyCompat} from '../scripts/compat/legacy.mjs';

// Unmodified Document.create from licensed Foundry 14.367, foundry.mjs:15145-15150.
// Only the database boundary is substituted; create's real delegation and return shape execute.
const nativeCreate = readFileSync(new URL('./fixtures/document-create-14.367.txt', import.meta.url), 'utf8');
function setup({overrideBatch = false, failure} = {}) {
  const calls = [];
  const backend = (receiver, data, operation, extra) => {
    if (failure) throw failure;
    assert.ok(data.every(d => typeof d.type !== 'number'), 'legacy type reached model validation');
    calls.push({receiver, data, operation, extra});
    return data.map((d, i) => ({id: `message-${i}`, _source: d}));
  };
  const Document = new Function('backend', `return class Document {
    ${nativeCreate}
    static async createDocuments(data = [], operation = {}, ...extra) { return backend(this, data, operation, extra); }
  }`)(backend);
  class Message extends Document { static get implementation() { return MessageImpl; } }
  class MessageImpl extends Message {}
  if (overrideBatch) Object.defineProperty(MessageImpl, 'createDocuments', {
    configurable: true, writable: true,
    value(data, ...args) {
      assert.ok(data.every(d => typeof d.type !== 'number'), 'legacy type reached system override');
      return Message.createDocuments.call(this, data, ...args);
    }
  });
  const g = {ChatMessage: Message, CONFIG: {ChatMessage: {documentClass: MessageImpl}}};
  const originalCreate = Message.create;
  const registrations = [];
  registerLegacyCompat({g, registerWrapper(target, callback) {
    const keys = target.split('.');
    const property = keys.pop();
    const owner = keys.reduce((o, key) => o[key], g);
    const original = owner[property];
    Object.defineProperty(owner, property, {configurable: true, writable: true,
      value: function(...args) { return callback.call(this, original, ...args); }});
    registrations.push(target);
  }});
  return {Message, MessageImpl, originalCreate, calls, registrations};
}

test('legacy chat compatibility leaves the native single-create receiver available to CotCT display', async () => {
  const {Message, MessageImpl, originalCreate} = setup();
  assert.equal(Object.hasOwn(Message, 'create'), false, 'base receiver must retain native inherited create');
  assert.equal(Object.hasOwn(MessageImpl, 'create'), false, 'system receiver must retain native inherited create');
  assert.equal(Message.create, originalCreate);
  assert.equal(MessageImpl.create, originalCreate);
  assert.equal(Message.implementation, MessageImpl);
  const result = await Message.create({type: 0, content: '旧导入器'});
  assert.deepEqual(result._source, {style: 0, content: '旧导入器'});
});

test('global and implementation single/array/batch entry points migrate before validation even with noHook', async () => {
  for (const implementation of [false, true]) for (const entry of ['single', 'array', 'batch']) {
    const {Message, MessageImpl, calls} = setup();
    const receiver = implementation ? MessageImpl : Message;
    const old = Object.freeze({type: 2, content: 'old', flags: Object.freeze({keep: true})});
    const modern = Object.freeze({type: 'base', style: 0, content: 'modern'});
    const operation = Object.freeze({noHook: true, render: false});
    const input = entry === 'single' ? old : Object.freeze([old, modern]);
    const result = await receiver[entry === 'batch' ? 'createDocuments' : 'create'](input, operation);
    const documents = entry === 'single' ? [result] : result;
    assert.deepEqual(documents[0]._source, {style: 2, content: 'old', flags: {keep: true}});
    assert.equal(documents[0]._source.flags, old.flags);
    if (entry !== 'single') assert.equal(documents[1]._source, modern);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].operation, operation);
    assert.equal(calls[0].receiver, entry === 'batch' ? receiver : MessageImpl);
    assert.equal(old.type, 2);
  }
});

test('system-owned createDocuments receives migrated data and preserves explicit styles and extra arguments', async () => {
  const {Message, MessageImpl, calls} = setup({overrideBatch: true});
  const operation = Object.freeze({noHook: true});
  const marker = {};
  const modern = Object.freeze({type: 'custom', style: 1});
  const result = await MessageImpl.createDocuments(Object.freeze([Object.freeze({type: 4, style: 0}), modern]), operation, marker);
  assert.deepEqual(result[0]._source, {style: 0});
  assert.equal(result[1]._source, modern);
  assert.equal(calls[0].receiver, MessageImpl);
  assert.equal(calls[0].operation, operation);
  assert.equal(calls[0].extra[0], marker);
  assert.deepEqual((await Message.create({type: 3}))._source, {style: 3});
});

test('a display wrapper can call the untouched native single-create and receive the same content acknowledgement', async () => {
  const {Message, originalCreate} = setup();
  assert.equal(Message.create, originalCreate);
  const data = Object.freeze({speaker: Object.freeze({alias: 'Zellara'}), content: '<p>Harrow</p>', type: 0});
  let receipt;
  Object.defineProperty(Message, 'create', {configurable: true, value: function(...args) {
    const promise = originalCreate.apply(this, args);
    promise.then(message => {receipt = message;});
    return promise;
  }});
  const message = await Message.create(data);
  assert.equal(receipt, message);
  assert.equal(message._source.content, data.content);
  assert.equal(message._source.speaker, data.speaker);
  assert.equal(message._source.style, 0);
});

test('database rejection is propagated through both chat entry points without replacing the error', async () => {
  const failure = new Error('database rejected');
  const {Message, MessageImpl} = setup({failure});
  await assert.rejects(Message.create({type: 0}), error => error === failure);
  await assert.rejects(MessageImpl.createDocuments([{type: 0}]), error => error === failure);
});
