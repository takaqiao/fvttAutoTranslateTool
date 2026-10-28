import test from 'node:test';
import assert from 'node:assert/strict';
import {adaptTokenizerEditor, installTokenizerSaveAdapter} from '../../scripts/foundry/tokenizer-save.mjs';

function editorFixture() {
  const calls = [];
  const editor = {actor: {id: 'actor-id'}, save: {
    saveToken(options) { calls.push({mode: 'token', options, receiver: this}); return 'token saved'; },
    saveAvatar(options) { calls.push({mode: 'avatar', options, receiver: this}); return 'avatar saved'; },
    async saveBoth(options) {
      calls.push({mode: 'both', options, receiver: this});
      await this.saveHeadless('avatar', ['avatar layer'], {...options, overrides: options.overrides.avatar});
      return 'both saved';
    },
    saveHeadless(mode, layers, options) { calls.push({mode, layers, options, receiver: this}); return 'headless saved'; },
  }};
  return {editor, calls};
}

test('each save receives a distinct filename while folder, target, options and receiver survive', () => {
  const {editor, calls} = editorFixture();
  const options = {overrides: {filename: 'same.webp', folder: 'custom', target: 'token'}, keepOpen: true};
  adaptTokenizerEditor(editor);
  adaptTokenizerEditor(editor);
  assert.equal(editor.save.saveToken(options), 'token saved');
  editor.save.saveToken(options);
  editor.save.saveAvatar(options);
  assert.equal(new Set(calls.map(call => call.options.overrides.filename)).size, 3);
  for (const call of calls) {
    assert.match(call.options.overrides.filename, /^same-(token|avatar)-[\w-]+\.webp$/);
    assert.equal(call.options.overrides.folder, 'custom');
    assert.equal(call.options.overrides.target, 'token');
    assert.equal(call.options.keepOpen, true);
    assert.equal(call.receiver, editor.save);
  }
  assert.equal(options.overrides.filename, 'same.webp');
});

test('saveBoth gives token and avatar different names and the nested headless call keeps its assigned version', async () => {
  const {editor, calls} = editorFixture();
  adaptTokenizerEditor(editor);
  const options = {overrides: {
    token: {filename: 'same.png', folder: 'same', target: 'prototype'},
    avatar: {filename: 'same.png', folder: 'same'},
  }, marker: 7};
  assert.equal(await editor.save.saveBoth(options), 'both saved');
  const pair = calls[0].options.overrides;
  assert.notEqual(pair.token.filename, pair.avatar.filename);
  assert.equal(pair.token.filename.replace('-token-', '-avatar-'), pair.avatar.filename);
  assert.equal(calls[1].options.overrides.filename, pair.avatar.filename);
  assert.deepEqual(calls[1].layers, ['avatar layer']);
  assert.equal(calls[1].options.marker, 7);
  assert.equal(pair.token.target, 'prototype');
  assert.equal(options.overrides.token.filename, 'same.png');
  editor.save.saveHeadless('avatar', ['other']);
  assert.notEqual(calls[2].options.overrides.filename, pair.avatar.filename);
});

test('a wildcard token save keeps its filename pattern and leaves its folder selection to Tokenizer', () => {
  const {editor, calls} = editorFixture();
  editor.isWildcard = true;
  editor.actor.prototypeToken = {texture: {src: 'custom/hero-*-token.webp'}};
  adaptTokenizerEditor(editor);
  editor.save.saveToken();
  editor.save.saveToken();
  for (const call of calls) {
    assert.match(call.options.overrides.filename, /^hero-token-[\w-]+-token\.webp$/);
    assert.equal(call.options.overrides.folder, undefined);
  }
  assert.notEqual(calls[0].options.overrides.filename, calls[1].options.overrides.filename);
});

test('the public editorOpen hook adapts supported instances and reports an unsupported controller', () => {
  const callbacks = new Map();
  globalThis.Hooks = {on: (name, callback) => callbacks.set(name, callback)};
  const warnings = [];
  globalThis.ui = {notifications: {warn: message => warnings.push(message)}};
  installTokenizerSaveAdapter();
  const open = callbacks.get('tokenizer-2.editorOpen');
  assert.equal(typeof open, 'function');
  const {editor, calls} = editorFixture();
  open({editor});
  editor.save.saveAvatar();
  assert.match(calls[0].options.overrides.filename, /avatar-/);
  open({editor: {save: {saveToken() {}}}});
  assert.equal(warnings.length, 1);
});
