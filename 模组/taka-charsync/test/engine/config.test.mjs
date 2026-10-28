import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadConfig } from '../../engine/config.mjs';

const base = {
  foundryAppDir: '.', main: {dataDir: 'main', statusUrl: 'http://localhost/api/status'},
  char: {dataDir: 'char', world: 'pf2creation'}, publishDir: 'charsyncTestPublish01',
  worlds: ['ecnu', 'sog'], stateDir: 'state', seedPath: 'seed.json',
};

test('campaignFolders accepts only explicit Foundry folder IDs mapped to configured worlds', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-config-'));
  const file = path.join(dir, 'config.json');
  const load = campaignFolders => {
    fs.writeFileSync(file, JSON.stringify({...base, campaignFolders}));
    return loadConfig(file);
  };
  try {
    assert.deepEqual(load(undefined).campaignFolders, {});
    assert.deepEqual(load({folderEcnu000001: 'ecnu'}).campaignFolders, {folderEcnu000001: 'ecnu'});
    for (const invalid of [[], null, 'ecnu', {'../escape': 'ecnu'}, {folderEcnu000001: 'missing'}, {folderEcnu000001: 1}]) {
      assert.throws(() => load(invalid), /campaignFolders/);
    }
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
});
