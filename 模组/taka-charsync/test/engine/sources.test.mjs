import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { cleanStaleSnapshots, fingerprint, mainActiveWorld, Throttle } from '../../engine/sources.mjs';

const HOUR = 60 * 60 * 1000;

let workDir;
let server;
let statusUrl;
let reply = {status: 200, body: {active: false}};

before(async () => {
  workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-sources-'));
  server = http.createServer((request, response) => {
    response.writeHead(reply.status, {'content-type': 'application/json'});
    response.end(JSON.stringify(reply.body));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  statusUrl = `http://127.0.0.1:${server.address().port}/api/status`;
});

after(async () => {
  await new Promise(resolve => server.close(resolve));
  if (workDir) fs.rmSync(workDir, {recursive: true, force: true});
});

const freshDir = name => {
  const dir = path.join(workDir, name);
  fs.mkdirSync(dir);
  return dir;
};

test('mainActiveWorld returns the running world id, or null when no world is running', async () => {
  reply = {status: 200, body: {active: true, version: '14.369', world: 'sog', system: 'pf2e'}};
  assert.equal(await mainActiveWorld(statusUrl), 'sog');
  reply = {status: 200, body: {active: false, version: '14.369'}};
  assert.equal(await mainActiveWorld(statusUrl), null);
});

test('mainActiveWorld throws on a non-2xx answer', async () => {
  reply = {status: 502, body: {}};
  await assert.rejects(mainActiveWorld(statusUrl), /502/);
});

test('mainActiveWorld throws when nothing listens on the port', async () => {
  const closed = http.createServer();
  await new Promise(resolve => closed.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${closed.address().port}/api/status`;
  await new Promise(resolve => closed.close(resolve));
  await assert.rejects(mainActiveWorld(url));
});

test('fingerprint changes when a file grows and ignores LOCK', () => {
  const dir = freshDir('db');
  fs.writeFileSync(path.join(dir, '000003.log'), 'a');
  fs.writeFileSync(path.join(dir, 'CURRENT'), 'MANIFEST-000002\n');
  fs.writeFileSync(path.join(dir, 'LOCK'), '');
  const before = fingerprint(dir);

  fs.appendFileSync(path.join(dir, '000003.log'), 'b');
  const grown = fingerprint(dir);
  assert.notEqual(grown, before);

  fs.writeFileSync(path.join(dir, 'LOCK'), 'held by foundry');
  fs.utimesSync(path.join(dir, 'LOCK'), new Date(Date.now() + HOUR), new Date(Date.now() + HOUR));
  assert.equal(fingerprint(dir), grown);
});

test('fingerprint throws for a missing directory', () => {
  assert.throws(() => fingerprint(path.join(workDir, 'no-such-db')), {code: 'ENOENT'});
});

test('Throttle is due for a new key, then only when the fingerprint changed and the interval passed', () => {
  const throttle = new Throttle(60_000);
  const t0 = 1_000_000;
  assert.equal(throttle.due('main:sog', 'fp1', t0), true);
  throttle.mark('main:sog', 'fp1', t0);

  assert.equal(throttle.due('main:sog', 'fp1', t0 + 10 * 60_000), false, 'same fingerprint is never due');
  assert.equal(throttle.due('main:sog', 'fp2', t0 + 30_000), false, 'changed, but read too recently');
  assert.equal(throttle.due('main:sog', 'fp2', t0 + 60_000), true);
  assert.equal(throttle.due('char', 'fp1', t0), true, 'keys are independent');
});

test('cleanStaleSnapshots removes only snapshot directories older than an hour', () => {
  const root = freshDir('tmp-root');
  const now = Date.now();
  const make = (name, ageMs) => {
    const dir = path.join(root, name);
    fs.mkdirSync(dir);
    fs.writeFileSync(path.join(dir, 'CURRENT'), 'x');
    fs.utimesSync(dir, new Date(now - ageMs), new Date(now - ageMs));
  };
  make('charsync-123-4', 2 * HOUR);
  make('charsync-123-5', 10 * 60 * 1000);
  make('charsync-test-abc', 2 * HOUR);
  make('charsync-dry', 2 * HOUR);

  assert.deepEqual(cleanStaleSnapshots(root, now), ['charsync-123-4']);
  assert.deepEqual(fs.readdirSync(root).sort(), ['charsync-123-5', 'charsync-dry', 'charsync-test-abc']);
});
