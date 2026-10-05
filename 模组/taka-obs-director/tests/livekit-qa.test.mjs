import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { parseHTML } from 'linkedom';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const relay = require('../tools/qa-relay.cjs');
const file = new URL('../tools/livekit-qa.mjs', import.meta.url);

test('QA publisher sends only a full raw clip to the native room, unpublishes and refuses changed identity', async () => {
  assert.ok(fs.existsSync(file), 'LiveKit helper must exist');
  const { createHelperScript } = await import(file);
  const writes = [], tracks = [{ stop() { writes.push('track-stop'); } }];
  let ended, source;
  class AudioContext {
    state = 'running';
    resume() { return Promise.resolve(); }
    decodeAudioData() { return Promise.resolve({ duration: 8.698 }); }
    createMediaStreamDestination() { return { stream: { getAudioTracks: () => tracks } }; }
    createBufferSource() { source = { connect(to) { assert.ok(to.stream); writes.push('stream-only'); }, start() { writes.push('start'); }, stop() { ended?.(); }, set onended(fn) { ended = fn; } }; return source; }
    close() { writes.push('context-close'); return Promise.resolve(); }
  }
  const participant = { trackPublications: new Map(), async publishTrack(track, options) { assert.equal(track, tracks[0]); assert.equal(options.dtx, false); assert.equal(options.source, 'microphone'); writes.push('publish'); return { track }; }, async unpublishTrack(track) { assert.equal(track, tracks[0]); writes.push('unpublish'); } };
  const room = { state: 'connected', name: 'obs-director-qa-cotct-test', localParticipant: participant, remoteParticipants: new Map(), on() {}, off() {} };
  const game = { ready: true, world: { id: 'obs-director-qa-cotct' }, user: { id: 'QAGM000000000001', isGM: true }, settings: { get: () => ({ room: room.name }) }, webrtc: { client: { _liveKitClient: { liveKitRoom: room } } } };
  const context = { globalThis: null, window: { __directorQA: { helperErrors: 0 } }, game, location: { hostname: '127.0.0.2' }, AudioContext, Date, Map, Set, JSON, URL, console, setInterval: () => 1, clearInterval() {}, document: { querySelectorAll: () => [], getElementById: () => null, createElement: () => ({ style: {}, append() {}, setAttribute() {}, addEventListener() {} }), body: { append() {} } }, getComputedStyle: () => ({}), fetch: async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(4) }) };
  context.globalThis = context; context.window.addEventListener = () => {};
  vm.runInNewContext(createHelperScript({ worldId: game.world.id, userId: game.user.id, room: room.name, userIds: [game.user.id], clip: 'gm' }), context);
  const api = context.window.__directorQA.controls;
  const result = await api.start();
  assert.equal(result.duration, 8.698); assert.equal(result.published, true);
  assert.deepEqual(writes, ['stream-only', 'publish', 'start']);
  await assert.rejects(api.start(), /already/);
  await api.stop(); assert.ok(writes.includes('unpublish')); assert.ok(writes.includes('track-stop'));
  await api.start(); ended(); await new Promise(r => setImmediate(r));
  assert.equal(writes.filter(v => v === 'unpublish').length, 2);
  game.world.id = 'cotct'; await assert.rejects(api.start(), /guard/);
  assert.equal(writes.filter(v => v === 'publish').length, 2);
  game.world.id = 'obs-director-qa-cotct'; game.settings.get = () => ({ room: 'production-room' }); await assert.rejects(api.start(), /guard/);
});

test('QA relay authenticates commands, bounds actions, requires helper session and refuses audio outside allowlist', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'director-clips-'));
  fs.writeFileSync(path.join(root, 'gm.wav'), Buffer.from('RIFF-fixture'));
  let active = true;
  const upstream = http.createServer((req, res) => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ active, world: 'obs-director-qa-cotct' })); });
  await new Promise(r => upstream.listen(0, '127.0.0.1', r));
  const server = relay.createRelay({ upstream: `http://127.0.0.1:${upstream.address().port}`, worldId: 'obs-director-qa-cotct', user: { id: 'QAGM000000000001', password: 'fixture' }, controlToken: 'private-controller-fixture', audioRoot: root });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (action, token = 'private-controller-fixture') => fetch(base + '/qa-control', { method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` }, body: JSON.stringify({ action }) });
  try {
    assert.equal((await post('start', 'wrong')).status, 403);
    assert.equal((await post('fake-speaking')).status, 403);
    const queued = await (await post('start')).json(); assert.equal(queued.id, 1);
    assert.equal((await fetch(base + '/qa-command')).status, 403);
    const command = await (await fetch(base + '/qa-command', { headers: { cookie: 'session=fixture' } })).json(); assert.deepEqual(command, { id: 1, action: 'start' });
    const ack = await fetch(base + '/qa-command-result', { method: 'POST', headers: { cookie: 'session=fixture', 'content-type': 'application/json' }, body: JSON.stringify({ id: 1, action: 'start', ok: true, duration: 8.698, room: 'secret', token: 'secret' }) });
    assert.equal(ack.status, 204);
    const latest = await (await fetch(base + '/qa-latest')).json(); assert.equal(latest.commands[0].duration, 8.698); assert.equal(JSON.stringify(latest).includes('secret'), false);
    assert.equal((await fetch(base + '/qa-clips/gm.wav')).status, 403);
    assert.equal(await (await fetch(base + '/qa-clips/gm.wav', { headers: { cookie: 'fixture' } })).text(), 'RIFF-fixture');
    assert.equal((await fetch(base + '/qa-clips/unknown.wav', { headers: { cookie: 'fixture' } })).status, 403);
    active = false; assert.equal((await post('stop')).status, 409);
  } finally { await new Promise(r => server.close(r)); await new Promise(r => upstream.close(r)); fs.rmSync(root, { recursive: true }); }
});

test('QA telemetry observes real SDK events, receiver stats and rendered CSS without writing speaking or layout', async () => {
  const { createHelperScript } = await import(file);
  const callbacks = new Map();
  const participant = { metadata: JSON.stringify({ fvttUserId: 'QAPL000000000001', secret: 'never-export' }), isSpeaking: false, audioTrackPublications: new Map([['audio', { track: { getRTCStatsReport: async () => new Map([['audio', { type: 'inbound-rtp', kind: 'audio', bytesReceived: 321 }]]) } }]]), on(name, callback) { callbacks.set(name, callback); }, off(name) { callbacks.delete(name); } };
  const roomCallbacks = new Map();
  const room = { state: 'connected', name: 'obs-director-qa-cotct-test', remoteParticipants: new Map([['remote', participant]]), localParticipant: { isSpeaking: false, on() {}, off() {} }, on(name, callback) { roomCallbacks.set(name, callback); }, off(name) { roomCallbacks.delete(name); } };
  const { document } = parseHTML('<html><body><div id="taka-obs-director" data-mode="combat"><div class="focus-name">Actual PC</div><div class="focus-hp">20/30</div><div class="cast-person" data-user-id="QAPL000000000001"><div class="portrait"><img class="figure"></div></div></div></body></html>');
  const original = document.body.innerHTML;
  let poll;
  const context = { game: { ready: true, world: { id: 'obs-director-qa-cotct' }, user: { id: 'OBSRecorder00001' }, settings: { get: () => ({ room: room.name }) }, webrtc: { client: { _liveKitClient: { liveKitRoom: room } } } }, window: { __directorQA: {}, addEventListener() {} }, document, location: { hostname: '127.0.0.1' }, Date, Map, Set, JSON, setInterval(fn) { poll = fn; return 1; }, clearInterval() {}, fetch: async () => ({ ok: true, json: async () => null }), getComputedStyle(el) { return { transform: el.className === 'portrait' ? 'matrix(1.06, 0, 0, 1.06, 0, -3)' : 'none', filter: el.className === 'figure' ? 'brightness(1.05) drop-shadow(rgb(255, 255, 255) 1px 0px 0px)' : 'none', outlineWidth: '0px', outlineColor: 'rgb(255, 255, 255)' }; } };
  context.globalThis = context;
  vm.runInNewContext(createHelperScript({ worldId: context.game.world.id, userId: context.game.user.id, userIds: ['OBSRecorder00001', 'QAPL000000000001'], room: room.name }), context);
  await poll(); participant.isSpeaking = true; callbacks.get('isSpeakingChanged')(true); roomCallbacks.get('activeSpeakersChanged')([participant]);
  const observed = JSON.parse(JSON.stringify(context.window.__directorQA.observe()));
  assert.deepEqual(observed.speakingUserIds, ['QAPL000000000001']); assert.equal(observed.portraits[0].scale, 1.06); assert.match(observed.portraits[0].filter, /drop-shadow/);
  assert.equal(observed.mode, 'combat'); assert.equal(observed.focusName, 'Actual PC'); assert.equal(observed.audioReceivedBytes[0].bytes, 321); assert.equal(observed.events.length, 2);
  assert.equal(document.body.innerHTML, original); assert.equal(JSON.stringify(observed).includes('never-export'), false);
  participant.audioTrackPublications.clear();
  const silent = JSON.parse(JSON.stringify(context.window.__directorQA.observe()));
  assert.deepEqual(silent.speakingUserIds, ['QAPL000000000001']); assert.deepEqual(silent.renderedSpeakingUserIds, []);
  assert.deepEqual(silent.audioPublications, [{ userId: 'QAPL000000000001', count: 0, unmuted: 0 }]);
  await assert.rejects(context.window.__directorQA.controls.start(), /Recorder/);
});
