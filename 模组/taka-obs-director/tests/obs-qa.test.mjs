import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
const file = new URL('../tools/obs-qa.mjs', import.meta.url);

test('owned OBS gate rejects devices, wrong source, monitoring or extra audio tracks before recording', async () => {
  assert.ok(fs.existsSync(file), 'OBS controller must exist');
  const { verifyOBS } = await import(file);
  const responses = { GetSceneCollectionList: { currentSceneCollectionName: 'FVTT Director QA' }, GetProfileList: { currentProfileName: 'FVTT Director QA' }, GetInputList: { inputs: [{ inputUuid: 'qa-source', inputKind: 'browser_source' }] }, GetSpecialInputs: { desktop1: null, mic1: null }, GetVideoSettings: { baseWidth: 1920, baseHeight: 1080, outputWidth: 1920, outputHeight: 1080, fpsNumerator: 30, fpsDenominator: 1 }, GetInputAudioTracks: { inputAudioTracks: { 1: true, 2: false, 3: false, 4: false, 5: false, 6: false } }, GetInputAudioMonitorType: { monitorType: 'OBS_MONITORING_TYPE_NONE' }, GetInputMute: { inputMuted: false }, GetRecordStatus: { outputActive: false }, GetInputSettings: { inputSettings: { reroute_audio: true } } };
  const calls = [], client = { request: async (name, data) => { calls.push([name, data]); return structuredClone(responses[name]); } };
  assert.equal((await verifyOBS(client, 'qa-source')).ready, true);
  for (const [name, field, bad] of [['GetSpecialInputs', 'mic1', 'real mic'], ['GetSceneCollectionList', 'currentSceneCollectionName', 'Other'], ['GetInputAudioMonitorType', 'monitorType', 'OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT']]) {
    const previous = responses[name][field]; responses[name][field] = bad; await assert.rejects(verifyOBS(client, 'qa-source'), /OBS/); responses[name][field] = previous;
  }
  responses.GetInputAudioTracks.inputAudioTracks[2] = true; await assert.rejects(verifyOBS(client, 'qa-source'), /OBS/);
  assert.equal(calls.some(([name]) => /^(Start|Stop|Set)/.test(name)), false);
});

test('OBS authentication stays inside protocol, request errors are redacted and ownership is exact', async () => {
  const { connectOBS, assertOwnedProcess } = await import(file);
  assertOwnedProcess({ pid: 40144, executablePath: path.resolve('owned/bin/64bit/obs64.exe') }, { pid: 40144, runtime: path.resolve('owned') });
  assert.throws(() => assertOwnedProcess({ pid: 40144, executablePath: path.resolve('other/obs64.exe') }, { pid: 40144, runtime: path.resolve('owned') }), /ownership/);
  const sent = [];
  class Socket extends EventTarget {
    constructor() { super(); queueMicrotask(() => this.message({ op: 0, d: { authentication: { salt: 'fixture', challenge: 'fixture' } } })); }
    message(packet) { const e = new Event('message'); e.data = JSON.stringify(packet); this.dispatchEvent(e); }
    send(value) { const packet = JSON.parse(value); sent.push(packet); queueMicrotask(() => packet.op === 1 ? this.message({ op: 2 }) : this.message({ op: 7, d: { requestId: packet.d.requestId, requestStatus: { result: false, code: 600, comment: 'password secret raw' } } })); }
    close() {}
  }
  const client = await connectOBS({ server_port: 4457, auth_required: true, server_password: 'never-print-password' }, Socket);
  assert.equal(sent[0].op, 1); assert.ok(sent[0].d.authentication); assert.equal(JSON.stringify(sent).includes('never-print-password'), false);
  await assert.rejects(client.request('GetRecordStatus'), error => /600/.test(error.message) && !/secret|password/.test(error.message)); client.close();
});

test('recording acknowledgement waits for actual state and preserves truthful receipt if postflight fails', async () => {
  const { acknowledgeRecording } = await import(file);
  assert.equal(typeof acknowledgeRecording, 'function', 'Recording state acknowledgement must be implemented');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'director-obs-ack-'));
  try {
    const calls = []; let polls = 0;
    const client = { request: async name => { calls.push(name); if (name === 'StartRecord') return {}; if (name === 'GetRecordStatus') return { outputActive: ++polls >= 3, outputPaused: false }; throw Error('Unexpected request'); } };
    const output = path.join(root, 'start.json');
    const result = await acknowledgeRecording(client, { action: 'start', output, sourceUuid: 'fixture', pause: async () => {}, verify: async () => ({ recording: true, paused: false }) });
    assert.equal(result.started, true); assert.equal(polls, 3); assert.equal(calls.filter(n => n === 'StartRecord').length, 1);
    const receipt = JSON.parse(fs.readFileSync(output)); assert.equal(receipt.startAcknowledged, true); assert.equal(receipt.recordingVerified, true);
    const failed = path.join(root, 'postflight-failed.json');
    await assert.rejects(acknowledgeRecording(client, { action: 'start', output: failed, sourceUuid: 'fixture', pause: async () => {}, verify: async () => { throw Error('OBS browser audio routing gate refused'); } }), /routing/);
    const incomplete = JSON.parse(fs.readFileSync(failed)); assert.equal(incomplete.startAcknowledged, true); assert.equal(incomplete.recordingVerified, false); assert.equal(incomplete.failure, 'postflight');
    const beforeCalls = calls.length;
    await assert.rejects(acknowledgeRecording(client, { action: 'start', output: path.join(root, 'missing/start.json'), sourceUuid: 'fixture' })); assert.equal(calls.length, beforeCalls, 'an unwritable receipt must refuse before StartRecord');
    const stoppedFile = path.join(root, 'stop.json');
    const stopped = { request: async name => name === 'StopRecord' ? { outputPath: 'fixture-recording.mkv' } : { outputActive: false } };
    await acknowledgeRecording(stopped, { action: 'stop', output: stoppedFile, sourceUuid: 'fixture', verify: async () => ({ recording: false, paused: false }) });
    const savedStop = JSON.parse(fs.readFileSync(stoppedFile)); assert.equal(savedStop.stopAcknowledged, true); assert.equal(savedStop.recordingVerified, true); assert.equal(savedStop.outputPath, 'fixture-recording.mkv');
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
