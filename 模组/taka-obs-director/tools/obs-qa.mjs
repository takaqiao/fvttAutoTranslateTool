import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { pathToFileURL } from 'node:url';
const sha = value => crypto.createHash('sha256').update(value).digest('base64');

export function assertOwnedProcess(process, expected) {
  if (!Number.isInteger(expected.pid) || expected.pid < 2 || process?.pid !== expected.pid || typeof process.executablePath !== 'string' || path.resolve(process.executablePath).toLowerCase() !== path.resolve(expected.runtime, 'bin/64bit/obs64.exe').toLowerCase()) throw Error('OBS process ownership refused');
}
export async function connectOBS(config, Socket = WebSocket) {
  if (config.server_port !== 4457 || config.auth_required !== true || !config.server_password) throw Error('Owned OBS authentication configuration refused');
  const socket = new Socket('ws://127.0.0.1:4457'), pending = new Map(); let sequence = 0;
  let identify, rejectIdentify;
  const identified = new Promise((resolve, reject) => { identify = resolve; rejectIdentify = reject; });
  const timer = setTimeout(() => { rejectIdentify(Error('OBS identify timed out')); socket.close(); }, 10000);
  function fail() { rejectIdentify(Error('OBS connection failed')); for (const { reject, timeout } of pending.values()) { clearTimeout(timeout); reject(Error('OBS connection closed')); } pending.clear(); }
  socket.addEventListener('error', fail); socket.addEventListener('close', fail);
  socket.addEventListener('message', ({ data }) => {
    try {
      const packet = JSON.parse(data);
      if (packet.op === 0) {
        const auth = packet.d?.authentication;
        if (!auth?.salt || !auth.challenge) { rejectIdentify(Error('OBS server authentication required')); socket.close(); return; }
        socket.send(JSON.stringify({ op: 1, d: { rpcVersion: 1, authentication: sha(sha(config.server_password + auth.salt) + auth.challenge), eventSubscriptions: 0 } }));
      } else if (packet.op === 2) { clearTimeout(timer); identify(); }
      else if (packet.op === 7) {
        const request = pending.get(packet.d?.requestId); if (!request) return;
        pending.delete(packet.d.requestId); clearTimeout(request.timeout);
        if (packet.d.requestStatus?.result === true) request.resolve(packet.d.responseData ?? {});
        else request.reject(Error(`OBS request refused (${Number(packet.d.requestStatus?.code) || 0})`));
      }
    } catch { fail(); }
  });
  try { await identified; } catch (error) { clearTimeout(timer); socket.close(); throw error; }
  return {
    request(requestType, requestData = {}) {
      const requestId = String(++sequence);
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => { pending.delete(requestId); reject(Error('OBS request timed out')); }, 10000);
        pending.set(requestId, { resolve, reject, timeout });
        socket.send(JSON.stringify({ op: 6, d: { requestType, requestId, requestData } }));
      });
    },
    close() { clearTimeout(timer); for (const request of pending.values()) { clearTimeout(request.timeout); request.reject(Error('OBS controller closed')); } pending.clear(); socket.close(); }
  };
}

export async function verifyOBS(client, sourceUuid) {
  const collection = await client.request('GetSceneCollectionList'), profile = await client.request('GetProfileList');
  const inputs = await client.request('GetInputList'), devices = await client.request('GetSpecialInputs');
  const video = await client.request('GetVideoSettings');
  const source = inputs.inputs?.[0];
  if (collection.currentSceneCollectionName !== 'FVTT Director QA' || profile.currentProfileName !== 'FVTT Director QA' || inputs.inputs?.length !== 1 || source.inputUuid !== sourceUuid || source.inputKind !== 'browser_source' || Object.values(devices).some(value => value !== null)) throw Error('OBS isolated collection/source/device gate refused');
  if (video.baseWidth !== 1920 || video.baseHeight !== 1080 || video.outputWidth !== 1920 || video.outputHeight !== 1080 || video.fpsNumerator / video.fpsDenominator !== 30) throw Error('OBS video gate refused');
  const input = { inputUuid: sourceUuid };
  const tracks = (await client.request('GetInputAudioTracks', input)).inputAudioTracks;
  const monitoring = await client.request('GetInputAudioMonitorType', input), muted = await client.request('GetInputMute', input), settings = await client.request('GetInputSettings', input);
  if (!tracks || tracks['1'] !== true || [2, 3, 4, 5, 6].some(key => tracks[String(key)] !== false) || monitoring.monitorType !== 'OBS_MONITORING_TYPE_NONE' || muted.inputMuted !== false || settings.inputSettings?.reroute_audio !== true) throw Error('OBS browser audio routing gate refused');
  const record = await client.request('GetRecordStatus');
  return { ready: true, sourceName: source.inputName, sourceUuid, recording: record.outputActive === true, paused: record.outputPaused === true, durationMs: record.outputDuration ?? 0, dimensions: [1920, 1080], fps: 30, track: 1, monitoringOff: true, devicesNull: true };
}
function argumentsOf(argv) { const a = {}; for (let i = 0; i < argv.length; i += 2) { if (!argv[i].startsWith('--') || argv[i + 1] === undefined) throw Error('Expected named arguments'); a[argv[i].slice(2)] = argv[i + 1]; } return a; }
function privateResult(file, value) {
  if (!file || fs.existsSync(file)) throw Error('New private output file required');
  fs.writeFileSync(file, JSON.stringify(value, null, 2), { flag: 'wx', mode: 0o600 });
}
export async function acknowledgeRecording(client, { action, output, sourceUuid, durationMsBeforeStop, pause = () => new Promise(resolve => setTimeout(resolve, 250)), verify = verifyOBS }) {
  if (!['start', 'stop'].includes(action) || !output || fs.existsSync(output)) throw Error('New private recording result required');
  const receipt = { requestedAt: Date.now(), recordingVerified: false, ...(action === 'start' ? { startAcknowledged: false } : { stopAcknowledged: false, durationMsBeforeStop }) };
  privateResult(output, receipt);
  let response;
  try { response = await client.request(action === 'start' ? 'StartRecord' : 'StopRecord'); }
  catch (error) { fs.writeFileSync(output, JSON.stringify({ ...receipt, failure: 'request' }, null, 2), { mode: 0o600 }); throw error; }
  Object.assign(receipt, { at: Date.now(), ...(action === 'start' ? { startAcknowledged: true } : { stopAcknowledged: true, outputPath: typeof response.outputPath === 'string' ? response.outputPath : undefined }) });
  fs.writeFileSync(output, JSON.stringify(receipt, null, 2), { mode: 0o600 });
  try {
    let settled = false;
    for (let attempt = 0; attempt < 40; attempt++) {
      const status = await client.request('GetRecordStatus');
      if (status.outputActive === (action === 'start') && status.outputPaused !== true) { settled = true; break; }
      await pause();
    }
    if (!settled) throw Error('OBS recording state acknowledgement timed out');
    const after = await verify(client, sourceUuid);
    if (after.recording !== (action === 'start') || after.paused || (action === 'stop' && typeof receipt.outputPath !== 'string')) throw Error('OBS recording postflight refused');
    fs.writeFileSync(output, JSON.stringify({ ...after, ...receipt, recordingVerified: true }, null, 2), { mode: 0o600 });
    return { [action === 'start' ? 'started' : 'stopped']: true, at: receipt.at, privateResultSaved: true };
  } catch (error) {
    fs.writeFileSync(output, JSON.stringify({ ...receipt, failure: 'postflight' }, null, 2), { mode: 0o600 });
    throw error;
  }
}
export async function runOBS({ action, runtime, pid, sourceUuid, output }) {
  if (!['status', 'start', 'stop', 'screenshot'].includes(action) || !runtime || !sourceUuid) throw Error('Bounded OBS action required');
  if (process.platform !== 'win32') throw Error('Owned OBS process check requires Windows');
  const number = Number(pid); if (!Number.isInteger(number) || number < 2) throw Error('Explicit owned OBS PID required');
  const { stdout } = await promisify(execFile)('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', `Get-CimInstance Win32_Process -Filter 'ProcessId = ${number}' | Select-Object @{N='pid';E={$_.ProcessId}},@{N='executablePath';E={$_.ExecutablePath}} | ConvertTo-Json -Compress`], { windowsHide: true });
  assertOwnedProcess(JSON.parse(stdout), { pid: number, runtime });
  const config = JSON.parse(fs.readFileSync(path.join(runtime, 'config/obs-studio/plugin_config/obs-websocket/config.json')));
  const client = await connectOBS(config);
  try {
    const before = await verifyOBS(client, sourceUuid);
    if (action === 'status') { const { sourceName, ...safe } = before; if (output) privateResult(output, safe); return safe; }
    if (!output || fs.existsSync(output)) throw Error('New private output file required');
    if (action === 'start') {
      if (before.recording) throw Error('OBS recording already active');
      return await acknowledgeRecording(client, { action, output, sourceUuid });
    }
    if (action === 'stop') {
      if (!before.recording) throw Error('OBS recording inactive');
      return await acknowledgeRecording(client, { action, output, sourceUuid, durationMsBeforeStop: before.durationMs });
    }
    const result = await client.request('GetSourceScreenshot', { sourceName: before.sourceName, imageFormat: 'png', imageWidth: 1920, imageHeight: 1080 });
    if (!/^data:image\/png;base64,/.test(result.imageData)) throw Error('OBS screenshot response refused');
    const bytes = Buffer.from(result.imageData.split(',')[1], 'base64');
    if (bytes.readUInt32BE(16) !== 1920 || bytes.readUInt32BE(20) !== 1080) throw Error('OBS screenshot dimensions refused');
    fs.writeFileSync(output, bytes, { flag: 'wx', mode: 0o600 }); return { screenshotSaved: true, bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') };
  } finally { client.close(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { const a = argumentsOf(process.argv.slice(2)); console.log(JSON.stringify(await runOBS({ action: a.action, runtime: a.runtime, pid: a.pid, sourceUuid: a['source-uuid'], output: a.output }))); }
  catch { console.error('Owned OBS operation refused; check isolated runtime, routing and current process ownership'); process.exitCode = 1; }
}
