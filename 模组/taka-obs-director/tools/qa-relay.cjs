const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const crypto = require('node:crypto');
const { args } = require('./deploy.cjs');
const LOOPBACK = /^127\.0\.0\.[1-7]$/;
const WORLD = /^obs-director-qa-[a-z]+$/;

function bootstrapScript({ worldId, userId }) {
  if (!WORLD.test(worldId) || !/^[A-Za-z0-9]{1,16}$/.test(userId)) throw Error('Invalid QA bootstrap identity');
  return `(() => {
    if (!/^127\\.0\\.0\\.[1-7]$/.test(location.hostname)) throw Error('QA origin refused');
    const state = window.__directorQA = { expectedWorld: ${JSON.stringify(worldId)}, expectedUser: ${JSON.stringify(userId)}, deviceAttempts: 0, helperErrors: 0 };
    localStorage.setItem('core.rtcClientSettings', JSON.stringify({ audioSrc: 'disabled', videoSrc: 'disabled', audioSink: 'default', muteAll: false, disableVideo: false, users: {} }));
    const denyDeviceCapture = async () => { state.deviceAttempts++; throw new DOMException('Real QA devices are disabled', 'NotAllowedError'); };
    if (navigator.mediaDevices) Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { configurable: false, get: () => denyDeviceCapture, set: () => {} });
    state.post = async (status) => { const response = await fetch('/qa-status', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(status) }); if (!response.ok) throw Error('QA status refused'); };
  })();`;
}
function statusScript({ worldId, userId, room }) {
  return `(() => {
    const timer = setInterval(async () => {
      if (!globalThis.game?.ready) return;
      try {
        const game = globalThis.game, qa = window.__directorQA, client = game.webrtc?.client?._liveKitClient, sdkRoom = client?.liveKitRoom;
        const actualRoom = game.settings.get('avclient-livekit', 'liveKitConnectionSettings')?.room;
        const breakout = game.webrtc?.client?.breakoutRoom ?? client?.breakoutRoom;
        const guardPassed = game.world.id === ${JSON.stringify(worldId)} && game.user.id === ${JSON.stringify(userId)} && actualRoom === ${JSON.stringify(room)} && (!breakout || breakout === actualRoom) && (sdkRoom?.state !== 'connected' || sdkRoom.name === actualRoom);
        const audio = [...document.querySelectorAll('audio.user-microphone-audio')];
        const attachedAudio = audio.filter(el => el.srcObject?.getAudioTracks?.().length > 0);
        const board = document.getElementById('board')?.getBoundingClientRect();
        await qa.post({ ...(qa.observe?.() ?? {}), worldId: game.world.id, userId: game.user.id, isGM: game.user.isGM, guardPassed, roomConnected: sdkRoom?.state === 'connected', localPublicationCount: sdkRoom?.localParticipant?.trackPublications?.size ?? -1, nativeAudioTrackNull: client?.audioTrack === null, nativeVideoTrackNull: client?.videoTrack === null, deviceAttempts: qa.deviceAttempts, subscribedAudioCount: attachedAudio.length, playbackErrors: attachedAudio.filter(el => el.error || el.paused).length, boardX: board?.x ?? 0, boardY: board?.y ?? 0, boardWidth: board?.width ?? 0, boardHeight: board?.height ?? 0, screenWidth: canvas?.screenDimensions?.[0] ?? 0, screenHeight: canvas?.screenDimensions?.[1] ?? 0, directorActive: document.body.classList.contains('taka-obs-director-active'), helperErrors: qa.helperErrors });
      } catch { window.__directorQA.helperErrors++; }
    }, 1000);
    window.addEventListener('unload', () => clearInterval(timer), { once: true });
  })();`;
}
const booleanFields = ['isGM', 'guardPassed', 'roomConnected', 'nativeAudioTrackNull', 'nativeVideoTrackNull', 'directorActive', 'publicFocus'];
const numberFields = ['localPublicationCount', 'deviceAttempts', 'subscribedAudioCount', 'playbackErrors', 'helperErrors', 'boardX', 'boardY', 'boardWidth', 'boardHeight', 'screenWidth', 'screenHeight'];
function sanitizeStatus(value, expected) {
  if (value?.worldId !== expected.worldId || value?.userId !== expected.userId || value.guardPassed !== true) throw Error('QA status identity or guard refused');
  const clean = { worldId: value.worldId, userId: value.userId };
  for (const field of booleanFields) if (typeof value[field] === 'boolean') clean[field] = value[field];
  for (const field of numberFields) if (Number.isFinite(value[field])) clean[field] = value[field];
  if (['explore', 'combat', 'story'].includes(value.mode)) clean.mode = value.mode;
  for (const field of ['focusName', 'focusHP', 'focusResourceText']) if (typeof value[field] === 'string') clean[field] = value[field].slice(0, 300);
  if (Array.isArray(value.audioReceivedBytes)) clean.audioReceivedBytes = value.audioReceivedBytes.slice(0, 6).filter((e) => expected.userIds?.includes(e.userId) && Number.isFinite(e.bytes) && e.bytes >= 0).map((e) => ({ userId: e.userId, bytes: e.bytes }));
  if (Array.isArray(value.portraits)) clean.portraits = value.portraits.slice(0, 6).filter((e) => expected.userIds?.includes(e.userId) && Number.isFinite(e.scale) && Number.isFinite(e.outlineWidth)).map((e) => ({ userId: e.userId, scale: e.scale, outlineWidth: e.outlineWidth, outlineColor: typeof e.outlineColor === 'string' ? e.outlineColor.slice(0, 60) : '', filter: typeof e.filter === 'string' ? e.filter.slice(0, 500) : '' }));
  if (Array.isArray(value.speakingUserIds)) clean.speakingUserIds = value.speakingUserIds.filter((id) => expected.userIds?.includes(id)).slice(0, 6);
  if (Array.isArray(value.renderedSpeakingUserIds)) clean.renderedSpeakingUserIds = value.renderedSpeakingUserIds.filter(id => expected.userIds?.includes(id)).slice(0, 6);
  if (Array.isArray(value.audioPublications)) clean.audioPublications = value.audioPublications.slice(0, 6).filter(e => expected.userIds?.includes(e.userId) && Number.isSafeInteger(e.count) && e.count >= 0 && Number.isSafeInteger(e.unmuted) && e.unmuted >= 0 && e.unmuted <= e.count).map(e => ({ userId: e.userId, count: e.count, unmuted: e.unmuted }));
  if (Array.isArray(value.events)) clean.events = value.events.slice(-30).filter((e) => expected.userIds?.includes(e.userId) && ['isSpeakingChanged', 'activeSpeakersChanged'].includes(e.type) && typeof e.speaking === 'boolean').map((e) => ({ userId: e.userId, type: e.type, speaking: e.speaking, at: Number.isFinite(e.at) ? e.at : 0 }));
  return clean;
}
async function body(req) { const chunks = []; let size = 0; for await (const chunk of req) { size += chunk.length; if (size > 32768) throw Error('QA request too large'); chunks.push(chunk); } return Buffer.concat(chunks); }
const ACTIONS = ['start', 'stop', 'explore', 'combat', 'next-turn', 'share-image', 'close-image', 'sound'];
function commandResult(value) {
  if (!Number.isSafeInteger(value?.id) || !ACTIONS.includes(value.action) || typeof value.ok !== 'boolean') throw Error('Invalid QA acknowledgement');
  const clean = { id: value.id, action: value.action, ok: value.ok };
  for (const key of ['at', 'startedAt', 'endedAt', 'duration']) if (Number.isFinite(value[key])) clean[key] = value[key];
  for (const key of ['published', 'unpublished']) if (typeof value[key] === 'boolean') clean[key] = value[key];
  return clean;
}
function createRelay({ upstream, worldId, user, room, userIds = [], helperScript = '', allowedHost, controlToken = '', audioRoot }) {
  const target = new URL(upstream);
  if (target.protocol !== 'http:' || target.hostname !== '127.0.0.1' || !WORLD.test(worldId) || !user?.id || !user.password || (allowedHost && !LOOPBACK.test(allowedHost))) throw Error('Relay must use a fixed QA user and loopback upstream');
  const status = { sequence: 0, latest: null, events: [], commands: [] };
  const queue = []; const issued = new Map(); let commandSequence = 0;
  const expected = { worldId, userId: user.id, userIds };
  async function activeWorld() {
    const response = await fetch(new URL('/api/status', target), { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw Error('QA status unavailable');
    const value = await response.json();
    return value.active === true && value.world === worldId;
  }
  const server = http.createServer(async (req, res) => {
    try {
      const requestUrl = new URL(req.url, target);
      if (!req.url.startsWith('/') || req.url.startsWith('//') || requestUrl.origin !== target.origin) { res.writeHead(403); return res.end('QA upstream route refused'); }
      const pathname = path.posix.normalize(decodeURIComponent(requestUrl.pathname).replace(/\\/g, '/')).replace(/\/+$/, '') || '/';
      const controllerPath = pathname.toLowerCase();
      if ((controllerPath === '/join' && req.method !== 'GET') || /^\/(setup|auth|quit)(?:\/|$)/.test(controllerPath)) { res.writeHead(403); return res.end('QA controller route refused'); }
      if (allowedHost && req.headers.host?.split(':')[0] !== allowedHost) { res.writeHead(403); return res.end('QA host refused'); }
      if (pathname === '/qa-latest' && req.method === 'GET') { res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store'); return res.end(JSON.stringify(status)); }
      if (!(await activeWorld())) { res.writeHead(409); return res.end('Expected QA world is inactive'); }
      if (pathname === '/qa-control') {
        const supplied = req.headers.authorization?.replace(/^Bearer /, '') ?? '';
        if (req.method !== 'POST' || !controlToken || supplied.length !== controlToken.length || !crypto.timingSafeEqual(Buffer.from(supplied), Buffer.from(controlToken))) throw Error('QA controller refused');
        const value = JSON.parse((await body(req)).toString());
        if (!ACTIONS.includes(value.action) || Object.keys(value).some(k => k !== 'action') || queue.length >= 16) throw Error('QA action refused');
        const command = { id: ++commandSequence, action: value.action }; queue.push(command); issued.set(command.id, command.action);
        res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store'); return res.end(JSON.stringify(command));
      }
      if (pathname === '/qa-command' || pathname === '/qa-command-result' || pathname.startsWith('/qa-clips/')) {
        if (!req.headers.cookie) throw Error('QA session required');
        if (pathname === '/qa-command' && req.method === 'GET') { res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store'); return res.end(JSON.stringify(queue.shift() ?? null)); }
        if (pathname === '/qa-command-result' && req.method === 'POST') {
          const clean = commandResult(JSON.parse((await body(req)).toString()));
          if (issued.get(clean.id) !== clean.action) throw Error('Unissued QA acknowledgement');
          issued.delete(clean.id); status.commands.push({ ...clean, receivedAt: new Date().toISOString() }); status.commands = status.commands.slice(-100);
          res.writeHead(204); return res.end();
        }
        const clip = /^\/qa-clips\/(gm|pl1|pl2)\.wav$/.exec(pathname);
        if (!clip || req.method !== 'GET' || !audioRoot) throw Error('QA clip refused');
        const root = fs.realpathSync(audioRoot), file = path.join(root, `${clip[1]}.wav`);
        if (fs.realpathSync(file) !== file || !fs.statSync(file).isFile()) throw Error('QA clip path refused');
        res.setHeader('Content-Type', 'audio/wav'); res.setHeader('Cache-Control', 'no-store'); return fs.createReadStream(file).pipe(res);
      }
      if (pathname.startsWith('/qa-login/')) {
        if (req.method !== 'GET' || pathname !== `/qa-login/${user.id}`) { res.writeHead(403); return res.end('Fixed QA user refused'); }
        const response = await fetch(new URL('/join', target), { method: 'POST', redirect: 'manual', headers: { 'Content-Type': 'application/json', Origin: target.origin }, body: JSON.stringify({ action: 'join', username: user.name, userId: user.id, password: user.password }), signal: AbortSignal.timeout(5000) });
        const result = await response.json();
        if (!response.ok || result.redirect !== '/game') { res.writeHead(502); return res.end('Native QA login failed'); }
        const cookies = response.headers.getSetCookie();
        if (!cookies.length) { res.writeHead(502); return res.end('Native QA session absent'); }
        res.writeHead(302, { 'Set-Cookie': cookies, Location: '/game', 'Cache-Control': 'no-store' }); return res.end();
      }
      if (pathname === '/qa-status' && req.method === 'POST') {
        if (!req.headers.cookie && allowedHost) { res.writeHead(403); return res.end('QA session required'); }
        const clean = sanitizeStatus(JSON.parse((await body(req)).toString()), expected);
        status.latest = { ...clean, receivedAt: new Date().toISOString(), sequence: ++status.sequence };
        status.events.push(...(clean.events ?? []).map((event) => ({ ...event, sequence: status.sequence })));
        status.events = status.events.slice(-300); res.writeHead(204); return res.end();
      }
      if (pathname === '/qa-helper.js' && req.method === 'GET') { res.setHeader('Content-Type', 'text/javascript'); res.setHeader('Cache-Control', 'no-store'); return res.end(helperScript); }
      const headers = { ...req.headers, host: target.host, 'accept-encoding': 'identity' };
      delete headers.origin;
      const proxy = http.request(new URL(req.url, target), { method: req.method, headers }, (response) => {
        const chunks = [];
        response.on('data', (chunk) => chunks.push(chunk));
        response.on('end', () => {
          let output = Buffer.concat(chunks); const responseHeaders = { ...response.headers };
          if (/text\/html/.test(responseHeaders['content-type'] ?? '')) {
            const bootstrap = bootstrapScript({ worldId, userId: user.id });
            const statusCode = room ? statusScript({ worldId, userId: user.id, room }) : '';
            // First head child guarantees the device guard precedes all app scripts.
            output = Buffer.from(output.toString().replace(/<head([^>]*)>/i, `<head$1><script>${bootstrap}</script><script>${statusCode}</script><script src="/qa-helper.js" defer></script>`));
          }
          delete responseHeaders['content-length']; delete responseHeaders['content-encoding']; responseHeaders['cache-control'] = 'no-store';
          res.writeHead(response.statusCode, responseHeaders); res.end(output);
        });
      });
      proxy.on('error', () => { if (!res.headersSent) res.writeHead(502); res.end('QA upstream unavailable'); });
      req.pipe(proxy);
    } catch { if (!res.headersSent) res.writeHead(403); res.end('QA request refused'); }
  });
  server.on('upgrade', async (req, socket, head) => {
    try {
      if ((allowedHost && req.headers.host?.split(':')[0] !== allowedHost) || !req.url.startsWith('/socket.io/') || new URL(req.url, target).origin !== target.origin || !(await activeWorld())) return socket.destroy();
      const proxy = http.request(new URL(req.url, target), { headers: { ...req.headers, host: target.host, origin: target.origin } });
      proxy.on('upgrade', (response, remote, remoteHead) => {
        socket.write(`HTTP/1.1 ${response.statusCode} ${response.statusMessage}\r\n${Object.entries(response.headers).map(([key, value]) => `${key}: ${value}`).join('\r\n')}\r\n\r\n`);
        if (remoteHead.length) socket.write(remoteHead); if (head.length) remote.write(head);
        remote.pipe(socket); socket.pipe(remote);
        remote.on('error', () => socket.destroy()); socket.on('error', () => remote.destroy());
      });
      proxy.on('error', () => socket.destroy()); proxy.end();
    } catch { socket.destroy(); }
  });
  return server;
}
module.exports = { createRelay, bootstrapScript, statusScript, sanitizeStatus, ACTIONS };
if (require.main === module) {
  try {
    const a = args(process.argv.slice(2)); if (!LOOPBACK.test(a.host) || !/^\d+$/.test(a.port)) throw Error('Explicit loopback listener required');
    const config = JSON.parse(fs.readFileSync(a.credentials)); const world = config.worlds.find((w) => w.worldId === a.world); const user = world?.users.find((u) => u.id === a.user);
    if (!user || !world.room.startsWith(`${world.worldId}-`)) throw Error('Fresh QA credential mapping absent');
    const control = a['control-config'] ? JSON.parse(fs.readFileSync(a['control-config'])) : null;
    if (control && (control.worldId !== world.worldId || control.userId !== user.id || !/^[a-f0-9]{64}$/.test(control.controlToken))) throw Error('QA controller identity mismatch');
    const server = createRelay({ upstream: a.upstream ?? 'http://127.0.0.1:30991', worldId: world.worldId, user, room: world.room, userIds: world.users.map((u) => u.id), allowedHost: a.host, helperScript: a.helper ? fs.readFileSync(a.helper, 'utf8') : '', controlToken: control?.controlToken, audioRoot: a['audio-root'] });
    server.listen(Number(a.port), a.host, () => console.log(JSON.stringify({ listening: true, host: a.host, port: Number(a.port), worldId: world.worldId, userId: user.id })));
    server.on('error', () => { console.error('QA relay listener failed'); process.exitCode = 1; });
  } catch { console.error('QA relay configuration refused'); process.exitCode = 1; }
}
