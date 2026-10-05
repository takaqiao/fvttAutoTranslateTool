import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';

const actions = ['start', 'stop', 'explore', 'combat', 'next-turn', 'share-image', 'close-image', 'sound'];

// Serialized into an isolated, fixed-user QA relay. Never included in delivery.
function installLiveKitQA(config) {
  if (!/^127\.0\.0\.[1-7]$/.test(location.hostname)) throw Error('QA origin refused');
  const qa = window.__directorQA;
  if (!qa) throw Error('QA bootstrap absent');
  let clipState = null, roomObserved = null, busy = false, statsBusy = false;
  const listeners = [], participantListeners = new Map(), events = [], received = new Map(), images = new Set();
  const imageSrc = 'icons/svg/book.svg';
  function native() {
    const game = globalThis.game, client = game?.webrtc?.client?._liveKitClient, room = client?.liveKitRoom;
    const configured = game?.ready && game.settings.get('avclient-livekit', 'liveKitConnectionSettings')?.room;
    const breakout = game?.webrtc?.client?.breakoutRoom ?? client?.breakoutRoom;
    if (!game?.ready || game.world.id !== config.worldId || game.user.id !== config.userId || configured !== config.room || (breakout && breakout !== config.room) || (room?.state === 'connected' && room.name !== config.room)) throw Error('QA identity/room guard refused');
    return { game, client, room };
  }
  function connected() { const state = native(); if (state.room?.state !== 'connected') throw Error('QA room disconnected'); return state; }
  function userId(participant) {
    const client = native().client;
    for (const [id, value] of client?.liveKitParticipants ?? []) if (value === participant && config.userIds.includes(id)) return id;
    try { const id = JSON.parse(participant?.metadata ?? '{}').fvttUserId; return config.userIds.includes(id) ? id : null; } catch { return null; }
  }
  function event(participant, type, speaking) { const id = userId(participant); if (id) { events.push({ userId: id, type, speaking: speaking === true, at: Date.now() }); if (events.length > 100) events.shift(); } }
  async function stop() {
    const current = clipState;
    if (!current) return { unpublished: true, endedAt: Date.now() };
    if (current.stopping) return current.stopping;
    current.stopping = (async () => {
      if (current.source) current.source.onended = null;
      try { current.source?.stop(); } catch {}
      try { if (current.published) await current.room.localParticipant.unpublishTrack(current.track); }
      finally { current.track?.stop(); await current.context.close(); if (clipState === current) clipState = null; }
      return { unpublished: true, endedAt: Date.now(), duration: current.duration, startedAt: current.startedAt };
    })();
    return current.stopping;
  }
  async function start() {
    const { room } = connected();
    if (!config.clip) throw Error('Recorder publication refused');
    if (clipState) throw Error('QA clip already active');
    const context = new AudioContext();
    const current = clipState = { room, context, published: false };
    try {
      await context.resume();
      const response = await fetch(`/qa-clips/${config.clip}.wav`); if (!response.ok) throw Error('QA clip unavailable');
      const buffer = await context.decodeAudioData(await response.arrayBuffer());
      if (!Number.isFinite(buffer.duration) || buffer.duration <= 0 || buffer.duration > 30) throw Error('Invalid QA duration');
      if (connected().room !== room || clipState !== current) throw Error('QA room replaced or clip cancelled');
      const destination = context.createMediaStreamDestination(), source = context.createBufferSource();
      source.buffer = buffer; source.connect(destination);
      const track = destination.stream.getAudioTracks()[0]; if (!track) throw Error('QA raw audio track absent');
      Object.assign(current, { source, track, duration: buffer.duration });
      await room.localParticipant.publishTrack(track, { name: `qa-clip-${config.clip}`, source: 'microphone', dtx: false });
      if (clipState !== current || connected().room !== room) { await room.localParticipant.unpublishTrack(track); track.stop(); throw Error('QA clip cancelled during publication'); }
      current.published = true; current.startedAt = Date.now();
      source.onended = () => { stop().then(result => { qa.lastClip = result; }).catch(() => { qa.helperErrors++; }); };
      source.start(); qa.lastClip = { published: true, startedAt: current.startedAt, duration: buffer.duration };
      return qa.lastClip;
    } catch (error) {
      if (current.source && current.track) await stop();
      else { await context.close(); if (clipState === current) clipState = null; }
      throw error;
    }
  }
  async function operation(action) {
    if (action === 'stop') { native(); return stop(); }
    if (action === 'start') return start();
    const { game } = native();
    if (action === 'close-image') { for (const app of [...images]) await app.close(); return { at: Date.now() }; }
    if (!game.user.isGM) throw Error('QA GM operation refused');
    const owned = [...game.combats].filter(c => c.flags?.['taka-obs-director']?.qaHelper === true);
    if (action === 'explore') {
      for (const combat of owned) await combat.delete();
      const scene = game.scenes.active ?? game.scenes.current; if (!scene) throw Error('QA scene absent'); await scene.view();
    } else if (action === 'combat') {
      if (game.combats.active && !owned.includes(game.combats.active)) throw Error('Unowned combat refused');
      const scene = game.scenes.active ?? game.scenes.current;
      if (!scene) throw Error('QA scene absent');
      const actors = new Set(config.userIds.map(id => game.users.get(id)?.character?.id).filter(Boolean));
      const combatants = [...scene.tokens].filter(token => actors.has(token.actorId)).map((token, index) => ({ tokenId: token.id, actorId: token.actorId, sceneId: scene.id, initiative: 20 - index }));
      if (!combatants.length) throw Error('Bound QA PC tokens absent');
      const combat = owned[0] ?? await game.combats.documentClass.create({ scene: scene.id, active: true, combatants, flags: { 'taka-obs-director': { qaHelper: true } } });
      if (!combat.started) await combat.startCombat();
    } else if (action === 'next-turn') {
      const combat = game.combats.active; if (!owned.includes(combat)) throw Error('Unowned combat refused'); await combat.nextTurn();
    } else if (action === 'share-image') {
      const app = new foundry.applications.apps.ImagePopout({ src: imageSrc, window: { title: 'QA native shared image' } });
      images.add(app); await app.render({ force: true }); app.shareImage({ users: config.userIds });
    } else if (action === 'sound') {
      const AudioHelper = foundry.audio.AudioHelper;
      await AudioHelper.play({ src: 'sounds/lock.wav', volume: 0.5, loop: false }, { recipients: config.userIds });
    } else throw Error('QA operation refused');
    return { at: Date.now() };
  }
  qa.controls = { start, stop, run: operation };
  qa.observe = () => {
    try {
      const { room } = native();
      const people = [room?.localParticipant, ...room?.remoteParticipants?.values?.() ?? []].filter(Boolean);
      const speakingUserIds = people.filter(p => p.isSpeaking === true).map(userId).filter(Boolean);
      const audioPublications = people.map(p => ({ userId: userId(p), count: p.audioTrackPublications?.size ?? 0, unmuted: [...p.audioTrackPublications?.values?.() ?? []].filter(publication => publication.isMuted !== true).length })).filter(p => p.userId);
      const renderedSpeakingUserIds = [...document.querySelectorAll('#taka-obs-director .cast-person.speaking')].map(el => el.dataset.userId).filter(id => config.userIds.includes(id));
      const portraits = [...document.querySelectorAll('#taka-obs-director .cast-person')].filter(el => config.userIds.includes(el.dataset.userId)).map(el => {
        const style = getComputedStyle(el.querySelector('.portrait, .gm-logo') ?? el), transform = style.transform;
        const matrix = transform?.match(/^matrix\(([^)]+)\)$/)?.[1].split(',').map(Number);
        const outline = getComputedStyle(el), figure = getComputedStyle(el.querySelector('.figure, .gm-logo') ?? el);
        return { userId: el.dataset.userId, scale: matrix ? Math.hypot(matrix[0], matrix[1]) : 1, outlineWidth: parseFloat(outline.outlineWidth) || 0, outlineColor: outline.outlineColor ?? '', filter: figure.filter ?? '' };
      });
      const root = document.getElementById('taka-obs-director');
      return { mode: root?.dataset.mode, focusName: root?.querySelector('.focus-name')?.textContent ?? '', focusHP: root?.querySelector('.focus-hp')?.textContent ?? '', focusResourceText: root?.querySelector('.counters')?.textContent ?? '', publicFocus: !!root?.querySelector('.focus-content') && !root.querySelector('.focus-hp'), speakingUserIds, renderedSpeakingUserIds, audioPublications, portraits, audioReceivedBytes: [...received].map(([userId, bytes]) => ({ userId, bytes })), events: events.splice(0, 30) };
    } catch { return {}; }
  };
  function detach() {
    for (const [emitter, name, callback] of listeners.splice(0)) emitter.off(name, callback);
    for (const [participant, callback] of participantListeners) participant.off('isSpeakingChanged', callback);
    participantListeners.clear(); roomObserved = null; received.clear();
  }
  async function telemetry() {
    const { room } = native(); if (!room) return;
    if (room !== roomObserved) {
      detach(); roomObserved = room;
      const callback = speakers => { for (const participant of [room.localParticipant, ...room.remoteParticipants.values()]) event(participant, 'activeSpeakersChanged', speakers.includes(participant)); };
      room.on('activeSpeakersChanged', callback); listeners.push([room, 'activeSpeakersChanged', callback]);
    }
    const people = [room.localParticipant, ...room.remoteParticipants.values()];
    for (const [participant, callback] of participantListeners) if (!people.includes(participant)) { participant.off('isSpeakingChanged', callback); participantListeners.delete(participant); }
    for (const participant of people) if (!participantListeners.has(participant)) { const callback = speaking => event(participant, 'isSpeakingChanged', speaking); participant.on('isSpeakingChanged', callback); participantListeners.set(participant, callback); }
    if (statsBusy) return; statsBusy = true;
    try {
      for (const participant of room.remoteParticipants.values()) {
        const id = userId(participant); if (!id) continue; let bytes = 0;
        for (const publication of participant.audioTrackPublications.values()) {
          const report = await publication.track?.getRTCStatsReport?.();
          report?.forEach(entry => { if (entry.type === 'inbound-rtp' && (entry.kind === 'audio' || entry.mediaType === 'audio') && Number.isFinite(entry.bytesReceived)) bytes += entry.bytesReceived; });
        }
        received.set(id, bytes);
      }
    } finally { statsBusy = false; }
  }
  let uiInstalled = false;
  function controlsUI() {
    if (uiInstalled) return; uiInstalled = true;
    globalThis.Hooks?.on('renderImagePopout', app => { if (app.options?.src === imageSrc) images.add(app); });
    globalThis.Hooks?.on('closeImagePopout', app => images.delete(app));
    if (!config.clip) return;
    const panel = document.createElement('aside'); panel.id = 'director-qa-controls'; panel.style.cssText = 'position:fixed;top:8px;left:8px;z-index:100000;background:#222;color:white;padding:6px;max-width:600px';
    const status = document.createElement('output'); status.setAttribute('aria-live', 'polite');
    for (const action of (globalThis.game.user.isGM ? ['start', 'stop', 'explore', 'combat', 'next-turn', 'share-image', 'close-image', 'sound'] : ['start', 'stop', 'close-image'])) {
      const button = document.createElement('button'); button.textContent = `QA ${action}`; button.setAttribute('data-qa-action', action);
      button.addEventListener('click', async () => { try { const result = await operation(action); status.textContent = JSON.stringify({ action, ok: true, ...result }); } catch { status.textContent = `${action}: refused`; qa.helperErrors++; } }); panel.append(button);
    }
    panel.append(status); document.body.append(panel);
  }
  const timer = setInterval(async () => {
    if (!globalThis.game?.ready) return;
    try {
      native(); controlsUI(); await telemetry();
      if (busy) return; busy = true;
      try {
        const response = await fetch('/qa-command'); if (!response.ok) return;
        const command = await response.json(); if (!command) return;
        let result;
        try { result = { ...command, ok: true, ...await operation(command.action) }; }
        catch { result = { ...command, ok: false, at: Date.now() }; qa.helperErrors++; }
        await fetch('/qa-command-result', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(result) });
      } finally { busy = false; }
    } catch { qa.helperErrors++; }
  }, 250);
  window.addEventListener('beforeunload', () => { clearInterval(timer); detach(); stop().catch(() => {}); }, { once: true });
}

export function createHelperScript(config) {
  if (!/^obs-director-qa-[a-z]+$/.test(config.worldId) || !/^[A-Za-z0-9]{1,16}$/.test(config.userId) || !config.userIds?.includes(config.userId) || !config.room?.startsWith(`${config.worldId}-`) || (config.clip && !['gm', 'pl1', 'pl2'].includes(config.clip))) throw Error('Invalid fixed QA helper configuration');
  const safe = { worldId: config.worldId, userId: config.userId, room: config.room, userIds: config.userIds, clip: config.clip ?? null };
  return `(${installLiveKitQA.toString()})(${JSON.stringify(safe)});`;
}

function argumentsOf(argv) { const a = {}; for (let i = 0; i < argv.length; i += 2) { if (!argv[i].startsWith('--') || argv[i + 1] === undefined) throw Error('Expected named arguments'); a[argv[i].slice(2)] = argv[i + 1]; } return a; }
export function prepareHelpers({ credentials, worldId, output }) {
  if (fs.existsSync(output)) throw Error('Helper output must be new');
  const world = JSON.parse(fs.readFileSync(credentials)).worlds.find(w => w.worldId === worldId);
  if (!world) throw Error('QA world absent');
  fs.mkdirSync(output, { recursive: true, mode: 0o700 });
  const clips = { QAGM000000000001: 'gm', QAPL000000000001: 'pl1', QAPL000000000002: 'pl2' };
  for (const user of world.users) {
    const config = { worldId, userId: user.id, room: world.room, userIds: world.users.map(u => u.id), clip: clips[user.id] ?? null };
    fs.writeFileSync(path.join(output, `${user.id}.js`), createHelperScript(config), { mode: 0o600 });
    fs.writeFileSync(path.join(output, `${user.id}.json`), JSON.stringify({ worldId, userId: user.id, controlToken: crypto.randomBytes(32).toString('hex') }), { mode: 0o600 });
  }
  return { worldId, seats: world.users.length, publishers: 3 };
}
export async function sendCommand({ config, url, action }) {
  if (!actions.includes(action)) throw Error('Unknown QA action');
  const target = new URL(url), privateConfig = JSON.parse(fs.readFileSync(config));
  if (target.protocol !== 'http:' || !/^127\.0\.0\.[1-7]$/.test(target.hostname) || target.username || target.password || target.pathname !== '/') throw Error('QA controller URL refused');
  const response = await fetch(new URL('/qa-control', target), { method: 'POST', headers: { authorization: `Bearer ${privateConfig.controlToken}`, 'content-type': 'application/json' }, body: JSON.stringify({ action }), signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw Error('QA command refused');
  const command = await response.json();
  for (let i = 0; i < 40; i++) {
    const latest = await fetch(new URL('/qa-latest', target), { signal: AbortSignal.timeout(5000) });
    const result = (await latest.json()).commands?.find(value => value.id === command.id);
    if (result) { if (!result.ok) throw Error('Native QA operation failed'); return result; }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw Error('QA acknowledgement timed out');
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const [command, ...argv] = process.argv.slice(2), a = argumentsOf(argv);
    if (command === 'prepare') console.log(JSON.stringify(prepareHelpers({ credentials: a.credentials, worldId: a.world, output: a.output })));
    else if (command === 'command') console.log(JSON.stringify(await sendCommand({ config: a.config, url: a.url, action: a.action })));
    else throw Error('Usage: prepare --credentials FILE --world QA_ID --output NEW_PRIVATE_DIR; command --config SEAT.json --url http://127.0.0.N:PORT --action ACTION');
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
