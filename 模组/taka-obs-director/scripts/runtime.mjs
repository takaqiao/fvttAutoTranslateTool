import { resolveMode, selectCastUsers, selectFocus } from './model.mjs';
import { collectActorView } from './pf2e.mjs';
import { mountDirector } from './view.mjs';
import { resolveSkin } from './skins.mjs';
import { applyRecorderPreferences } from './client.mjs';
import { suspendRecorderViewportBroadcast } from './viewport.mjs';
import { resolvePortraitLayout } from './portrait-layout.mjs';
import { portraitDefaults } from './portrait-defaults.mjs';
import { createPortraitMenu } from './portrait-editor.mjs';

export const MODULE_ID = 'taka-obs-director';
const RESIZE_TARGET = 'foundry.canvas.Canvas.prototype._onResize';
const ACTIVE_CLASS = 'taka-obs-director-active';
const string = (value) => typeof value === 'string' ? value : '';
function setting(game, key, fallback) {
  try { return game.settings.get(MODULE_ID, key) ?? fallback; } catch { return fallback; }
}
export function recorderUserId(game) {
  const configured = string(setting(game, 'recorderUserId', ''));
  if (configured) return configured;
  try { return string(game.settings.get('obs-utils', 'obsModeUser')); } catch { return ''; }
}
export function isDirectorRecorder(game) {
  return setting(game, 'enabled', false) === true && game.system?.id === 'pf2e'
    && game.view === 'game' && game.user?.isGM === false
    && !!recorderUserId(game) && game.user.id === recorderUserId(game);
}
export function registerSettings(game) {
  const register = (key, definition) => game.settings.register(MODULE_ID, key, { config: true, ...definition });
  register('enabled', { name: 'Enable OBS director', scope: 'world', type: Boolean, default: false });
  register('recorderUserId', { name: 'Recorder user ID', hint: 'Empty uses the OBS Utils recorder user.', scope: 'world', type: String, default: '' });
  register('seatOrder', { name: 'Cast user IDs in seat order', scope: 'world', type: Array, default: [] });
  register('portraitLayouts', { name: '角色卡立绘校准数据', scope: 'world', type: Object, default: {}, config: false });
  const menu = createPortraitMenu({ game, document: globalThis.document, DialogV2: globalThis.foundry?.applications?.api?.DialogV2, defaults: portraitDefaults });
  if (menu && game.settings.registerMenu) game.settings.registerMenu(MODULE_ID, 'portraitCalibration', { name: '角色卡立绘校准', label: '调整立绘位置', icon: 'fa-solid fa-up-down-left-right', type: menu, restricted: true });
  register('portraitOverrides', { name: 'Original portraits by Actor ID', scope: 'world', type: Object, default: {} });
  register('skin', { name: 'World artwork', scope: 'world', type: String, default: 'auto', choices: { auto: 'Automatic', cotct: 'CotCT', sog: 'SoG', fotrp: 'FotRP', av: 'AV', bob: 'BoB' } });
  register('manualMode', { name: 'Director mode on this client', scope: 'client', type: String, default: 'auto', choices: { auto: 'Automatic', explore: 'Exploration', combat: 'Combat', story: 'Story image' } });
  register('resourceDetail', { name: '录像侧栏显示', scope: 'world', type: String, default: 'compact', choices: { compact: '简洁（最高三个法术环阶）', full: '详细（全部法术环阶）' } });
}

/** Owns one recorder layout and its native client preferences. */
export function startDirector({ game, canvas, ui, hooks, document, libWrapper = globalThis.libWrapper }) {
  if (!isDirectorRecorder(game) || !libWrapper?.register || !libWrapper?.unregister) return () => {};
  const configuredSkin = () => resolveSkin(setting(game, 'skin', 'auto')) ?? resolveSkin(game.world?.id) ?? resolveSkin('cotct');
  const view = mountDirector(document, { skin: configuredSkin().id });
  const board = document.getElementById('board');
  const originalBoardStyle = board?.getAttribute('style');
  const ownedHooks = [], images = new Map(), snapshots = new Map();
  const pendingActors = new Set(), actorVersions = new Map();
  const speaking = new Set(), offlineUsers = new Set();
  const participants = new Map(), roomListeners = [];
  let generation = 0, queued = false, disposed = false, room = null, roomConnected = false;
  let renderPending = false;
  let viewportApplied = false, canvasAvailable = canvas?.ready === true, resizeObserver = null, cameraObserver = null;
  document.body.classList.add(ACTIVE_CLASS);

  const on = (name, handler) => { hooks.on(name, handler); ownedHooks.push([name, handler]); };
  const users = () => game.users?.contents ?? Array.from(game.users?.values?.() ?? game.users ?? []);
  const userById = (id) => game.users?.get?.(id) ?? users().find((user) => user.id === id);
  function castUsers() {
    const order = setting(game, 'seatOrder', []);
    return selectCastUsers({ users: users(), recorderId: recorderUserId(game), seatOrder: Array.isArray(order) ? order : [], connectedUserIds: roomConnected ? [...participants.values()].map((entry) => entry.id).filter((id) => !offlineUsers.has(id)) : [] });
  }
  function assigned(actor, requestGeneration) {
    return !disposed && generation === requestGeneration && isDirectorRecorder(game)
      && castUsers().some((user) => !user.isGM && user.character === actor && user.character?.id === actor?.id);
  }
  function observed(actor, requestGeneration) {
    try { return assigned(actor, requestGeneration) && actor.testUserPermission?.(game.user, 'OBSERVER') === true && actor.type === 'character'; } catch { return false; }
  }
  function publicFocus() {
    const combatant = game.combat?.started === true ? game.combat.combatant : null;
    const allowedActorIds = castUsers().filter((user) => !user.isGM).map((user) => user.character?.id).filter(Boolean);
    return selectFocus({ combatant, allowedActorIds, viewer: game.user });
  }
  const currentMode = () => resolveMode({ manualMode: setting(game, 'manualMode', 'auto'), publicImages: images, combatStarted: game.combat?.started === true });
  function render(clear = false) {
    if (disposed) return;
    const current = castUsers();
    const actorSnapshot = (actor) => !clear && observed(actor, generation) ? snapshots.get(actor) ?? null : null;
    const focus = clear ? null : publicFocus();
    const focusView = focus?.actor && actorSnapshot(focus.actor);
    const publicView = focus ? { name: focus.name, portrait: focus.actor ? null : focus.portrait, publicOnly: true } : null;
    const portraitLayout = focusView ? resolvePortraitLayout({ worldId: game.world?.id, actorId: focusView.actorId, source: focusView.portrait, overrides: setting(game, 'portraitLayouts', {}), defaults: portraitDefaults }) : null;
    const mode = currentMode();
    const story = [...images.values()].at(-1) ?? null;
    view.render({
      skin: configuredSkin().id, worldId: game.world?.id,
      resourceDetail: setting(game, 'resourceDetail', 'compact'),
      mode,
      cast: current.map((user) => {
        const data = user.isGM ? null : actorSnapshot(user.character);
        return { userId: user.id, role: user.isGM ? 'gm' : 'pc', online: user.active === true || (roomConnected && [...participants.values()].some((entry) => entry.id === user.id) && !offlineUsers.has(user.id)), name: data?.name ?? string(user.name), portrait: data?.portrait ?? null, hp: data?.hp ?? null };
      }),
      focus: focus ? { ...(focusView || publicView), portraitLayout } : null, story, speakingUserIds: [...speaking], scale: 1.06,
    });
    syncViewport();
  }
  function collect() {
    if (disposed) return;
    const requestGeneration = generation;
    const actors = [...pendingActors];
    pendingActors.clear();
    if (renderPending) render();
    renderPending = false;
    const override = setting(game, 'portraitOverrides', {});
    actors.filter(actor => observed(actor, requestGeneration)).forEach(actor => {
      const version = actorVersions.get(actor);
      const current = () => !disposed && generation === requestGeneration && actorVersions.get(actor) === version;
      const portraitOverride = override && Object.hasOwn(override, actor.id) ? string(override[actor.id]) : '';
      collectActorView(actor, {
        viewer: game.user, allowedActorIds: castUsers().filter((user) => !user.isGM).map((user) => user.character?.id).filter(Boolean),
        isStillAssigned: (document) => current() && assigned(document, requestGeneration), portraitOverride,
        localize: (label) => game.i18n?.localize?.(label) ?? label,
      }).catch(() => null).then((snapshot) => {
        if (!current()) return;
        if (snapshot && observed(actor, requestGeneration)) snapshots.set(actor, snapshot);
        else snapshots.delete(actor);
        // Completed actors can paint without waiting for another sheet. The
        // shared microtask coalesces results that finish in the same turn.
        renderPending = true;
        queueCollect();
      });
    });
  }
  function queueCollect() {
    if (queued || disposed) return;
    queued = true;
    queueMicrotask(() => { queued = false; collect(); });
  }
  function invalidateActor(actor) {
    actorVersions.set(actor, (actorVersions.get(actor) ?? 0) + 1);
    snapshots.delete(actor);
  }
  function checkAccess() {
    if (disposed) return false;
    if (!isDirectorRecorder(game)) { dispose(); return false; }
    let cleared = false;
    for (const actor of snapshots.keys()) if (!observed(actor, generation)) {
      invalidateActor(actor);
      cleared = true;
    }
    if (cleared) render();
    return true;
  }
  function refreshActors(actors, deleted = false) {
    if (!checkAccess()) return;
    let denied = false;
    for (const actor of actors) {
      invalidateActor(actor);
      pendingActors.delete(actor);
      if (!deleted && observed(actor, generation)) pendingActors.add(actor);
      else denied = true;
    }
    if (denied) render();
    if (pendingActors.size) { renderPending = true; queueCollect(); }
  }
  function refreshPublic() {
    if (!checkAccess()) return;
    renderPending = true;
    queueCollect();
  }
  function refreshActor(actor, deleted = false) {
    const relevant = castUsers().some(user => !user.isGM && user.character === actor);
    if (relevant) refreshActors([actor], deleted);
    else checkAccess();
  }
  function refreshItem(item) {
    refreshActor(item?.parent);
  }
  function refreshToken(token, changes = {}) {
    // Coordinates/visibility only change the public focus. Never read the
    // token's actor getter: an unrelated token may represent a private NPC.
    if (Object.keys(changes).some(key => key === 'delta' || key.startsWith('delta.') || key === 'actorData' || key.startsWith('actorData.'))) {
      const actors = castUsers().filter(user => !user.isGM && user.character?.id === token?.actorId).map(user => user.character);
      if (actors.length) refreshActors(new Set(actors));
    }
    refreshPublic();
  }
  function refreshTime() {
    if (!checkAccess()) return;
    const actors = new Set(castUsers().filter(user => !user.isGM).map(user => user.character).filter(Boolean));
    const timed = [...actors].filter(actor => {
      if (!observed(actor, generation)) return false;
      const items = actor.itemTypes;
      const active = actor.conditions?.active;
      return !!(items?.effect?.length || items?.condition?.length || active?.length || active?.size);
    });
    if (timed.length) refreshActors(timed);
  }
  function refresh() {
    if (disposed) return;
    if (!isDirectorRecorder(game)) { dispose(); return; }
    generation++;
    snapshots.clear();
    pendingActors.clear(); actorVersions.clear();
    render(true);
    for (const user of castUsers()) if (!user.isGM && user.character && observed(user.character, generation)) pendingActors.add(user.character);
    renderPending = false;
    queueCollect();
  }

  function syncViewport(force = false) {
    if (disposed || !canvasAvailable || canvas?.ready !== true) return;
    const bounds = view.sceneBounds(), renderer = canvas.app?.renderer;
    if (!bounds || !renderer || !canvas.stage || !board) return;
    if (![bounds.left, bounds.top, bounds.width, bounds.height].every(Number.isFinite) || bounds.width <= 0 || bounds.height <= 0) return;
    const dimensions = canvas.screenDimensions;
    const changed = force || !viewportApplied || dimensions?.[0] !== bounds.width || dimensions?.[1] !== bounds.height;
    board.style.position = 'fixed'; board.style.left = `${bounds.left}px`; board.style.top = `${bounds.top}px`;
    board.style.width = `${bounds.width}px`; board.style.height = `${bounds.height}px`;
    if (changed) {
      const pivot = { x: canvas.stage.pivot.x, y: canvas.stage.pivot.y };
      renderer.resize(bounds.width, bounds.height);
      canvas.screenDimensions = [bounds.width, bounds.height];
      canvas.stage.position.set(bounds.width / 2, bounds.height / 2);
      canvas.pan(pivot);
    }
    viewportApplied = true;
  }
  try {
    libWrapper.register(MODULE_ID, RESIZE_TARGET, function (wrapped, ...args) {
      const result = wrapped(...args);
      if (!disposed && this === canvas && isDirectorRecorder(game)) syncViewport(true);
      return result;
    }, 'WRAPPER');
  } catch {
    view.destroy(); document.body.classList.remove(ACTIVE_CLASS);
    ui?.notifications?.warn?.('OBS director canvas hook is unavailable.');
    return () => {};
  }
  const ResizeObserver = document.defaultView?.ResizeObserver ?? globalThis.ResizeObserver;
  if (ResizeObserver) { resizeObserver = new ResizeObserver(() => syncViewport()); resizeObserver.observe(document.getElementById('taka-obs-director')); }

  function updateSpeaking(activeParticipants) {
    if (disposed) return;
    speaking.clear();
    if (roomConnected) {
      const active = activeParticipants ? new Set(activeParticipants) : null;
      for (const [participant, { id }] of participants) {
        const user = userById(id);
        const audio = participant.audioTrackPublications;
        const hasAudio = !(audio instanceof Map) || [...audio.values()].some(publication => publication.isMuted !== true);
        if (hasAudio && user && user.active !== false && !offlineUsers.has(id) && (active ? active.has(participant) : participant.isSpeaking === true)) speaking.add(id);
      }
    }
    view.setSpeaking([...speaking]);
  }
  function participantId(participant, map) {
    if (map instanceof Map) for (const [id, value] of map) if (value === participant && userById(id)) return id;
    try { const id = JSON.parse(participant.metadata).fvttUserId; return typeof id === 'string' && userById(id) ? id : null; } catch { return null; }
  }
  function attachParticipant(participant, map) {
    const id = participantId(participant, map);
    const existing = participants.get(participant);
    if (existing?.id === id) return;
    if (existing) detachParticipant(participant);
    if (!id || typeof participant.on !== 'function') return;
    const handler = () => updateSpeaking();
    participant.on('isSpeakingChanged', handler);
    participants.set(participant, { id, handler });
  }
  function detachParticipant(participant) {
    const entry = participants.get(participant);
    if (entry) participant.off?.('isSpeakingChanged', entry.handler);
    participants.delete(participant);
  }
  function detachRoom() {
    for (const [event, handler] of roomListeners.splice(0)) room?.off?.(event, handler);
    for (const participant of [...participants.keys()]) detachParticipant(participant);
    room = null; roomConnected = false;
    updateSpeaking();
  }
  function reconcileLiveKit() {
    if (disposed) return;
    const previousSeats = castUsers().map((user) => user.id).join(',');
    const client = game.webrtc?.client?._liveKitClient;
    const nextRoom = client?.liveKitRoom ?? null;
    if (nextRoom !== room) {
      detachRoom(); room = nextRoom;
      if (room?.on) {
        roomConnected = room.state === 'connected';
        const listen = (event, handler) => { room.on(event, handler); roomListeners.push([event, handler]); };
        listen('participantConnected', (participant) => { attachParticipant(participant, client.liveKitParticipants); updateSpeaking(); refresh(); });
        listen('participantDisconnected', (participant) => { detachParticipant(participant); updateSpeaking(); refresh(); });
        listen('activeSpeakersChanged', (active) => updateSpeaking(active));
        for (const event of ['trackPublished', 'trackUnpublished', 'trackMuted', 'trackUnmuted']) listen(event, () => updateSpeaking());
        for (const event of ['connected', 'reconnected']) listen(event, () => {
          roomConnected = true;
          for (const id of offlineUsers) if (userById(id)?.active === true) offlineUsers.delete(id);
          reconcileLiveKit(); updateSpeaking(); refresh();
        });
        for (const event of ['disconnected', 'reconnecting']) listen(event, () => { roomConnected = false; for (const participant of [...participants.keys()]) detachParticipant(participant); updateSpeaking(); refresh(); });
      }
    }
    if (roomConnected && client?.liveKitParticipants instanceof Map) {
      const currentParticipants = new Set([...client.liveKitParticipants.values(), ...(room.remoteParticipants?.values?.() ?? []), ...(room.localParticipant ? [room.localParticipant] : [])]);
      for (const participant of [...participants.keys()]) if (!currentParticipants.has(participant)) detachParticipant(participant);
      for (const participant of currentParticipants) attachParticipant(participant, client.liveKitParticipants);
    }
    updateSpeaking();
    observeCameras();
    if (previousSeats !== castUsers().map((user) => user.id).join(',')) refresh();
  }
  function observeCameras() {
    cameraObserver?.disconnect(); cameraObserver = null;
    const cameras = ui?.webrtc?.element ?? document.getElementById('camera-views');
    const MutationObserver = document.defaultView?.MutationObserver ?? globalThis.MutationObserver;
    if (!cameras || !MutationObserver || participants.size) return;
    const read = () => {
      if (disposed || participants.size || room && !roomConnected) return;
      speaking.clear();
      for (const node of cameras.querySelectorAll('.camera-view.speaking[data-user]')) {
        const id = node.dataset.user;
        if (userById(id)?.active !== false && userById(id) && !offlineUsers.has(id)) speaking.add(id);
      }
      view.setSpeaking([...speaking]);
    };
    cameraObserver = new MutationObserver(read);
    cameraObserver.observe(cameras, { attributes: true, attributeFilter: ['class'], childList: true, subtree: true });
    read();
  }
  on('liveKitClientAvailable', reconcileLiveKit);
  on('liveKitClientInitialized', reconcileLiveKit);
  on('renderCameraViews', () => { reconcileLiveKit(); });
  on('userConnected', (user, connected) => {
    if (connected === false) offlineUsers.add(user.id); else offlineUsers.delete(user.id);
    updateSpeaking(); refresh();
  });
  on('updateUser', () => { updateSpeaking(); refresh(); });
  on('deleteUser', () => { updateSpeaking(); refresh(); });
  on('createUser', refresh);
  for (const event of ['createActor', 'updateActor']) on(event, actor => refreshActor(actor));
  on('deleteActor', actor => refreshActor(actor, true));
  for (const event of ['createItem', 'updateItem', 'deleteItem']) on(event, refreshItem);
  for (const event of ['createCombat', 'updateCombat', 'deleteCombat', 'createCombatant', 'updateCombatant', 'deleteCombatant', 'deleteToken']) on(event, refreshPublic);
  on('updateToken', refreshToken);
  on('updateWorldTime', refreshTime);
  on('updateSetting', changed => {
    if ([`${MODULE_ID}.enabled`, `${MODULE_ID}.recorderUserId`, `${MODULE_ID}.seatOrder`, `${MODULE_ID}.portraitOverrides`, 'obs-utils.obsModeUser'].includes(changed.key)) refresh();
    else if (changed.key?.startsWith(`${MODULE_ID}.`)) refreshPublic();
  });
  on('canvasReady', () => { canvasAvailable = true; syncViewport(true); refresh(); });
  on('canvasTearDown', () => { canvasAvailable = false; viewportApplied = false; generation++; snapshots.clear(); pendingActors.clear(); actorVersions.clear(); renderPending = false; render(true); });
  on('renderImagePopout', (app) => {
    const src = string(app.options?.src);
    if (!app.id || !src) return;
    images.delete(app.id); images.set(app.id, { src, title: string(app.title) }); refreshPublic();
  });
  on('closeImagePopout', (app) => { if (images.delete(app.id)) refreshPublic(); });
  const window = document.defaultView;
  window?.addEventListener?.('beforeunload', dispose);
  window?.addEventListener?.('resize', syncWindow);
  function syncWindow() { syncViewport(); }
  // The plugin can replace its room during routing/reconnect without a new
  // Foundry hook. This timer only discovers listeners, never directs the map.
  const recorderPreferences = applyRecorderPreferences({ game, isActive: () => !disposed && isDirectorRecorder(game) });
  const restoreViewportBroadcast = suspendRecorderViewportBroadcast({ game, hooks, isActive: () => !disposed && isDirectorRecorder(game) });
  const roomTimer = setInterval(reconcileLiveKit, 1000);
  roomTimer.unref?.();

  function dispose() {
    if (disposed) return;
    disposed = true; generation++; snapshots.clear(); images.clear(); speaking.clear();
    void recorderPreferences.dispose(); restoreViewportBroadcast();
    pendingActors.clear(); actorVersions.clear();
    for (const [name, handler] of ownedHooks.splice(0)) hooks.off(name, handler);
    clearInterval(roomTimer); detachRoom(); resizeObserver?.disconnect(); cameraObserver?.disconnect();
    window?.removeEventListener?.('beforeunload', dispose); window?.removeEventListener?.('resize', syncWindow);
    libWrapper?.unregister?.(MODULE_ID, RESIZE_TARGET);
    view.destroy(); document.body.classList.remove(ACTIVE_CLASS);
    if (board) { if (originalBoardStyle === null) board.removeAttribute('style'); else board.setAttribute('style', originalBoardStyle); }
    if (viewportApplied && canvas?.ready === true) canvas._onResize?.();
  }
  reconcileLiveKit(); refresh();
  return dispose;
}
