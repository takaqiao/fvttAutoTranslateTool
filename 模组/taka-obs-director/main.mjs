import { MODULE_ID, isDirectorRecorder, recorderUserId, registerSettings, startDirector } from './scripts/runtime.mjs';

/** Keep activation separate from the active client's data/image lifecycle. */
export function installDirector({ game: injectedGame, canvas, ui, hooks, document, libWrapper }) {
  let stop = null, ready = false, recorder = null;
  const listeners = [];
  const on = (name, handler) => { hooks.on(name, handler); listeners.push([name, handler]); };
  function reconcile() {
    const game = injectedGame ?? globalThis.game;
    const allowed = ready && isDirectorRecorder(game);
    const nextRecorder = recorderUserId(game);
    if (stop && (!allowed || nextRecorder !== recorder)) { stop(); stop = null; }
    if (allowed && !stop) {
      recorder = nextRecorder;
      stop = startDirector({ game, canvas: canvas ?? globalThis.canvas, ui: ui ?? globalThis.ui, hooks, document, libWrapper: libWrapper === undefined ? globalThis.libWrapper : libWrapper });
    }
  }
  on('init', () => registerSettings(injectedGame ?? globalThis.game));
  on('ready', () => { ready = true; reconcile(); });
  on('updateSetting', (setting) => {
    if ([`${MODULE_ID}.enabled`, `${MODULE_ID}.recorderUserId`, 'obs-utils.obsModeUser'].includes(setting.key)) reconcile();
  });
  on('updateUser', reconcile);
  function dispose() {
    stop?.(); stop = null;
    for (const [name, handler] of listeners.splice(0)) hooks.off(name, handler);
    document.defaultView?.removeEventListener?.('beforeunload', dispose);
  }
  document.defaultView?.addEventListener?.('beforeunload', dispose);
  return dispose;
}

if (globalThis.Hooks && globalThis.document) {
  installDirector({ hooks: globalThis.Hooks, document: globalThis.document });
}
