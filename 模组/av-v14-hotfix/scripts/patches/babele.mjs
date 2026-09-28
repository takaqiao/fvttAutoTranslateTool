/** Foundry v14 / Babele on-demand adapter. No upstream source or prototype edits. */
const TARGET = 'pf2e_compendium_chn';
const PACK_TARGET = 'foundry.documents.collections.CompendiumCollection.prototype.indexDocument';
const SIGNATURES = Object.freeze({pack: '00f29d8bebbb7ae9', index: 'f5af42c0e64fa640', ready: 'cd61f663b9cd8c1e', preserveIndexFlags: '2f7f5cccb742870f'});
const captures = new WeakMap();
const registrations = new WeakMap();

// An exact normalized-source compatibility gate, not a security hash. Never executes source text.
function signature(fn) {
  if (typeof fn !== 'function') return null;
  let hash = 0xcbf29ce484222325n;
  for (const char of Function.prototype.toString.call(fn).replace(/\s/g, '')) {
    hash ^= BigInt(char.charCodeAt(0));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

// libWrapper 1.13.5.1 snapshots this effective priority into each wrapper at
// registration. Bare IDs are not module keys; malformed entries are ignored.
function wrapperPriority(g, owner) {
  const configuration = g.game?.settings?.get?.('lib-wrapper', 'module-priorities');
  if (!configuration || typeof configuration !== 'object' || Array.isArray(configuration)) {
    throw new Error('Unrecognized libWrapper module-priorities configuration.');
  }
  const key = `module:${owner}`;
  for (const [section, base] of [['prioritized', 10000], ['deprioritized', -10000]]) {
    const entries = configuration[section];
    if (!entries) continue;
    if (typeof entries !== 'object' || Array.isArray(entries)) throw new Error('Malformed libWrapper priority section.');
    if (!Object.hasOwn(entries, key)) continue;
    const entry = entries[key];
    if (!entry || typeof entry.id !== 'string' || typeof entry.title !== 'string' || typeof entry.index !== 'number') continue;
    const priority = base - entry.index;
    if (!Number.isFinite(priority)) throw new Error('Nonfinite libWrapper module priority is unsupported.');
    return priority;
  }
  return 0;
}

/** Call at ESM evaluation, before libWrapper.Ready. A late/already wrapped capture is refused. */
export function captureBabeleCore(g = globalThis) {
  if (captures.has(g)) return captures.get(g);
  const proto = g.foundry?.documents?.collections?.CompendiumCollection?.prototype;
  const native = proto && Object.getOwnPropertyDescriptor(proto, 'indexDocument')?.value;
  const capture = {g, proto, native, valid: signature(native) === SIGNATURES.pack,
    knownRegistered: false, babeleRegistered: false, unknown: null, registrations: 0, babeleRegistrations: 0,
    wrapperOrder: [], priorityError: null, listeners: new Set(), hooks: []};
  captures.set(g, capture);
  const changed = () => { for (const listener of capture.listeners) listener(); };
  const add = (name, callback) => {
    if (typeof g.Hooks?.on === 'function') capture.hooks.push([name, g.Hooks.on(name, callback)]);
  };
  const relevant = target => typeof target === 'string'
    && (/\.indexDocument$/.test(target) || /DocumentIndex\.prototype\.(index|ready)$/.test(target));
  add('libWrapper.Register', (owner, target, type) => {
    if (!relevant(target)) return;
    let typeName;
    try { typeName = String(type); } catch { typeName = '(unreadable)'; }
    if (owner === TARGET && target === PACK_TARGET && typeName === 'WRAPPER' && capture.registrations++ === 0) {
      capture.knownRegistered = true;
    } else if (owner === 'babele' && target === PACK_TARGET && typeName === 'WRAPPER' && capture.babeleRegistrations++ === 0) {
      // Babele 2.9.1 delegates this exact wrapper to its exported preserveIndexFlags.
      // Registration alone is insufficient: capability also verifies that helper.
      capture.babeleRegistered = true;
    } else capture.unknown = `Unverified wrapper registered: ${owner} / ${target} / ${typeName}`;
    if (!capture.unknown) {
      try { capture.wrapperOrder.push({owner, priority: wrapperPriority(g, owner)}); }
      catch (error) { capture.priorityError = `Cannot verify libWrapper priorities: ${String(error)}`; }
    }
    changed();
  });
  add('libWrapper.Unregister', (owner, target) => {
    if ((owner === TARGET || owner === 'babele') && target === PACK_TARGET) {
      if (owner === TARGET) capture.knownRegistered = false;
      else capture.babeleRegistered = false;
      capture.unknown = 'The verified upstream indexDocument wrapper was unregistered.';
      changed();
    }
  });
  add('libWrapper.UnregisterAll', owner => {
    if ((owner === TARGET && capture.knownRegistered) || (owner === 'babele' && capture.babeleRegistered)) {
      if (owner === TARGET) capture.knownRegistered = false;
      else capture.babeleRegistered = false;
      capture.unknown = 'The verified upstream wrappers were unregistered.';
      changed();
    }
  });
  add('updateSetting', setting => {
    if (setting?.key !== 'lib-wrapper.module-priorities') return;
    // onChange updates libWrapper's priority map, but does not re-sort existing
    // wrapper records. Refuse further bypass until reload establishes a new chain.
    capture.priorityError = 'libWrapper module priorities changed; reload to verify the wrapper order.';
    changed();
  });
  if (!capture.hooks.length) capture.valid = false;
  return capture;
}

// Capture the unwrapped method while all module ESM entry points are still loading.
const earlyCapture = captureBabeleCore(globalThis);

function mode(g) {
  for (const namespace of ['babele', TARGET]) {
    try {
      const value = g.game?.settings?.get?.(namespace, 'loadingMode');
      if (value === 'full' || value === 'ondemand') return value;
    } catch { /* A setting may not yet be registered during module loading. */ }
  }
  return 'ondemand';
}

/** Install only on known, uncustomized pack / document-index instances. */
export function registerBabeleIndex({moduleId = 'av-v14-hotfix', g = globalThis, capture = earlyCapture, preserveIndexFlags} = {}) {
  const prior = registrations.get(g);
  if (prior?.game === g.game && prior.moduleId === moduleId) return prior.api;
  const verifiedPreserveIndexFlags = signature(preserveIndexFlags) === SIGNATURES.preserveIndexFlags;
  let disposed = false, closing = false, indexControl = null, entryTimer = null, fullTimer = null;
  let running = null, queue = new Map(), lastError = null;
  const fullRunning = new Set();
  const packs = new Map(), hooks = [], skippedPacks = new Set();
  const metrics = {nativeCalls: 0, batches: 0, replaced: 0, removed: 0, fullRebuilds: 0};
  const capability = () => {
    if (disposed) return ['disposed', 'Adapter disposed.'];
    if (!g.game?.modules?.get?.(TARGET)?.active) return ['inactive', 'Translation module is not active.'];
    if (mode(g) === 'full') return ['not-applicable', 'Effective Babele loading mode is full; retain the upstream bypass.'];
    if (!g.game?.babele?.__ondemandPatch) return ['waiting', 'Babele on-demand facade has not initialized.'];
    if (capture.g !== g || !capture.valid) return ['unsupported', 'No verified pre-libWrapper native indexDocument capture.'];
    if (capture.unknown) return ['unsupported', capture.unknown];
    if (g.game.release?.generation !== 14) return ['unsupported', 'Only Foundry generation 14 is supported.'];
    // 3.2.1 ships byte-identical babele.js and babele-ondemand-patch.js to the
    // audited 3.1.2 release. Keep an exact allowlist; future releases still fall back.
    for (const [id, versions] of [[TARGET, ['3.1.2', '3.2.1']], ['babele', ['2.9.1']], ['lib-wrapper', ['1.13.5.1']]]) {
      const module = g.game.modules.get(id);
      if (!module?.active || !versions.includes(module.version)) return ['unsupported', `Expected active ${id} ${versions.join(' or ')}.`];
    }
    if (!capture.knownRegistered || !capture.babeleRegistered) return ['waiting', 'Waiting for both verified upstream libWrapper registrations.'];
    if (capture.priorityError) return ['unsupported', capture.priorityError];
    const [babeleWrapper, titleWrapper] = capture.wrapperOrder;
    if (capture.wrapperOrder.length !== 2 || babeleWrapper?.owner !== 'babele' || titleWrapper?.owner !== TARGET) {
      return ['unsupported', 'Unverified Babele wrapper registration order; expected Babele before title translation.'];
    }
    if (babeleWrapper.priority !== titleWrapper.priority) {
      return ['unsupported', 'Different effective libWrapper priorities are unsupported; retain the original wrapper chain.'];
    }
    if (!verifiedPreserveIndexFlags) return ['unsupported', 'Babele preserveIndexFlags helper is missing or its source fingerprint differs from 2.9.1.'];
    if (typeof g.game.babele.translateIndexTitles !== 'function') return ['unsupported', 'Public title translation API is unavailable.'];
    if (typeof g.setTimeout !== 'function' || typeof g.clearTimeout !== 'function') return ['unsupported', 'Timer APIs unavailable.'];
    return ['supported', null];
  };
  const supported = () => capability()[0] === 'supported';
  const fallbackMethod = (receiver, name, args) => Reflect.get(Object.getPrototypeOf(receiver), name, receiver).apply(receiver, args);
  const clear = index => {
    for (const key of Object.keys(index.trees ?? {})) delete index.trees[key];
    for (const key of Object.keys(index.uuids ?? {})) delete index.uuids[key];
  };

  function attachIndex() {
    const index = g.game?.documentIndex;
    if (indexControl?.index === index) return true;
    if (indexControl) { lastError = 'DocumentIndex instance changed; reload to re-verify.'; return false; }
    if (!index || Object.hasOwn(index, 'index') || Object.hasOwn(index, 'ready')) {
      lastError = 'DocumentIndex has an unknown own index/ready override.'; return false;
    }
    const proto = Object.getPrototypeOf(index), nativeIndex = proto?.index;
    const nativeReady = Object.getOwnPropertyDescriptor(proto, 'ready')?.get;
    if (signature(nativeIndex) !== SIGNATURES.index || signature(nativeReady) !== SIGNATURES.ready) {
      lastError = 'DocumentIndex index/ready source fingerprint differs from Foundry 14.367.'; return false;
    }
    if (!Object.isExtensible(index)) { lastError = 'DocumentIndex instance is not extensible.'; return false; }
    const control = {index, nativeIndex, nativeReady, pending: null, revision: 0};
    function indexAdapter(...args) {
      if (this !== index || !supported()) return fallbackMethod(this, 'index', args);
      const previous = control.pending;
      let finish;
      const barrier = new Promise(resolve => { finish = resolve; });
      control.pending = barrier;
      control.revision++;
      // Mark the barrier synchronously. Clear inside this serialized operation, after prior builds.
      return (async () => {
        try {
          await previous;
          await nativeReady.call(index);
          clear(index);
          metrics.fullRebuilds++;
          return await nativeIndex.apply(index, args);
        } finally {
          finish();
          if (control.pending === barrier) control.pending = null;
        }
      })();
    }
    function readyAdapter() {
      return control.pending ?? Reflect.get(Object.getPrototypeOf(this), 'ready', this);
    }
    Object.defineProperty(index, 'index', {value: indexAdapter, configurable: true, writable: true});
    Object.defineProperty(index, 'ready', {get: readyAdapter, configurable: true});
    Object.assign(control, {indexAdapter, readyAdapter});
    indexControl = control;
    return true;
  }

  async function fullRebuild() {
    const index = g.game?.documentIndex;
    if (!index || typeof index.index !== 'function') return;
    await index.ready;
    clear(index);
    await index.index();
  }
  function scheduleFull() {
    if (disposed || fullTimer !== null) return;
    fullTimer = g.setTimeout(() => {
      fullTimer = null;
      const promise = fullRebuild().catch(error => { lastError = `Full fallback failed: ${String(error)}`; })
        .finally(() => { fullRunning.delete(promise); });
      fullRunning.add(promise);
      return promise;
    }, 50);
  }
  function schedule(document) {
    const index = g.game?.documentIndex;
    if (!document?.uuid || !document?.pack || typeof index?.replaceDocument !== 'function'
      || typeof index?.removeDocument !== 'function' || index !== indexControl?.index) {
      scheduleFull(); return;
    }
    queue.set(document.uuid, document);
    if (entryTimer !== null || running) return;
    entryTimer = g.setTimeout(() => {
      entryTimer = null;
      running = flushEntries().finally(() => { running = null; if (queue.size) schedule(queue.values().next().value); });
      return running;
    }, 50);
  }
  async function flushEntries() {
    const control = indexControl, index = control?.index;
    let batch;
    try {
      let revision;
      do {
        revision = control.revision;
        await control.pending;
        await index.ready;
      } while (revision !== control.revision);
      if (g.game.documentIndex !== index || !supported()) throw new Error('Index or compatibility state changed while waiting.');
      batch = queue;
      queue = new Map();
      metrics.batches++;
      for (const document of batch.values()) {
        const pack = g.game.packs.get(document.pack);
        if (pack?.index?.get(document.id)) { index.replaceDocument(document); metrics.replaced++; }
        else { index.removeDocument(document); metrics.removed++; }
      }
    } catch (error) {
      if (!batch) queue.clear();
      lastError = `Incremental fallback: ${String(error)}`;
      scheduleFull();
    }
  }
  function attachPack(pack) {
    if (!pack || packs.has(pack)) return;
    if (Object.hasOwn(pack, 'indexDocument') || Object.getPrototypeOf(pack) !== capture.proto || !Object.isExtensible(pack)) {
      skippedPacks.add(pack.collection ?? '(unknown)'); return;
    }
    function indexDocumentAdapter(...args) {
      // Unknown prototype wrappers always get the normal prototype dispatch, never get skipped.
      if (closing || this !== pack || !supported() || g.game.documentIndex !== indexControl?.index) {
        return fallbackMethod(this, 'indexDocument', args);
      }
      // Keep Babele's original merge of prior index fields and document.flags.babele.
      // The exported helper calls core exactly once and preserves its native return.
      const out = preserveIndexFlags(this, (...nativeArgs) => capture.native.apply(this, nativeArgs), args);
      metrics.nativeCalls++;
      try {
        const document = args[0], id = document?.id ?? document?._id;
        const packId = typeof this.collection === 'string' ? this.collection : this.metadata?.id;
        const state = g.game.babele?.__ondemandPatch;
        if (!id || !packId || !state) return out;
        const entry = this.index?.get?.(id);
        if (!entry) return out;
        if (entry.originalName == null) entry.originalName = document?.originalName ?? document?.name ?? entry.name;
        g.game.babele.translateIndexTitles([entry], packId);
        schedule(document);
      } catch (error) {
        // Upstream catches translation errors too; do not call the native method twice.
        lastError = `Title translation failed: ${String(error)}`;
      }
      return out;
    }
    Object.defineProperty(pack, 'indexDocument', {value: indexDocumentAdapter, configurable: true, writable: true});
    packs.set(pack, indexDocumentAdapter);
  }
  function scan() {
    if (closing || !supported() || !attachIndex()) return api.status();
    for (const pack of g.game.packs ?? []) attachPack(pack);
    return api.status();
  }
  const add = (name, callback) => {
    if (typeof g.Hooks?.on === 'function') hooks.push([name, g.Hooks.on(name, callback)]);
  };
  const api = Object.freeze({
    scan,
    status() {
      let [state, reason] = capability();
      if (state === 'supported') {
        state = indexControl ? 'applied' : 'unsupported';
        reason = indexControl ? 'Verified per-pack instance adapter with serialized full-build barrier.' : lastError;
      }
      return {moduleId, target: TARGET, mode: mode(g), state, reason, applied: state === 'applied', implemented: true,
        attachedPacks: packs.size, skippedPacks: [...skippedPacks], queued: queue.size,
        entryPending: entryTimer !== null || !!running, fullPending: fullTimer !== null || fullRunning.size > 0 || !!indexControl?.pending,
        metrics: {...metrics}, lastError,
        limitation: 'Optimized pack instances preserve the verified Babele helper and title translation while replacing rebuild scheduling; unknown registrations disable optimization.'};
    },
    async dispose() {
      if (disposed || closing) return;
      closing = true;
      for (const [name, id] of hooks) g.Hooks.off(name, id);
      capture.listeners.delete(scan);
      if (entryTimer !== null) { g.clearTimeout(entryTimer); entryTimer = null; }
      if (running) await running;
      if (queue.size) await flushEntries();
      if (fullTimer !== null) { g.clearTimeout(fullTimer); fullTimer = null; await fullRebuild(); }
      await Promise.all(fullRunning);
      if (indexControl?.pending) await indexControl.pending;
      for (const [pack, adapter] of packs) if (Object.getOwnPropertyDescriptor(pack, 'indexDocument')?.value === adapter) delete pack.indexDocument;
      if (indexControl) {
        const {index, indexAdapter, readyAdapter} = indexControl;
        if (Object.getOwnPropertyDescriptor(index, 'index')?.value === indexAdapter) delete index.index;
        if (Object.getOwnPropertyDescriptor(index, 'ready')?.get === readyAdapter) delete index.ready;
      }
      for (const [name, id] of capture.hooks) g.Hooks.off(name, id);
      capture.hooks.length = 0;
      capture.valid = false;
      disposed = true;
    }
  });
  registrations.set(g, {game: g.game, moduleId, api});
  capture.listeners.add(scan);
  add('ready', scan);
  add('babele.ready', scan);
  add('createCompendium', () => scan());
  scan();
  return api;
}
