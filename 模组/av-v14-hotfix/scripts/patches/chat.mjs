/**
 * Foundry v14. Coalesce the known ChatLog deletion completion, while
 * retaining the native public method, rendering queue, animation and closure.
 * No prototypes or global DOM/Animation/Promise methods are modified.
 */
const completionSignature = "()=>{li.remove();this.#onScrollLog();if(this.isPopout)this._refit();}";
const functionToString = Function.prototype.toString;

function isKnownCompletion(callback) {
  if (typeof callback !== "function") return false;
  const source = Reflect.apply(functionToString, callback, [])
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\r\n]*/g, "")
    .replace(/\s+/g, "");
  return source === completionSignature;
}

function isDeletionAnimation(frames, options) {
  return frames && Object.keys(frames).length === 1 && Array.isArray(frames.height)
    && frames.height.length === 2 && /^-?[\d.]+px$/.test(frames.height[0]) && frames.height[1] === "0"
    && options && Object.keys(options).length === 2 && options.duration === 100 && options.easing === "ease";
}

// Restore only our own temporary property; do not overwrite another module's edit.
function restoreProperty(object, key, replacement, descriptor) {
  if (Object.getOwnPropertyDescriptor(object, key)?.value !== replacement) return;
  if (descriptor) Object.defineProperty(object, key, descriptor);
  else delete object[key];
}

export function createChatDeleteWrapper({
  getVersion = () => globalThis.game?.version,
  schedule = (callback, delay) => setTimeout(callback, delay)
} = {}) {
  const nodeAdapters = new WeakMap();
  const batches = new WeakMap();

  function flush(app, batch) {
    if (batches.get(app) !== batch) return;
    batches.delete(app);
    const records = batch.records;
    const last = records.at(-1);
    try {
      const nodes = new Set(records.map(record => record.node));
      if (app.rendered) nodes.delete(last.node);
      for (const node of nodes) node.remove();
    } catch (error) {
      // Removal failed before any native completion ran: still give each card
      // its native cleanup, and surface failure on the continuation promises.
      for (const record of records) {
        try { Reflect.apply(record.callback, record.receiver, record.args); }
        catch (cleanupError) { record.reject(cleanupError); continue; }
        record.reject(error);
      }
      return;
    }
    try {
      // The retained native closure is the only way private scroll state is
      // touched. It still loads older messages and restores the scroll anchor.
      if (app.rendered) Reflect.apply(last.callback, last.receiver, last.args);
      for (const record of records) record.resolve(undefined);
    } catch (error) {
      // A completion that already ran must not be replayed if layout/refit fails.
      for (const record of records) record.reject(error);
    }
  }

  function complete(app, node, callback, receiver, args) {
    return new Promise((resolve, reject) => {
      let batch = batches.get(app);
      const fresh = !batch;
      if (fresh) {
        batch = {records: []};
        batches.set(app, batch);
      }
      batch.records.push({node, callback, receiver, args, resolve, reject});
      if (!fresh) return;
      try { schedule(() => flush(app, batch), 0); }
      catch {
        // Scheduling is optional optimization: a failure uses every original
        // callback, so it cannot leave successful animations unremoved.
        batches.delete(app);
        for (const record of batch.records) {
          try { record.resolve(Reflect.apply(record.callback, record.receiver, record.args)); }
          catch (error) { record.reject(error); }
        }
      }
    });
  }

  function adaptFinished(animation, app, node) {
    const finished = animation?.finished;
    if (!finished || typeof finished.then !== "function") return;
    const descriptor = Object.getOwnPropertyDescriptor(finished, "then");
    if (descriptor && !descriptor.configurable) return;
    const originalThen = finished.then;
    function then(onFulfilled, onRejected) {
      restore();
      const adapted = isKnownCompletion(onFulfilled)
        ? function(...args) { return complete(app, node, onFulfilled, this, args); }
        : onFulfilled;
      return Reflect.apply(originalThen, this, [adapted, onRejected]);
    }
    function restore() { restoreProperty(finished, "then", then, descriptor); }
    try {
      Object.defineProperty(finished, "then", {value: then, configurable: true, writable: true});
      // The core reads .finished.then synchronously. An unused adapter should
      // not survive that turn even if another consumer never calls .then.
      queueMicrotask(restore);
    } catch { restore(); }
  }

  function acquire(node, app) {
    let state = nodeAdapters.get(node);
    if (state) {
      if (state.app !== app || node.animate !== state.animate) return null;
      state.references++;
    } else {
      const original = node.animate;
      const descriptor = Object.getOwnPropertyDescriptor(node, "animate");
      if (typeof original !== "function" || (descriptor && !descriptor.configurable)) return null;
      state = {app, references: 1, original, descriptor};
      state.animate = function(...args) {
        const animation = Reflect.apply(original, this, args);
        if (state.references > 0 && this === node && isDeletionAnimation(...args)) {
          // Reading or decorating a third-party Animation must not break its
          // native return. Unsupported objects simply retain their own behavior.
          try { adaptFinished(animation, app, node); } catch { /* Native fallback. */ }
        }
        return animation;
      };
      try { Object.defineProperty(node, "animate", {value: state.animate, configurable: true, writable: true}); }
      catch { return null; }
      nodeAdapters.set(node, state);
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      if (--state.references) return;
      restoreProperty(node, "animate", state.animate, state.descriptor);
      nodeAdapters.delete(node);
    };
  }

  return function deleteMessage(wrapped, messageId, ...args) {
    // A card rendered only later by the native queue deliberately falls back.
    if (Number.parseInt(getVersion(),10) !== 14 || !this.rendered || !/^[A-Za-z0-9_-]+$/.test(messageId)) {
      return wrapped(messageId, ...args);
    }
    const node = this.element?.querySelector(`.message[data-message-id="${messageId}"]`);
    const release = node ? acquire(node, this) : null;
    if (!release) return wrapped(messageId, ...args);
    let result;
    try { result = wrapped(messageId, ...args); }
    catch (error) { release(); throw error; }
    // The continuation preserves the exact value/error and leaves an ignored
    // rejection visible. It intentionally returns a new Promise, with one
    // additional settlement microtask; observing and returning the original
    // Promise would silently consume an otherwise-unhandled queue rejection.
    try {
      return result.then(value => {release(); return value;}, error => {release(); throw error;});
    } catch (error) { release(); throw error; }
  };
}

export function installChatDeleteCoalescing({moduleId, libWrapper = globalThis.libWrapper, ...options} = {}) {
  if (!moduleId || typeof libWrapper?.register !== "function") return false;
  if (Number.parseInt(options.getVersion?.() ?? globalThis.game?.version,10) !== 14) return false;
  if (typeof globalThis.foundry?.applications?.sidebar?.tabs?.ChatLog?.prototype?.deleteMessage !== 'function') return false;
  libWrapper.register(moduleId, "foundry.applications.sidebar.tabs.ChatLog.prototype.deleteMessage",
    createChatDeleteWrapper(options), "WRAPPER");
  return true;
}
