// Exact Foundry 14.367 live records. One internal off mapping is shared by the
// features in this module; unknown runtime wrappers are never treated as native.
const brokers = new WeakMap();
const functionToString = Function.prototype.toString;
const HOOK_SOURCES = {on:'82df7097',once:'ad035069',off:'00d0df67',call:'3a1d0e81',callAll:'782db789'};
function fingerprint(fn) {
  if (typeof fn !== 'function') return null;
  let hash = 2166136261;
  for (const character of Reflect.apply(functionToString,fn,[])) hash = Math.imul(hash ^ character.charCodeAt(0),16777619);
  return (hash >>> 0).toString(16).padStart(8,'0');
}

function ownsOff(Hooks, broker) {
  const descriptor = Object.getOwnPropertyDescriptor(Hooks,'off');
  return descriptor?.value === broker.off && descriptor.writable === broker.descriptor.writable
    && descriptor.configurable === broker.descriptor.configurable && descriptor.enumerable === broker.descriptor.enumerable;
}

function knownHooks(Hooks, broker) {
  const events = Object.getOwnPropertyDescriptor(Hooks,'events');
  const off = Object.getOwnPropertyDescriptor(Hooks,'off');
  return fingerprint(Hooks) === '05559ab9' && fingerprint(events?.get) === 'd22fc400'
    && !events.set && off?.writable && off.configurable
    && Object.entries(HOOK_SOURCES).every(([key,source]) => {
      const method = Object.getOwnPropertyDescriptor(Hooks,key)?.value;
      return key === 'off' && broker ? ownsOff(Hooks,broker) : fingerprint(method) === source;
    });
}

function createBroker(Hooks) {
  const descriptor = Object.getOwnPropertyDescriptor(Hooks,'off'), nativeOff = descriptor.value;
  const records = new Map();
  function off(hook, fn, ...args) {
    if (this === Hooks && typeof fn === 'function') {
      let hasAlias = false;
      for (const record of records.values()) {
        if (record.hook === hook && record.fn === fn
          && Object.getOwnPropertyDescriptor(record.entry,'fn')?.value === record.replacement) { hasAlias = true; break; }
      }
      // Native off(function) removes the first matching registration. Include
      // owned original aliases in that same live order, including later copies.
      // Unrelated selectors do not inspect records before the native method.
      const list = hasAlias && Object.getOwnPropertyDescriptor(Hooks.events,hook)?.value;
      if (Array.isArray(list)) {
        for (const entry of list) {
          if (entry.fn === fn) break;
          const record = records.get(entry);
          if (record?.fn === fn && entry.fn === record.replacement) { fn = record.replacement; break; }
        }
      }
    }
    return Reflect.apply(nativeOff,this,[hook,fn,...args]);
  }
  return {descriptor,off,records};
}

/** Adapt one feature's complete callback group without replacing native records. */
export function adaptNativeHooks({Hooks, callbacks}) {
  const skipped = (reason,detail={}) => ({status:'skipped',reason,...detail});
  if (!Hooks || !Array.isArray(callbacks) || !callbacks.length) return skipped('hook-api-unavailable');
  let broker = brokers.get(Hooks);
  if (!knownHooks(Hooks,broker)) return skipped('core-hooks-fingerprint-mismatch');
  const events = Hooks.events;
  if (events !== Hooks.events) return skipped('non-live-hooks-events');
  const originals = [];
  for (const {hook,source,wrap} of callbacks) {
    const list = Object.getOwnPropertyDescriptor(events,hook)?.value;
    if (!Array.isArray(list) || list !== Hooks.events[hook]) return skipped('unsupported-hook-records',{hook});
    const matches = list.filter(entry => {
      if (!entry || typeof entry !== 'object') return false;
      const fn = Object.getOwnPropertyDescriptor(entry,'fn')?.value;
      return typeof fn === 'function' && Reflect.apply(functionToString,fn,[]) === source;
    });
    if (matches.length !== 1) return skipped('callback-fingerprint-mismatch',{hook,matches:matches.length});
    const entry = matches[0], descriptors = Object.getOwnPropertyDescriptors(entry), descriptor = descriptors.fn;
    if (descriptors.hook?.value !== hook || !Number.isSafeInteger(descriptors.id?.value) || descriptors.id.value <= 0
      || descriptors.once?.value !== false || !descriptor.writable || typeof wrap !== 'function'
      || broker?.records.has(entry) || originals.some(record => record.entry === entry)) return skipped('unsupported-hook-record',{hook});
    originals.push({hook,list,entry,descriptor,fn:descriptor.value,wrap});
  }
  broker ??= createBroker(Hooks);
  const restoreRecords = () => {
    for (const record of originals) {
      // An in-flight native snapshot may still reference a removed record.
      // Restore it in place, but never reinsert it or overwrite a later edit.
      const descriptor = Object.getOwnPropertyDescriptor(record.entry,'fn');
      if (descriptor?.value === record.replacement && descriptor.writable === record.descriptor.writable
        && descriptor.configurable === record.descriptor.configurable && descriptor.enumerable === record.descriptor.enumerable) {
        Object.defineProperty(record.entry,'fn',record.descriptor);
      }
      // A later freeze or descriptor edit owns the retained callback. It must
      // not keep this feature's aliases, or prevent another feature restoring.
      broker.records.delete(record.entry);
    }
    if (!broker.records.size) {
      if (ownsOff(Hooks,broker)) Object.defineProperty(Hooks,'off',broker.descriptor);
      if (brokers.get(Hooks) === broker) brokers.delete(Hooks);
    }
  };
  try {
    for (const record of originals) {
      record.replacement = record.wrap(record.fn);
      if (typeof record.replacement !== 'function') throw new TypeError('Callback adaptation must return a function');
    }
    Object.defineProperty(Hooks,'off',{...broker.descriptor,value:broker.off});
    for (const record of originals) {
      Object.defineProperty(record.entry,'fn',{...record.descriptor,value:record.replacement});
      broker.records.set(record.entry,record);
    }
  } catch (error) {
    restoreRecords();
    return skipped('record-adaptation-failed',{message:String(error?.message ?? error)});
  }
  brokers.set(Hooks,broker);
  let restored = false;
  return {status:'installed',hooks:originals.map(({hook,entry})=>({hook,id:entry.id})),restore() {
    if (restored) return;
    restoreRecords();
    restored = true;
  }};
}
