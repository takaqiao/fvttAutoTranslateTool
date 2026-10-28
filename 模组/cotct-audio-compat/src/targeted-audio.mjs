const CHANNEL = "module.soundscape-adventure";

export function createTargetedAudio({ game, paths, onError = console.error }) {
  const socket = game.socket;
  const world = game.world;
  const worldId = world.id;
  const user = game.user;
  const userId = user.id;
  const boundToGlobal = globalThis.game === game;
  const trustedPaths = new Set(paths);
  const managed = new Map();
  const received = new Map();
  const dispatching = new Set();
  let disposed = false;

  function clear() {
    managed.clear();
    received.clear();
  }

  function isCurrent() {
    return !disposed && socket.connected === true && game.socket === socket
      && game.world === world && world.id === worldId
      && game.user === user && user.id === userId
      && (!boundToGlobal || globalThis.game === game);
  }

  function isGM(id) {
    return game.users.get(id)?.isGM === true;
  }

  function validRecipients(userIds) {
    return Array.isArray(userIds) && userIds.length > 0
      && userIds.every(id => typeof id === "string" && id.length > 0)
      && new Set(userIds).size === userIds.length;
  }

  function removePending(key, entry, userIds) {
    for (const id of userIds) entry.pending.delete(id);
    if (!entry.pending.size) managed.delete(key);
  }

  function observe(message, senderId, outgoing) {
    if (!isCurrent()) {
      clear();
      return;
    }
    if (!message || typeof message.key !== "string" || !message.key
      || typeof senderId !== "string" || message.senderId !== senderId) return;

    const { action, key } = message;
    if (action === "play") {
      if (!isGM(senderId) || !trustedPaths.has(message.src) || !validRecipients(message.userIds)) return;
      const entry = managed.get(key) ?? received.get(key);
      if (entry && (entry.senderId !== senderId || entry.src !== message.src
        || entry.userIds.length !== message.userIds.length
        || !entry.userIds.every((id, index) => id === message.userIds[index]))) return;
      const record = entry ?? {
        senderId, src: message.src, userIds: [...message.userIds], pending: new Set(message.userIds)
      };
      if (user.isGM && isGM(userId)) managed.set(key, record);
      if (!outgoing && record.userIds.includes(userId)) received.set(key, record);
      return;
    }

    if (action === "ended") {
      const entry = managed.get(key);
      if (entry && message.targetId === entry.senderId && entry.userIds.includes(senderId)) {
        removePending(key, entry, [senderId]);
      }
      const local = received.get(key);
      if (local && senderId === userId && message.targetId === local.senderId) received.delete(key);
      return;
    }

    if (action !== "stop" || !isGM(senderId) || !validRecipients(message.userIds)) return;
    const entry = managed.get(key) ?? received.get(key);
    if (!entry || !message.userIds.every(id => entry.userIds.includes(id))) return;

    // Our own outgoing observation runs before emit can fail or self-delivery arrives.
    if (managed.has(key) && !(outgoing && dispatching.has(key))) {
      removePending(key, entry, message.userIds);
    }
    const local = received.get(key);
    if (outgoing || !local || !message.userIds.includes(userId)) return;

    // Upstream stopLocal removes its Sound before the stop event can send ended.
    const ended = { action: "ended", key, targetId: local.senderId, senderId: userId };
    if (local.senderId === userId) socket.emit(CHANNEL, ended, { recipients: [userId] });
    else socket.emit(CHANNEL, ended);
    received.delete(key);
  }

  function safelyObserve(message, senderId, outgoing) {
    try {
      observe(message, senderId, outgoing);
    } catch (error) {
      // Socket.IO calls outgoing listeners inside native emit; reporting cannot break it.
      try { onError(error); } catch { /* Keep the native socket dispatch intact. */ }
    }
  }

  const onMessage = (message, senderId) => safelyObserve(message, senderId, false);
  const onOutgoing = (channel, message) => {
    if (channel === CHANNEL) safelyObserve(message, userId, true);
  };
  socket.on(CHANNEL, onMessage);
  socket.on("disconnect", clear);
  socket.onAnyOutgoing(onOutgoing);

  function stop() {
    if (!isCurrent() || !user.isGM || !isGM(userId)) {
      if (!isCurrent()) clear();
      else managed.clear();
      throw new Error("Targeted audio STOP requires the original connected GM session.");
    }
    const errors = [];
    let stopped = 0;
    for (const [key, entry] of [...managed]) {
      dispatching.add(key);
      try {
        socket.emit(CHANNEL, {
          action: "stop", key, userIds: [...entry.userIds], senderId: userId, fadeOut: 0
        }, { recipients: [...entry.userIds] });
        managed.delete(key);
        stopped++;
      } catch (error) {
        errors.push(error);
      } finally {
        dispatching.delete(key);
      }
    }
    if (errors.length) throw new AggregateError(errors, "Failed to stop some targeted audio.");
    return stopped;
  }

  function hasActive() {
    if (!isCurrent() || !user.isGM || !isGM(userId)) {
      if (!isCurrent()) clear();
      else managed.clear();
      return false;
    }
    return managed.size > 0;
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    clear();
    socket.off(CHANNEL, onMessage);
    socket.off("disconnect", clear);
    socket.offAnyOutgoing(onOutgoing);
  }

  return { stop, hasActive, dispose };
}
