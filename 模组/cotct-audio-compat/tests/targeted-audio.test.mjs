import assert from "node:assert/strict";
import { test } from "node:test";
import { createTargetedAudio } from "../src/targeted-audio.mjs";

const CHANNEL = "module.soundscape-adventure";
const PATH = "modules/cotct-audio/sounds/bell.ogg";
const OTHER_PATH = "modules/cotct-audio/sounds/bell.ogg.backup";

// Simulated authenticated transport, including Foundry's explicit self-recipient option.
function createTable({ queued = false } = {}) {
  const users = new Map([
    ["gm", { id: "gm", isGM: true }],
    ["gm2", { id: "gm2", isGM: true }],
    ["target", { id: "target", isGM: false }],
    ["bystander", { id: "bystander", isGM: false }]
  ]);
  const clients = new Map();
  const packets = [];
  const pending = [];
  function flush() {
    while (pending.length) pending.shift()();
  }
  for (const user of users.values()) {
    const listeners = new Map();
    const outgoing = new Set();
    const socket = {
      connected: true,
      fail: null,
      on(event, listener) {
        if (!listeners.has(event)) listeners.set(event, new Set());
        listeners.get(event).add(listener);
      },
      off(event, listener) { listeners.get(event)?.delete(listener); },
      onAnyOutgoing(listener) { outgoing.add(listener); },
      offAnyOutgoing(listener) { outgoing.delete(listener); },
      receive(event, ...args) {
        for (const listener of [...(listeners.get(event) ?? [])]) listener(...args);
      },
      emit(event, message, options) {
        for (const listener of [...outgoing]) listener(event, message, options);
        if (socket.fail?.(message)) throw new Error(`send failed: ${message.key}`);
        const packet = structuredClone({ event, message, options, transportSenderId: user.id });
        packets.push(packet);
        pending.push(() => {
          for (const [id, client] of clients) {
            const selected = packet.options?.recipients
              ? packet.options.recipients.includes(id)
              : id !== user.id;
            if (selected && client.game.socket.connected) {
              client.game.socket.receive(event, structuredClone(packet.message), user.id);
            }
          }
        });
        if (!queued) flush();
        return socket;
      }
    };
    const game = { user, users, world: { id: "qa" }, socket };
    clients.set(user.id, { game });
  }
  const errors = [];
  function install(id, paths = [PATH], onError = error => errors.push(error)) {
    const client = clients.get(id);
    client.audio = createTargetedAudio({ game: client.game, paths, onError });
    return client;
  }
  function play({ sender = "gm", key = "key-1", userIds = ["target"], src = PATH, ...extra } = {}) {
    const message = { action: "play", key, userIds, src, senderId: sender, ...extra };
    clients.get(sender).game.socket.emit(CHANNEL, message);
    return message;
  }
  return { clients, packets, errors, install, play, flush };
}

test("STOP reaches remote targets with the original recipients and is idempotent", () => {
  const table = createTable();
  const gm = table.install("gm");
  table.install("target");
  table.play();
  assert.equal(gm.audio.hasActive(), true);
  assert.equal(gm.audio.stop(), 1);
  const stop = table.packets.find(packet => packet.message.action === "stop");
  assert.deepEqual(stop.message, {
    action: "stop", key: "key-1", userIds: ["target"], senderId: "gm", fadeOut: 0
  });
  assert.deepEqual(stop.options, { recipients: ["target"] });
  assert.equal(gm.audio.hasActive(), false);
  assert.equal(gm.audio.stop(), 0);
});

test("a recipient GM can stop itself and the other original recipients", () => {
  const table = createTable({ queued: true });
  const gm = table.install("gm");
  const gm2 = table.install("gm2");
  table.install("target");
  table.play({ userIds: ["gm2", "target"] });
  table.flush();
  assert.equal(gm2.audio.stop(), 1);
  assert.equal(gm2.audio.hasActive(), false);
  table.flush();
  assert.equal(gm.audio.hasActive(), false);
  assert.deepEqual(table.packets.filter(packet => packet.message.action === "ended")
    .map(packet => [packet.transportSenderId, packet.message.senderId, packet.message.targetId]).sort(),
  [["gm2", "gm2", "gm"], ["target", "target", "gm"]]);
});

test("only exact trusted paths become managed", () => {
  const table = createTable();
  const gm = table.install("gm");
  table.play({ src: OTHER_PATH });
  table.play({ key: "other-module", src: "modules/unrelated/bell.ogg" });
  assert.equal(gm.audio.stop(), 0);
  assert.equal(table.packets.some(packet => packet.message.action === "stop"), false);
});

test("trusted paths and original play recipients are snapshots", () => {
  const table = createTable();
  const paths = new Set([PATH]);
  const gm = table.install("gm", paths);
  paths.add(OTHER_PATH);
  paths.delete(PATH);
  const userIds = ["target"];
  const message = table.play({ userIds });
  userIds.push("bystander");
  message.src = OTHER_PATH;
  table.play({ key: "not-trusted", src: OTHER_PATH });
  assert.equal(gm.audio.stop(), 1);
  assert.deepEqual(table.packets.find(packet => packet.message.action === "stop").message.userIds, ["target"]);
});

test("a known key cannot be rebound to a different sender, source or recipients", () => {
  const table = createTable();
  const gm = table.install("gm", [PATH, OTHER_PATH]);
  table.play();
  table.play({ sender: "gm2", userIds: ["bystander"] });
  table.play({ src: OTHER_PATH });
  table.play({ userIds: ["bystander"] });
  assert.equal(gm.audio.stop(), 1);
  assert.deepEqual(table.packets.find(packet => packet.message.action === "stop").message.userIds, ["target"]);
});

test("incoming play requires a real GM transport sender matching the payload", () => {
  const table = createTable();
  const gm2 = table.install("gm2");
  const play = { action: "play", key: "forged", src: PATH, userIds: ["target"], senderId: "gm" };
  for (const sender of ["target", undefined, "missing"]) gm2.game.socket.receive(CHANNEL, play, sender);
  table.play({ sender: "target", key: "not-gm" });
  assert.equal(gm2.audio.stop(), 0);
});

test("outgoing play cannot impersonate another GM", () => {
  const table = createTable();
  const gm = table.install("gm");
  gm.game.socket.emit(CHANNEL, {
    action: "play", key: "forged", src: PATH, userIds: ["target"], senderId: "gm2"
  });
  assert.equal(gm.audio.stop(), 0);
});

test("non-target players do not acknowledge or retain a targeted play", () => {
  const table = createTable();
  const gm = table.install("gm");
  const bystander = table.install("bystander");
  table.play();
  assert.equal(bystander.audio.hasActive(), false);
  gm.audio.stop();
  assert.equal(table.packets.some(packet => packet.message.action === "ended"), false);
});

test("player UI queries and a rejected STOP preserve its pending acknowledgement", () => {
  const table = createTable();
  const gm = table.install("gm");
  const target = table.install("target");
  table.play();
  assert.equal(target.audio.hasActive(), false);
  assert.throws(() => target.audio.stop());
  gm.audio.stop();
  assert.equal(table.packets.filter(packet => packet.message.action === "ended"
    && packet.message.senderId === "target").length, 1);
});

test("forged and unbound stop messages never produce an ended acknowledgement", () => {
  const table = createTable();
  table.install("target");
  table.play();
  const target = table.clients.get("target");
  for (const [sender, patch] of [
    ["bystander", {}], [undefined, {}], ["gm", { key: "unknown" }],
    ["gm", { userIds: ["target", "bystander"] }], ["gm", { senderId: "gm2" }]
  ]) {
    target.game.socket.receive(CHANNEL, {
      action: "stop", key: "key-1", userIds: ["target"], senderId: "gm", ...patch
    }, sender);
  }
  assert.equal(table.packets.some(packet => packet.message.action === "ended"), false);
});

test("natural ended accepts only original recipients with authenticated sender and target", () => {
  const table = createTable();
  const gm = table.install("gm");
  table.play({ userIds: ["target", "gm2"] });
  const ended = { action: "ended", key: "key-1", senderId: "target", targetId: "gm" };
  for (const [sender, patch] of [
    ["bystander", {}], [undefined, {}], ["target", { targetId: "gm2" }],
    ["bystander", { senderId: "bystander" }], ["target", { key: "unknown" }]
  ]) gm.game.socket.receive(CHANNEL, { ...ended, ...patch }, sender);
  assert.equal(gm.audio.hasActive(), true);
  gm.game.socket.receive(CHANNEL, ended, "target");
  assert.equal(gm.audio.hasActive(), true);
  gm.game.socket.receive(CHANNEL, { ...ended, senderId: "gm2" }, "gm2");
  assert.equal(gm.audio.hasActive(), false);
});

test("natural end does not change the original STOP recipient list", () => {
  const table = createTable();
  const gm = table.install("gm");
  table.play({ userIds: ["target", "gm2"] });
  gm.game.socket.receive(CHANNEL, {
    action: "ended", key: "key-1", senderId: "target", targetId: "gm"
  }, "target");
  assert.equal(gm.audio.stop(), 1);
  assert.deepEqual(table.packets.find(packet => packet.message.action === "stop").message.userIds, ["target", "gm2"]);
});

test("native stop may address the remaining subset after a natural end", () => {
  const table = createTable();
  const gm = table.install("gm");
  table.install("target");
  table.play({ userIds: ["target", "gm2"] });
  table.clients.get("gm2").game.socket.emit(CHANNEL, {
    action: "ended", key: "key-1", senderId: "gm2", targetId: "gm"
  });
  gm.game.socket.emit(CHANNEL, { action: "stop", key: "key-1", userIds: ["target"], senderId: "gm" });
  assert.equal(gm.audio.hasActive(), false);
  assert.equal(table.packets.filter(packet => packet.message.action === "ended" && packet.message.senderId === "target").length, 1);
});

test("recipient acknowledgements clear the soundboard model and allow another send", () => {
  const table = createTable();
  const gm = table.install("gm");
  const target = table.install("target");
  const active = new Map();
  const playing = new Set();
  // Protocol state model only; native Soundscape integration is checked separately.
  gm.game.socket.on(CHANNEL, message => {
    if (message.action !== "ended" || message.targetId !== "gm") return;
    const recipients = active.get(message.key);
    recipients?.delete(message.senderId);
    if (recipients?.size === 0) active.delete(message.key);
  });
  target.game.socket.on(CHANNEL, message => {
    if (!message.userIds?.includes("target")) return;
    if (message.action === "play") playing.add(message.key);
    if (message.action === "stop") playing.delete(message.key);
  });
  for (const key of ["first-send", "second-send"]) {
    assert.equal(active.size, 0);
    active.set(key, new Set(["target"]));
    table.play({ key });
    assert.equal(playing.has(key), true);
    assert.equal(gm.audio.stop(), 1);
    assert.equal(playing.size, 0);
    assert.equal(active.size, 0);
  }
});

test("STOP attempts other keys after a dispatch failure and retries the failed key", () => {
  const table = createTable();
  const gm = table.install("gm");
  table.play({ key: "first" });
  table.play({ key: "second" });
  gm.game.socket.fail = message => message.action === "stop" && message.key === "first";
  assert.throws(() => gm.audio.stop(), error => error instanceof AggregateError && error.errors.length === 1);
  assert.deepEqual(table.packets.filter(packet => packet.message.action === "stop").map(packet => packet.message.key), ["second"]);
  assert.equal(gm.audio.hasActive(), true);
  gm.game.socket.fail = null;
  assert.equal(gm.audio.stop(), 1);
  assert.equal(gm.audio.hasActive(), false);
});

test("disconnect forgets records and rejects STOP until connected", () => {
  const table = createTable();
  const gm = table.install("gm");
  const target = table.install("target");
  table.play();
  for (const client of [gm, target]) {
    client.game.socket.connected = false;
    client.game.socket.receive("disconnect", "transport close");
  }
  assert.throws(() => gm.audio.stop());
  assert.equal(gm.audio.hasActive(), false);
  gm.game.socket.connected = true;
  target.game.socket.connected = true;
  assert.equal(gm.audio.stop(), 0);
  target.game.socket.receive(CHANNEL, { action: "stop", key: "key-1", userIds: ["target"], senderId: "gm" }, "gm");
  assert.equal(table.packets.some(packet => packet.message.action === "ended"), false);
});

test("STOP rejects non-GM, changed world, changed socket and changed user", () => {
  for (const change of [
    game => { game.user.isGM = false; },
    game => { game.world = { id: "other" }; },
    game => { game.world.id = "other"; },
    game => { game.socket = { ...game.socket }; },
    game => { game.user = game.users.get("gm2"); }
  ]) {
    const table = createTable();
    const gm = table.install("gm");
    table.play();
    change(gm.game);
    assert.throws(() => gm.audio.stop());
    assert.equal(gm.audio.hasActive(), false);
  }
  const table = createTable();
  assert.throws(() => table.install("target").audio.stop());
});

test("dispose removes only its listeners and rejects later STOP", () => {
  const table = createTable();
  const gm = table.install("gm");
  let messages = 0;
  let outgoing = 0;
  gm.game.socket.on(CHANNEL, () => { messages++; });
  gm.game.socket.onAnyOutgoing(() => { outgoing++; });
  table.play();
  gm.audio.dispose();
  gm.audio.dispose();
  table.play({ sender: "gm2" });
  table.play();
  assert.equal(messages, 1);
  assert.equal(outgoing, 2);
  assert.equal(gm.audio.hasActive(), false);
  assert.throws(() => gm.audio.stop());
});

test("an instance bound to global game rejects a replacement game session", () => {
  const table = createTable();
  const previousGame = globalThis.game;
  try {
    globalThis.game = table.clients.get("gm").game;
    const gm = table.install("gm");
    table.play();
    globalThis.game = { ...gm.game };
    assert.throws(() => gm.audio.stop());
    assert.equal(gm.audio.hasActive(), false);
  } finally {
    if (previousGame === undefined) delete globalThis.game;
    else globalThis.game = previousGame;
  }
});

test("observer errors and a throwing error callback do not break native emit", () => {
  const table = createTable();
  const gm = table.install("gm", [PATH], () => { throw new Error("report failed"); });
  const users = gm.game.users;
  gm.game.users = { get() { throw new Error("lookup failed"); } };
  assert.doesNotThrow(() => table.play());
  gm.game.users = users;
  assert.equal(gm.audio.hasActive(), false);
  table.play({ key: "valid" });
  assert.equal(gm.audio.stop(), 1);
});
