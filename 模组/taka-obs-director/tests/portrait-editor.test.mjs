import test from 'node:test';
import assert from 'node:assert/strict';
import { parseHTML } from 'linkedom';

const editor = await import('../scripts/portrait-editor.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
const moduleId = 'taka-obs-director';

function fixture() {
  const { document, window } = parseHTML('<html><body></body></html>');
  Object.defineProperty(document, 'baseURI', { value: 'https://table.example/foundry/' });
  const actor = (id, name) => ({ id, name, img: `assets/${id}.webp` });
  const actors = Array.from({ length: 5 }, (_, index) => actor(`actor-${index}`, index === 0 ? '<img onerror=evil>' : `角色${index}`));
  for (const item of actors) Object.defineProperty(item, 'system', { get() { throw new Error('no actor stats'); } });
  const users = new Map(actors.map((character, index) => [`user-${index}`, { id: `user-${index}`, isGM: false, active: index !== 4, name: `玩家${index}`, character }]));
  users.set('other', { id: 'other', isGM: false, character: actor('other', '其他玩家') });
  users.set('gm', { id: 'gm', isGM: true, character: actor('gm-character', 'GM角色') });
  const values = new Map([
    ['seatOrder', [...users.keys()]], ['portraitOverrides', { 'actor-1': 'assets/original-one.png' }],
    ['portraitLayouts', { 'actor-0': { source: actors[0].img, x: 12, y: -4, scale: 1.2 }, unrelated: { source: 'untouched.webp', x: 8, y: 3, scale: 2 } }],
    ['skin', 'auto'],
  ]);
  values.set('seatOrder', Array.from({ length: 5 }, (_, index) => `user-${index}`).concat('gm'));
  const writes = [];
  const game = {
    user: { id: 'gm', isGM: true }, world: { id: 'cotct' }, users,
    settings: {
      get(namespace, key) { assert.equal(namespace, moduleId); return values.get(key); },
      async set(namespace, key, value) { assert.equal(namespace, moduleId); assert.equal(key, 'portraitLayouts'); writes.push(value); values.set(key, value); },
    },
  };
  const defaults = { cotct: { 'actor-0': { source: actors[0].img, x: -5, y: 6, scale: 1.1 } } };
  return { document, window, actors, users, values, game, defaults, writes };
}

function controller(f) {
  assert.equal(typeof editor.createPortraitController, 'function', 'createPortraitController must be implemented');
  const control = editor.createPortraitController(f);
  const host = f.document.createElement('form');
  host.innerHTML = control.content.innerHTML;
  f.document.body.append(host);
  control.bind(host);
  return { control, host };
}

function input(f, host, name, value) {
  const element = host.querySelector(`[name="${name}"]`);
  element.value = String(value);
  element.dispatchEvent(new f.window.Event('input', { bubbles: true }));
}

function selectActor(f, host, actorId) {
  const select = host.querySelector('[name="actorId"]');
  for (const option of select.options) option.removeAttribute('selected');
  [...select.options].find(option => option.value === actorId).setAttribute('selected', '');
  select.dispatchEvent(new f.window.Event('change', { bubbles: true }));
}

test('all configured players including an offline fifth can calibrate original images without actor stats', () => {
  const f = fixture(), { control, host } = controller(f);
  assert.equal(control.content.attributes.length, 0, 'native DialogV2 requires an attribute-free outer div');
  const select = host.querySelector('[name="actorId"]');
  assert.equal(select.options.length, 5);
  assert.equal(select.options[4].value, 'actor-4');
  assert.equal(host.querySelector('img.portrait-calibration-image').getAttribute('src'), 'assets/actor-0.webp');
  assert.equal(host.querySelector('img[onerror]'), null);
  assert.equal(host.querySelector('[name="x"]').value, '12');
  selectActor(f, host, 'actor-1');
  assert.equal(host.querySelector('img.portrait-calibration-image').getAttribute('src'), 'assets/original-one.png');
  assert.deepEqual(f.writes, []);
});

test('editing one actor merges current world settings without replacing another GM calibration', async () => {
  const f = fixture(), { control, host } = controller(f);
  input(f, host, 'x', 30);
  f.values.set('portraitLayouts', { ...f.values.get('portraitLayouts'), fresh: { source: 'fresh.webp', x: 9, y: -2, scale: 1.3 }, 'actor-1': { source: 'assets/original-one.png', x: -20, y: 0, scale: 1 } });
  assert.equal(await control.save(host), true);
  assert.equal(f.writes.length, 1);
  assert.deepEqual(f.writes[0]['actor-0'], { source: 'assets/actor-0.webp', x: 30, y: -4, scale: 1.2 });
  assert.deepEqual(f.writes[0]['actor-1'], { source: 'assets/original-one.png', x: -20, y: 0, scale: 1 });
  assert.deepEqual(f.writes[0].fresh, { source: 'fresh.webp', x: 9, y: -2, scale: 1.3 });
  assert.deepEqual(f.writes[0].unrelated, { source: 'untouched.webp', x: 8, y: 3, scale: 2 });
});

test('restoring the preset affects only the selected draft and cancellation never writes', async () => {
  const f = fixture(), { control, host } = controller(f);
  host.querySelector('[data-action="restore"]').dispatchEvent(new f.window.Event('click', { bubbles: true }));
  assert.equal(host.querySelector('[name="x"]').value, '-5');
  assert.equal(host.querySelector('[name="y"]').value, '6');
  assert.equal(host.querySelector('[name="scale"]').value, '1.1');
  control.dispose();
  assert.equal(await control.save(host), false);
  assert.deepEqual(f.writes, []);
});

test('unchanged dialogs have no writes and numeric UI edits are clamped before persistence', async () => {
  const f = fixture(), { control, host } = controller(f);
  assert.equal(await control.save(host), true);
  assert.deepEqual(f.writes, []);
  input(f, host, 'x', 999);
  input(f, host, 'y', -999);
  input(f, host, 'scale', 99);
  assert.equal(await control.save(host), true);
  assert.deepEqual(f.writes[0]['actor-0'], { source: 'assets/actor-0.webp', x: 75, y: -75, scale: 3 });
});

test('GM, account and world identity changes refuse stale calibration writes', async () => {
  for (const change of [
    f => { f.game.user.isGM = false; }, f => { f.game.user.id = 'another-gm'; }, f => { f.game.world.id = 'sog'; },
  ]) {
    const f = fixture(), { control, host } = controller(f);
    input(f, host, 'x', 24);
    change(f);
    assert.equal(await control.save(host), false);
    assert.deepEqual(f.writes, []);
  }
});

test('source, assignment and seat changes refuse the entire save instead of writing stale actors', async () => {
  for (const change of [
    f => { f.actors[0].img = 'assets/new.webp'; },
    f => { f.values.set('portraitOverrides', { 'actor-0': 'assets/new-source.webp' }); },
    f => { f.users.get('user-0').character = { ...f.actors[0] }; },
    f => { f.values.set('seatOrder', ['user-1', 'user-2']); },
  ]) {
    const f = fixture(), { control, host } = controller(f);
    input(f, host, 'x', 24);
    change(f);
    assert.equal(await control.save(host), false);
    assert.deepEqual(f.writes, []);
  }
});

test('serialized render binds actual controls, and later rerenders remove old listeners', async () => {
  const f = fixture(), { control, host } = controller(f);
  const next = f.document.createElement('form');
  next.innerHTML = control.content.innerHTML;
  control.bind(next);
  input(f, host, 'x', 55);
  input(f, next, 'y', 18);
  await control.save(next);
  assert.deepEqual(f.writes[0]['actor-0'], { source: 'assets/actor-0.webp', x: 12, y: 18, scale: 1.2 });
});

test('pointer dragging changes slot percentages and retains the complete original image', async () => {
  const f = fixture(), { control, host } = controller(f);
  const slot = host.querySelector('.portrait-calibration-slot');
  slot.getBoundingClientRect = () => ({ width: 200, height: 100 });
  const pointer = (type, x, y) => {
    const event = new f.window.Event(type, { bubbles: true });
    Object.assign(event, { pointerId: 1, clientX: x, clientY: y, button: 0 });
    slot.dispatchEvent(event);
  };
  pointer('pointerdown', 10, 20);
  pointer('pointermove', 50, 10);
  pointer('pointerup', 50, 10);
  await control.save(host);
  assert.deepEqual(f.writes[0]['actor-0'], { source: 'assets/actor-0.webp', x: 32, y: -14, scale: 1.2 });
  assert.equal(host.querySelector('.portrait-calibration-image').getAttribute('src'), 'assets/actor-0.webp');
});

test('non-GM dialogs never inspect configured users or images and cannot save', async () => {
  const f = fixture();
  f.game.user.isGM = false;
  Object.defineProperty(f.game, 'users', { get() { throw new Error('not authorized'); } });
  const { control, host } = controller(f);
  assert.equal(host.querySelector('[name="actorId"]'), null);
  assert.equal(await control.save(host), false);
  assert.deepEqual(f.writes, []);
});

test('native DialogV2 subclass uses render event bindings and save/cancel button.form callbacks', async () => {
  const f = fixture();
  class Dialog extends EventTarget {
    constructor(options) { super(); this.options = options; this.options.content = options.content.innerHTML; this.closed = false; }
    render() {
      this.element = f.document.createElement('dialog');
      const form = f.document.createElement('form');
      form.innerHTML = this.options.content;
      this.element.append(form);
      this.dispatchEvent(new Event('render'));
      return this;
    }
    async close() { this.closed = true; this.dispatchEvent(new Event('close')); }
  }
  assert.equal(typeof editor.createPortraitMenu, 'function', 'createPortraitMenu must be implemented');
  assert.equal(editor.createPortraitMenu({ ...f, DialogV2: null }), null);
  const Menu = editor.createPortraitMenu({ ...f, DialogV2: Dialog });
  const menu = new Menu().render();
  const form = menu.element.querySelector('form');
  input(f, form, 'x', 25);
  const save = menu.options.buttons.find(button => button.action === 'save');
  await save.callback({}, { form }, menu);
  assert.equal(menu.closed, true);
  assert.equal(f.writes[0]['actor-0'].x, 25);

  const canceled = new Menu().render();
  input(f, canceled.element, 'x', 60);
  const cancel = canceled.options.buttons.find(button => button.action === 'cancel');
  await cancel.callback({}, { form: canceled.element.querySelector('form') }, canceled);
  assert.equal(canceled.closed, true);
  assert.equal(f.writes.length, 1);
});

test('all edited assignments are checked before any actor calibration is written', async () => {
  const f = fixture(), { control, host } = controller(f);
  input(f, host, 'x', 30);
  selectActor(f, host, 'actor-1');
  input(f, host, 'y', -20);
  f.users.get('user-0').character = { ...f.actors[0] };
  assert.equal(await control.save(host), false);
  assert.deepEqual(f.writes, []);
});

test('merge preserves own prototype-shaped keys without invoking settings accessors', async () => {
  const f = fixture(), { control, host } = controller(f);
  input(f, host, 'x', 25);
  const latest = JSON.parse('{"__proto__":{"source":"old.webp","scale":1},"constructor":{"source":"other.webp","scale":2}}');
  f.values.set('portraitLayouts', latest);
  assert.equal(await control.save(host), true);
  assert.equal(Object.getPrototypeOf(f.writes[0]), Object.prototype);
  assert.equal(Object.hasOwn(f.writes[0], '__proto__'), true);
  assert.deepEqual(f.writes[0].__proto__, { source: 'old.webp', scale: 1 });
  assert.deepEqual(f.writes[0].constructor, { source: 'other.webp', scale: 2 });

  const failed = fixture(), next = controller(failed);
  input(failed, next.host, 'x', 10);
  let calls = 0;
  const accessor = {};
  Object.defineProperty(accessor, 'private', { enumerable: true, get() { calls++; throw new Error('private settings'); } });
  failed.values.set('portraitLayouts', accessor);
  assert.equal(await next.control.save(next.host), false);
  assert.equal(calls, 0);
  assert.deepEqual(failed.writes, []);
});

test('a failed latest-settings read leaves the dialog draft local instead of replacing settings', async () => {
  const f = fixture(), { control, host } = controller(f);
  input(f, host, 'x', 22);
  const read = f.game.settings.get;
  f.game.settings.get = (namespace, key) => {
    if (key === 'portraitLayouts') throw new Error('secret storage path');
    return read(namespace, key);
  };
  assert.equal(await control.save(host), false);
  assert.deepEqual(f.writes, []);
  assert.equal(host.querySelector('.portrait-calibration-status').textContent, '保存失败，请重新打开窗口后重试。');
  assert.ok(!host.textContent.includes('secret storage path'));
});

test('portrait overrides must own a safe image string and cannot evaluate a private accessor', () => {
  const f = fixture();
  let reads = 0;
  const overrides = Object.create({ 'actor-1': 'https://example.test/inherited.webp' });
  Object.defineProperty(overrides, 'actor-0', { get() { reads++; throw new Error('private override'); } });
  overrides['actor-2'] = 'javascript:evil()';
  f.values.set('portraitOverrides', overrides);
  const { host } = controller(f);
  assert.equal(reads, 0);
  assert.equal(host.querySelector('.portrait-calibration-image').getAttribute('src'), 'assets/actor-0.webp');
  selectActor(f, host, 'actor-2');
  assert.equal(host.querySelector('.portrait-calibration-image').getAttribute('src'), 'assets/actor-2.webp');
});

test('native range step rounding cannot turn untouched preset coordinates into a save', async () => {
  const f = fixture();
  f.values.set('portraitLayouts', { 'actor-0': { source: 'assets/actor-0.webp', x: -0.747, y: -8.994, scale: 1.0234 } });
  assert.equal(typeof editor.createPortraitController, 'function');
  const control = editor.createPortraitController(f);
  const host = f.document.createElement('form');
  host.innerHTML = control.content.innerHTML;
  // Linkedom does not sanitize native range values. Model the observed browser
  // step rounding at that boundary while exercising the real controller.
  for (const input of host.querySelectorAll('input[type="range"]')) {
    let value = '';
    Object.defineProperty(input, 'value', {
      get: () => value,
      set(raw) {
        const min = Number(input.getAttribute('min')), step = Number(input.getAttribute('step'));
        value = String(min + Math.round((Number(raw) - min) / step) * step);
      },
    });
  }
  control.bind(host);
  assert.equal(await control.save(host), true);
  assert.deepEqual(f.writes, []);
});
