import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseHTML } from 'linkedom';
import { hudIcon } from '../scripts/icons.mjs';

const view = await import('../scripts/view.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
function setup(configureDocument = () => {}) {
  assert.equal(typeof view.mountDirector, 'function', 'mountDirector must be implemented');
  const { document } = parseHTML('<html><body><audio id="livekit"></audio></body></html>');
  configureDocument(document);
  const director = view.mountDirector(document, { skin: 'cotct', assetRoot: '/module/assets/' });
  return { document, director, root: document.getElementById('taka-obs-director') };
}
function loadPortrait(image, width, height) {
  Object.defineProperties(image, {
    naturalWidth: { value: width, configurable: true },
    naturalHeight: { value: height, configurable: true },
    complete: { value: true, configurable: true },
  });
  image.dispatchEvent(new image.ownerDocument.defaultView.Event('load'));
}
const actor = (overrides = {}) => ({
  actorId: 'pc1', name: '米什帕特', portrait: '/portrait.png', publicOnly: false,
  hp: { value: 0, max: 82, temp: 8 },
  stats: [{ label: 'AC', value: 27 }, { label: '强韧', value: 12 }, { label: '反射', value: null }, { label: '意志', value: 0 }, { label: '察觉', value: -1 }, { label: '地面', value: 0 }],
  shield: { hp: { value: 0, max: 20 }, hardness: 5, raised: true, broken: true },
  counters: [{ id: 'focus', label: '专注', value: 2, max: 3 }],
  classDc: { label: '职业 DC', value: 26 },
  spellcasting: [{ id: 'arcane', label: '奥术施法', dc: 25, kind: 'prepared', groups: [{ rank: 1, label: '1环', value: 0, max: 3, unlimited: false }] }],
  conditions: [{ id: 'fear', label: '惊惧 1' }], ...overrides,
});
const state = (overrides = {}) => ({
  worldId: 'cotct', mode: 'combat', cast: [
    { userId: 'gm', role: 'gm', name: 'GM', online: true },
    { userId: 'one', role: 'pc', name: '甲', portrait: '/one.png', hp: { value: 0, max: 10, temp: 0 }, online: true },
    { userId: 'two', role: 'pc', name: '乙', portrait: '/two.png', hp: null, online: false },
  ], focus: actor(), story: null, speakingUserIds: [], ...overrides,
});

test('focus uses original art with manual calibration while bottom cast keeps its layout', () => {
  const { director, root } = setup();
  const tokenPortrait = 'data:image/png;base64,bmF0aXZlLXJpbmc=';
  director.render(state({ focus: actor({ tokenPortrait, portraitLayout: { x: -4, y: 2, scale: 1.05 } }) }));
  const figure = root.querySelector('.focus-header .focus-portrait');
  assert.equal(figure.getAttribute('src'), '/portrait.png');
  assert.equal(figure.style.getPropertyValue('--portrait-shift-x'), '-4%');
  assert.equal(figure.style.getPropertyValue('--portrait-shift-y'), '2%');
  assert.equal(figure.style.getPropertyValue('--portrait-scale'), '1.05');
  assert.ok(root.querySelector('.focus-portrait-slot .focus-portrait'));
  assert.deepEqual([...root.querySelectorAll('.cast .figure')].map((image) => image.getAttribute('src')), ['/one.png', '/two.png']);
  assert.deepEqual([...root.querySelectorAll('.cast .haze')].map((image) => image.getAttribute('src')), ['/one.png', '/two.png']);
  assert.equal(root.querySelector('.focus-portrait-slot .haze'), null);
  assert.equal(root.querySelector('.focus-portrait-slot .ornate-frame'), null);
});

test('missing original portrait retains an empty slot and accessible character data', () => {
  const { director, root } = setup();
  const focus = actor({ portrait: null });
  Object.defineProperty(focus, 'tokenPortrait', { get() { throw new Error('Token art must not be used'); } });
  assert.doesNotThrow(() => director.render(state({ focus })));
  assert.ok(root.querySelector('.focus-portrait-slot'));
  assert.equal(root.querySelector('.focus-header .focus-portrait'), null);
  assert.equal(root.querySelector('.focus-header .focus-name').textContent, '米什帕特');
  assert.equal(root.querySelector('.focus-header .hp-value').textContent, '0/82');
  assert.equal(root.querySelector('.cast .figure').getAttribute('src'), '/one.png');
});

test('focus switches clear the previous portrait and its calibration', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ portrait: '/first.webp', portraitLayout: { x: 10, y: -5, scale: 1.2 } }) }));
  assert.equal(root.querySelector('.focus-portrait').getAttribute('src'), '/first.webp');
  director.render(state({ focus: actor({ portrait: null }) }));
  assert.equal(root.querySelector('.focus-portrait'), null);
  assert.equal(root.querySelector('.focus-portrait-slot').childElementCount, 0);
  director.render(state({ focus: { name: '公开战斗员', portrait: '/public.webp', publicOnly: true } }));
  assert.equal(root.querySelector('.focus-portrait').getAttribute('src'), '/public.webp');
  assert.equal(root.querySelector('.focus-portrait').style.getPropertyValue('--portrait-scale'), '1');
  assert.equal(root.querySelector('.focus-data'), null);
});

test('material URLs use the document base rather than the director stylesheet directory', () => {
  for (const [baseURI, assetRoot, expectedRoot] of [
    ['https://table.example/game', undefined, 'https://table.example/modules/taka-obs-director/assets/'],
    ['https://table.example/foundry/', undefined, 'https://table.example/foundry/modules/taka-obs-director/assets/'],
    ['http://127.0.0.1:8974/fixture/', '../module/assets/', 'http://127.0.0.1:8974/module/assets/'],
    ['http://127.0.0.1:8974/fixture/', '/module/assets/', 'http://127.0.0.1:8974/module/assets/'],
    ['https://table.example/game', 'https://assets.example/director/', 'https://assets.example/director/'],
  ]) {
    const { document } = parseHTML('<html><body></body></html>');
    Object.defineProperty(document, 'baseURI', { value: baseURI });
    const director = view.mountDirector(document, { assetRoot });
    try {
      director.render(state());
      const root = document.getElementById('taka-obs-director');
      const stylesheetUrl = new URL('modules/taka-obs-director/styles/director.css', baseURI);
      for (const [property, file] of [['--focus-material', 'focus-material-atlas.png'], ['--cast-material', 'cast-nameplate-atlas.png']]) {
        const css = root.style.getPropertyValue(property);
        const url = /^url\("(.*)"\)$/.exec(css)?.[1];
        assert.ok(url, `${property} must contain a CSS URL`);
        assert.equal(new URL(url, stylesheetUrl).href, `${expectedRoot}textures/${file}`);
      }
    } finally { director.destroy(); }
  }
});

test('material URLs support document URL fallback when baseURI is unavailable', () => {
  const { document } = parseHTML('<html><body></body></html>');
  Object.defineProperty(document, 'URL', { value: 'https://table.example/foundry/game' });
  const director = view.mountDirector(document);
  try {
    director.render(state());
    assert.equal(document.getElementById('taka-obs-director').style.getPropertyValue('--focus-material'), 'url("https://table.example/foundry/modules/taka-obs-director/assets/textures/focus-material-atlas.png")');
  } finally { director.destroy(); }
});

test('switching caster to noncaster clears old shield, pools, sources and conditions while retaining HP0', () => {
  const { director, root } = setup();
  director.render(state());
  assert.match(root.querySelector('.focus-panel').textContent, /0\/82/);
  assert.match(root.querySelector('.cast').textContent, /0\/10/);
  assert.match(root.textContent, /奥术施法/);
  director.render(state({ focus: actor({ name: '战士', spellcasting: [], counters: [], shield: null, classDc: null, conditions: [] }) }));
  assert.match(root.querySelector('.focus-panel').textContent, /0\/82/);
  for (const stale of ['奥术施法', '专注', '惊惧', '硬度', '职业 DC']) assert.ok(!root.textContent.includes(stale), stale);
});

test('two simultaneous speakers highlight independently without replacing focus or portraits', () => {
  const { director, root } = setup();
  director.render(state());
  const panel = root.querySelector('.focus-panel');
  const first = root.querySelector('[data-user-id="one"]');
  director.setSpeaking(['one', 'two']);
  assert.equal(root.querySelectorAll('.cast-person.speaking').length, 2);
  assert.equal(root.querySelector('.focus-panel'), panel);
  assert.equal(root.querySelector('[data-user-id="one"]'), first);
  assert.match(panel.textContent, /米什帕特/);
  director.setSpeaking(['gm']);
  assert.equal(root.querySelectorAll('.speaking').length, 1);
  assert.ok(root.querySelector('[data-user-id="gm"]').classList.contains('speaking'));
});

test('unrelated state refreshes preserve the mounted cast portrait and its loaded images', () => {
  const { director, root } = setup();
  director.render(state());
  const person = root.querySelector('[data-user-id="one"]');
  const portrait = person.querySelector('.portrait');
  const figure = portrait.querySelector('.figure');
  const haze = portrait.querySelector('.haze');
  loadPortrait(figure, 512, 512);
  director.render(state({ cast: state().cast.map((seat) => seat.userId === 'one'
    ? { ...seat, name: '甲的新名字', hp: { value: 6, max: 10, temp: 2 } } : seat),
  focus: actor({ hp: { value: 20, max: 82, temp: 0 } }) }));
  const updated = root.querySelector('[data-user-id="one"]');
  assert.ok(updated === person, 'a state refresh must preserve the mounted seat');
  assert.ok(updated.querySelector('.portrait') === portrait, 'the portrait transition target must stay mounted');
  assert.ok(updated.querySelector('.figure') === figure, 'unchanged image sources must keep their loaded figure');
  assert.ok(updated.querySelector('.haze') === haze, 'unchanged image sources must keep their loaded haze');
  assert.equal(updated.querySelector('.cast-name').getAttribute('aria-label'), '甲的新名字');
  assert.equal(updated.querySelector('.hp-value').textContent, '6/10');
  assert.equal(updated.querySelector('.temp-value').textContent, '2');
});

test('loaded square portrait classification survives a refresh before another image load', () => {
  const { director, root } = setup();
  director.render(state());
  loadPortrait(root.querySelector('[data-user-id="one"] .figure'), 512, 512);
  loadPortrait(root.querySelector('[data-user-id="two"] .figure'), 512, 1024);
  assert.equal(root.querySelector('[data-user-id="one"] .portrait').classList.contains('square'), true);
  director.render(state({ mode: 'explore', focus: null }));
  assert.equal(root.querySelector('[data-user-id="one"] .portrait').classList.contains('square'), true,
    'a loaded square avatar must not briefly use the tall silhouette on refresh');
  assert.equal(root.querySelector('[data-user-id="two"] .portrait').classList.contains('square'), false);
});

test('speaking changes across state refreshes keep the PC and GM transition targets mounted', () => {
  const { director, root } = setup();
  director.render(state());
  const person = root.querySelector('[data-user-id="one"]');
  const portrait = person.querySelector('.portrait');
  const gm = root.querySelector('[data-user-id="gm"]');
  const logo = gm.querySelector('.gm-logo');
  for (const speakingUserIds of [['one', 'gm'], ['one', 'gm'], ['gm'], []]) {
    director.render(state({ speakingUserIds }));
    assert.ok(root.querySelector('[data-user-id="one"] .portrait') === portrait,
      'ordinary render updates must not restart the PC speaking transition');
    assert.ok(root.querySelector('[data-user-id="gm"] .gm-logo') === logo,
      'ordinary render updates must not restart the GM speaking transition');
    assert.equal(person.classList.contains('speaking'), speakingUserIds.includes('one'));
    assert.equal(gm.classList.contains('speaking'), speakingUserIds.includes('gm'));
  }
});

test('changed or revoked cast portraits remove old image sources and their square classification', () => {
  const { director, root } = setup();
  director.render(state());
  const first = root.querySelector('[data-user-id="one"] .figure');
  loadPortrait(first, 512, 512);
  const withPortrait = (portrait) => state({ cast: [{ userId: 'one', role: 'pc', name: '甲', portrait }] });
  director.render(withPortrait('/replacement.webp'));
  const replacement = root.querySelector('.cast .figure');
  assert.equal(first.isConnected, false);
  assert.equal(root.querySelector('img[src="/one.png"]'), null);
  assert.equal(replacement.getAttribute('src'), '/replacement.webp');
  assert.equal(root.querySelector('.cast .haze').getAttribute('src'), '/replacement.webp');
  assert.equal(root.querySelector('.cast .portrait').classList.contains('square'), false);
  loadPortrait(replacement, 512, 1024);
  director.render(withPortrait(null));
  assert.equal(replacement.isConnected, false);
  assert.equal(root.querySelector('.cast .portrait'), null);
  assert.equal(root.querySelector('img[src="/replacement.webp"]'), null);
  director.render(withPortrait('/replacement.webp'));
  assert.ok(root.querySelector('.cast .figure') !== replacement,
    'restored permission must not resurrect a detached portrait from a cache');
  director.render(state({ cast: [] }));
  assert.equal(root.querySelector('.cast').childElementCount, 0);
  assert.equal(root.querySelector('.gm-stage').childElementCount, 0);
});

test('reordered offline seats and a fifth PC preserve portraits while updating placement and availability', () => {
  const { director, root } = setup();
  director.render(state());
  const one = root.querySelector('[data-user-id="one"]');
  const two = root.querySelector('[data-user-id="two"]');
  const figure = one.querySelector('.figure');
  loadPortrait(figure, 512, 512);
  const [gm, first, second] = state().cast;
  const additions = ['three', 'four', 'five'].map((userId) => ({ userId, role: 'pc', name: userId, portrait: `/${userId}.webp` }));
  director.render(state({ cast: [{ ...second, online: true }, ...additions, { ...first, online: false }, gm],
    speakingUserIds: ['one', 'five'] }));
  assert.deepEqual([...root.querySelector('.cast').children].map((node) => node.dataset.userId),
    ['two', 'three', 'four', 'five', 'one']);
  assert.equal(root.querySelector('.cast').style.getPropertyValue('--pc-count'), '5');
  assert.ok(root.querySelector('[data-user-id="one"]') === one, 'reordering must move the existing seat');
  assert.ok(root.querySelector('[data-user-id="two"]') === two, 'online updates must preserve the seat');
  assert.ok(one.querySelector('.figure') === figure);
  assert.equal(one.classList.contains('offline'), true);
  assert.equal(two.classList.contains('offline'), false);
  assert.equal(one.classList.contains('speaking'), true);
  assert.ok(root.querySelector('.cast [data-user-id="five"].speaking'));
  assert.ok(root.querySelector('.gm-stage [data-user-id="gm"]'));
  director.render(state({ cast: [gm, second] }));
  assert.equal(one.isConnected, false);
  director.render(state());
  assert.ok(root.querySelector('[data-user-id="one"] .figure') !== figure,
    'removed seats must not retain images for later reuse');
});

test('theme updates refresh GM art and title without replacing its speaking transition target', () => {
  const { director, root } = setup();
  director.render(state({ speakingUserIds: ['gm'] }));
  const gm = root.querySelector('.gm-seat');
  const logo = gm.querySelector('.gm-logo');
  for (const skin of ['sog', 'fotrp', 'av', 'bob', 'cotct']) {
    director.render(state({ skin, speakingUserIds: ['gm'] }));
    assert.ok(root.querySelector('.gm-seat') === gm, 'theme changes must preserve the mounted GM seat');
    assert.ok(root.querySelector('.gm-logo') === logo, 'theme changes must preserve the GM transition target');
    assert.equal(logo.querySelector('.gm-logo-base').getAttribute('src'), `/module/assets/logos/${skin}.webp`);
    assert.equal(logo.querySelectorAll('.gm-logo-title').length, skin === 'cotct' ? 1 : 0);
    assert.equal(gm.classList.contains('speaking'), true);
  }
});

test('already complete cast images are classified without waiting for a later load event', () => {
  const { director, root } = setup((document) => {
    const createElement = document.createElement.bind(document);
    document.createElement = (tag) => {
      const node = createElement(tag);
      if (tag === 'img') Object.defineProperties(node, {
        complete: { value: true }, naturalWidth: { value: 512 }, naturalHeight: { value: 512 },
      });
      return node;
    };
  });
  director.render(state());
  assert.equal(root.querySelector('[data-user-id="one"] .portrait').classList.contains('square'), true,
    'a cached complete square image must use its classified size on the first render');
});

test('new cast portraits stay hidden until their natural dimensions select the correct silhouette', () => {
  const { director, root } = setup();
  director.render(state());
  const portrait = root.querySelector('[data-user-id="one"] .portrait');
  const figure = portrait.querySelector('.figure');
  assert.equal(portrait.hidden, true, 'an unclassified image must not flash at the tall portrait size');
  loadPortrait(figure, 0, 0);
  assert.equal(portrait.hidden, true, 'missing dimensions must not reveal or classify a failed image');
  loadPortrait(figure, 512, 512);
  assert.equal(portrait.hidden, false);
  assert.equal(portrait.classList.contains('square'), true);
  const tall = root.querySelector('[data-user-id="two"] .portrait');
  loadPortrait(tall.querySelector('.figure'), 512, 1024);
  assert.equal(tall.hidden, false);
  assert.equal(tall.classList.contains('square'), false);
});

test('refreshing a retained portrait registers the current full name for marquee measurement', () => {
  let pendingFrame;
  const animations = [];
  const { director, root } = setup((document) => {
    Object.defineProperty(document, 'defaultView', { value: {
      requestAnimationFrame(callback) { pendingFrame = callback; return 1; },
      cancelAnimationFrame() { pendingFrame = null; },
      matchMedia() { return { matches: false }; },
    } });
    Object.defineProperty(document, 'timeline', { value: { currentTime: 1000 } });
    const createElement = document.createElement.bind(document);
    document.createElement = (tag) => {
      const node = createElement(tag);
      Object.defineProperty(node, 'clientWidth', { value: 120 });
      node.getBoundingClientRect = () => ({ width: 240 });
      node.animate = () => {
        const animation = { cancelled: false, cancel() { this.cancelled = true; } };
        animations.push({ node, animation });
        return animation;
      };
      return node;
    };
  });
  const cast = [{ userId: 'one', role: 'pc', name: 'Alexandria von Pantalaimon', portrait: '/one.png' }];
  director.render(state({ cast, focus: null }));
  pendingFrame();
  assert.equal(root.querySelector('.cast-name').classList.contains('scrolling'), true);
  const firstAnimation = animations[0].animation;
  const fullName = 'Alexandria von Pantalaimon 米什帕特 <img src=x>';
  director.render(state({ cast: [{ ...cast[0], name: fullName }], focus: null }));
  pendingFrame();
  const currentName = root.querySelector('.cast-name');
  assert.equal(currentName.getAttribute('aria-label'), fullName);
  assert.equal(currentName.querySelector('.name-text').textContent, fullName);
  assert.equal(currentName.querySelector('img'), null);
  assert.equal(currentName.classList.contains('scrolling'), true);
  assert.equal(firstAnimation.cancelled, true);
  assert.equal(animations.length, 2, 'the current name must be measured and animated after each refresh');
  assert.ok(animations[1].node === currentName.querySelector('.name-text'));
  assert.equal(animations[1].node.isConnected, true);
  director.destroy();
  assert.equal(animations[1].animation.cancelled, true);
});

test('all five worlds select original frame and GM logo and remove CotCT title on skin changes', () => {
  const { director, root } = setup();
  for (const [worldId, skin] of [['cotct', 'cotct'], ['sog', 'sog'], ['pnvfcgjbf2cjp7gz', 'fotrp'], ['-', 'av'], ['ujx5r8oipw7ercdr', 'bob']]) {
    director.render(state({ worldId }));
    assert.equal(root.querySelector('.ornate-frame').getAttribute('src'), `/module/assets/frames/${skin}-frame.png`);
    assert.equal(root.querySelector('.gm-logo-base').getAttribute('src'), `/module/assets/logos/${skin}.webp`);
    assert.equal(root.querySelectorAll('.gm-logo-title').length, skin === 'cotct' ? 1 : 0);
  }
});

test('user names, conditions, source labels and image title render as text without headings or demo controls', () => {
  const { director, root } = setup();
  director.render(state({ mode: 'story', cast: [{ userId: 'one', role: 'pc', name: '<img src=x onerror=bad()>', portrait: '/one.png' }],
    focus: actor({ name: '<script>bad()</script>', conditions: [{ id: 'c', label: '<b>惊惧</b>' }] }),
    story: { src: '/public.jpg', title: '<em>公开插画</em>' }, round: 3 }));
  assert.equal(root.querySelectorAll('script, b, em, button').length, 0);
  assert.match(root.textContent, /<img src=x onerror=bad\(\)>/);
  assert.equal(root.querySelector('.story-image').getAttribute('alt'), '<em>公开插画</em>');
  assert.ok(!root.textContent.includes('第三轮'));
  assert.ok(!root.textContent.includes('猩红王座'));
});

test('resource rendering distinguishes inventory, charges, unlimited and focus costs', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ spellcasting: [
    { id: 'items', label: '法杖与魔杖', dc: null, kind: 'items', groups: [
      { rank: null, label: '魔杖 · 库存', value: 4, max: null, unlimited: false },
      { rank: 2, label: '魔杖 · 充能', value: 0, max: 1, unlimited: false },
    ] },
    { id: 'focus', label: '专注法术', dc: 24, kind: 'focus', groups: [{ rank: 1, label: '专注 1', value: null, max: null, unlimited: false }, { rank: 0, label: '戏法', value: null, max: null, unlimited: true }] },
    { id: 'unknown', label: '未知能力', dc: null, kind: 'unknown', groups: [{ rank: null, label: '未知次数', value: null, max: null, unlimited: false }] },
  ] }) }));
  const sources = root.querySelectorAll('.spell-source');
  assert.match(sources[0].textContent, /库存4/);
  assert.match(sources[0].textContent, /0\/1/);
  assert.match(sources[1].textContent, /专注 1/);
  assert.match(sources[1].textContent, /无限/);
  assert.match(sources[2].textContent, /未知次数/);
  assert.ok(!root.textContent.includes('/null'));
  assert.equal(root.querySelectorAll('[data-counter-id="focus"]').length, 1);
});

test('raised-only shield and public focus never invent HP or private details', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ shield: { hp: null, hardness: null, raised: true, broken: false } }) }));
  assert.equal(root.querySelector('.shield').textContent, '');
  assert.equal(root.querySelector('.shield').getAttribute('aria-label'), '举盾');
  assert.equal(root.querySelector('.shield .shield-state').getAttribute('aria-label'), '举盾');
  director.render(state({ focus: actor({ publicOnly: true }) }));
  assert.equal(root.querySelector('.focus-panel').textContent, '米什帕特');
  director.render(state({ focus: null }));
  assert.equal(root.querySelector('.focus-panel').hidden, false);
  assert.equal(root.querySelector('.focus-panel').textContent, '');
});

test('empty sources disappear while real DCs and distinct inventory captions remain accessible', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ spellcasting: [
    { id: 'empty', label: '空施法源', dc: null, kind: 'items', groups: [] },
    { id: 'dc-only', label: '只有DC', dc: 24, kind: 'prepared', groups: [] },
    { id: 'wand', label: '冰流魔杖', dc: null, kind: 'items', groups: [
      { label: '冰流魔杖 · 库存', value: 3, max: null },
      { label: '冰流魔杖 · 本件次数', value: 0, max: 1 },
      { label: '另一魔杖 · 库存', value: 2, max: null }
    ] }
  ] }) }));
  assert.deepEqual([...root.querySelectorAll('.spell-source')].map(n => n.dataset.sourceId), ['dc-only', 'wand']);
  assert.equal(root.querySelector('[data-source-id="dc-only"] .source-dc').textContent, '24');
  assert.equal(root.querySelector('[data-source-id="dc-only"] .source-dc').title, '施法 DC 24');
  const labels = [...root.querySelector('[data-source-id="wand"]').querySelectorAll('.resource-label')];
  assert.deepEqual(labels.map(n => n.textContent), ['库存', '本件次数', '另一魔杖 · 库存']);
  assert.equal(labels[0].title, '冰流魔杖 · 库存');
  assert.equal(labels[0].getAttribute('aria-label'), '冰流魔杖 · 库存');
  assert.deepEqual([...root.querySelector('[data-source-id="wand"]').querySelectorAll('.resource-value')].map(n => n.textContent), ['3', '0/1', '2']);
});

test('speeds show only icons and values while retaining full titles and accessible names', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ stats: [
    { label: 'AC', value: 27 }, { label: '强韧', value: 12 }, { label: '反射', value: 0 }, { label: '意志', value: 0 }, { label: '察觉', value: 5 }, { label: '地面', value: 0 }, { label: '飞行', value: 30 }
  ] }) }));
  const speeds = [...root.querySelectorAll('.stat')].slice(5);
  assert.deepEqual(speeds.map(n => n.textContent), ['0', '30']);
  assert.deepEqual(speeds.map(n => n.title), ['地面', '飞行']);
  assert.deepEqual(speeds.map(n => n.getAttribute('aria-label')), ['地面 0', '飞行 30']);
  assert.ok(speeds.every(n => n.querySelector('svg') && !n.querySelector('.detail-label')));
});

test('statistic DCs do not depend on translated labels and explicit skin works in imported worlds', () => {
  const { director, root } = setup();
  director.render(state({ worldId: 'new-import', skin: 'av', focus: actor({ stats: [
    { label: 'AC', value: 27 }, { label: 'Fortitude', value: 12 }, { label: 'Reflex', value: 0 },
    { label: 'Will', value: -1 }, { label: 'Perception', value: 5 }, { label: 'Land', value: 0 },
  ] }) }));
  assert.equal(root.querySelector('.gm-logo-base').getAttribute('src'), '/module/assets/logos/av.webp');
  assert.deepEqual([...root.querySelectorAll('.stat .detail-value')].map((node) => node.textContent), ['27', '22', '10', '9', '15', '0']);
});

test('unknown and Object prototype skin names cannot supply art outside the approved whitelist', () => {
  const { director, root } = setup();
  for (const key of ['constructor', 'toString', '__proto__', 'lastdayhope']) {
    assert.doesNotThrow(() => director.render(state({ worldId: key, skin: key })));
    assert.equal(root.querySelector('.gm-logo-base').getAttribute('src'), '/module/assets/logos/cotct.webp');
  }
});

test('scene bounds include viewport stage offset for explore, combat and story; destroy preserves audio', () => {
  const { director, root, document } = setup();
  root.getBoundingClientRect = () => ({ left: 35, top: 50, width: 1920, height: 1080 });
  for (const [mode, width] of [['explore', 1704.96], ['combat', 1332.48], ['story', 1704.96]]) {
    director.render(state({ mode }));
    const bounds = director.sceneBounds();
    for (const [key, expected] of Object.entries({ left: 142.52, top: 50, width, height: 804.6 })) assert.ok(Math.abs(bounds[key] - expected) < 0.00001);
  }
  director.destroy();
  assert.equal(document.getElementById('taka-obs-director'), null);
  assert.ok(document.getElementById('livekit'));
  director.render(state());
  assert.equal(document.getElementById('taka-obs-director'), null);
});

test('compact keeps each source’s highest three supported ranks, depleted pools and non-slot meanings', () => {
  const { director, root } = setup();
  const groups = [5, 1, 4, 2, 3].map((rank) => ({ rank, label: `${rank}环`, value: rank === 4 ? 0 : 2, max: 3 }));
  const extra = [{ rank: 0, label: '戏法', unlimited: true }, { rank: 9, label: '未知能力', value: null, max: null }];
  director.render(state({ focus: actor({ spellcasting: ['prepared', 'spontaneous', 'flexible'].map((kind, i) => ({
    id: `source${i}`, label: `来源${i}`, kind, dc: 25 + i, groups: [...groups, ...extra],
  })) }) }));
  for (const source of root.querySelectorAll('.spell-source')) {
    assert.equal(source.querySelectorAll('.resource').length, 5);
    assert.deepEqual([...source.querySelectorAll('[data-rank]')].map((node) => node.dataset.rank), ['3', '4', '5']);
    assert.match(source.textContent, /4环0\/3/);
    assert.match(source.textContent, /戏法无限/);
    assert.match(source.textContent, /未知能力/);
  }
  assert.deepEqual([...root.querySelectorAll('.source-dc')].map((node) => node.textContent), ['25', '26', '27']);
  assert.deepEqual([...root.querySelectorAll('.source-dc')].map((node) => node.getAttribute('aria-label')), ['施法 DC 25', '施法 DC 26', '施法 DC 27']);
  director.render(state({ resourceDetail: 'full', focus: actor({ spellcasting: [{ id: 'a', label: '来源', kind: 'prepared', groups }] }) }));
  assert.deepEqual([...root.querySelectorAll('[data-rank]')].map((node) => node.dataset.rank), ['1', '2', '3', '4', '5']);
});

test('rank selection does not merge equal groups or hide innate uses and independent focus costs', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ spellcasting: [
    { id: 'a', kind: 'prepared', groups: [1, 2, 3, 4, 4].map((rank) => ({ rank, label: `${rank}环`, value: 0, max: 2 })) },
    { id: 'b', kind: 'innate', groups: [1, 2, 3, 4, 5].map((rank) => ({ rank, label: `法术${rank}`, value: 0, max: 1 })) },
    { id: 'c', kind: 'focus', dc: 23, groups: [{ rank: 5, label: '领域 · 专注 1' }] },
    { id: 'd', kind: 'focus', dc: 24, groups: [{ rank: 5, label: '能力 · 专注 2' }] },
  ] }) }));
  assert.deepEqual([...root.querySelector('[data-source-id="a"]').querySelectorAll('[data-rank]')].map((node) => node.dataset.rank), ['2', '3', '4', '4']);
  assert.equal(root.querySelector('[data-source-id="b"]').querySelectorAll('.resource').length, 5);
  assert.equal(root.querySelectorAll('[data-counter-id="focus"]').length, 1);
  assert.match(root.querySelector('[data-source-id="c"]').textContent, /专注 1/);
  assert.match(root.querySelector('[data-source-id="d"]').textContent, /专注 2/);
});

test('standard hero and focus pools show only glyphs including depleted zero; other counters retain real numbers', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ counters: [
    { id: 'hero', label: '英雄点', value: 0, max: 3 }, { id: 'focus', label: '专注', value: 0, max: 3 },
    { id: 'large', label: '资源', value: 8, max: 12 }, { id: 'over', label: '资源', value: 4, max: 3 },
    { id: 'unknown', label: '未知资源', value: 7, max: null }, { id: 'fraction', label: '资源', value: 1.5, max: 3 },
  ] }) }));
  for (const id of ['hero', 'focus']) {
    const row = root.querySelector(`[data-counter-id="${id}"]`);
    assert.equal(row.querySelectorAll('.resource-glyph').length, 3);
    assert.equal(row.querySelectorAll('.resource-glyph.filled').length, 0);
    assert.equal(row.textContent, '');
    assert.match(row.getAttribute('aria-label'), /0\/3/);
    assert.match(row.title, /0\/3/);
  }
  assert.equal(root.querySelector('[data-counter-id="large"]').querySelectorAll('.pip').length, 0);
  assert.match(root.querySelector('[data-counter-id="large"]').textContent, /8\/12/);
  assert.equal(root.querySelector('[data-counter-id="over"]').querySelectorAll('.pip').length, 0);
  assert.match(root.querySelector('[data-counter-id="over"]').textContent, /4\/3/);
  assert.equal(root.querySelector('[data-counter-id="unknown"] .detail-value').textContent, '7');
  assert.equal(root.querySelector('[data-counter-id="fraction"]').querySelectorAll('.pip').length, 0);
  assert.match(root.querySelector('.temp-hp').textContent, /8/);
});

test('public-only focus reads only permitted identity and preserves the full portrait header', () => {
  const { director, root } = setup();
  const focus = { name: '公开战斗员', portrait: '/public.png', publicOnly: true };
  for (const key of ['hp', 'stats', 'counters', 'classDc', 'shield', 'spellcasting', 'conditions']) Object.defineProperty(focus, key, { get() { throw new Error(`Private ${key}`); } });
  assert.doesNotThrow(() => director.render(state({ focus })));
  assert.equal(root.querySelector('.focus-panel').textContent, '公开战斗员');
  assert.ok(root.querySelector('.focus-portrait'), 'public portrait must remain recognizable in the header');
  assert.equal(root.querySelector('.focus-portrait').getAttribute('src'), '/public.png');
  assert.equal(root.querySelector('.focus-art'), null);
});

test('defense and awareness strips show only icons and values with full accessible descriptions', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ stats: [
    { label: 'AC', value: 27 }, { label: 'Fortitude', value: 12 }, { label: 'Reflex', value: 0 }, { label: 'Will', value: -1 },
    { label: 'Perception', value: 5 }, { label: '飞行', value: 30 }, { label: '特殊跃迁', value: 15 },
  ] }) }));
  assert.ok(root.querySelector('.defense-strip'), 'defense must have one aligned strip');
  assert.equal(root.querySelector('.defense-strip').querySelectorAll('.stat').length, 4);
  assert.equal(root.querySelectorAll('.stat .detail-label').length, 0);
  assert.deepEqual([...root.querySelectorAll('.defense-strip .stat')].map((node) => node.textContent), ['27', '22', '10', '9']);
  assert.equal(root.querySelector('.awareness-strip').textContent, '153015');
  assert.deepEqual([...root.querySelectorAll('.defense-strip .stat')].map((node) => node.getAttribute('aria-label')), ['AC 27', 'Fortitude DC 22（修正值 +12）', 'Reflex DC 10（修正值 +0）', 'Will DC 9（修正值 -1）']);
  assert.equal(root.querySelectorAll('.awareness-strip .stat')[2].getAttribute('aria-label'), '特殊跃迁 15');
  assert.ok(!root.querySelector('.awareness-strip').textContent.includes('步'));
  assert.equal(root.querySelector('.defense-strip .hud-icon').getAttribute('aria-label'), 'AC');
});

test('only save and perception modifiers become DCs without mutating snapshots or fabricating absent values', () => {
  const { director, root } = setup();
  const stats = Object.freeze([
    { label: 'AC', value: 27 }, { label: '<b>Fortitude</b>', value: 12 }, { label: 'Reflex', value: 0 },
    { label: 'Will', value: -1 }, { label: 'Perception', value: null }, { label: 'Land', value: 0 },
  ].map(Object.freeze));
  const focus = Object.freeze(actor({ stats }));
  director.render(state({ focus }));
  assert.deepEqual([...root.querySelectorAll('.stat .detail-value')].map((node) => node.textContent), ['27', '22', '10', '9', '0']);
  assert.equal(root.querySelectorAll('.stat').length, 5);
  assert.equal(root.querySelector('.awareness-strip').textContent, '0');
  assert.equal(root.querySelector('.class-dc .detail-value').textContent, '26');
  assert.equal(root.querySelector('.source-dc').textContent, '25');
  assert.equal(stats[1].value, 12);
  assert.equal(root.querySelectorAll('b').length, 0);
  assert.equal(root.querySelectorAll('.stat')[1].title, '<b>Fortitude</b> DC 22（修正值 +12）');
});

test('authoritative statistic DC wins over ten plus modifier and may exist without a modifier', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ stats: [
    { label: 'AC', value: 27, dc: 37 }, { label: 'Fortitude', value: 12, dc: 24 },
    { label: 'Reflex', value: 0, dc: NaN }, { label: 'Will', value: -1, dc: null },
    { label: 'Perception', value: null, dc: 18 }, { label: 'Land', value: 0, dc: 10 },
  ] }) }));
  assert.deepEqual([...root.querySelectorAll('.stat .detail-value')].map((node) => node.textContent), ['27', '24', '10', '9', '18', '0']);
  assert.equal(root.querySelector('.awareness-strip .stat').title, 'Perception DC 18');
});

test('native HUD icons use verified core Free classes and retain original vectors for non-native surfaces', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ stats: actor().stats.map((stat, index) => index === 2 ? { ...stat, value: 10 } : stat) }) }));
  const classes = [...root.querySelectorAll('.defense-strip .hud-icon-native')].map((node) => node.className);
  assert.deepEqual(classes, ['hud-icon-native fa-solid fa-shield', 'hud-icon-native fa-solid fa-chess-rook', 'hud-icon-native fa-solid fa-person-running', 'hud-icon-native fa-solid fa-brain']);
  assert.equal(root.querySelectorAll('.defense-strip .hud-icon-fallback').length, 4);
  assert.equal(root.querySelector('.awareness-strip .hud-icon-native').className, 'hud-icon-native fa-solid fa-eye');
  assert.ok(root.querySelector('.class-dc .hud-icon'), 'class DC needs its own resource symbol');
  assert.equal(root.querySelector('.class-dc .hud-icon-native'), null, 'class DC must not borrow the Will statistic icon');
  assert.equal(root.querySelectorAll('.defense-strip .hud-icon')[1].getAttribute('aria-label'), '强韧 DC 22（修正值 +12）');
  const { document } = parseHTML('<html><body></body></html>');
  for (const name of ['unknown', 'constructor', '__proto__']) {
    const icon = hudIcon(document, name, '未知资源');
    assert.equal(icon.querySelector('.hud-icon-native'), null);
    assert.equal(icon.querySelector('path').getAttribute('d'), hudIcon(document, 'pool', '资源').querySelector('path').getAttribute('d'));
  }
});

test('each approved skin selects its exact atlas column without accepting arbitrary materials', () => {
  const { director, root } = setup();
  for (const [skin, position] of [['cotct', '0%'], ['sog', '25%'], ['fotrp', '50%'], ['av', '75%'], ['bob', '100%']]) {
    director.render(state({ skin }));
    assert.equal(root.style.getPropertyValue('--focus-material'), 'url("/module/assets/textures/focus-material-atlas.png")');
    assert.equal(root.style.getPropertyValue('--focus-material-position'), position);
    assert.equal(root.style.getPropertyValue('--cast-material'), 'url("/module/assets/textures/cast-nameplate-atlas.png")');
  }
});

test('native hero-points and semantic movement types select their approved icons across labels', () => {
  const { director, root } = setup();
  for (const labels of [['Land Speed', 'Flight'], ['地面速度', '飞行速度']]) {
    director.render(state({ focus: actor({ counters: [{ id: 'hero-points', label: 'Hero Points', value: 0, max: 3 }], stats: [...actor().stats.slice(0, 5), { label: labels[0], value: 25, movementType: 'land' }, { label: labels[1], value: 30, movementType: 'fly' }] }) }));
    const hero = root.querySelector('[data-counter-id="hero-points"]');
    assert.equal(hero.querySelector('.detail-label'), null); assert.equal(hero.querySelector('.hud-icon path').getAttribute('d'), hudIcon(root.ownerDocument, 'hero', '').querySelector('path').getAttribute('d')); assert.equal(hero.title, 'Hero Points 0/3');
    const speeds = [...root.querySelectorAll('.awareness-strip .stat')].slice(1);
    assert.match(speeds[0].querySelector('.hud-icon-native').className, /fa-shoe-prints/); assert.match(speeds[1].querySelector('.hud-icon-native').className, /fa-feather/);
    assert.equal(speeds[0].getAttribute('aria-label'), `${labels[0]} 25`);
  }
});

test('native GM speaking transform scales only the GM logo and preserves centered placement', () => {
  const { director, root } = setup(); director.render(state()); director.setSpeaking(['gm']);
  const css = fs.readFileSync(new URL('../styles/director.css', import.meta.url), 'utf8');
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]+)\}/g)];
  const speaking = rules.find(([_, selector, body]) => selector.trim() === '#taka-obs-director .gm-seat.speaking .gm-logo' && /transform\s*:\s*translateX\(-50%\) scale\(var\(--speech-scale\)\)/.test(body));
  assert.ok(speaking, 'GM must scale with the approved speaking variable while centered');
  assert.ok(root.querySelector(speaking[1].trim()));
  director.setSpeaking([]); assert.equal(root.querySelector(speaking[1].trim()), null);
  assert.match(css, /\.gm-logo\s*\{[^}]*transition:\s*transform 180ms ease/);
});

test('GM lives in the top decoration stage for every skin while only PCs occupy the bottom cast', () => {
  const { director, root } = setup();
  const seats = [...state().cast, ...['three', 'four', 'five'].map((userId) => ({ userId, role: 'pc', name: userId }))];
  for (const skin of ['cotct', 'sog', 'fotrp', 'av', 'bob']) {
    director.render(state({ skin, cast: seats }));
    assert.ok(root.querySelector('.gm-stage [data-user-id="gm"]'), 'GM must occupy the top stage');
    assert.equal(root.querySelector('.cast .gm-seat'), null);
    assert.equal(root.querySelector('.cast').childElementCount, 5);
    director.setSpeaking(['gm', 'one', 'five']);
    assert.equal(root.querySelectorAll('.speaking').length, 3);
    assert.ok(root.querySelector('.gm-stage .gm-seat.speaking'));
    assert.equal(root.querySelector('.focus-name').textContent, '米什帕特');
  }
  director.render(state({ cast: seats.filter((seat) => seat.role !== 'gm') }));
  assert.equal(root.querySelector('.gm-stage').childElementCount, 0);
  assert.equal(root.querySelector('.gm-stage').hidden, true);
});

test('portrait name and HP remain outside the scrolling data area and combat empty state retains its slot', () => {
  const { director, root } = setup();
  director.render(state());
  assert.ok(root.querySelector('.focus-header .focus-portrait'));
  assert.ok(root.querySelector('.focus-header .focus-name'));
  assert.ok(root.querySelector('.focus-header .focus-hp'));
  assert.ok(root.querySelector('.focus-data .defense-strip'));
  assert.ok(root.querySelector('.focus-data .spell-sources'));
  assert.equal(root.querySelector('.focus-data .focus-hp'), null);
  director.render(state({ focus: null }));
  assert.equal(root.querySelector('.focus-panel').hidden, false);
  assert.equal(root.querySelector('.focus-panel').childElementCount, 0);
  director.render(state({ mode: 'explore' }));
  assert.equal(root.querySelector('.focus-panel').hidden, true);
});

test('resource glyphs preserve partial pools and shield states retain health hardness and truthful accessible data', () => {
  const { director, root } = setup();
  director.render(state({ focus: actor({ counters: [
    { id: 'hero-points', label: '英雄点', value: 1, max: 3 }, { id: 'focus', label: '专注', value: 2, max: 3 },
    { id: 'hero', label: '英雄点', value: 4, max: 3 },
  ] }) }));
  assert.equal(root.querySelector('[data-counter-id="hero-points"]').querySelectorAll('.resource-glyph.filled').length, 1);
  assert.equal(root.querySelector('[data-counter-id="focus"]').querySelectorAll('.resource-glyph.filled').length, 2);
  assert.equal(root.querySelector('[data-counter-id="hero"]').querySelectorAll('.resource-glyph').length, 0);
  assert.equal(root.querySelector('[data-counter-id="hero"]').textContent, '4/3');
  const shield = root.querySelector('.shield');
  assert.match(shield.getAttribute('aria-label'), /举盾.*破损.*0\/20.*硬度 5/);
  assert.match(shield.title, /举盾.*破损.*0\/20.*硬度 5/);
  assert.equal(shield.querySelectorAll('.shield-state').length, 2);
  assert.match(shield.textContent, /0\/20/);
  director.render(state({ focus: actor({ shield: { raised: true, broken: false, hp: null, hardness: null } }) }));
  assert.equal(root.querySelector('.shield').textContent, '');
  assert.equal(root.querySelector('.shield').querySelectorAll('.shield-state').length, 1);
});

test('DC and temporary health use symbols and numbers while keeping truthful accessible names', () => {
  const { director, root } = setup();
  director.render(state());
  const classDc = root.querySelector('.class-dc');
  assert.equal(classDc.textContent, '26');
  assert.match(classDc.getAttribute('aria-label'), /DC.*26/);
  const spellDc = root.querySelector('.source-dc');
  assert.equal(spellDc.textContent, '25');
  assert.equal(spellDc.getAttribute('aria-label'), '施法 DC 25');
  for (const temporary of root.querySelectorAll('.temp-hp')) {
    assert.equal(temporary.textContent, '8');
    assert.equal(temporary.getAttribute('aria-label'), '临时生命 8');
    assert.equal(temporary.title, '临时生命 8');
  }
  assert.notEqual(classDc.querySelector('path').getAttribute('d'), root.querySelector('.defense-strip .stat:last-child path').getAttribute('d'));
});

test('long names travel at 24px per second with two-second rests, while fitting and unmeasured names stay still', () => {
  assert.equal(typeof view.nameMotion, 'function');
  for (const dimensions of [[100, 120], [120.5, 120], [200, 0], [NaN, 120], [200, Infinity]]) assert.equal(view.nameMotion(...dimensions), null);
  const motion = view.nameMotion(192, 120);
  assert.equal(motion.duration, 10000);
  assert.deepEqual(motion.frames.map(frame => frame.offset), [0, .2, .5, .7, 1]);
  assert.deepEqual(motion.frames.map(frame => frame.transform), ['translateX(0px)', 'translateX(0px)', 'translateX(-72px)', 'translateX(-72px)', 'translateX(0px)']);
});

test('cast and focus preserve complete untrusted names inside a separate stationary plaque', () => {
  const { director, root } = setup();
  const fullName = 'Alexandria von Pantalaimon 米什帕特·潘塔拉多姆 <img src=x>';
  director.render(state({ cast: [{ userId: 'one', role: 'pc', name: fullName }], focus: actor({ name: fullName }) }));
  for (const selector of ['.cast-name', '.focus-name-text .name-viewport']) {
    const viewport = root.querySelector(selector);
    assert.ok(viewport);
    assert.equal(viewport.querySelector('.name-text').textContent, fullName);
    assert.equal(viewport.getAttribute('aria-label'), fullName);
    assert.equal(viewport.querySelector('img'), null);
  }
  assert.equal(root.querySelector('.focus-name-text').firstElementChild.className, 'name-viewport');
  director.destroy();
  assert.equal(root.isConnected, false);
});
