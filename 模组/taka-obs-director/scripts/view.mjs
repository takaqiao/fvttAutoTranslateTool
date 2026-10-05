import { sceneRect } from './model.mjs';
import { resolveSkin } from './skins.mjs';
import { hudIcon } from './icons.mjs';

const number = (value) => Number.isFinite(value);
const text = (value) => typeof value === 'string' ? value : '';
const list = (value) => Array.isArray(value) ? value : [];
const ratio = (hp) => hp && number(hp.value) && number(hp.max) && hp.max > 0;
const slotGroup = (source, group) => ['prepared', 'spontaneous', 'flexible'].includes(source.kind)
  && Number.isInteger(group.rank) && group.rank > 0 && number(group.max) && group.max > 0;

function resourceGroups(source, detailMode) {
  const groups = list(source.groups);
  const ranks = [...new Set(groups.filter((group) => slotGroup(source, group)).map((group) => group.rank))].sort((a, b) => a - b);
  const selected = new Set(detailMode === 'full' ? ranks : ranks.slice(-3));
  return groups.filter((group) => !slotGroup(source, group) || selected.has(group.rank))
    .sort((a, b) => (number(a.rank) ? a.rank : Infinity) - (number(b.rank) ? b.rank : Infinity));
}

export function nameMotion(textWidth, viewportWidth) {
  if (!number(textWidth) || !number(viewportWidth) || viewportWidth <= 0 || textWidth - viewportWidth <= 1) return null;
  const distance = textWidth - viewportWidth;
  const travel = distance / 24 * 1000;
  const duration = 4000 + travel * 2;
  const start = 'translateX(0px)';
  const end = `translateX(${-distance}px)`;
  return { duration, frames: [
    { transform: start, offset: 0 }, { transform: start, offset: 2000 / duration },
    { transform: end, offset: (2000 + travel) / duration }, { transform: end, offset: (4000 + travel) / duration },
    { transform: start, offset: 1 },
  ] };
}

function nameFingerprint(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) hash = Math.imul(hash ^ value.charCodeAt(i), 16777619);
  return `${value.length}:${hash >>> 0}`;
}

/** Renders already-authorized plain snapshots. No Foundry documents or media tracks. */
export function mountDirector(document, { skin = 'cotct', assetRoot = 'modules/taka-obs-director/assets/' } = {}) {
  if (document.getElementById('taka-obs-director')) throw new Error('Director is already mounted');
  const el = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  };
  const img = (className, src, alt = '') => {
    const node = el('img', className);
    node.src = src;
    node.alt = alt;
    node.draggable = false;
    return node;
  };
  const asset = (path) => `${String(assetRoot).replace(/\/?$/, '/')}${path}`;
  const materialUrl = (path) => {
    const src = asset(path);
    // CSS variables resolve relative URLs from the stylesheet, not the page.
    try { return new URL(src, document.baseURI ?? document.URL).href; }
    catch { return src; }
  };
  const root = el('div', 'taka-director');
  root.id = 'taka-obs-director';
  const scene = el('div', 'scene-zone');
  const frame = img('ornate-frame', '');
  const panel = el('section', 'focus-panel');
  const cast = el('div', 'cast');
  const gmStage = el('div', 'gm-stage');
  root.append(scene, frame, panel, cast, gmStage);
  document.body.append(root);
  let mode = 'explore';
  let destroyed = false;
  let speaking = new Set();
  let currentSkin = resolveSkin(skin) ?? resolveSkin('cotct');
  let resourceDetail = 'compact';
  const window = document.defaultView;
  const requestFrame = window?.requestAnimationFrame?.bind(window);
  const reducedMotion = window?.matchMedia?.('(prefers-reduced-motion: reduce)');
  const canAnimateNames = !!requestFrame && typeof root.animate === 'function';
  const nameEpochs = new Map();
  const activeNames = new Map();
  let names = [];
  let nameFrame = null;
  const nameObserver = window?.ResizeObserver ? new window.ResizeObserver(scheduleNames) : null;

  function stopNames() {
    for (const entry of activeNames.values()) entry.animation.cancel();
    activeNames.clear();
  }

  function measureNames() {
    nameFrame = null;
    if (destroyed) return;
    const staticNames = !canAnimateNames || reducedMotion?.matches === true;
    root.classList.toggle('static-names', staticNames);
    if (staticNames) {
      stopNames();
      for (const name of names) name.viewport.classList.remove('scrolling');
      return;
    }
    const now = document.timeline?.currentTime;
    if (!number(now)) return;
    for (const name of names) {
      if (!name.viewport.isConnected) continue;
      const width = name.viewport.clientWidth;
      const textWidth = name.label.getBoundingClientRect().width;
      const motion = nameMotion(textWidth, width);
      const active = activeNames.get(name.viewport);
      if (active && Math.abs(active.width - width) < .5 && Math.abs(active.textWidth - textWidth) < .5) continue;
      active?.animation.cancel();
      activeNames.delete(name.viewport);
      name.viewport.classList.toggle('scrolling', !!motion);
      if (!motion) continue;
      let epoch = nameEpochs.get(name.key);
      if (!epoch || Math.abs(epoch.width - width) >= .5 || Math.abs(epoch.textWidth - textWidth) >= .5) epoch = { width, textWidth, startedAt: now };
      // Keep only timing/measurements across refreshes, never old names or actor data.
      nameEpochs.delete(name.key);
      nameEpochs.set(name.key, epoch);
      const animation = name.label.animate(motion.frames, { duration: motion.duration, iterations: Infinity, easing: 'linear' });
      animation.startTime = epoch.startedAt;
      activeNames.set(name.viewport, { animation, width, textWidth });
    }
    while (nameEpochs.size > 12) nameEpochs.delete(nameEpochs.keys().next().value);
  }

  function scheduleNames() {
    if (destroyed || nameFrame !== null || !requestFrame) return;
    nameFrame = requestFrame(measureNames);
  }

  function nameViewport(value, slot, className = '') {
    const fullName = text(value);
    const viewport = el('span', `name-viewport${className ? ` ${className}` : ''}`);
    viewport.title = fullName;
    viewport.setAttribute('aria-label', fullName);
    const label = el('span', 'name-text', fullName);
    label.setAttribute('aria-hidden', 'true');
    viewport.append(label);
    names.push({ viewport, label, key: `${slot}:${nameFingerprint(fullName)}` });
    return viewport;
  }

  root.classList.toggle('static-names', !canAnimateNames || reducedMotion?.matches === true);
  reducedMotion?.addEventListener?.('change', scheduleNames);
  document.fonts?.addEventListener?.('loadingdone', scheduleNames);
  document.fonts?.ready?.then(scheduleNames);

  function portrait(src) {
    const box = el('div', 'portrait');
    box.hidden = true;
    const haze = img('haze', src);
    const figure = img('figure', src);
    // Square avatars use a smaller silhouette, without a character-name exception.
    const classify = () => {
      if (!(figure.naturalWidth > 0) || !(figure.naturalHeight > 0)) return;
      box.classList.toggle('square', figure.naturalWidth / figure.naturalHeight >= 0.9);
      box.hidden = false;
    };
    figure.addEventListener('load', classify);
    if (figure.complete) classify();
    box.append(haze, figure);
    return box;
  }

  function hpRow(hp, focus = false) {
    if (!ratio(hp)) return null;
    const row = el('div', focus ? 'focus-hp' : 'hp-row');
    row.append(el('span', 'hp-value', `${hp.value}/${hp.max}`));
    const bar = el('span', 'hp-bar');
    bar.style.setProperty('--hp-ratio', `${Math.max(0, Math.min(100, hp.value / hp.max * 100))}%`);
    row.append(bar);
    if (number(hp.temp) && hp.temp > 0) {
      const temporary = el('span', 'temp-hp');
      temporary.title = `临时生命 ${hp.temp}`;
      temporary.setAttribute('aria-label', temporary.title);
      temporary.append(hudIcon(document, 'temporary', '临时生命'), el('span', 'temp-value', hp.temp));
      row.append(temporary);
    }
    return row;
  }

  function renderCast(seats) {
    // Reuse only currently mounted seats; removed or denied portraits are not cached.
    const mounted = new Map([...cast.children, ...gmStage.children].map((person) => [
      `${person.classList.contains('gm-seat') ? 'gm' : 'pc'}:${person.dataset.userId}`, person,
    ]));
    const retained = new Set();
    let pcIndex = 0;
    let gmIndex = 0;
    cast.style.setProperty('--pc-count', String(Math.max(1, seats.filter((seat) => seat.role !== 'gm').length)));
    gmStage.hidden = !seats.some((seat) => seat.role === 'gm');
    for (const seat of seats) {
      const role = seat.role === 'gm' ? 'gm' : 'pc';
      const userId = text(seat.userId);
      const key = `${role}:${userId}`;
      const person = mounted.get(key) ?? el('div', 'cast-person');
      mounted.delete(key);
      retained.add(person);
      person.dataset.userId = userId;
      person.classList.toggle('offline', seat.online === false);
      if (role === 'gm') {
        person.classList.add('gm-seat');
        let logo = person.querySelector('.gm-logo');
        if (!logo) {
          logo = el('div', 'gm-logo');
          person.append(logo);
        }
        if (logo.dataset.skin !== currentSkin.id) {
          logo.replaceChildren(img('gm-logo-base', asset(currentSkin.logo)));
          if (currentSkin.title) logo.append(img('gm-logo-title', asset(currentSkin.title)));
          logo.dataset.skin = currentSkin.id;
        }
      } else {
        const src = text(seat.portrait);
        const previous = person.querySelector('.portrait');
        if (!src || previous?.querySelector('.figure')?.getAttribute('src') !== src) {
          previous?.remove();
          if (src) person.prepend(portrait(src));
        }
        person.querySelector('.caption')?.remove();
        const caption = el('div', 'caption');
        const name = nameViewport(seat.name, `cast:${userId}`, 'cast-name');
        caption.append(name);
        const hp = hpRow(seat.hp);
        if (hp) caption.append(hp);
        person.append(caption);
      }
      const stage = role === 'gm' ? gmStage : cast;
      const index = role === 'gm' ? gmIndex++ : pcIndex++;
      if (stage.children[index] !== person) stage.insertBefore(person, stage.children[index] ?? null);
    }
    for (const person of [...cast.children, ...gmStage.children]) if (!retained.has(person)) person.remove();
    setSpeaking([...speaking]);
  }

  function detail(label, value, className = 'detail-row', iconName = null, fullLabel = label) {
    const row = el('div', className);
    if (iconName) row.append(hudIcon(document, iconName, text(fullLabel)));
    row.append(el('span', 'detail-label', text(label)), el('strong', 'detail-value', value));
    row.title = text(fullLabel);
    return row;
  }

  function renderFocus(focus) {
    panel.replaceChildren();
    panel.hidden = mode !== 'combat';
    if (!focus) return;
    const header = el('div', 'focus-header');
    const portraitSlot = el('div', 'focus-portrait-slot');
    if (text(focus.portrait)) {
      const figure = img('focus-portrait', focus.portrait);
      const layout = focus.portraitLayout;
      figure.style.setProperty('--portrait-shift-x', `${number(layout?.x) ? Math.min(75, Math.max(-75, layout.x)) : 0}%`);
      figure.style.setProperty('--portrait-shift-y', `${number(layout?.y) ? Math.min(75, Math.max(-75, layout.y)) : 0}%`);
      figure.style.setProperty('--portrait-scale', String(number(layout?.scale) ? Math.min(3, Math.max(.25, layout.scale)) : 1));
      portraitSlot.append(figure);
    }
    header.append(portraitSlot);
    const name = el('h2', 'focus-name');
    const plaque = el('span', 'focus-name-text');
    plaque.append(nameViewport(focus.name, 'focus'));
    name.append(plaque);
    name.title = text(focus.name);
    header.append(name);
    panel.append(header);
    if (focus.publicOnly === true) return;
    const hp = hpRow(focus.hp, true);
    if (hp) header.append(hp);
    const content = el('div', 'focus-data');
    panel.append(content);
    const conditions = el('div', 'conditions');
    for (const condition of list(focus.conditions)) conditions.append(el('span', 'condition', text(condition.label)));
    if (conditions.childElementCount) content.append(conditions);
    const defense = el('div', 'defense-strip');
    const awareness = el('div', 'awareness-strip');
    for (const [index, stat] of list(focus.stats).entries()) {
      // Only saves/perception are modifiers; AC and speeds are already absolute.
      const modifier = index > 0 && index < 5;
      if (!number(stat.value) && !(modifier && number(stat.dc))) continue;
      const icons = ['shield', 'fortitude', 'reflex', 'will', 'eye'];
      const movements = { '地面': 'boot', land: 'boot', Land: 'boot', '飞行': 'wing', fly: 'wing', Fly: 'wing' };
      const movementKey = text(stat.movementType) || stat.label;
      const movement = Object.hasOwn(movements, movementKey) ? movements[movementKey] : Object.hasOwn(movements, stat.label) ? movements[stat.label] : 'pool';
      const value = modifier ? (number(stat.dc) ? stat.dc : stat.value + 10) : stat.value;
      const modifierLabel = number(stat.value) ? `（修正值 ${stat.value >= 0 ? '+' : ''}${stat.value}）` : '';
      const fullLabel = modifier ? `${text(stat.label)} DC ${value}${modifierLabel}` : stat.label;
      const row = detail('', String(value), 'stat', index < 5 ? icons[index] : movement, fullLabel);
      row.querySelector('.detail-label').remove();
      row.setAttribute('aria-label', modifier ? fullLabel : `${text(fullLabel)} ${value}`);
      (index < 4 ? defense : awareness).append(row);
    }
    if (defense.childElementCount) content.append(defense);
    if (awareness.childElementCount) content.append(awareness);
    const counters = el('div', 'counters');
    for (const counter of list(focus.counters)) {
      if (!number(counter.value)) continue;
      const hero = ['hero', 'hero-points'].includes(counter.id);
      const icon = hero ? 'hero' : counter.id === 'focus' ? 'focus' : 'pool';
      const value = number(counter.max) ? `${counter.value}/${counter.max}` : String(counter.value);
      const description = `${text(counter.label)} ${value}`;
      const row = el('div', 'detail-row counter');
      row.dataset.counterId = text(counter.id);
      row.title = description;
      row.setAttribute('aria-label', description);
      if ((hero || counter.id === 'focus') && Number.isInteger(counter.max) && counter.max > 0 && counter.max <= 5 && Number.isInteger(counter.value) && counter.value >= 0 && counter.value <= counter.max) {
        for (let i = 0; i < counter.max; i++) {
          const glyph = hudIcon(document, icon, text(counter.label));
          glyph.classList.add('resource-glyph', i < counter.value ? 'filled' : 'empty');
          glyph.setAttribute('aria-hidden', 'true');
          row.append(glyph);
        }
      } else {
        row.append(hudIcon(document, icon, text(counter.label)));
        if (!hero && counter.id !== 'focus') row.append(el('span', 'detail-label', text(counter.label)));
        row.append(el('strong', 'detail-value', value));
      }
      counters.append(row);
    }
    if (counters.childElementCount) content.append(counters);
    const equipment = el('div', 'equipment');
    if (focus.classDc && number(focus.classDc.value)) {
      const description = `${text(focus.classDc.label)} ${focus.classDc.value}`;
      const dc = detail('', focus.classDc.value, 'class-dc', 'dc', description);
      dc.querySelector('.detail-label').remove();
      dc.setAttribute('aria-label', description);
      equipment.append(dc);
    }
    if (focus.shield) {
      const shield = focus.shield;
      const parts = [];
      if (shield.raised) parts.push('举盾');
      if (shield.broken) parts.push('破损');
      if (ratio(shield.hp)) parts.push(`盾牌 ${shield.hp.value}/${shield.hp.max}`);
      if (number(shield.hardness)) parts.push(`硬度 ${shield.hardness}`);
      if (parts.length) {
        const chip = el('div', 'shield');
        chip.title = parts.join(' · ');
        chip.setAttribute('aria-label', chip.title);
        if (ratio(shield.hp)) chip.append(hudIcon(document, 'shield', '盾牌'), el('span', 'shield-hp', `${shield.hp.value}/${shield.hp.max}`));
        if (number(shield.hardness)) chip.append(hudIcon(document, 'hardness', '硬度'), el('span', 'shield-hardness', String(shield.hardness)));
        for (const [active, icon, label] of [[shield.raised, 'raised', '举盾'], [shield.broken, 'broken', '破损']]) {
          if (!active) continue;
          const state = hudIcon(document, icon, label);
          state.classList.add('shield-state');
          chip.append(state);
        }
        equipment.append(chip);
      }
    }
    if (equipment.childElementCount) content.append(equipment);
    const sources = el('div', 'spell-sources');
    for (const source of list(focus.spellcasting)) {
      if (!number(source.dc) && list(source.groups).length === 0) continue;
      const section = el('section', 'spell-source');
      section.dataset.sourceId = text(source.id);
      const heading = el('div', 'source-heading');
      const label = el('span', 'source-label', text(source.label));
      label.title = text(source.label);
      heading.append(label);
      if (number(source.dc)) {
        const dc = el('strong', 'source-dc');
        dc.title = `施法 DC ${source.dc}`;
        dc.setAttribute('aria-label', dc.title);
        dc.append(hudIcon(document, 'dc', '施法 DC'), el('span', 'source-dc-value', source.dc));
        heading.append(dc);
      }
      section.append(heading);
      const groups = el('div', 'resource-grid');
      for (const group of resourceGroups(source, resourceDetail)) {
        const cell = el('div', 'resource');
        const isSlot = slotGroup(source, group);
        if (isSlot) cell.dataset.rank = String(group.rank);
        cell.classList.toggle('slot', isSlot);
        cell.classList.toggle('wide', !isSlot && text(group.label).length > 5);
        const prefix = `${text(source.label)} · `;
        const caption = source.kind === 'items' && text(source.label) && text(group.label).startsWith(prefix) ? group.label.slice(prefix.length) : text(group.label);
        const label = el('span', 'resource-label', isSlot ? `${group.rank}环` : caption);
        label.title = text(group.label);
        label.setAttribute('aria-label', text(group.label));
        cell.append(label);
        if (number(group.value)) cell.append(el('strong', 'resource-value', number(group.max) ? `${group.value}/${group.max}` : String(group.value)));
        else if (group.unlimited === true) cell.append(el('strong', 'resource-value unlimited', '无限'));
        groups.append(cell);
      }
      section.append(groups);
      sources.append(section);
    }
    if (sources.childElementCount) content.append(sources);
  }

  function setSpeaking(ids = []) {
    if (destroyed) return;
    speaking = new Set(ids);
    for (const person of [...cast.children, ...gmStage.children]) person.classList.toggle('speaking', speaking.has(person.dataset.userId));
  }

  function render(state = {}) {
    if (destroyed) return;
    stopNames();
    nameObserver?.disconnect();
    names = [];
    currentSkin = resolveSkin(state.skin) ?? resolveSkin(state.worldId) ?? resolveSkin(skin) ?? currentSkin;
    root.dataset.skin = currentSkin.id;
    root.style.setProperty('--focus-material', `url("${materialUrl(currentSkin.material)}")`);
    root.style.setProperty('--focus-material-position', currentSkin.materialPosition);
    root.style.setProperty('--focus-portrait-x', currentSkin.portraitCenter.x);
    root.style.setProperty('--focus-portrait-y', currentSkin.portraitCenter.y);
    root.style.setProperty('--cast-material', `url("${materialUrl(currentSkin.nameplate)}")`);
    root.style.setProperty('--cast-material-size', currentSkin.nameplateSize);
    root.style.setProperty('--cast-material-position', currentSkin.nameplatePosition);
    resourceDetail = state.resourceDetail === 'full' ? 'full' : 'compact';
    root.dataset.resourceDetail = resourceDetail;
    for (const [key, value] of Object.entries(currentSkin.palette)) root.style.setProperty(`--${key}`, value);
    mode = ['explore', 'combat', 'story'].includes(state.mode) ? state.mode : 'explore';
    root.dataset.mode = mode;
    root.style.setProperty('--speech-scale', number(state.scale) ? String(state.scale) : '1.06');
    root.style.setProperty('--portrait-halo', number(state.halo) ? String(state.halo) : '.18');
    frame.src = asset(currentSkin.frame);
    scene.replaceChildren();
    if (mode === 'story' && text(state.story?.src)) scene.append(img('story-image', state.story.src, text(state.story.title)));
    speaking = new Set(state.speakingUserIds ?? []);
    renderCast(list(state.cast));
    renderFocus(state.focus);
    for (const name of names) nameObserver?.observe(name.viewport);
    scheduleNames();
  }

  function sceneBounds() {
    if (destroyed) return null;
    const bounds = root.getBoundingClientRect();
    const rect = sceneRect({ width: bounds.width, height: bounds.height, mode });
    return rect ? { ...rect, left: bounds.left + rect.left, top: bounds.top + rect.top } : null;
  }

  function destroy() {
    destroyed = true;
    speaking.clear();
    stopNames();
    nameObserver?.disconnect();
    if (nameFrame !== null) window?.cancelAnimationFrame?.(nameFrame);
    reducedMotion?.removeEventListener?.('change', scheduleNames);
    document.fonts?.removeEventListener?.('loadingdone', scheduleNames);
    nameEpochs.clear();
    names = [];
    root.remove();
  }
  render();
  return { render, setSpeaking, sceneBounds, destroy };
}
