import { normalizePortraitLayout, resolvePortraitLayout } from './portrait-layout.mjs';
import { resolveSkin } from './skins.mjs';

const MODULE_ID = 'taka-obs-director';
const text = value => typeof value === 'string' ? value : '';
function ownValue(object, key) {
  if (!object || typeof object !== 'object' || Array.isArray(object)) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  return descriptor && Object.hasOwn(descriptor, 'value') ? descriptor.value : undefined;
}
function imageSource(value) {
  if (typeof value !== 'string' || !value || /[\u0000-\u001f]/.test(value)) return '';
  if (/^[a-z][a-z0-9+.-]*:/i.test(value) && !/^https?:\/\//i.test(value)
    && !/^data:image\/(?:png|webp|jpeg|gif);base64,[A-Za-z0-9+/=]+$/.test(value)) return '';
  return value;
}
function sourceFor(actor, overrides) {
  return imageSource(ownValue(overrides, actor.id)) || imageSource(actor.img);
}
function seatIds(game) {
  const order = game.settings.get(MODULE_ID, 'seatOrder');
  return new Set((Array.isArray(order) ? order : []).flatMap(value => text(value).split(',')).map(value => value.trim()).filter(Boolean));
}
function userById(game, id) {
  return game.users?.get?.(id) ?? game.users?.contents?.find(user => user.id === id);
}

/** Keep drafts separate from world settings until the GM explicitly saves. */
export function createPortraitController({ game, document, defaults } = {}) {
  const userId = game.user?.id, worldId = game.world?.id;
  const entries = new Map(), drafts = new Map(), edited = new Set(), listeners = [];
  let mounted = null, selected = '', disposed = false, saving = false, drag = null;
  const active = () => !disposed && game.user?.isGM === true && !!userId && game.user.id === userId
    && !!worldId && game.world?.id === worldId;
  const readSetting = (key, fallback) => {
    try { return game.settings.get(MODULE_ID, key) ?? fallback; } catch { return fallback; }
  };
  const content = document.createElement('div');
  function node(tag, className, label) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (label !== undefined) result.textContent = label;
    return result;
  }
  if (active()) {
    const overrides = readSetting('portraitOverrides', {}), layouts = readSetting('portraitLayouts', {});
    try {
      for (const id of seatIds(game)) {
        const user = userById(game, id), actor = user?.character;
        if (user?.isGM !== false || !actor?.id || entries.has(actor.id)) continue;
        const source = sourceFor(actor, overrides);
        if (!source) continue;
        entries.set(actor.id, { userId: id, actor, source, name: `${text(user.name)} · ${text(actor.name)}` });
        drafts.set(actor.id, resolvePortraitLayout({ worldId, actorId: actor.id, source, overrides: layouts, defaults }));
      }
    } catch { entries.clear(); drafts.clear(); }
  }
  const panel = node('div', 'portrait-calibration');
  content.append(panel);
  if (entries.size) {
    const label = node('label', 'portrait-calibration-character', '角色');
    const select = node('select');
    select.name = 'actorId';
    for (const [id, entry] of entries) {
      const option = node('option', '', entry.name);
      option.value = id;
      select.append(option);
    }
    selected = entries.keys().next().value;
    label.append(select);
    panel.append(label);
    const preview = node('div', 'portrait-calibration-preview');
    const skin = resolveSkin(readSetting('skin', 'auto')) ?? resolveSkin(worldId) ?? resolveSkin('cotct');
    const center = skin.portraitCenter ?? { x: '50%', y: '19.62%' };
    try {
      const material = new URL(`modules/${MODULE_ID}/assets/${skin.material}`, document.baseURI || document.URL);
      preview.style.setProperty('--portrait-material', `url("${material.href}")`);
    } catch { /* A document without a base URL can still edit the original image. */ }
    preview.style.setProperty('--portrait-material-position', skin.materialPosition);
    preview.style.setProperty('--portrait-center-x', center.x);
    preview.style.setProperty('--portrait-center-y', `${Number.parseFloat(center.y) * 841 / 330}%`);
    preview.append(node('div', 'portrait-calibration-material'));
    const slot = node('div', 'portrait-calibration-slot'), image = node('img', 'portrait-calibration-image');
    image.draggable = false;
    slot.append(image);
    preview.append(slot);
    panel.append(preview, node('p', 'portrait-calibration-hint', '拖动原图或调整滑块。仅更改录像角色槽的位置与大小。'));
    const controls = node('div', 'portrait-calibration-controls');
    for (const [name, caption, min, max, step] of [
      ['x', '水平位置', -75, 75, 0.001], ['y', '垂直位置', -75, 75, 0.001], ['scale', '大小', 0.25, 3, 0.001],
    ]) {
      const field = node('label', 'portrait-calibration-field'), input = node('input'), output = node('output');
      input.type = 'range'; input.name = name;
      input.setAttribute('min', String(min)); input.setAttribute('max', String(max)); input.setAttribute('step', String(step));
      output.setAttribute('data-value', name);
      field.append(node('span', '', caption), input, output);
      controls.append(field);
    }
    const reset = node('button', '', '恢复预设');
    reset.type = 'button'; reset.setAttribute('data-action', 'restore');
    panel.append(controls, reset);
  }
  const message = node('p', 'portrait-calibration-status', entries.size ? '' : '当前没有可校准的玩家原图。请由 GM 在对应世界打开。');
  message.setAttribute('role', 'status');
  panel.append(message);

  function status(value) {
    const target = mounted?.querySelector('.portrait-calibration-status');
    if (target) target.textContent = value;
  }
  function reflect() {
    const entry = entries.get(selected), layout = drafts.get(selected);
    if (!mounted || !entry || !layout) return;
    const select = mounted.querySelector('[name="actorId"]');
    if (select) for (const option of select.options) option.selected = option.value === selected;
    const image = mounted.querySelector('.portrait-calibration-image'), slot = mounted.querySelector('.portrait-calibration-slot');
    image.setAttribute('src', entry.source);
    image.alt = entry.name;
    slot.style.setProperty('--portrait-layout-x', `${layout.x}%`);
    slot.style.setProperty('--portrait-layout-y', `${layout.y}%`);
    slot.style.setProperty('--portrait-layout-scale', String(layout.scale));
    for (const name of ['x', 'y', 'scale']) {
      mounted.querySelector(`[name="${name}"]`).value = String(layout[name]);
      mounted.querySelector(`[data-value="${name}"]`).textContent = name === 'scale' ? `${Math.round(layout.scale * 100)}%` : `${layout[name]}%`;
    }
  }
  function update(values) {
    if (!active() || saving || !entries.has(selected)) return;
    const source = entries.get(selected).source;
    const next = normalizePortraitLayout({ source, ...values }, { source }), previous = drafts.get(selected);
    if (['x', 'y', 'scale'].some(key => next[key] !== previous[key])) {
      drafts.set(selected, next);
      edited.add(selected);
    }
    reflect(); status('');
  }
  function readInputs(root = mounted) {
    if (!root || !entries.has(selected)) return;
    const values = {};
    for (const name of ['x', 'y', 'scale']) {
      const input = root.querySelector(`[name="${name}"]`);
      if (!input) return;
      values[name] = Number(input.value);
    }
    update(values);
  }
  function unbind() {
    for (const [element, event, handler] of listeners.splice(0)) element.removeEventListener(event, handler);
    drag = null;
  }
  function bind(element) {
    unbind();
    mounted = element?.querySelector('.portrait-calibration') ?? null;
    if (!mounted || !entries.size) return;
    const on = (target, event, handler) => { target.addEventListener(event, handler); listeners.push([target, event, handler]); };
    on(mounted.querySelector('[name="actorId"]'), 'change', event => {
      if (!entries.has(event.target.value)) return;
      selected = event.target.value; reflect(); status('');
    });
    for (const name of ['x', 'y', 'scale']) on(mounted.querySelector(`[name="${name}"]`), 'input', () => readInputs());
    on(mounted.querySelector('[data-action="restore"]'), 'click', event => {
      event.preventDefault();
      if (!active() || saving) return;
      update(resolvePortraitLayout({ worldId, actorId: selected, source: entries.get(selected).source, defaults }));
    });
    const slot = mounted.querySelector('.portrait-calibration-slot');
    on(slot, 'pointerdown', event => {
      if (!active() || saving || event.button !== 0 || !Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return;
      const bounds = slot.getBoundingClientRect();
      if (!(bounds.width > 0 && bounds.height > 0)) return;
      event.preventDefault();
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY, width: bounds.width, height: bounds.height, layout: { ...drafts.get(selected) } };
      slot.setPointerCapture?.(event.pointerId);
    });
    on(slot, 'pointermove', event => {
      if (!drag || event.pointerId !== drag.id) return;
      update({ ...drag.layout, x: drag.layout.x + (event.clientX - drag.x) / drag.width * 100, y: drag.layout.y + (event.clientY - drag.y) / drag.height * 100 });
    });
    const endDrag = event => {
      if (drag && event.pointerId === drag.id) { slot.releasePointerCapture?.(drag.id); drag = null; }
    };
    on(slot, 'pointerup', endDrag); on(slot, 'pointercancel', endDrag);
    reflect();
  }
  async function save(form) {
    if (!active() || saving) { status('账号或世界已改变，请重新打开校准窗口。'); return false; }
    if (edited.has(selected)) readInputs(form);
    if (!edited.size) return true;
    try {
      const ids = seatIds(game), overrides = game.settings.get(MODULE_ID, 'portraitOverrides');
      for (const id of edited) {
        const entry = entries.get(id), user = userById(game, entry.userId);
        if (!ids.has(entry.userId) || user?.isGM !== false || user.character !== entry.actor
          || sourceFor(user.character, overrides) !== entry.source) {
          status('角色绑定或原图已改变，请重新打开校准窗口。'); return false;
        }
      }
      const latest = game.settings.get(MODULE_ID, 'portraitLayouts');
      if (!latest || typeof latest !== 'object' || Array.isArray(latest)) throw new TypeError('Invalid portrait layouts');
      const values = Object.entries(Object.getOwnPropertyDescriptors(latest));
      if (values.some(([, descriptor]) => !Object.hasOwn(descriptor, 'value'))) throw new TypeError('Invalid portrait layouts');
      const merged = Object.fromEntries(values.map(([key, descriptor]) => [key, descriptor.value]));
      for (const id of edited) Object.defineProperty(merged, id, {
        value: { source: entries.get(id).source, ...drafts.get(id) }, enumerable: true, configurable: true, writable: true,
      });
      if (!active()) return false;
      saving = true;
      await game.settings.set(MODULE_ID, 'portraitLayouts', merged);
      edited.clear(); status('已保存。');
      return true;
    } catch {
      status('保存失败，请重新打开窗口后重试。');
      return false;
    } finally { saving = false; }
  }
  function dispose() { disposed = true; unbind(); mounted = null; drafts.clear(); edited.clear(); }
  return { content, bind, save, dispose, canEdit: active };
}

export function createPortraitMenu({ game, document, DialogV2, defaults } = {}) {
  if (typeof DialogV2 !== 'function' || typeof document?.createElement !== 'function') return null;
  return class PortraitCalibrationMenu extends DialogV2 {
    constructor() {
      const controller = createPortraitController({ game, document, defaults });
      super({
        window: { title: '录像角色原图校准' }, position: { width: 520 }, form: { closeOnSubmit: false }, content: controller.content,
        buttons: [
          { action: 'save', label: '保存', default: true, disabled: !controller.canEdit(), callback: async (_event, button, dialog) => {
            const saved = await controller.save(button.form);
            if (saved) await dialog.close();
            return saved;
          } },
          { action: 'cancel', label: '取消', callback: async (_event, _button, dialog) => { await dialog.close(); return false; } },
        ],
      });
      this.addEventListener('render', () => controller.bind(this.element));
      this.addEventListener('close', () => controller.dispose(), { once: true });
    }
  };
}
