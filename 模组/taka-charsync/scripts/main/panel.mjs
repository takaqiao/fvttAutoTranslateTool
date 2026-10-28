// 主服 GM 的同步面板、角色目录上的入口按钮、登录时的待处理弹框和定时检查。面板上的操作只在 activeGM 的客户端可用，
// 和定时检查一样经 `exclusive` 排队写卡。
import { OWN_FLAG } from '../lib/rules.mjs';
import { formatClock, formatTime } from '../lib/time.mjs';
import { getCopies, getMeta } from '../foundry/published.mjs';
import {
  applyNew, applyPending, campaignCards, exclusive, ignoreCopy, importNewFromWorking, lastCheck, otherCards,
  previewPending, pullWorking, replacementActors, replaceTarget, runApplyCheck, undoLast, unmappedFamiliars,
} from './apply.mjs';

const TITLE = '角色卡同步';
const CHECK_INTERVAL_MS = 60_000;
const PULL = '用建卡服现在的版本覆盖';

const REASONS = {
  paused: '暂停：主服和建卡服的规则版本不一致，先不套用',
  stale: '谱系已失效：主服这张卡的等级已经变过，这份副本接不上了',
  pred: '前驱版本对不上：副本不是在主服现在这一版上做的',
  'same-level': '同级改动：等级没变，要 GM 确认',
  gap: '跳级副本：一次升不止一级，要 GM 确认',
  session: '有玩家在线或正在战斗，等 GM 套用',
  conflict: '合并有冲突，手动套用时按归属取值',
  error: '套用失败',
  force: '玩家点了「整卡同步」，套用会整卡覆盖',
  new: '建卡服提交的新卡',
};

const escape = text => foundry.utils.escapeHTML(String(text));
const errorMessage = error => (error instanceof Error ? error.message : String(error));
const actionable = item => item.reason !== 'paused';

function notifyError(prefix, error) {
  console.error(`taka-charsync | ${prefix}`, error);
  ui.notifications.error(`${prefix}：${escape(errorMessage(error))}`);
}

function copyLabel(copy) {
  if (copy.kind === 'new') return `新卡「${copy.name}」`;
  if (copy.kind === 'force') return '整卡副本';
  return `${copy.level} 级副本`;
}

function describe(item) {
  const reason = REASONS[item.reason] ?? item.reason;
  const message = item.message ? `（${item.message}）` : '';
  return item.copy ? `${copyLabel(item.copy)}：${reason}${message}` : `${reason}${message}`;
}

/** 检查一次（`runApplyCheck`），再刷新目录按钮上的角标和开着的面板。返回检查结果。 */
export async function checkNow(trigger) {
  const result = await runApplyCheck({trigger});
  refreshBadge();
  if (panel?.rendered) await panel.render();
  return result;
}

const pendingCount = () => (lastCheck()?.pending ?? []).filter(actionable).length;

function refreshBadge() {
  const count = pendingCount();
  for (const badge of document.querySelectorAll('.taka-charsync-panel-button .taka-charsync-badge')) {
    badge.textContent = String(count);
    badge.hidden = count === 0;
  }
}

/** `renderActorDirectory` 钩子：GM 在角色目录顶部的按钮下面多一行「角色卡同步」，角标是待处理的项数。 */
export function addPanelButton(app, html) {
  if (!game.user.isGM) return;
  const actions = html.querySelector('.directory-header .header-actions');
  if (!actions || html.querySelector('.taka-charsync-panel-button')) return;
  const row = document.createElement('div');
  row.className = 'header-actions action-buttons flexrow';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'taka-charsync-panel-button';
  const count = pendingCount();
  button.innerHTML = `<i class="fa-solid fa-arrows-rotate" inert></i><span>${TITLE}</span>`
    + `<span class="taka-charsync-badge"${count ? '' : ' hidden'}>${count}</span>`;
  button.addEventListener('click', () => openPanel());
  row.append(button);
  actions.after(row);
}

/** GM 登录后调一次：马上检查，有要处理的就弹框；之后每 60 秒检查一次，新出现的只加角标。 */
export function startChecks() {
  // 模组启用了、世界还没配同步数据目录：每分钟都会读失败，提示一次就不再检查
  if (!game.settings.get('taka-charsync', 'publishDir')) {
    console.warn('taka-charsync | 本世界没有设置同步数据目录（publishDir），不检查建卡服的副本');
    return;
  }
  checkNow('ready')
    .then(result => (result.pending.some(actionable) ? showPendingDialog(result.pending) : undefined))
    .catch(error => console.error('taka-charsync | 登录时的同步检查失败', error));
  setInterval(() => {
    checkNow('tick').catch(error => console.error('taka-charsync | 定时同步检查失败', error));
  }, CHECK_INTERVAL_MS);
}

// 弹框里「全部套用」会处理的项：谱系失效的只能忽略，报过错的（套用失败、版本读不到）到面板上逐个处理
const appliesInBulk = item => item.copy && item.copy.kind !== 'new' && item.reason !== 'stale' && item.reason !== 'paused' && !item.message;

async function confirmNewCard({name, cardId, copy = null}, replaceId) {
  const meta = await getMeta();
  if (!meta) throw new Error('读不到同步数据，请刷新同步面板');
  const companions = copy
    ? await Promise.all((copy.companions ?? []).map(async entry => {
      const companion = (await getCopies(entry.sourceId)).find(candidate => candidate.copyId === entry.copyId && candidate.rev === entry.rev);
      return companion?.name ?? entry.sourceId;
    }))
    : (await unmappedFamiliars(meta, cardId)).map(entry => entry.working.doc.name);
  const archived = replaceId ? replacementActors(meta, replaceId).map(actor => actor.name) : [];
  const names = values => values.map(value => `「${escape(value)}」`).join('、');
  return {
    window: {title: replaceId ? '替换角色卡' : '新增角色卡'},
    content: `<p>将在本团新建主人「${escape(name)}」${companions.length ? `及同伴${names(companions)}` : ''}，所有者使用卡主在本团的账号。</p>`
      + (archived.length ? `<p>旧卡及同伴${names(archived)}将一起移进「charsync 归档」，保留旧卡内容并停止同步。</p>` : ''),
    yes: {label: replaceId ? '新建并归档旧卡' : '新增整组卡'}, no: {label: '取消'},
  };
}

function confirmUndo(actor) {
  const applied = actor._source.flags?.[OWN_FLAG]?.applied;
  const names = ids => ids.map(id => `「${escape(game.actors.get(id)?.name ?? id)}」`).join('、');
  return {
    window: {title: '恢复到套用前'},
    content: `<p>恢复「${escape(actor.name)}」${applied?.companions?.length ? `及同伴${names(applied.companions)}` : ''}到最后一次套用前，套用后的改动会丢失。</p>`
      + (applied?.createdCompanions?.length ? `<p>那次新建的同伴${names(applied.createdCompanions)}将被删除。</p>` : ''),
    yes: {label: '恢复'}, no: {label: '取消'},
  };
}

async function applyAll(items) {
  const failures = [];
  let done = 0;
  for (const item of items.filter(appliesInBulk)) {
    try {
      // 已经套用过的（'already'）也算处理完了
      await exclusive(() => applyPending(item));
      done++;
    } catch (error) {
      console.error('taka-charsync | 套用失败', error);
      failures.push(errorMessage(error));
    }
  }
  await checkNow('tick');
  if (failures.length) ui.notifications.error(`套用了 ${done} 项，${failures.length} 项失败：${escape(failures.join('；'))}`);
  else ui.notifications.info(`套用了 ${done} 项`);
}

/** 登录时的待处理弹框：全部套用 / 逐张看（打开面板）/ 以后再说。规则版本不一致的暂停项不列。 */
export async function showPendingDialog(pending) {
  const items = pending.filter(actionable);
  if (!items.length) return;
  const meta = await getMeta();
  const lines = items.map(item => {
    if (item.copy?.kind === 'new') {
      const replaceId = meta ? replaceTarget(meta, item.copy) : null;
      const replaced = replaceId ? `，玩家选了替换〈${game.actors.get(replaceId).name}〉` : '';
      return `<li>${escape(`${describe(item)}${replaced}`)}</li>`;
    }
    const actor = game.actors.get(item.actorId);
    return `<li>${escape(`${actor ? `「${actor.name}」` : ''}${describe(item)}`)}</li>`;
  });
  const choice = await foundry.applications.api.DialogV2.wait({
    window: {title: `${TITLE}：有 ${items.length} 项待处理`},
    content: `<p>建卡服提交的这些副本要你处理：</p><ul>${lines.join('')}</ul>`
      + '<p>「全部套用」不处理新卡、谱系已失效和报过错的项，它们留在面板里逐张确认。</p>',
    buttons: [
      {action: 'all', label: '全部套用'},
      {action: 'review', label: '逐张看', default: true},
      {action: 'later', label: '以后再说'},
    ],
  });
  try {
    if (choice === 'all') await applyAll(items);
    else if (choice === 'review') await openPanel();
  } catch (error) {
    notifyError('处理待处理的副本失败', error);
  }
}

let panel = null;

/** 打开（已开着就刷新并提到最前）同步面板。 */
export async function openPanel() {
  panel ??= new SyncPanel();
  if (!lastCheck() && game.users.activeGM?.isSelf) {
    try {
      await checkNow('tick');
    } catch (error) {
      notifyError('同步检查失败', error);
    }
  }
  await panel.render({force: true});
  panel.bringToFront();
  return panel;
}

const {ApplicationV2} = foundry.applications.api;

class SyncPanel extends ApplicationV2 {
  static DEFAULT_OPTIONS = {
    id: 'taka-charsync-panel',
    classes: ['taka-charsync-panel'],
    window: {title: TITLE, icon: 'fa-solid fa-arrows-rotate', resizable: true},
    position: {width: 600, height: 640},
    actions: {
      run(event, target) {
        return this.runAction(target);
      },
    },
  };

  // 本次渲染的按钮，按钮上的 data-index 指向这里
  #handlers = [];
  #busy = false;

  #button(label, handler) {
    const index = this.#handlers.push(handler) - 1;
    const disabled = this.#busy || !game.users.activeGM?.isSelf ? ' disabled' : '';
    return `<button type="button" data-action="run" data-index="${index}"${disabled}>${escape(label)}</button>`;
  }

  async _renderHTML() {
    this.#handlers = [];
    const body = document.createElement('div');
    body.className = 'taka-charsync-panel-body';
    let meta;
    try {
      meta = await getMeta();
    } catch (error) {
      body.innerHTML = `<p>读不到同步数据：${escape(errorMessage(error))}</p>`;
      return body;
    }
    if (!meta) {
      body.innerHTML = '<p>还读不到同步数据（meta.json），引擎可能还没发布过。</p>';
      return body;
    }
    const check = lastCheck();
    const pending = check?.pending ?? [];
    const parts = [];
    if (!game.users.activeGM?.isSelf) {
      const name = game.users.activeGM?.name ?? '另一位 GM';
      parts.push(`<p class="taka-charsync-note">副本由 ${escape(name)} 的客户端套用，这里只能看。</p>`);
    } else if (check) {
      parts.push(`<p class="taka-charsync-note">上次检查：${formatTime(check.at)}</p>`);
    }

    const cards = campaignCards(meta);
    parts.push('<section><h3>本团的卡</h3>');
    if (!cards.length) parts.push('<p>本团还没有对应到建卡服的卡。</p>');
    for (const entry of cards) parts.push(this.#cardRow(meta, entry, check));
    parts.push('</section>');

    const newItems = pending.filter(item => item.reason === 'new');
    if (newItems.length) {
      parts.push('<section><h3>建卡服提交的新卡</h3>');
      for (const item of newItems) parts.push(this.#newCopyRow(meta, item));
      parts.push('</section>');
    }

    const submitted = new Set(newItems.map(item => item.copy.cardId));
    const others = (await otherCards(meta)).filter(card => !submitted.has(card.cardId));
    if (others.length) {
      parts.push('<section><h3>建卡服上本团玩家的其他卡</h3>');
      for (const card of others) parts.push(this.#otherRow(meta, card, cards));
      parts.push('</section>');
    }
    body.innerHTML = parts.join('');
    return body;
  }

  _replaceHTML(result, content) {
    content.replaceChildren(result);
  }

  #cardRow(meta, {cardId, card, actor}, check) {
    const source = actor._source;
    const level = source.system.details?.level?.value ?? '?';
    const xp = source.system.details?.xp;
    const {applied} = source.flags?.[OWN_FLAG] ?? {};
    const items = (check?.pending ?? []).filter(item => item.actorId === actor.id);
    const waiting = (check?.waiting ?? []).find(item => item.actorId === actor.id);

    const status = [];
    if (applied) status.push(`最近一次套用：${formatTime(applied.at)}`);
    if (waiting) status.push(`${copyLabel(waiting.copy)}等经验满`);
    if (!items.length && !waiting) status.push('没有待处理的副本');
    const companions = (card.companions ?? []).map(id => meta.cards[id]?.mainId && game.actors.get(meta.cards[id].mainId)?.name)
      .filter(Boolean);

    const entries = items.map(item => {
      const buttons = [];
      if (item.reason === 'stale') {
        buttons.push(this.#button('忽略此副本', {run: () => ignoreCopy(actor, item.copy)}));
      } else if (item.copy) {
        buttons.push(this.#button('套用', {run: () => applyPending(item)}));
        buttons.push(this.#button('查看变动', {preview: item}));
      }
      return `<li class="taka-charsync-item" data-reason="${escape(item.reason)}"><span>${escape(describe(item))}</span>`
        + `<span class="taka-charsync-actions">${buttons.join('')}</span></li>`;
    });

    const actions = [];
    if (applied?.undoPath) actions.push(this.#button('恢复到套用前', {confirm: () => confirmUndo(actor), run: () => undoLast(actor)}));
    const working = meta.working?.[cardId];
    if (working) {
      actions.push(this.#button(`${PULL}（读取于 ${formatClock(working.readAt)}）`, {
        confirm: {
          window: {title: PULL},
          content: `<p>把「${escape(actor.name)}」整张换成建卡服现在的版本（同伴一起）：主服上没有同步回建卡服的改动都会丢掉，`
            + '经验保持主服现在的值，血量照建卡服。覆盖后可以在面板上「恢复到套用前」。</p>',
          yes: {label: '覆盖'},
          no: {label: '取消'},
        },
        run: () => pullWorking(actor),
      }));
    }

    return `<div class="taka-charsync-card" data-actor-id="${actor.id}">`
      + `<header><strong>${escape(actor.name)}</strong><span>等级 ${escape(level)}`
      + `${xp ? ` · 经验 ${escape(xp.value)}/${escape(xp.max)}` : ''}</span></header>`
      + `<p class="taka-charsync-status">${escape(status.join('；'))}</p>`
      + (companions.length ? `<p class="taka-charsync-status">同伴：${escape(companions.join('、'))}</p>` : '')
      + (entries.length ? `<ul class="taka-charsync-pending">${entries.join('')}</ul>` : '')
      + (actions.length ? `<div class="taka-charsync-actions">${actions.join('')}</div>` : '')
      + '</div>';
  }

  #newCopyRow(meta, item) {
    const {copy} = item;
    const confirmation = replaceId => () => confirmNewCard({name: copy.name, cardId: copy.cardId, copy}, replaceId);
    const buttons = [this.#button('新增', {confirm: confirmation(null), run: () => applyNew(copy, {replaceId: null, meta})})];
    const replaceId = replaceTarget(meta, copy);
    if (replaceId) {
      const name = game.actors.get(replaceId).name;
      buttons.push(this.#button(`替换〈${name}〉`, {confirm: confirmation(replaceId), run: () => applyNew(copy, {replaceId, meta})}));
    }
    buttons.push(this.#button('查看变动', {preview: item}));
    return `<div class="taka-charsync-card" data-card-id="${escape(copy.cardId)}">`
      + `<header><strong>${escape(copy.name)}</strong><span>等级 ${escape(copy.level)}</span></header>`
      + `<p class="taka-charsync-status">${escape(REASONS.new)}</p>`
      + `<div class="taka-charsync-actions">${buttons.join('')}</div></div>`;
  }

  #otherRow(meta, card, campaign) {
    const confirmation = replaceId => () => confirmNewCard(card, replaceId);
    const buttons = [this.#button('新增', {confirm: confirmation(null), run: () => importNewFromWorking(card.cardId, {replaceId: null})})];
    for (const {card: mapped, actor} of campaign) {
      if (mapped.charOwner !== card.owner) continue;
      buttons.push(this.#button(`替换〈${actor.name}〉`, {confirm: confirmation(actor.id), run: () => importNewFromWorking(card.cardId, {replaceId: actor.id})}));
    }
    return `<div class="taka-charsync-card taka-charsync-other" data-card-id="${escape(card.cardId)}">`
      + `<header><strong>${escape(card.name)}</strong><span>等级 ${escape(card.level ?? '?')}</span></header>`
      + `<p class="taka-charsync-status">建卡服读取于 ${formatTime(card.readAt)}，还没提交过</p>`
      + `<div class="taka-charsync-actions">${buttons.join('')}</div></div>`;
  }

  async runAction(target) {
    const handler = this.#handlers[Number(target.dataset.index)];
    if (!handler || this.#busy) return;
    if (handler.preview) return showPreview(handler.preview);
    this.#busy = true;
    for (const button of this.element.querySelectorAll('button[data-action="run"]')) button.disabled = true;
    try {
      if (handler.confirm) {
        const confirmation = typeof handler.confirm === 'function' ? await handler.confirm() : handler.confirm;
        if (!(await foundry.applications.api.DialogV2.confirm(confirmation))) return;
      }
      if ((await exclusive(handler.run)) === 'already') ui.notifications.info('这份副本已经套用过了');
      await checkNow('tick');
    } catch (error) {
      notifyError('同步操作失败', error);
    } finally {
      this.#busy = false;
      if (this.rendered) await this.render();
    }
  }
}

async function showPreview(item) {
  try {
    const html = await previewPending(item);
    await foundry.applications.api.DialogV2.prompt({
      window: {title: `变动：${copyLabel(item.copy)}`},
      content: html || '<p>没有变动。</p>',
      ok: {label: '关闭'},
    });
  } catch (error) {
    notifyError('读不出这份副本的变动', error);
  }
}
