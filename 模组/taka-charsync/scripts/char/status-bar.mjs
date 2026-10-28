// 建卡服：角色卡、魔宠卡内容顶部的同步状态条。卡面每次打开时先让卡跟上主服，开着期间不再跟上。
import { normalizeCopy, refKey, syncAllowed } from '../lib/chain.mjs';
import { describeChanges } from '../lib/change-summary.mjs';
import { projectBuildChanges } from '../lib/build-sync.mjs';
import { formatSummary } from '../lib/merge.mjs';
import { OWN_FLAG } from '../lib/rules.mjs';
import { formatTime } from '../lib/time.mjs';
import { cleanDoc } from '../foundry/import.mjs';
import { getMainLatest } from '../foundry/published.mjs';
import {
  catchUpWithNotice, discardWithNotice, levelOf, masterIdOf, notifyFailure, onCharSide, ownsCard,
} from './catch-up.mjs';
import { actorRuleState, selfCheck } from './rules-override.mjs';
import { loadRuleMeta, ruleProblem } from './rules-session.mjs';
import {
  canWithdraw, companionsOf, copyActorsOf, copyOf, newCardWorlds, ownsNewCard, replaceableCards, submitForce, submitLevel, submitNew,
  withdraw,
} from './submit.mjs';

const DISCARD = '放弃草稿，恢复成主服版';
const FORCE = '整卡同步到主服';
const NEW_CARD = '作为新卡提交';
const WITHDRAW = '撤回';
const submitLabel = (level, mainLevel) => level === mainLevel ? '提交本级修改' : `提交 ${level} 级升级`;
const NO_CHANGES = {level: null, added: [], removed: [], changed: [], skillRanks: []};

// 与普通套用使用同一范围，库存及消耗差异不算待提交的构筑。
const draftChanges = (mainDoc, cardDoc) => describeChanges(mainDoc, projectBuildChanges(mainDoc, cardDoc));

// 卡面开着时每次重绘都要比一次，主服那份清洗过的文档按发布条目缓存
const cleanedMainDocs = new WeakMap();

function cleanedMainDoc(actor, card, mainLatest) {
  if (!cleanedMainDocs.has(mainLatest)) {
    cleanedMainDocs.set(mainLatest, cleanDoc(onCharSide(mainLatest.doc, masterIdOf(actor, card))));
  }
  return cleanedMainDocs.get(mainLatest);
}

// 卡主在已对应的卡上能点的：有草稿时提交当前等级（不低于主服的）和放弃草稿，整卡同步一直有。
// 同伴随主人提交，只能放弃草稿
function cardActions(actor, card, mainLatest, {draft}) {
  if (card.companionOf) return draft ? [DISCARD] : [];
  const level = levelOf(actor._source);
  const actions = [];
  if (draft && level >= mainLatest.level) actions.push(submitLabel(level, mainLatest.level));
  if (draft) actions.push(DISCARD);
  actions.push(FORCE);
  return actions;
}

/**
 * 卡的同步状态。`meta`、`mainLatest` 是发布目录里的 `meta.json` 和主服最新版，`copies` 是这张卡在建卡服上的副本记号。
 * `text` 是一句话；`details` 是已转义的变动或冲突 HTML，`summary` 是当前草稿的单行摘要。
 * `pending` 按副本 id 和 rev 排除已处理的提交；`editedAfterSubmission` 为 null 时尚不能比较冻结副本。
 * `actions` 是要给的按钮名，只给卡主（副本只给提交者，还没对应的新卡看 ownership）：按钮会建、改副本或改卡，
 * GM 和别的玩家只看状态。卡上的 `base` 不是主服最新版（还没跟上）时也算 `paused`。
 *
 * @returns {{state: 'synced'|'draft'|'submitted'|'conflict'|'paused'|'unmapped'|'copy'|'backup',
 *   text: string, details: string, summary: string, draft: boolean, actions: string[],
 *   pending: object[], relations: object[], editedAfterSubmission: boolean|null}}
 */
export function cardStatus(actor, {meta, copies = [], mainLatest, companions = []}) {
  const own = actor._source.flags?.[OWN_FLAG] ?? {};
  const card = meta?.cards?.[actor.id];
  const masterId = own.copy?.companionOf ?? card?.companionOf
    ?? (actor.type === 'familiar' ? actor._source.system.master?.id : null);
  const relation = (actorId, role) => ({
    actorId, role, name: game.actors.get(actorId)?.name ?? actorId,
    automatic: role === 'master' ? actor.type === 'familiar' : game.actors.get(actorId)?.type === 'familiar',
  });
  const relations = masterId ? [relation(masterId, 'master')]
    : companions.map(({actor: companion}) => relation(companion.id, 'companion'));
  const pending = pendingCopies(copies, mainLatest, Boolean(card));
  const editedAfterSubmission = own.copy ? null : changedSinceSubmission(actor, pending, companions, masterId);
  const status = (state, text, {actions = [], details = '', summary = '', draft = false,
    owner = ownsCard(actor, meta)} = {}) => ({
    state, text, details, summary, draft, pending, relations, editedAfterSubmission, actions: owner ? actions : [],
  });
  if (own.copy) {
    const {copy} = own;
    const processed = !pendingCopies([copy], mainLatest, false).length;
    const state = copy.withdrawn ? '已撤回，等待引擎发布' : processed ? '主服已处理' : pendingText([copy], mainLatest);
    return status('copy', `${copyLabel(copy, mainLatest)}的冻结副本：${state}；要修改请打开原卡`, {
      actions: processed ? [] : [WITHDRAW], owner: canWithdraw(copy),
    });
  }
  if (own.backupOf) {
    return status('backup', `这是「${game.actors.get(own.backupOf)?.name ?? own.backupOf}」的草稿备份，不参与同步`);
  }
  if (!card) {
    return status(pending.length ? 'submitted' : 'unmapped', pending.length ? pendingText(pending, mainLatest)
      : masterId ? '这张同伴卡随主人一起提交' : '这张卡还没有对应的主服卡', {
      actions: !masterId && newCardWorlds(meta).length ? [NEW_CARD] : [], owner: ownsNewCard(actor, meta),
    });
  }
  if (!syncAllowed(meta, card.world)) return status('paused', '主服和建卡服的规则版本不一致，同步暂停');
  if (!mainLatest) return status('paused', '还没读到这张卡的主服版本，同步暂停');
  if (own.base?.kind !== 'main' || own.base.hash !== mainLatest.hash) {
    return status('paused', `还没跟上主服 ${formatTime(mainLatest.readAt)} 的版本`);
  }
  const changes = draftChanges(cleanedMainDoc(actor, card, mainLatest), actor.toObject());
  const companionChanges = companions.flatMap(({actor: companion, mainLatest: latest}) => {
    if (!latest) return [{name: companion.name, html: '<p>尚未同步到主服，将随主人提交。</p>', text: '新同伴'}];
    const change = draftChanges(cleanedMainDoc(companion, meta.cards[companion.id], latest), companion.toObject());
    return change.changed ? [{name: companion.name, ...change}] : [];
  });
  const draft = changes.changed || companionChanges.length > 0;
  const details = changes.html + companionChanges.map(change =>
    `<p>${foundry.utils.escapeHTML(change.name)}：</p>${change.html}`).join('');
  const summary = [changes.changed ? changes.text : '', ...companionChanges.map(change => `${change.name}：${change.text}`)]
    .filter(Boolean).join('；');
  const conflicts = own.conflicts?.list ?? [];
  const draftActions = cardActions(actor, card, mainLatest, {draft: draft || conflicts.length > 0});
  if (conflicts.length) {
    return status('conflict', `冲突：上次跟上主服时有 ${conflicts.length} 处两边都改了`, {
      actions: draftActions, details: formatSummary(NO_CHANGES, conflicts) + details, summary, draft,
    });
  }
  if (pending.length) {
    return status('submitted', pendingText(pending, mainLatest), {actions: draftActions, details, summary, draft});
  }
  if (!draft) {
    return status('synced', '构筑和外观已和主服一致', {actions: cardActions(actor, card, mainLatest, {draft: false})});
  }
  return status('draft', '当前草稿有改动', {actions: draftActions, details, summary, draft});
}

function pendingCopies(copies, mainLatest, mapped) {
  const applied = refKey(mainLatest?.applied);
  const undone = new Set(mainLatest?.doc.flags?.[OWN_FLAG]?.undone ?? []);
  return copies.filter(copy => {
    const key = refKey({kind: 'copy', copyId: copy.copyId, rev: copy.rev});
    return !copy.withdrawn && key !== applied && !undone.has(key)
      && !(copy.kind === 'new' && mapped)
      && !(copy.kind === 'level' && mainLatest?.level != null && copy.level < mainLatest.level);
  }).sort((a, b) => a.level - b.level || a.rev - b.rev);
}

function copyLabel(copy, mainLatest) {
  if (copy.kind === 'force') return '整卡同步';
  if (copy.kind === 'new') return `${copy.level} 级新卡`;
  return copy.level === mainLatest?.level ? '本级修改' : `${copy.level} 级升级`;
}

function pendingText(copies, mainLatest) {
  const gm = copies.filter(copy => copy.kind !== 'level' || copy.level <= (mainLatest?.level ?? copy.level));
  const upgrades = copies.filter(copy => !gm.includes(copy));
  return [gm.length ? `已提交${gm.map(copy => copyLabel(copy, mainLatest)).join('、')}，待 GM 确认` : '',
    upgrades.length ? `已提交 ${upgrades.map(copy => copy.level).join('、')} 级，待主服升级` : ''].filter(Boolean).join('；');
}

function changedSinceSubmission(actor, pending, companions, masterId) {
  if (!pending.length) return null;
  const latest = pending.reduce((a, b) => a.rev > b.rev ? a : b);
  const frozen = game.actors.get(latest.copyId);
  if (!frozen || copyOf(frozen)?.rev !== latest.rev) return null;
  const changed = (working, submitted, master) => {
    const before = cleanDoc(normalizeCopy(submitted._source, {actorId: working.id, masterId: master}));
    const after = working.toObject();
    return describeChanges(before, latest.kind === 'level' ? projectBuildChanges(before, after) : after).changed;
  };
  if (changed(actor, frozen, masterId)) return true;
  for (const {actor: companion} of companions) {
    const ref = latest.companions.find(ref => ref.sourceId === companion.id);
    if (!ref) return true;
    const submitted = game.actors.get(ref.copyId);
    if (!submitted || copyOf(submitted)?.rev !== ref.rev) return null;
    if (changed(companion, submitted, actor.id)) return true;
  }
  return false;
}

const sessions = new WeakMap();

function paragraph(text, className) {
  const element = document.createElement('p');
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function button(label, onClick) {
  const element = document.createElement('button');
  element.type = 'button';
  element.textContent = label;
  element.addEventListener('click', () => onClick(element));
  return element;
}

function campaignLine(actor, meta) {
  const state = actorRuleState(actor);
  const title = meta?.worlds?.[state.world]?.title ?? state.world;
  return paragraph(state.ok ? `按${title}团规` : `暂不可编辑构筑或提交：${ruleProblem(state.reason)}`,
    state.ok ? 'taka-charsync-campaign' : 'taka-charsync-warning');
}

function disclosure(app, key, label, className) {
  const details = document.createElement('details');
  details.className = className;
  const session = sessions.get(app);
  details.open = Boolean(session?.expanded?.[key]);
  const summary = document.createElement('summary');
  summary.textContent = label;
  details.append(summary);
  details.addEventListener('toggle', () => {
    if (sessions.get(app) === session) session.expanded[key] = details.open;
  });
  return details;
}

function refreshButton(app) {
  return button('刷新状态', async element => {
    element.disabled = true;
    const session = sessions.get(app);
    if (session) await loadStatus(app, session, {catchUp: false});
  });
}

function appendRelations(bar, relations) {
  for (const relation of relations) {
    const master = relation.role === 'master';
    const line = paragraph(`${master ? '主人' : '同伴'}：${relation.name}（${relation.automatic ? '魔宠自动随主人提交' : '已配置为随主人提交'}）`,
      'taka-charsync-relation');
    const actor = game.actors.get(relation.actorId);
    const open = button(master ? '打开主人卡' : '打开同伴卡', () => actor.sheet.render(true));
    open.disabled = !actor;
    line.append(open);
    bar.append(line);
  }
}

function appendPending(bar, app, status, mainLatest) {
  if (!status.pending.length || copyOf(app.actor)) return;
  const details = disclosure(app, 'submissions', `提交记录（${status.pending.length} 份待处理）`, 'taka-charsync-submissions');
  details.append(paragraph('工作卡可以继续编辑；修改后要重新提交，才会更新交给主服的冻结副本。'));
  for (const copy of status.pending) {
    const row = document.createElement('div');
    row.className = 'taka-charsync-submission';
    const label = paragraph(copyLabel(copy, mainLatest));
    const copyActor = game.actors.get(copy.copyId);
    const open = button('查看提交', () => copyActor.sheet.render(true));
    open.disabled = !copyActor;
    row.append(label, open);
    if (copyActor && canWithdraw(copy)) row.append(button(WITHDRAW, element => confirmWithdraw(app, element, copyActor, copy)));
    details.append(row);
  }
  bar.append(details);
}

function buildBar(app, data) {
  const bar = document.createElement('section');
  bar.className = 'taka-charsync-bar';
  if (!data) {
    bar.dataset.state = 'loading';
    bar.append(paragraph('正在核对主服版本……'));
    return bar;
  }
  if (data.error) {
    bar.dataset.state = 'error';
    bar.append(paragraph(`读不到同步数据：${data.error.message ?? data.error}`));
    bar.append(refreshButton(app));
    return bar;
  }
  // 副本取建卡服本地的：刚提交的要等引擎下一轮才进发布目录
  const status = cardStatus(app.actor, {...data, copies: copyActorsOf(app.actor.id).map(copyOf)});
  bar.dataset.state = status.state;
  const header = document.createElement('div');
  header.className = 'taka-charsync-header';
  header.append(paragraph(status.text), refreshButton(app));
  bar.append(header);
  if (status.editedAfterSubmission !== null) {
    const latest = status.pending.reduce((a, b) => a.rev > b.rev ? a : b);
    const scope = latest.kind === 'level' ? '构筑或外观' : '工作卡';
    bar.append(paragraph(status.editedAfterSubmission
      ? `提交后${scope}又有修改，需重新提交才会交给主服。`
      : `当前${latest.kind === 'level' ? '构筑和外观' : '工作卡'}与最近提交的冻结副本一致。`, 'taka-charsync-submission-note'));
  }
  if (status.details) {
    if (status.state === 'conflict') bar.insertAdjacentHTML('beforeend', status.details);
    else {
      const details = disclosure(app, 'changes', `当前草稿：${status.summary || '查看改动'}`, 'taka-charsync-details');
      details.insertAdjacentHTML('beforeend', status.details);
      bar.append(details);
    }
  }
  appendRelations(bar, status.relations);
  appendPending(bar, app, status, data.mainLatest);
  const frozen = copyOf(app.actor);
  if (frozen) {
    const original = game.actors.get(frozen.cardId);
    const open = button('打开工作卡', () => original.sheet.render(true));
    open.disabled = !original;
    bar.append(open);
  }
  const name = foundry.utils.escapeHTML(app.actor.name);
  const submit = submitLabel(levelOf(app.actor._source), data.mainLatest?.level);
  const handlers = {
    [DISCARD]: element => confirmDiscard(app, element),
    [submit]: element => runAction(app, element, '提交', submitLevel,
      copy => `已提交${name}的${copyLabel(copy, data.mainLatest)}${copy.level === data.mainLatest?.level ? '，待 GM 确认' : ''}`),
    [FORCE]: element => confirmForce(app, element),
    [NEW_CARD]: element => promptNewCard(app, element, data.meta),
    [WITHDRAW]: element => confirmWithdraw(app, element),
  };
  if (status.actions.length) {
    const actions = document.createElement('div');
    actions.className = 'taka-charsync-actions';
    const secondary = disclosure(app, 'secondary', '其他操作', 'taka-charsync-secondary');
    const secondaryActions = document.createElement('div');
    secondaryActions.className = 'taka-charsync-actions';
    for (const label of status.actions) {
      const action = button(label, handlers[label]);
      if (label === submit) action.title = '提交构筑及头像、默认 Token 外观；库存、金币和资源消耗以主服为准';
      if ([FORCE, NEW_CARD, submit].includes(label)) {
        action.disabled = !actorRuleState(app.actor).ok || selfCheck(app.actor).length > 0;
      }
      if ([DISCARD, FORCE].includes(label)) secondaryActions.append(action);
      else actions.append(action);
    }
    if (actions.children.length) bar.append(actions);
    if (secondaryActions.children.length) {
      secondary.append(secondaryActions);
      bar.append(secondary);
    }
  }
  const campaign = campaignLine(app.actor, data.meta);
  if (campaign) bar.append(campaign);
  return bar;
}

// AppV1 重绘时整段换掉 .window-content 的内容，状态条每次都要重插
function insertBar(app, data) {
  const content = app.element[0]?.querySelector('.window-content');
  if (!content) return;
  content.querySelector(':scope > .taka-charsync-bar')?.remove();
  content.prepend(buildBar(app, data));
}

async function statusData(actor, meta) {
  const card = meta?.cards?.[copyOf(actor)?.cardId ?? actor.id];
  const mainLatest = card ? await getMainLatest(card.world, card.mainId) : null;
  const companions = await Promise.all(companionsOf(actor, meta).map(async companion => {
    const mapped = meta?.cards?.[companion.id];
    return {actor: companion, mainLatest: mapped ? await getMainLatest(mapped.world, mapped.mainId) : null};
  }));
  return {meta, mainLatest, companions};
}

// 读发布数据算状态；`catchUp` 时当前用户是卡主的卡先跟上主服
async function loadStatus(app, session, {catchUp}) {
  if (session.loading) return;
  session.loading = true;
  const {actor} = app;
  try {
    const meta = await loadRuleMeta();
    if (catchUp && actorRuleState(actor).ok && ownsCard(actor, meta)) await catchUpWithNotice(actor, meta);
    session.data = await statusData(actor, meta);
  } catch (error) {
    console.error(`taka-charsync | 读不到 ${actor.name} 的同步状态`, error);
    session.data = {error};
  } finally {
    session.loading = false;
  }
  if (sessions.get(app) === session && app.rendered) insertBar(app, session.data);
}

async function confirmDiscard(app, element) {
  const {actor} = app;
  const confirmed = await foundry.applications.api.DialogV2.confirm({
    window: {title: DISCARD},
    content: `<p>先把「${foundry.utils.escapeHTML(actor.name)}」现在的样子另存成一张草稿备份卡，再把它恢复成主服的版本。</p><p>已经提交的副本不受影响；要收回提交，请在副本上点「撤回」。</p>`,
    yes: {label: '放弃草稿'},
    no: {label: '取消'},
  });
  if (!confirmed) return;
  element.disabled = true;
  await discardWithNotice(actor);
  const session = sessions.get(app);
  if (session) await loadStatus(app, session, {catchUp: false});
}

// 提交、撤回：先禁用按钮，做完提示 `notice(结果)`（省略或没做时不提示）、刷新状态条；出错时提示，不抛出
async function runAction(app, element, what, action, notice) {
  const {actor} = app;
  element.disabled = true;
  try {
    const result = await action(actor);
    if (result && notice) ui.notifications.info(notice(result));
  } catch (error) {
    notifyFailure(actor, what, error);
  }
  const session = sessions.get(app);
  if (session) await loadStatus(app, session, {catchUp: false});
}

async function confirmForce(app, element) {
  const confirmed = await foundry.applications.api.DialogV2.confirm({
    window: {title: FORCE},
    content: '<p>这会在主服整张覆盖，血量、物品、金币也换成这里的；主服经验保留。主人和同伴一起提交，要 GM 点套用才生效。</p>',
    yes: {label: '提交'},
    no: {label: '取消'},
  });
  if (!confirmed) return;
  const name = foundry.utils.escapeHTML(app.actor.name);
  await runAction(app, element, '整卡同步', submitForce, () => `已提交${name}的整卡同步，等主服 GM 点套用`);
}

async function confirmWithdraw(app, element, copyActor = app.actor, expectedCopy = copyOf(copyActor)) {
  const confirmed = await foundry.applications.api.DialogV2.confirm({
    window: {title: WITHDRAW},
    content: '<p>这份副本、随它提交的同伴及本卡更高级的同类副本会一起标记撤回；要恢复得在工作卡上重新提交。</p><p>撤回需要等同步引擎发布后主服才会看到；已经套用的结果不会自动撤销。</p>',
    yes: {label: '撤回'},
    no: {label: '取消'},
  });
  if (!confirmed) return;
  await runAction(app, element, '撤回', async () => {
    const current = copyOf(copyActor);
    if (!current || current.rev !== expectedCopy.rev) throw new Error('提交副本已更新，请刷新状态后重试');
    await withdraw(copyActor);
    return true;
  }, () => '已标记撤回，等待同步引擎发布到主服');
}

// 替换下拉只列选中的团里的卡
function limitReplaceable(form) {
  const {world, replaces} = form.elements;
  const update = () => {
    for (const option of replaces.options) option.hidden = Boolean(option.value) && option.dataset.world !== world.value;
    if (replaces.selectedOptions[0]?.hidden) replaces.value = '';
  };
  world.addEventListener('change', update);
  update();
}

async function promptNewCard(app, element, meta) {
  const escape = foundry.utils.escapeHTML;
  const title = world => meta.worlds?.[world]?.title ?? world;
  const option = (value, label, world) => {
    const worldAttribute = world ? ` data-world="${escape(world)}"` : '';
    return `<option value="${escape(value)}"${worldAttribute}>${escape(label)}</option>`;
  };
  const worlds = newCardWorlds(meta).filter(world => world === actorRuleState(app.actor).world);
  const replaceable = worlds.flatMap(world => replaceableCards(meta, world)
    .map(cardId => option(cardId, game.actors.get(cardId)?.name ?? cardId, world)));
  // 传元素而不是字符串：字符串内容会被 cleanHTML 过一遍
  const content = document.createElement('div');
  content.innerHTML = `
    <div class="form-group"><label>提交到</label><select name="world">${worlds.map(world => option(world, title(world))).join('')}</select></div>
    <div class="form-group"><label>替换</label><select name="replaces">${option('', '不替换，新增一张')}${replaceable.join('')}</select></div>`;
  const companions = companionsOf(app.actor, meta);
  if (companions.length) content.insertAdjacentHTML('beforeend',
    `<p>一起提交的同伴：${companions.map(companion => escape(companion.name)).join('、')}。GM 确认后会一起导入。</p>`);
  const choice = await foundry.applications.api.DialogV2.input({
    window: {title: NEW_CARD},
    content,
    ok: {label: '提交'},
    render: (event, dialog) => limitReplaceable(dialog.element.querySelector('form')),
  });
  if (!choice) return;
  const submit = actor => submitNew(actor, {targetWorld: choice.world, replaces: choice.replaces || null});
  const notice = () => `已把${escape(app.actor.name)}作为新卡提交到${escape(title(choice.world))}，等主服 GM 处理`;
  await runAction(app, element, '作为新卡提交', submit, notice);
}

/**
 * `renderCharacterSheetPF2e`、`renderFamiliarSheetPF2e` 钩子。卡面每次打开（关着到开着，不是开着时重绘）时，
 * 当前用户是卡主的卡先跟上主服，GM 打开只看状态；状态条先显示「正在核对」，读完发布数据再换成状态。
 */
export function renderStatusBar(app) {
  const {actor} = app;
  if (!actor?.isOwner || actor.isToken || app.options.classes.includes('attack-popout')) return;
  let session = sessions.get(app);
  if (!session) {
    session = {data: null, expanded: {}};
    sessions.set(app, session);
    loadStatus(app, session, {catchUp: true});
  } else if (session.data?.error && actorRuleState(actor).ok) {
    // 规则连接恢复会重绘已打开卡面；清掉上一次断网留下的状态读取错误。
    loadStatus(app, session, {catchUp: false});
  }
  insertBar(app, session.data);
}

/** `closeCharacterSheetPF2e`、`closeFamiliarSheetPF2e` 钩子：下次打开重新跟上主服。 */
export function endStatusSession(app) {
  sessions.delete(app);
}
