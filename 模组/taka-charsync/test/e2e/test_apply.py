"""主服套用建卡服提交的副本：登录时与定时自动套用、提交后改动、撤回、开团中等 GM 点、两个 GM 同时在线、
规则版本暂停、卡上已有非法条目、主人与魔宠一起套用、同伴撤回时整组拒绝、谱系失效、撤回点目录缺失与更新被丢弃、
玩家整卡同步、GM 直接拉建卡服版本、新卡替换旧卡（提交过的和没提交过的），以及建卡服新建的魔宠：
写入中途失败时整张恢复，再套用时以原 id 新建、撤回时删掉。

卡从夹具运行时挑：L6、不施法、带 ChoiceSet 专长的卡（A）；带幻灵的卡（B）；魔宠和它的主人（C）。
「等 61 秒」只在用例 1 真等定时器，其余「下一次检查也不会套上」直接调模组的 `api.runApplyCheck`。

用法：python -X utf8 test_apply.py
"""
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from harness import CAMP_WORLD, MODULE_ID, PUBLISH_DIR, Env, PageLogOnFailure, fixtures_dir

SUBMIT_URL = f'/modules/{MODULE_ID}/scripts/char/submit.mjs'
IMPORT_URL = f'/modules/{MODULE_ID}/scripts/foundry/import.mjs'
API = f"game.modules.get('{MODULE_ID}').api"
WAIT_MS = 60_000
# 定时检查每 60 秒一次
TIMER_WAIT_MS = 90_000

PANEL = '#taka-charsync-panel'
DIALOG = '.application.dialog'
ARCHIVE = 'charsync 归档'
PULL = '用建卡服现在的版本覆盖'
FIRST_PICK = 'E2E 七级专长甲'
SECOND_PICK = 'E2E 七级专长乙'
FORCE_FEAT = 'E2E 整卡专长'
FORCE_ITEM = 'E2E 整卡物品'
PULLED_ITEM = 'E2E 拉取物品'
NEW_CARD_NAME = 'E2E 新卡'
OTHER_CARD_NAME = 'E2E 没提交的卡'
INVALID_ITEM = 'E2E 非法专长'
NEW_FAMILIAR_NAME = 'E2E 新魔宠'
# 登录时那次检查做完后，弹框还要再读一次 meta、渲染出来
DIALOG_GRACE_MS = 3_000
# 用例 5、10、15 故意使套用失败，模组照常记错误；对应测试同时核对失败原因和卡状态。
EXPECTED_ERRORS = ('整卡导入被 Foundry 丢弃', '的副本缺失、已撤回或版本不符')

GRANT_ACTOR_CREATE = """async () => {
  const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
  const {PLAYER, TRUSTED, ASSISTANT, GAMEMASTER} = CONST.USER_ROLES;
  permissions.ACTOR_CREATE = [PLAYER, TRUSTED, ASSISTANT, GAMEMASTER];
  await game.settings.set('core', 'permissions', permissions);
}"""

# 建卡服（P1）
SET_LEVEL = """async ([id, level]) => { await game.actors.get(id).update({'system.details.level.value': level}); }"""
SET_HP = """async ([id, value]) => { await game.actors.get(id).update({'system.attributes.hp.value': value}); }"""
ADD_FEAT = """async ([actorId, name]) => {
  const [item] = await game.actors.get(actorId).createEmbeddedDocuments('Item', [
    {name, type: 'feat', system: {category: 'general', level: {value: 1}}},
  ]);
  return item.id;
}"""
ADD_EQUIPMENT = """async ([actorId, name]) => {
  const [item] = await game.actors.get(actorId).createEmbeddedDocuments('Item', [{name, type: 'equipment'}]);
  return item.id;
}"""
DELETE_ITEM = """async ([actorId, itemId]) => { await game.actors.get(actorId).deleteEmbeddedDocuments('Item', [itemId]); }"""
# PF2e Item.deleteDocuments 会过滤 invalid id；GM 在测试内精确替换 items，清理本次注入的坏行。
CLEAN_INVALID_FIXTURE_ITEM = """async ([actorId, itemId, name]) => {
  const actor = game.actors.get(actorId);
  const matches = actor._source.items.filter(item => item._id === itemId);
  if (!game.user.isGM || matches.length !== 1 || matches[0].name !== name || matches[0].type !== 'feat'
      || !Number.isNaN(matches[0].system.level.value) || !actor.items.invalidDocumentIds.has(itemId)) {
    throw new Error('只能由 GM 清理本测试注入的非法专长');
  }
  const items = foundry.utils.deepClone(actor._source.items).filter(item => item._id !== itemId);
  const updated = await actor.update({items}, {diff: false, recursive: false});
  if (!updated) throw new Error('GM 清理测试非法专长的更新被拒绝');
}"""
# 把一项受训的技能提一级；没有受训的就把第一项提到受训
RAISE_SKILL = """async id => {
  const actor = game.actors.get(id);
  const skills = actor._source.system.skills;
  const key = Object.keys(skills).find(skill => skills[skill].rank === 1) ?? Object.keys(skills)[0];
  await actor.update({[`system.skills.${key}.rank`]: (skills[key].rank ?? 0) + 1});
  return key;
}"""
CREATE_CARD = """async name => (await Actor.create({name, type: 'character', folder: 'CharsyncCamp0001'})).id"""
CREATE_FAMILIAR = """async ([name, masterId]) => (await Actor.create({name, type: 'familiar', system: {master: {id: masterId}}})).id"""
SUBMIT_LEVEL = """async ([url, id]) => (await import(url)).submitLevel(game.actors.get(id))"""
SUBMIT_FORCE = """async ([url, id]) => (await import(url)).submitForce(game.actors.get(id))"""
SUBMIT_NEW = """async ([url, id, world, replaces]) =>
  (await import(url)).submitNew(game.actors.get(id), {targetWorld: world, replaces})"""
CHAR_DOC = """id => game.actors.get(id).toObject()"""
SOURCE_ITEMS = """id => game.actors.get(id)._source.items"""
BASE_IS = """([id, hash]) => game.actors.get(id)?._source.flags['taka-charsync']?.base?.hash === hash"""
BACKUP_COUNT = """id => game.actors.filter(actor => actor._source.flags['taka-charsync']?.backupOf === id).length"""

# 两边都能用
ITEM_LABELS = """id => game.actors.get(id)._source.items.map(item => [item.type, item.name])"""
SKILL_RANKS = """id => Object.fromEntries(Object.entries(game.actors.get(id)._source.system.skills ?? {})
  .map(([skill, value]) => [skill, value.rank ?? 0]))"""

# 主服（GM）
MAIN_STATE = """id => {
  const actor = game.actors.get(id);
  if (!actor) return null;
  const source = actor._source;
  const own = source.flags['taka-charsync'] ?? {};
  return {
    name: actor.name,
    level: source.system.details?.level?.value ?? null,
    xp: source.system.details?.xp ?? null,
    hp: source.system.attributes?.hp?.value ?? null,
    hpMax: actor.system.attributes?.hp?.max ?? null,
    applied: own.applied ?? null,
    undone: own.undone ?? [],
    retired: own.retired ?? false,
    charId: own.charId ?? null,
    folder: actor.folder?.name ?? null,
    ownership: source.ownership,
    master: source.system.master?.id ?? null,
    hasMaster: actor.type === 'familiar' ? Boolean(actor.master) : null,
    itemIds: source.items.map(item => item._id),
    itemNames: source.items.map(item => item.name),
  };
}"""
FILL_XP = """async id => {
  const actor = game.actors.get(id);
  await actor.update({'system.details.xp.value': actor._source.system.details.xp.max});
}"""
SET_MAIN_LEVEL = """async ([id, level]) => {
  await game.actors.get(id).update({'system.details.level.value': level}, {diff: false});
}"""
RUN_CHECK = f"""async () => JSON.parse(JSON.stringify(await {API}.runApplyCheck({{trigger: 'tick'}})))"""
START_CHECK = f"""() => {{ window.e2eCheck = {API}.runApplyCheck({{trigger: 'tick'}}).then(r => JSON.parse(JSON.stringify(r))); }}"""
CHECK_RESULT = """() => window.e2eCheck"""
READY_CHECKED = f"""() => {API}.lastCheck()?.trigger === 'ready'"""
LAST_CHECK = f"""() => {API}.lastCheck()"""
IS_ACTIVE_GM = """() => game.users.activeGM?.isSelf === true"""
USER_ACTIVE = """([name, active]) => game.users.getName(name).active === active"""
USER_ID = """name => game.users.getName(name).id"""
WHISPERS = """id => game.messages
  .filter(message => message.flags['taka-charsync']?.actorId === id)
  .map(message => ({gmOnly: message.whisper.length > 0 && message.whisper.every(user => game.users.get(user)?.isGM)}))"""
WHISPER_COUNT_IS = """([id, count]) => game.messages.filter(message => message.flags['taka-charsync']?.actorId === id).length >= count"""
LAST_WHISPER_HAS = """([id, texts]) => {
  const message = game.messages.filter(message => message.flags['taka-charsync']?.actorId === id).at(-1);
  return texts.every(text => message?.content.includes(foundry.utils.escapeHTML(text)));
}"""
# 已渲染的窗口：AppV2 带框的，加上 AppV1 的
OPEN_WINDOWS = """() => [
  ...[...foundry.applications.instances.values()].filter(app => app.rendered && app.hasFrame).map(app => app.constructor.name),
  ...Object.values(ui.windows).filter(app => app.rendered).map(app => app.constructor.name),
]"""
CHOICE_SELECTIONS = """id => Object.fromEntries(game.actors.get(id)._source.items
  .filter(item => (item.system.rules ?? []).some(rule => rule.key === 'ChoiceSet'))
  .map(item => [item._id, item.flags?.pf2e?.rulesSelections ?? null]))"""
EXISTS = """id => game.actors.has(id)"""
EXISTS_NOT = """id => !game.actors.has(id)"""
RETIRED = """id => game.actors.get(id)?._source.flags['taka-charsync']?.retired === true"""
UNDONE_HAS = """([id, key]) => game.actors.get(id)._source.flags['taka-charsync']?.undone?.includes(key) ?? false"""
# 让这张卡的 update 一律返回空（客户端校验没过时 Foundry 就这样），数调用次数；改经验绕过它
BREAK_UPDATE = """id => {
  const actor = game.actors.get(id);
  window.e2eUpdateCalls = 0;
  actor.update = async () => { window.e2eUpdateCalls++; return undefined; };
}"""
FILL_XP_PAST_BREAK = """async id => {
  const actor = game.actors.get(id);
  await Object.getPrototypeOf(actor).update.call(actor, {'system.details.xp.value': actor._source.system.details.xp.max});
}"""
RESTORE_UPDATE = """id => { delete game.actors.get(id).update; return window.e2eUpdateCalls; }"""
# 同上，但改在原型上：这个 id 的卡还没建出来时也生效
BREAK_ACTOR_UPDATE = """id => {
  const proto = foundry.documents.Actor.prototype;
  const own = Object.getOwnPropertyDescriptor(proto, 'update');
  const original = proto.update;
  window.e2eUpdateCalls = 0;
  window.e2eRestoreActorUpdate = () => {
    if (own) Object.defineProperty(proto, 'update', own);
    else delete proto.update;
    return window.e2eUpdateCalls;
  };
  proto.update = async function (...args) {
    if (this.id === id) {
      window.e2eUpdateCalls++;
      return undefined;
    }
    return original.apply(this, args);
  };
}"""
RESTORE_ACTOR_UPDATE = """() => window.e2eRestoreActorUpdate()"""

SHOW_ACTORS = """() => ui.actors.activate()"""
BADGE_SHOWN = """() => {
  const badge = document.querySelector('.taka-charsync-panel-button .taka-charsync-badge');
  return Boolean(badge) && !badge.hidden && Number(badge.textContent) > 0;
}"""
ROW_TEXT = f"""id => document.querySelector(`{PANEL} [data-actor-id="${{id}}"]`)?.innerText ?? null"""
ENTRY_BUTTONS = f"""([id, reason]) => [...document.querySelectorAll(`{PANEL} [data-actor-id="${{id}}"] [data-reason="${{reason}}"] button`)]
  .map(button => button.textContent)"""
ROW_BUTTONS = f"""id => [...document.querySelectorAll(`{PANEL} [data-actor-id="${{id}}"] button`)].map(button => button.textContent)"""

# 主服卡和一份建卡服文档比，返回不同的路径。不比的：每张卡自己的 id、所有者、文件夹、排序、_stats（条目和效果上
# 是 _stats、所有者、排序，整卡导入保留卡上的），同步记号，以及整卡覆盖时保留主服的经验
DIFF_FROM_DOC = """async ([importUrl, id, doc]) => {
  const {cleanDoc} = await import(importUrl);
  const embedded = list => Object.fromEntries((list ?? []).map(entry => {
    const out = foundry.utils.deepClone(entry);
    for (const key of ['_id', '_stats', 'ownership', 'sort']) delete out[key];
    if (Array.isArray(out.effects)) out.effects = embedded(out.effects);
    return [entry._id, out];
  }));
  const project = source => {
    const system = foundry.utils.deepClone(source.system);
    delete system.details.xp;
    const flags = foundry.utils.deepClone(source.flags ?? {});
    delete flags['taka-charsync'];
    return {
      name: source.name, img: source.img, system, flags, prototypeToken: source.prototypeToken,
      items: embedded(source.items), effects: embedded(source.effects),
    };
  };
  const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  // 可选字段缺着、为 null、为 undefined 算一样：建卡服客户端内存里的条目把它们记成 null，发布的文档里缺着这些键
  const canon = value => JSON.stringify(value ?? null, (key, inner) => isObject(inner)
    ? Object.fromEntries(Object.keys(inner).sort().filter(k => inner[k] != null).map(k => [k, inner[k]])) : inner);
  const diff = [];
  const walk = (a, b, path) => {
    if (canon(a) === canon(b)) return;
    if (isObject(a) && isObject(b)) {
      for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) walk(a[key], b[key], path ? `${path}.${key}` : key);
      return;
    }
    diff.push(path);
  };
  walk(project(game.actors.get(id).toObject()), project(cleanDoc(doc)), '');
  return diff;
}"""


def is_module_error(line: str) -> bool:
    return (line.startswith(('error:', 'pageerror:')) and MODULE_ID in line
            and not any(message in line for message in EXPECTED_ERRORS))


def copy_ref(copy: dict) -> dict:
    return {'kind': 'copy', 'copyId': copy['copyId'], 'rev': copy['rev']}


def ref_key(copy: dict) -> str:
    return f"copy:{copy['copyId']}:{copy['rev']}"


def choice_selections(doc: dict) -> dict:
    return {
        item['_id']: item.get('flags', {}).get('pf2e', {}).get('rulesSelections')
        for item in doc['items'] if any(rule.get('key') == 'ChoiceSet' for rule in item['system'].get('rules', []))
    }


class ApplyTest(PageLogOnFailure, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-e2e-')), fixtures_dir())
        cls.addClassCleanup(env.close)
        actors = env.actors

        def level(card_id):
            return actors[card_id]['system'].get('details', {}).get('level', {}).get('value')

        def casts(card_id):
            return any(item['type'] == 'spellcastingEntry' for item in actors[card_id]['items'])

        cls.card = next(c for c in env.main_cards if level(c) == 6 and not casts(c) and choice_selections(actors[c]))
        summoner = next(card for card in env.seed_cards if card.get('companions'))
        cls.retrain, cls.eidolons = summoner['charId'], summoner['companions']
        cls.familiar = next(actor_id for actor_id, doc in actors.items() if doc['type'] == 'familiar')
        cls.master = actors[cls.familiar]['system']['master']['id']
        cls.master_level = level(cls.master)

        env.build()
        env.start('main')
        env.start('char')
        env.seed()
        gm, _ = env.login('char', 'Gamemaster')
        try:
            gm.evaluate(GRANT_ACTOR_CREATE)
        finally:
            gm.context.close()
        env.stop('main')
        cls.run_engine()

        cls.client_logs = []
        cls.expected_module_errors = {}
        cls.p1 = None
        cls.gm = None
        cls.login_p1()
        meta = cls.read_published('meta.json')
        for actor_id in (cls.card, cls.retrain, *cls.eidolons, cls.master, cls.familiar):
            cls.p1.wait_for_function(BASE_IS, arg=[actor_id, meta['bootstrap'][actor_id]['hash']], timeout=WAIT_MS)

    @classmethod
    def run_engine(cls):
        report = cls.env.engine_once()
        if report['errors']:
            raise AssertionError(f"引擎一轮有错误：{report['errors']}")

    @classmethod
    def login_p1(cls):
        if cls.p1 is not None:
            cls.p1.context.close()
        cls.p1, logs = cls.env.login('char', 'P1')
        cls.client_logs.append(logs)

    @classmethod
    def read_published(cls, rel, which='char'):
        return json.loads((cls.env.data_dir(which) / PUBLISH_DIR / rel).read_text(encoding='utf-8'))

    def login(self, which, user):
        page, logs = self.env.login(which, user)
        self.client_logs.append(logs)
        return page

    def gm_login(self):
        """以 GM 登进主服（团世界要已经开着），等登录时那一次检查跑完。"""
        self.close_gm()
        gm = type(self).gm = self.login('main', 'Gamemaster')
        gm.wait_for_function(READY_CHECKED, timeout=WAIT_MS)
        return gm

    def close_gm(self):
        if self.gm is not None:
            self.gm.context.close()
            type(self).gm = None

    def stop_main(self):
        self.close_gm()
        self.env.stop('main')

    def latest(self, actor_id):
        return self.read_published(f'main/{CAMP_WORLD}/{actor_id}.json')

    def main_file(self, rel):
        return self.env.data_dir('main') / PUBLISH_DIR / rel

    def undo_files(self):
        folder = self.main_file(f'undo/{CAMP_WORLD}')
        return sorted(path.name for path in folder.glob('*.json')) if folder.is_dir() else []

    def main_state(self, actor_id):
        return self.gm.evaluate(MAIN_STATE, actor_id)

    def whisper_count(self, actor_id):
        return len(self.gm.evaluate(WHISPERS, actor_id))

    def wait_whispers(self, actor_id, count, timeout=WAIT_MS):
        self.gm.wait_for_function(WHISPER_COUNT_IS, arg=[actor_id, count], timeout=timeout)

    def submit_level(self, actor_id, level):
        self.p1.evaluate(SET_LEVEL, [actor_id, level])
        return self.p1.evaluate(SUBMIT_LEVEL, [SUBMIT_URL, actor_id])

    def open_panel(self):
        self.gm.evaluate(SHOW_ACTORS)
        self.gm.click('.taka-charsync-panel-button')
        self.gm.wait_for_selector(f'{PANEL} .taka-charsync-panel-body', timeout=WAIT_MS)

    def click_panel(self, selector):
        self.gm.click(f'{PANEL} {selector}')

    def confirm_action(self, title, *names, accept=True):
        dialog = self.gm.locator(f'{DIALOG}:has(button[data-action="yes"])')
        dialog.wait_for(timeout=WAIT_MS)
        content = dialog.inner_text()
        for text in (title, *names):
            self.assertIn(text, content)
        action = 'yes' if accept else 'no'
        dialog.locator(f'button[data-action="{action}"]').click()
        dialog.wait_for(state='hidden', timeout=WAIT_MS)

    def test_01_ready_and_timer_apply_next_level(self):
        card = self.card
        copy7 = self.submit_level(card, 7)
        copy8 = type(self).copy8 = self.submit_level(card, 8)

        # 经验拉满时副本还没发布，这一场里不会被套上
        self.env.start('main')
        self.gm_login()
        before = self.main_state(card)
        self.assertEqual((before['level'], before['xp']['value']), (6, 0))
        self.gm.evaluate(FILL_XP, card)
        self.stop_main()
        self.run_engine()
        undo_before = self.undo_files()

        self.env.start('main')
        self.gm_login()
        after = self.main_state(card)
        self.assertEqual(after['level'], 7)
        self.assertEqual(after['xp']['value'], 0)
        self.assertGreater(after['hpMax'], before['hpMax'])
        self.assertEqual(after['hp'], min(after['hpMax'], before['hp'] + after['hpMax'] - before['hpMax']))
        self.assertEqual(after['applied']['ref'], copy_ref(copy7))
        self.assertEqual(len(self.undo_files()), len(undo_before) + 1)
        whispers = self.gm.evaluate(WHISPERS, card)
        self.assertEqual(len(whispers), 1)
        self.assertTrue(whispers[0]['gmOnly'])
        self.gm.wait_for_timeout(DIALOG_GRACE_MS)
        self.assertEqual(self.gm.evaluate(OPEN_WINDOWS), [])
        # 带 ChoiceSet 的专长：套用时没有弹窗，选择值与副本一致
        history = self.read_published(f"history/copy-{copy7['copyId']}-{copy7['rev']}.json", 'main')
        self.assertTrue(choice_selections(history))
        self.assertEqual(self.gm.evaluate(CHOICE_SELECTIONS, card), choice_selections(history))

        self.gm.evaluate(FILL_XP, card)
        self.wait_whispers(card, 2, timeout=TIMER_WAIT_MS)
        later = self.main_state(card)
        self.assertEqual((later['level'], later['xp']['value']), (8, 0))
        self.assertEqual(later['applied']['ref'], copy_ref(copy8))
        self.assertEqual(len(self.undo_files()), len(undo_before) + 2)

        # 建卡服那张跟上主服时以套用的副本为基线合并，不重置、不建草稿备份
        backups = self.p1.evaluate(BACKUP_COUNT, card)
        self.stop_main()
        self.run_engine()
        latest = self.latest(card)
        self.assertEqual(latest['applied'], copy_ref(copy8))
        self.login_p1()
        self.p1.wait_for_function(BASE_IS, arg=[card, latest['hash']], timeout=WAIT_MS)
        self.assertEqual(self.p1.evaluate(BACKUP_COUNT, card), backups)

    def test_02_undo_restores_previous_state_and_ignores_copy(self):
        card = self.card
        self.env.start('main')
        self.gm_login()
        before = self.main_state(card)
        self.assertEqual(before['level'], 8)
        self.open_panel()
        self.click_panel(f'[data-actor-id="{card}"] button:text-is("恢复到套用前")')
        self.assertEqual(self.main_state(card), before)
        self.confirm_action('恢复到套用前', before['name'], '套用后的改动会丢失')
        self.gm.wait_for_function(UNDONE_HAS, arg=[card, ref_key(self.copy8)], timeout=WAIT_MS)
        state = self.main_state(card)
        self.assertEqual(state['level'], 7)
        self.assertEqual(state['xp']['value'], state['xp']['max'])
        self.assertEqual(state['applied']['ref']['kind'], 'copy')
        self.assertNotEqual(state['applied']['ref']['copyId'], self.copy8['copyId'])

        result = self.gm.evaluate(RUN_CHECK)
        self.assertNotIn(card, result['applied'])
        self.assertEqual(self.main_state(card)['level'], 7)

    def test_03_resubmitted_level_applies_without_duplicates(self):
        card = self.retrain
        self.p1.evaluate(SET_LEVEL, [card, 7])
        first = self.p1.evaluate(ADD_FEAT, [card, FIRST_PICK])
        self.p1.evaluate(SUBMIT_LEVEL, [SUBMIT_URL, card])
        # 交了 7 级后换掉这一级的专长、提一项技能，再交 7 级和 8 级
        self.p1.evaluate(DELETE_ITEM, [card, first])
        self.p1.evaluate(ADD_FEAT, [card, SECOND_PICK])
        self.p1.evaluate(RAISE_SKILL, card)
        self.p1.evaluate(SUBMIT_LEVEL, [SUBMIT_URL, card])
        self.submit_level(card, 8)
        self.run_engine()

        whispers = self.whisper_count(card)
        for level in (7, 8):
            self.gm.evaluate(FILL_XP, card)
            self.gm.evaluate(RUN_CHECK)
            whispers += 1
            self.wait_whispers(card, whispers)
            self.assertEqual(self.main_state(card)['level'], level)

        main_items = Counter(map(tuple, self.gm.evaluate(ITEM_LABELS, card)))
        self.assertEqual(main_items, Counter(map(tuple, self.p1.evaluate(ITEM_LABELS, card))))
        self.assertEqual((main_items[('feat', FIRST_PICK)], main_items[('feat', SECOND_PICK)]), (0, 1))
        self.assertEqual(self.gm.evaluate(SKILL_RANKS, card), self.p1.evaluate(SKILL_RANKS, card))

    def test_04_players_online_leave_it_to_gm_and_familiar_follows(self):
        master, familiar = self.master, self.familiar
        level = self.master_level + 1
        copy = self.submit_level(master, level)
        self.run_engine()
        p1_main = self.login('main', 'P1')
        try:
            self.gm.wait_for_function(USER_ACTIVE, arg=['P1', True], timeout=WAIT_MS)
            self.gm.evaluate(FILL_XP, master)
            result = self.gm.evaluate(RUN_CHECK)
            self.assertEqual(self.main_state(master)['level'], level - 1)
            self.assertIn({'actorId': master, 'reason': 'session', 'copy': copy}, result['pending'])
            self.gm.evaluate(SHOW_ACTORS)
            self.assertTrue(self.gm.evaluate(BADGE_SHOWN))

            whispers = self.whisper_count(master)
            companion_before = self.main_state(familiar)
            self.open_panel()
            self.click_panel(f'[data-actor-id="{master}"] [data-reason="session"] button:text-is("套用")')
            self.wait_whispers(master, whispers + 1)
        finally:
            p1_main.context.close()
        state = self.main_state(master)
        self.assertEqual(state['level'], level)
        self.assertEqual(state['applied']['ref'], copy_ref(copy))
        companion = self.main_state(familiar)
        self.assertEqual(companion['master'], master)
        self.assertTrue(companion['hasMaster'])
        self.assertEqual(companion['applied']['ref']['copyId'], copy['companions'][0]['copyId'])
        # 魔宠的上限随主人的等级涨，当前血量补上差值
        self.assertGreater(companion['hpMax'], companion_before['hpMax'])
        gained = companion_before['hp'] + companion['hpMax'] - companion_before['hpMax']
        self.assertEqual(companion['hp'], min(companion['hpMax'], max(0, gained)))

    def test_05_withdrawn_companion_copy_rejects_the_whole_group(self):
        master, familiar = self.master, self.familiar
        copy = self.submit_level(master, self.master_level + 2)
        self.run_engine()
        # 单独撤回魔宠副本：发布数据不完整时，主人与魔宠都不能被写入。
        listed_file = self.main_file(f'copies/{familiar}.json')
        original = listed_file.read_bytes()
        listed = json.loads(original)
        companion_ref = copy['companions'][0]
        matches = [entry for entry in listed if entry['copyId'] == companion_ref['copyId']]
        self.assertEqual(len(matches), 1)
        matches[0]['withdrawn'] = True
        listed_file.write_text(json.dumps(listed, ensure_ascii=False), encoding='utf-8')
        try:
            self.gm.wait_for_function(USER_ACTIVE, arg=['P1', False], timeout=WAIT_MS)
            self.gm.evaluate(FILL_XP, master)
            before = {actor_id: self.gm.evaluate(CHAR_DOC, actor_id) for actor_id in (master, familiar)}
            whispers = self.whisper_count(master)
            undo_before = self.undo_files()
            result = self.gm.evaluate(RUN_CHECK)
            pending = [item for item in result['pending'] if item['actorId'] == master]
            self.assertEqual([item['reason'] for item in pending], ['error'])
            self.assertIn('同伴', pending[0]['message'])
            self.assertIn('副本缺失、已撤回或版本不符', pending[0]['message'])
            self.assertNotIn(master, result['applied'])
            for actor_id, doc in before.items():
                self.assertEqual(self.gm.evaluate(CHAR_DOC, actor_id), doc, actor_id)
            self.assertEqual(self.undo_files(), undo_before)
            self.assertEqual(self.whisper_count(master), whispers)
        finally:
            listed_file.write_bytes(original)

        # 恢复发布状态后手动重试，清掉本会话的失败记号，并保持后续顺序测试的升级前驱。
        self.open_panel()
        self.click_panel(f'[data-actor-id="{master}"] [data-reason="error"] button:text-is("套用")')
        self.wait_whispers(master, whispers + 1)
        state = self.main_state(master)
        companion = self.main_state(familiar)
        self.assertEqual((state['level'], state['xp']['value']), (self.master_level + 2, 0))
        self.assertEqual(state['applied']['ref'], copy_ref(copy))
        self.assertEqual(companion['applied']['ref'], {
            'kind': 'copy', 'copyId': companion_ref['copyId'], 'rev': companion_ref['rev'],
        })
        self.assertEqual(companion['master'], master)
        self.assertTrue(companion['hasMaster'])
        self.assertTrue(self.gm.evaluate(LAST_WHISPER_HAS, [master, [companion['name'], '一起写入']]))
        self.assertEqual(len(self.undo_files()), len(undo_before) + 2)

    def test_06_only_active_gm_applies(self):
        card = self.retrain
        self.submit_level(card, 9)
        self.run_engine()
        whispers = self.whisper_count(card)
        gm2 = self.login('main', 'GM2')
        try:
            self.gm.evaluate(FILL_XP, card)
            pages = {'Gamemaster': self.gm, 'GM2': gm2}
            for page in pages.values():
                page.evaluate(START_CHECK)
            results = {name: page.evaluate(CHECK_RESULT) for name, page in pages.items()}
            active = next(name for name, page in pages.items() if page.evaluate(IS_ACTIVE_GM))
            other = next(name for name in pages if name != active)
            self.wait_whispers(card, whispers + 1)
            first = self.main_state(card)['applied']
            self.assertEqual(results[other], {'applied': [], 'pending': [], 'waiting': []})
            self.assertIsNone(pages[other].evaluate(LAST_CHECK))

            for page in pages.values():
                page.evaluate(START_CHECK)
            for page in pages.values():
                page.evaluate(CHECK_RESULT)
            self.assertEqual(self.main_state(card)['applied'], first)
            self.assertEqual(gm2.evaluate(MAIN_STATE, card)['applied'], first)
            self.assertEqual(self.whisper_count(card), whispers + 1)
            self.assertEqual(self.main_state(card)['level'], 9)
        finally:
            gm2.context.close()
        self.gm.wait_for_function(IS_ACTIVE_GM, timeout=WAIT_MS)

    def test_07_schema_mismatch_pauses_applying(self):
        card = self.retrain
        self.submit_level(card, 10)
        self.run_engine()
        meta_file = self.main_file('meta.json')
        original = meta_file.read_text(encoding='utf-8')
        meta = json.loads(original)
        meta['char']['schema'] = 'e2e-mismatch'
        meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        try:
            self.gm.evaluate(FILL_XP, card)
            result = self.gm.evaluate(RUN_CHECK)
            self.assertEqual(self.main_state(card)['level'], 9)
            self.assertEqual({item['reason'] for item in result['pending']}, {'paused'})
            self.open_panel()
            self.assertIn('暂停', self.gm.evaluate(ROW_TEXT, card))
        finally:
            meta_file.write_text(original, encoding='utf-8')
        whispers = self.whisper_count(card)
        self.gm.evaluate(RUN_CHECK)
        self.wait_whispers(card, whispers + 1)
        self.assertEqual(self.main_state(card)['level'], 10)

    def test_08_invalid_item_on_main_card_blocks_ordinary_apply(self):
        master = self.master
        # 经验先拉满、副本后发布，这一场里不会被套上
        self.gm.evaluate(FILL_XP, master)
        before = self.main_state(master)
        items_before = self.gm.evaluate(SOURCE_ITEMS, master)
        familiar_before = self.main_state(self.familiar)
        whispers = self.whisper_count(master)
        undo_before = {name: self.main_file(f'undo/{CAMP_WORLD}/{name}').read_bytes() for name in self.undo_files()}
        self.stop_main()
        level = self.master_level + 3
        copy = self.submit_level(master, level)
        self.run_engine()
        invalid = self.env.inject_invalid_item('main', master)

        log_start = len(self.client_logs)
        self.env.start('main')
        self.gm_login()
        result = self.gm.evaluate(LAST_CHECK)
        pending = [item for item in result['pending'] if item['actorId'] == master]
        message = f'「{before["name"]}」有非法条目（{INVALID_ITEM}），请先修复；普通同步不会清理这些条目'
        self.assertEqual(pending, [{'actorId': master, 'reason': 'error', 'copy': copy, 'message': message}])
        self.assertNotIn(master, result['applied'])
        state = self.main_state(master)
        for key in before.keys() - {'itemIds', 'itemNames'}:
            self.assertEqual(state[key], before[key], key)
        items = self.gm.evaluate(SOURCE_ITEMS, master)
        self.assertEqual([item for item in items if item['_id'] != invalid], items_before)
        injected = [item for item in items if item['_id'] == invalid]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]['name'], INVALID_ITEM)
        self.assertTrue(self.gm.evaluate('''([id, itemId]) => Number.isNaN(
          game.actors.get(id)._source.items.find(item => item._id === itemId).system.level.value)''', [master, invalid]))
        self.assertTrue(self.gm.evaluate('([id, itemId]) => game.actors.get(id).items.invalidDocumentIds.has(itemId)',
                                        [master, invalid]))
        self.assertEqual(self.main_state(self.familiar), familiar_before)
        self.assertEqual(self.whisper_count(master), whispers)
        self.assertEqual(self.undo_files(), sorted(undo_before))
        for name, snapshot in undo_before.items():
            self.assertEqual(self.main_file(f'undo/{CAMP_WORLD}/{name}').read_bytes(), snapshot, name)

        # 只登记这次注入引起的完整错误；之后同样的错误或其他模块错误仍会使末项失败。
        errors = [(log_index, index, line) for log_index, logs in enumerate(self.client_logs)
                  for index, line in enumerate(logs) if log_index >= log_start and is_module_error(line)]
        self.assertEqual(len(errors), 1, errors)
        log_index, index, line = errors[0]
        parts = line.splitlines()
        self.assertEqual(parts[0], f'error: {MODULE_ID} | {before["name"]} 自动套用失败：{message} Error: {message}')
        self.assertTrue(parts[1:] and all(part.startswith('    at ') for part in parts[1:]), line)
        self.expected_module_errors[(log_index, index)] = line

        # GM 显式清理坏夹具；重登清除本会话的失败记号，再让普通套用重试。
        before_cleanup = self.gm.evaluate(CHAR_DOC, master)
        self.gm.evaluate(CLEAN_INVALID_FIXTURE_ITEM, [master, invalid, INVALID_ITEM])
        after_cleanup = self.gm.evaluate(CHAR_DOC, master)
        # GM 整体替换 items 会重建条目的创建/修改时间；其余字段仍须完全相同。
        def without_write_times(items):
            return [{**item, '_stats': {key: value for key, value in item.get('_stats', {}).items()
                                      if key not in ('createdTime', 'modifiedTime')}} for item in items]
        self.assertEqual(without_write_times(self.gm.evaluate(SOURCE_ITEMS, master)), without_write_times(items_before))
        for key in ('system', 'flags'):
            self.assertEqual(after_cleanup[key], before_cleanup[key], key)
        self.assertNotIn(invalid, self.main_state(master)['itemIds'])
        # Foundry 重算不清旧 invalidDocumentIds；重登后再核对集合，不能手改缓存。
        self.gm_login()
        self.assertFalse(self.gm.evaluate('([id, itemId]) => game.actors.get(id).items.invalidDocumentIds.has(itemId)',
                                         [master, invalid]))
        self.wait_whispers(master, whispers + 1)
        state = self.main_state(master)
        self.assertEqual((state['level'], state['xp']['value']), (level, 0))
        self.assertEqual(state['applied']['ref'], copy_ref(copy))
        self.assertNotIn(invalid, state['itemIds'])
        self.assertEqual(len(self.undo_files()), len(undo_before) + 2)

    def test_09_gm_level_up_makes_copy_stale(self):
        card = self.retrain
        copy = self.submit_level(card, 11)
        self.run_engine()
        self.gm.evaluate(SET_MAIN_LEVEL, [card, 11])
        result = self.gm.evaluate(RUN_CHECK)
        self.assertIn({'actorId': card, 'reason': 'stale', 'copy': copy}, result['pending'])

        self.open_panel()
        self.assertIn('谱系已失效', self.gm.evaluate(ROW_TEXT, card))
        self.assertEqual(self.gm.evaluate(ENTRY_BUTTONS, [card, 'stale']), ['忽略此副本'])
        self.assertNotIn('套用', self.gm.evaluate(ROW_BUTTONS, card))
        self.click_panel(f'[data-actor-id="{card}"] [data-reason="stale"] button:text-is("忽略此副本")')
        self.gm.wait_for_function(UNDONE_HAS, arg=[card, ref_key(copy)], timeout=WAIT_MS)
        result = self.gm.evaluate(RUN_CHECK)
        self.assertEqual([item for item in result['pending'] if item['actorId'] == card], [])

    def test_10_missing_undo_folder_and_dropped_update(self):
        master = self.master
        self.submit_level(master, self.master_level + 4)
        self.run_engine()
        # 引擎每轮都会建撤回点目录，所以在引擎之后删
        shutil.rmtree(self.main_file(f'undo/{CAMP_WORLD}'))
        whispers = self.whisper_count(master)
        self.gm.evaluate(FILL_XP, master)
        self.gm.evaluate(RUN_CHECK)
        self.wait_whispers(master, whispers + 1)
        self.assertEqual(self.main_state(master)['level'], self.master_level + 4)
        # 撤回点主人一份，一起写的魔宠一份
        self.assertEqual(sorted(name.split('-')[0] for name in self.undo_files()), sorted([master, self.familiar]))

        self.submit_level(master, self.master_level + 5)
        self.run_engine()
        self.gm.evaluate(BREAK_UPDATE, master)
        try:
            self.gm.evaluate(FILL_XP_PAST_BREAK, master)
            before = self.main_state(master)
            results = [self.gm.evaluate(RUN_CHECK) for _ in range(2)]
            after = self.main_state(master)
        finally:
            calls = self.gm.evaluate(RESTORE_UPDATE, master)
        for result in results:
            self.assertEqual([item['reason'] for item in result['pending'] if item['actorId'] == master], ['error'])
        self.assertEqual(calls, 1)
        self.assertEqual(self.whisper_count(master), whispers + 1)
        self.assertEqual((after['level'], after['hp'], after['xp']), (before['level'], before['hp'], before['xp']))
        self.assertEqual(after['xp']['value'], after['xp']['max'])

    def test_11_force_copy_overwrites_whole_card(self):
        card = self.card
        p1_main = self.login('main', 'P1')
        try:
            self.p1.evaluate(ADD_FEAT, [card, FORCE_FEAT])
            self.p1.evaluate(SET_HP, [card, 1])
            self.p1.evaluate(ADD_EQUIPMENT, [card, FORCE_ITEM])
            force = self.p1.evaluate(SUBMIT_FORCE, [SUBMIT_URL, card])
            self.run_engine()
            before = self.main_state(card)
            undo_before = len(self.undo_files())
            result = self.gm.evaluate(RUN_CHECK)
            self.assertIn({'actorId': card, 'reason': 'force', 'copy': force}, result['pending'])

            whispers = self.whisper_count(card)
            self.open_panel()
            self.click_panel(f'[data-actor-id="{card}"] [data-reason="force"] button:text-is("套用")')
            self.wait_whispers(card, whispers + 1)
        finally:
            p1_main.context.close()
        after = self.main_state(card)
        self.assertEqual(after['applied']['ref'], copy_ref(force))
        # 套过的整卡副本记进 undone，之后主服再套别的也不会再提它
        self.assertIn(ref_key(force), after['undone'])
        self.assertEqual(after['xp'], before['xp'])
        self.assertEqual(after['hp'], 1)
        self.assertTrue({FORCE_FEAT, FORCE_ITEM} <= set(after['itemNames']))
        self.assertEqual(self.gm.evaluate(DIFF_FROM_DOC, [IMPORT_URL, card, self.p1.evaluate(CHAR_DOC, card)]), [])
        self.assertEqual(len(self.undo_files()), undo_before + 1)

    def test_12_gm_pulls_current_char_version(self):
        card = self.card
        self.p1.evaluate(ADD_EQUIPMENT, [card, PULLED_ITEM])
        self.run_engine()
        working = self.read_published(f'working/{card}.json', 'main')
        history = f"history/char-{working['hash']}.json"
        before = self.main_state(card)
        whispers = self.whisper_count(card)
        self.open_panel()
        pull = self.gm.locator(f'{PANEL} [data-actor-id="{card}"] button:has-text("{PULL}")')
        self.assertRegex(pull.inner_text(), rf'^{PULL}（读取于 \d\d:\d\d）$')
        pull.click()
        self.gm.locator(DIALOG).locator('button[data-action="yes"]').click()
        self.wait_whispers(card, whispers + 1)
        after = self.main_state(card)
        self.assertEqual(after['applied']['ref'], {'kind': 'char', 'cardId': card, 'hash': working['hash']})
        self.assertEqual(after['xp'], before['xp'])
        self.assertIn(PULLED_ITEM, after['itemNames'])
        self.assertEqual(self.gm.evaluate(DIFF_FROM_DOC, [IMPORT_URL, card, working['doc']]), [])
        self.assertTrue(self.main_file(history).is_file())
        result = self.gm.evaluate(RUN_CHECK)
        self.assertEqual([item for item in result['pending'] if item['actorId'] == card], [])

        backups = self.p1.evaluate(BACKUP_COUNT, card)
        self.stop_main()
        self.run_engine()
        self.assertTrue((self.env.data_dir('char') / PUBLISH_DIR / history).is_file())
        latest = self.latest(card)
        self.login_p1()
        self.p1.wait_for_function(BASE_IS, arg=[card, latest['hash']], timeout=WAIT_MS)
        self.assertEqual(self.p1.evaluate(BACKUP_COUNT, card), backups)

    def test_13_new_card_copy_replaces_old_card(self):
        old_id = self.retrain
        new_card = self.p1.evaluate(CREATE_CARD, NEW_CARD_NAME)
        copy = self.p1.evaluate(SUBMIT_NEW, [SUBMIT_URL, new_card, CAMP_WORLD, old_id])
        self.run_engine()
        self.env.start('main')
        self.gm_login()
        dialog = self.gm.locator(DIALOG)
        dialog.wait_for(timeout=WAIT_MS)
        self.assertIn(NEW_CARD_NAME, dialog.inner_text())
        dialog.locator('button[data-action="review"]').click()
        self.gm.wait_for_selector(f'{PANEL} .taka-charsync-panel-body', timeout=WAIT_MS)
        old = self.main_state(old_id)
        old_companions = {actor_id: self.main_state(actor_id) for actor_id in self.eidolons}
        self.click_panel(f'[data-card-id="{new_card}"] button:text-is("新增")')
        self.confirm_action('新增角色卡', NEW_CARD_NAME, accept=False)
        self.assertFalse(self.gm.evaluate(EXISTS, new_card))
        self.assertEqual(self.main_state(old_id), old)
        self.click_panel(f'[data-card-id="{new_card}"] button:text-is("替换〈{old["name"]}〉")')
        self.assertFalse(self.gm.evaluate(EXISTS, new_card))
        self.assertEqual(self.main_state(old_id), old)
        self.confirm_action('替换角色卡', NEW_CARD_NAME, old['name'],
                            *(companion['name'] for companion in old_companions.values()), '归档')
        self.gm.wait_for_function(RETIRED, arg=old_id, timeout=WAIT_MS)

        created = self.main_state(new_card)
        p1_user = self.gm.evaluate(USER_ID, 'P1')
        self.assertEqual(created['ownership'][p1_user], 3)
        self.assertEqual(created['ownership']['default'], old['ownership']['default'])
        self.assertEqual(created['folder'], old['folder'])
        self.assertEqual(created['charId'], new_card)
        self.assertEqual(created['applied']['ref'], copy_ref(copy))
        retired = self.main_state(old_id)
        self.assertEqual((retired['folder'], retired['retired']), (ARCHIVE, True))
        for actor_id, companion in old_companions.items():
            self.gm.wait_for_function(RETIRED, arg=actor_id, timeout=WAIT_MS)
            archived = self.main_state(actor_id)
            self.assertEqual((archived['folder'], archived['retired']), (ARCHIVE, True))
            self.assertEqual(archived['itemIds'], companion['itemIds'])

        self.stop_main()
        self.run_engine()
        cards = self.read_published('meta.json')['cards']
        self.assertEqual((cards[new_card]['world'], cards[new_card]['mainId']), (CAMP_WORLD, new_card))
        self.assertNotIn(old_id, cards)
        for actor_id in old_companions:
            self.assertNotIn(actor_id, cards)

    def test_14_gm_imports_unsubmitted_char_card(self):
        old_id = self.master
        other = self.p1.evaluate(CREATE_CARD, OTHER_CARD_NAME)
        self.run_engine()
        working = self.read_published(f'working/{other}.json', 'main')
        self.env.start('main')
        self.gm_login()
        old = self.main_state(old_id)
        old_familiar = self.main_state(self.familiar)
        self.open_panel()
        self.click_panel(f'.taka-charsync-other[data-card-id="{other}"] button:text-is("替换〈{old["name"]}〉")')
        self.assertFalse(self.gm.evaluate(EXISTS, other))
        self.assertEqual(self.main_state(old_id), old)
        self.confirm_action('替换角色卡', OTHER_CARD_NAME, old['name'], old_familiar['name'], '归档')
        self.gm.wait_for_function(RETIRED, arg=old_id, timeout=WAIT_MS)

        created = self.main_state(other)
        self.assertEqual(created['ownership'][self.gm.evaluate(USER_ID, 'P1')], 3)
        self.assertEqual(created['charId'], other)
        self.assertEqual(created['applied']['ref'], {'kind': 'char', 'cardId': other, 'hash': working['hash']})
        retired = self.main_state(old_id)
        self.assertEqual((retired['folder'], retired['retired']), (ARCHIVE, True))
        self.gm.wait_for_function(RETIRED, arg=self.familiar, timeout=WAIT_MS)
        archived_familiar = self.main_state(self.familiar)
        self.assertEqual((archived_familiar['folder'], archived_familiar['retired']), (ARCHIVE, True))
        self.assertEqual(archived_familiar['itemIds'], old_familiar['itemIds'])
        self.assertTrue(self.main_file(f"history/char-{working['hash']}.json").is_file())

    def test_15_new_familiar_rolls_back_then_is_created_with_its_id(self):
        card = self.card
        level = self.main_state(card)['level'] + 1
        familiar = self.p1.evaluate(CREATE_FAMILIAR, [NEW_FAMILIAR_NAME, card])
        # 新魔宠的整卡导入一律失败：这时主人已经写过，要整张恢复
        self.gm.evaluate(BREAK_ACTOR_UPDATE, familiar)
        try:
            copy = self.submit_level(card, level)
            self.run_engine()
            self.gm.evaluate(FILL_XP, card)
            before = self.main_state(card)
            whispers = self.whisper_count(card)
            result = self.gm.evaluate(RUN_CHECK)
            after = self.main_state(card)
        finally:
            calls = self.gm.evaluate(RESTORE_ACTOR_UPDATE)
        self.assertGreaterEqual(calls, 1)
        entries = [item for item in result['pending'] if item['actorId'] == card]
        self.assertEqual([item['reason'] for item in entries], ['error'])
        self.assertIn('恢复到套用前', entries[0]['message'])
        for key in ('level', 'xp', 'hp', 'applied', 'undone', 'itemIds'):
            self.assertEqual(after[key], before[key], key)
        self.assertFalse(self.gm.evaluate(EXISTS, familiar))
        self.assertEqual(self.whisper_count(card), whispers)

        # 手动再套一次：魔宠以建卡服的 id 新建，主人是主服这张卡
        self.open_panel()
        self.click_panel(f'[data-actor-id="{card}"] [data-reason="error"] button:text-is("套用")')
        self.wait_whispers(card, whispers + 1)
        state = self.main_state(card)
        self.assertEqual((state['level'], state['xp']['value']), (level, 0))
        self.assertEqual(state['applied']['ref'], copy_ref(copy))
        created = self.main_state(familiar)
        self.assertEqual((created['master'], created['charId']), (card, familiar))
        self.assertTrue(created['hasMaster'])
        self.assertEqual(created['applied']['ref']['copyId'], copy['companions'][0]['copyId'])
        # 用例 11 套过的整卡副本不再出现
        result = self.gm.evaluate(RUN_CHECK)
        self.assertEqual([item for item in result['pending'] if item['actorId'] == card], [])

        # 引擎认它作同伴，不是独立主卡
        self.stop_main()
        self.run_engine()
        cards = self.read_published('meta.json')['cards']
        self.assertEqual((cards[familiar]['companionOf'], cards[familiar]['mainId']), (card, familiar))
        self.assertIn(familiar, cards[card]['companions'])

        self.env.start('main')
        self.gm_login()
        self.open_panel()
        self.click_panel(f'[data-actor-id="{card}"] button:text-is("恢复到套用前")')
        self.assertTrue(self.gm.evaluate(EXISTS, familiar))
        self.confirm_action('恢复到套用前', state['name'], NEW_FAMILIAR_NAME, '删除')
        self.gm.wait_for_function(EXISTS_NOT, arg=familiar, timeout=WAIT_MS)
        restored = self.main_state(card)
        self.assertEqual((restored['level'], restored['applied']), (before['level'], before['applied']))

    def test_16_no_module_errors(self):
        errors = [line for log_index, logs in enumerate(self.client_logs) for index, line in enumerate(logs)
                  if is_module_error(line) and self.expected_module_errors.get((log_index, index)) != line]
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
