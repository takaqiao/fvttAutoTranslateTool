"""建卡服提交冻结副本：升级副本的前驱链、同级重交与撤回更高级、副本锁、两个标签页同时提交、同伴一起冻结、
引擎发布副本、作为新卡提交、整卡同步、GM 清理、卡被重置后撤回未套用的副本、只有卡主能提交和撤回、撤回按钮、
放弃草稿不撤回副本，以及过不了校验没进副本的条目要提示。

用法：python -X utf8 test_submit.py
"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from harness import CAMP_WORLD, MODULE_ID, PUBLISH_DIR, Env, fixtures_dir

CHAIN_URL = f'/modules/{MODULE_ID}/scripts/lib/chain.mjs'
SUBMIT_URL = f'/modules/{MODULE_ID}/scripts/char/submit.mjs'
PUBLISHED_URL = f'/modules/{MODULE_ID}/scripts/foundry/published.mjs'
WAIT_MS = 60_000

FORCE = '整卡同步到主服'
NEW_CARD = '作为新卡提交'
WITHDRAW = '撤回'
LOCK_NOTICE = '已提交的副本不能修改，要改请改原卡后重新提交'
SAME_LEVEL_FEAT = 'E2E 同级改动'
RETRAINED_FEAT = 'E2E 重训专长'
NEW_CARD_NAME = 'E2E 新卡'
SUBMITTED_FEAT = 'E2E 提交过的专长'
DISCARDED_FEAT = 'E2E 放弃的专长'
DISCARD = '放弃草稿，恢复成主服版'

# 建卡服默认只有助理 GM 以上能建 actor；线上 pf2creation 放开给了玩家，提交副本要用到
GRANT_ACTOR_CREATE = """async () => {
  const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
  const {PLAYER, TRUSTED, ASSISTANT, GAMEMASTER} = CONST.USER_ROLES;
  permissions.ACTOR_CREATE = [PLAYER, TRUSTED, ASSISTANT, GAMEMASTER];
  await game.settings.set('core', 'permissions', permissions);
}"""

SET_LEVEL = """async ([id, level]) => { await game.actors.get(id).update({'system.details.level.value': level}); }"""

# diff:false 时 PF2e 不补职业特性：无头浏览器里没人点 ChoiceSet 的弹窗，这里只要等级变了
SET_MAIN_LEVEL = """async ([id, level]) => {
  await game.actors.get(id).update({'system.details.level.value': level}, {diff: false});
}"""

# 模拟主服套用、忽略副本（套用本身是主服模式的事）：每项是 [卡 id, 要写的改动]
UPDATE_MAIN = """async updates => {
  for (const [id, changes] of updates) await game.actors.get(id).update(changes, {diff: false});
}"""

ADD_FEAT = """async ([actorId, name]) => {
  const [item] = await game.actors.get(actorId).createEmbeddedDocuments('Item', [
    {name, type: 'feat', system: {category: 'general', level: {value: 1}}},
  ]);
  return item.id;
}"""

CREATE_CARD = """async name => (await Actor.create({name, type: 'character', folder: 'CharsyncCamp0001'})).id"""

COPY_ID = """async ([url, cardId, level]) => (await import(url)).copyIdFor(cardId, level)"""
FORCE_COPY_ID = """async ([url, cardId]) => (await import(url)).forceCopyIdFor(cardId)"""

ACTOR_STATE = """id => {
  const actor = game.actors.get(id);
  if (!actor) return null;
  const own = actor._source.flags['taka-charsync'] ?? {};
  return {
    name: actor.name,
    tokenName: actor._source.prototypeToken.name,
    role: actor.folder?.getFlag('taka-charsync', 'role') ?? null,
    copy: own.copy ?? null,
    base: own.base ?? null,
    level: actor._source.system.details?.level?.value ?? null,
    master: actor._source.system.master?.id ?? null,
    itemNames: actor._source.items.map(item => item.name),
    itemCount: actor.items.size,
  };
}"""

COPIES_OF = """cardId => game.actors
  .filter(actor => actor._source.flags['taka-charsync']?.copy?.cardId === cardId)
  .map(actor => actor.id)"""

LEVEL_COPY_COUNT = """([cardId, level]) => game.actors.filter(actor => {
  const copy = actor._source.flags['taka-charsync']?.copy;
  return copy?.cardId === cardId && copy.kind === 'level' && copy.level === level;
}).length"""

ITEM_COUNTS = """ids => ids.map(id => game.actors.get(id)?.items.size ?? null)"""

EXISTING = """ids => ids.filter(id => game.actors.has(id))"""

ALL_WITHDRAWN = """ids => ids.every(id => game.actors.get(id)?._source.flags['taka-charsync']?.copy?.withdrawn === true)"""

BACKUP_COUNT = """id => game.actors.filter(actor => actor._source.flags['taka-charsync']?.backupOf === id).length"""

BASE_IS = """([id, hash]) => game.actors.get(id)?._source.flags['taka-charsync']?.base?.hash === hash"""

LOCKED_EDITS = """async id => {
  const actor = game.actors.get(id);
  const updated = await actor.update({'system.details.level.value': 9});
  const created = await actor.createEmbeddedDocuments('Item', [{name: 'E2E 锁住的专长', type: 'feat'}]);
  return {updated: Boolean(updated), created: created.length, level: actor._source.system.details.level.value, items: actor.items.size};
}"""

RENAME = """async ([id, name]) => {
  const actor = game.actors.get(id);
  await actor.update({name});
  return actor.name;
}"""

SUBMIT_LEVEL = """async ([url, id]) => (await import(url)).submitLevel(game.actors.get(id))"""
SUBMIT_FORCE = """async ([url, id]) => (await import(url)).submitForce(game.actors.get(id))"""
SUBMIT_NEW = """async ([url, id, world]) => (await import(url)).submitNew(game.actors.get(id), {targetWorld: world})"""
WITHDRAW_COPY = """async ([url, id]) => { await (await import(url)).withdraw(game.actors.get(id)); }"""

# 两个标签页约定同一时刻调用：模块先载好，到点再提交，结果挂在 window 上
START_SUBMIT = """([url, id, at]) => {
  window.e2eSubmit = import(url)
    .then(module => new Promise(resolve => setTimeout(resolve, at - Date.now()))
      .then(() => module.submitLevel(game.actors.get(id))))
    .then(copy => ({copy}), error => ({error: String(error)}));
}"""
SUBMIT_RESULT = """() => window.e2eSubmit"""

CLEANUP = """async ([submitUrl, publishedUrl]) => {
  const [{cleanupCopies}, {getMeta}] = await Promise.all([import(submitUrl), import(publishedUrl)]);
  try {
    return await cleanupCopies(await getMeta());
  } catch (error) {
    return {error: String(error)};
  }
}"""

NOT_OWNER_CALLS = """async ([url, id, copyId, newCardId, world]) => {
  const {submitLevel, submitForce, submitNew, withdraw} = await import(url);
  const actor = game.actors.get(id);
  return {
    level: await submitLevel(actor),
    force: await submitForce(actor),
    withdraw: (await withdraw(game.actors.get(copyId))) ?? null,
    new: await submitNew(game.actors.get(newCardId), {targetWorld: world}),
  };
}"""

ALL_COPIES = """() => Object.fromEntries(game.actors
  .filter(actor => actor._source.flags['taka-charsync']?.copy)
  .map(actor => [actor.id, actor._source.flags['taka-charsync'].copy]))"""

FIRST_FEAT = """id => {
  const feat = game.actors.get(id).items.find(item => item.type === 'feat');
  return feat ? {id: feat.id, name: feat.name} : null;
}"""
ITEM_LEVEL = """([id, itemId]) => game.actors.get(id)._source.items.find(item => item._id === itemId)?.system.level.value"""

# 模拟卡上有一条过不了校验的条目：只改本页内存里的来源数据，提交完换回去，不写库
SUBMIT_WITH_INVALID_ITEM = """async ([url, id, itemId]) => {
  const source = game.actors.get(id)._source.items.find(item => item._id === itemId);
  const original = source.system.level.value;
  source.system.level.value = 'abc';
  try {
    return await (await import(url)).submitLevel(game.actors.get(id));
  } finally {
    source.system.level.value = original;
  }
}"""

SHOW_ACTORS = """() => ui.actors.activate()"""
HAS_CLEANUP_BUTTON = """() => Boolean(document.querySelector('.taka-charsync-cleanup button'))"""

OPEN_SHEET = """id => { game.actors.get(id).sheet.render(true); }"""
CLOSE_SHEET = """async id => { await game.actors.get(id).sheet.close(); }"""
SHEET_ID = """id => game.actors.get(id).sheet.id"""
BAR_SELECTOR = "game.actors.get(id).sheet.element?.[0]?.querySelector('.window-content > .taka-charsync-bar')"
BAR_SETTLED = f"""id => {{
  const bar = {BAR_SELECTOR};
  return Boolean(bar) && bar.dataset.state !== 'loading';
}}"""
BAR_IN_STATE = f"""([id, state]) => {BAR_SELECTOR}?.dataset.state === state"""
BAR = f"""id => {{
  const bar = {BAR_SELECTOR};
  return bar && {{state: bar.dataset.state, text: bar.innerText, buttons: [...bar.querySelectorAll('button')].map(b => b.textContent)}};
}}"""

NOTICE_HAS = """text => document.querySelector('#notifications')?.innerText.includes(text) ?? false"""
DIALOG = '.application.dialog'


def is_module_error(line: str) -> bool:
    return line.startswith(('error:', 'pageerror:')) and MODULE_ID in line


def ref_key(copy: dict) -> str:
    return f"copy:{copy['copyId']}:{copy['rev']}"


def copy_ref(copy: dict) -> dict:
    return {'kind': 'copy', 'copyId': copy['copyId'], 'rev': copy['rev']}


class SubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-e2e-')), fixtures_dir())
        cls.addClassCleanup(env.close)
        actors = env.actors

        def level(card_id):
            return actors[card_id]['system'].get('details', {}).get('level', {}).get('value')

        def casts(card_id):
            return any(item['type'] == 'spellcastingEntry' for item in actors[card_id]['items'])

        cls.card = next(c for c in env.main_cards if level(c) == 6 and not casts(c))
        same_level = next(card for card in env.seed_cards if card['charId'] != cls.card and level(card['charId']) == 6)
        cls.same_level = same_level['charId']
        cls.same_level_companions = same_level.get('companions', [])
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
        cls.login_p1()
        cls.p1_user = cls.p1.evaluate('() => game.user.id')
        meta = cls.read_published('meta.json')
        for actor_id in (cls.card, cls.same_level, *cls.same_level_companions, cls.master, cls.familiar):
            cls.p1.wait_for_function(BASE_IS, arg=[actor_id, meta['bootstrap'][actor_id]['hash']], timeout=WAIT_MS)
        cls.copy_ids = {}

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

    def login(self, user):
        page, logs = self.env.login('char', user)
        self.client_logs.append(logs)
        return page

    @classmethod
    def read_published(cls, rel):
        return json.loads((cls.env.data_dir('char') / PUBLISH_DIR / rel).read_text(encoding='utf-8'))

    def published_exists(self, rel):
        return (self.env.data_dir('char') / PUBLISH_DIR / rel).is_file()

    def latest(self, actor_id):
        return self.read_published(f'main/{CAMP_WORLD}/{actor_id}.json')

    def edit_main(self, script, arg):
        """起主服世界，以 GM 跑 `script`，再停掉世界、跑引擎一轮。"""
        self.env.start('main')
        page, _ = self.env.login('main', 'Gamemaster')
        try:
            page.evaluate(script, arg)
        finally:
            page.context.close()
        self.env.stop('main')
        self.run_engine()

    def state(self, actor_id, page=None):
        return (page or self.p1).evaluate(ACTOR_STATE, actor_id)

    def copy_id(self, card_id, level):
        key = (card_id, level)
        if key not in self.copy_ids:
            self.copy_ids[key] = self.p1.evaluate(COPY_ID, [CHAIN_URL, card_id, level])
        return self.copy_ids[key]

    def open_sheet(self, actor_id, state=None, page=None):
        page = page or self.p1
        page.evaluate(OPEN_SHEET, actor_id)
        if state:
            page.wait_for_function(BAR_IN_STATE, arg=[actor_id, state], timeout=WAIT_MS)
        else:
            page.wait_for_function(BAR_SETTLED, arg=actor_id, timeout=WAIT_MS)
        return page.evaluate(BAR, actor_id)

    def close_sheet(self, actor_id, page=None):
        (page or self.p1).evaluate(CLOSE_SHEET, actor_id)

    def click_bar(self, actor_id, label, page=None):
        page = page or self.p1
        # 常驻通知会压在卡面按钮上
        page.evaluate('() => ui.notifications.clear()')
        sheet = page.evaluate(SHEET_ID, actor_id)
        bar = page.locator(f'#{sheet} .taka-charsync-bar')
        if label in (FORCE, DISCARD):
            secondary = bar.locator('.taka-charsync-secondary')
            if secondary.get_attribute('open') is None:
                secondary.locator(':scope > summary').click()
        bar.get_by_role('button', name=label, exact=True).click()

    def wait_notice(self, text, page=None):
        (page or self.p1).wait_for_function(NOTICE_HAS, arg=text, timeout=WAIT_MS)

    def submit(self, actor_id, level):
        same_level = level == self.latest(actor_id)['level']
        kind = '本级修改' if same_level else f'{level} 级升级'
        self.click_bar(actor_id, '提交本级修改' if same_level else f'提交 {level} 级升级')
        self.wait_notice(f'已提交{self.state(actor_id)["name"]}的{kind}')
        self.p1.wait_for_function(BAR_IN_STATE, arg=[actor_id, 'submitted'], timeout=WAIT_MS)
        self.assertIn('待 GM 确认' if same_level else '待主服升级', self.p1.evaluate(BAR, actor_id)['text'])

    def test_01_level_copies_chain_onto_unabsorbed_previous_level(self):
        card = self.state(self.card)
        self.p1.evaluate(SET_LEVEL, [self.card, 7])
        self.open_sheet(self.card, state='draft')
        self.submit(self.card, 7)
        self.p1.evaluate(SET_LEVEL, [self.card, 8])
        self.submit(self.card, 8)
        self.close_sheet(self.card)

        copy7, copy8 = (self.state(self.copy_id(self.card, level)) for level in (7, 8))
        for level, copy in ((7, copy7), (8, copy8)):
            self.assertEqual(copy['role'], 'submitted')
            self.assertEqual(copy['name'], f"{card['name']}〔已提交 {level} 级〕")
            self.assertEqual(copy['tokenName'], card['tokenName'])
            self.assertIsNone(copy['base'])
            meta = copy['copy']
            self.assertEqual(meta['copyId'], self.copy_id(self.card, level))
            self.assertEqual((meta['cardId'], meta['level'], meta['kind']), (self.card, level, 'level'))
            self.assertEqual((meta['targetWorld'], meta['replaces'], meta['withdrawn']), (CAMP_WORLD, None, False))
            self.assertEqual((meta['name'], meta['submittedBy']), (card['name'], self.p1_user))
            self.assertEqual((meta['companionOf'], meta['companions']), (None, []))
        self.assertEqual(copy7['copy']['pred'], card['base'])
        self.assertEqual(copy8['copy']['pred'], copy_ref(copy7['copy']))
        self.assertEqual(set(self.p1.evaluate(COPIES_OF, self.card)), {copy7['copy']['copyId'], copy8['copy']['copyId']})

        # 和主服同级的提交：前驱是卡的基线，不是任何副本；同伴随主人一起冻结
        same_level = self.state(self.same_level)
        self.p1.evaluate(ADD_FEAT, [self.same_level, SAME_LEVEL_FEAT])
        self.open_sheet(self.same_level, state='draft')
        self.submit(self.same_level, 6)
        self.close_sheet(self.same_level)
        copy6 = self.state(self.copy_id(self.same_level, 6))['copy']
        self.assertEqual(copy6['pred'], same_level['base'])
        companions = []
        for companion_id in self.same_level_companions:
            companion_copy = self.state(self.copy_id(companion_id, 6))['copy']
            self.assertEqual(companion_copy['pred'], self.state(companion_id)['base'])
            self.assertEqual(companion_copy['companionOf'], copy6['copyId'])
            companions.append({'copyId': companion_copy['copyId'], 'rev': companion_copy['rev'], 'sourceId': companion_id})
        self.assertEqual(copy6['companions'], companions)

    def test_02_resubmitting_a_level_replaces_it_and_withdraws_higher(self):
        copy7_id, copy8_id = self.copy_id(self.card, 7), self.copy_id(self.card, 8)
        before = self.state(copy7_id)['copy']
        self.p1.evaluate(SET_LEVEL, [self.card, 7])
        self.p1.evaluate(ADD_FEAT, [self.card, RETRAINED_FEAT])
        self.open_sheet(self.card, state='submitted')
        self.submit(self.card, 7)
        self.close_sheet(self.card)

        copy7 = self.state(copy7_id)
        self.assertGreater(copy7['copy']['rev'], before['rev'])
        self.assertEqual(copy7['level'], 7)
        self.assertIn(RETRAINED_FEAT, copy7['itemNames'])
        self.assertEqual(copy7['itemCount'], self.state(self.card)['itemCount'])
        self.assertEqual(copy7['role'], 'submitted')
        copy8 = self.state(copy8_id)
        self.assertTrue(copy8['copy']['withdrawn'])
        self.assertEqual(copy8['role'], 'withdrawn')
        self.assertEqual(set(self.p1.evaluate(COPIES_OF, self.card)), {copy7_id, copy8_id})

    def test_03_copies_are_locked_for_players(self):
        copy7_id, copy8_id = self.copy_id(self.card, 7), self.copy_id(self.card, 8)
        before = self.state(copy7_id)
        self.p1.evaluate('() => ui.notifications.clear()')
        result = self.p1.evaluate(LOCKED_EDITS, copy7_id)
        self.assertEqual(result, {'updated': False, 'created': 0, 'level': 7, 'items': before['itemCount']})
        self.wait_notice(LOCK_NOTICE)

        gm = self.login('Gamemaster')
        try:
            renamed = gm.evaluate(RENAME, [copy8_id, 'E2E GM 改过的副本'])
        finally:
            gm.context.close()
        self.assertEqual(renamed, 'E2E GM 改过的副本')

    def test_04_two_tabs_submitting_together_leave_one_copy(self):
        second = self.login('P1')
        try:
            at = int(time.time() * 1000) + 3000
            for page in (self.p1, second):
                page.evaluate(START_SUBMIT, [SUBMIT_URL, self.master, at])
            results = [page.evaluate(SUBMIT_RESULT) for page in (self.p1, second)]
        finally:
            second.context.close()
        for result in results:
            self.assertIn('copy', result, result)

        copy_id = self.copy_id(self.master, self.master_level)
        familiar_copy_id = self.copy_id(self.familiar, self.master_level)
        gm = self.login('Gamemaster')
        try:
            count = gm.evaluate(LEVEL_COPY_COUNT, [self.master, self.master_level])
            items = gm.evaluate(ITEM_COUNTS, [copy_id, self.master, familiar_copy_id, self.familiar])
        finally:
            gm.context.close()
        self.assertEqual(count, 1)
        self.assertEqual(items[0], items[1])
        self.assertEqual(items[2], items[3])

    def test_05_familiar_is_frozen_with_its_master(self):
        copy_id = self.copy_id(self.master, self.master_level)
        familiar_copy_id = self.copy_id(self.familiar, self.master_level)
        master_copy = self.p1.evaluate(SUBMIT_LEVEL, [SUBMIT_URL, self.master])
        familiar_copy = self.state(familiar_copy_id)
        self.assertEqual(familiar_copy['master'], copy_id)
        self.assertEqual(familiar_copy['role'], 'submitted')
        meta = familiar_copy['copy']
        self.assertEqual((meta['cardId'], meta['companionOf'], meta['level']), (self.familiar, copy_id, self.master_level))
        self.assertEqual(meta['pred'], self.state(self.familiar)['base'])
        self.assertEqual(master_copy, self.state(copy_id)['copy'])
        self.assertIn({'copyId': familiar_copy_id, 'rev': meta['rev'], 'sourceId': self.familiar}, master_copy['companions'])
        self.assertEqual(self.state(self.familiar)['master'], self.master)

    def test_06_engine_publishes_copies_and_history(self):
        self.run_engine()
        meta = self.read_published('meta.json')
        expected = {
            self.card: [7, 8],
            self.same_level: [6],
            **{companion_id: [6] for companion_id in self.same_level_companions},
            self.master: [self.master_level],
            self.familiar: [self.master_level],
        }
        for card_id, levels in expected.items():
            listed = {copy['copyId']: copy for copy in self.read_published(f'copies/{card_id}.json')}
            for level in levels:
                local = self.state(self.copy_id(card_id, level))['copy']
                self.assertEqual(listed.get(local['copyId']), local)
                self.assertTrue(self.published_exists(f"history/copy-{local['copyId']}-{local['rev']}.json"))
                self.assertIn(ref_key(local), meta['published'])
        withdrawn = self.read_published(f'copies/{self.card}.json')
        self.assertTrue(next(copy for copy in withdrawn if copy['copyId'] == self.copy_id(self.card, 8))['withdrawn'])

    def test_07_new_card_is_submitted_to_a_campaign(self):
        new_card = type(self).new_card = self.p1.evaluate(CREATE_CARD, NEW_CARD_NAME)
        level = self.state(new_card)['level']
        self.open_sheet(new_card, state='unmapped')
        self.click_bar(new_card, NEW_CARD)
        dialog = self.p1.locator(DIALOG)
        dialog.locator('select[name="world"]').select_option(CAMP_WORLD)
        replaceable = dialog.locator('select[name="replaces"] option').evaluate_all('options => options.map(o => o.value)')
        self.assertEqual(set(replaceable), {'', *self.env.main_cards})
        dialog.locator('select[name="replaces"]').select_option(self.same_level)
        dialog.locator('button[data-action="ok"]').click()
        self.wait_notice('作为新卡提交到')
        self.p1.wait_for_function(BAR_IN_STATE, arg=[new_card, 'submitted'], timeout=WAIT_MS)
        self.assertIn('待 GM 确认', self.p1.evaluate(BAR, new_card)['text'])
        self.close_sheet(new_card)

        copy = self.state(self.copy_id(new_card, level))
        self.assertEqual(copy['role'], 'submitted')
        meta = copy['copy']
        self.assertEqual((meta['kind'], meta['targetWorld'], meta['replaces']), ('new', CAMP_WORLD, self.same_level))
        self.assertEqual((meta['cardId'], meta['level'], meta['pred']), (new_card, level, None))

    def test_08_force_sync_keeps_one_copy_per_card(self):
        force_id = type(self).force_id = self.p1.evaluate(FORCE_COPY_ID, [CHAIN_URL, self.card])
        copy7_id = self.copy_id(self.card, 7)
        copy7_before = self.state(copy7_id)['copy']
        card = self.state(self.card)
        self.open_sheet(self.card)
        revs = []
        for _ in range(2):
            self.click_bar(self.card, FORCE)
            dialog = self.p1.locator(DIALOG)
            self.assertIn('这会在主服整张覆盖，血量、物品、金币也换成这里的；主服经验保留。'
                          '主人和同伴一起提交，要 GM 点套用才生效。', dialog.inner_text())
            dialog.locator('button[data-action="yes"]').click()
            self.wait_notice('整卡同步')
            revs.append(self.state(force_id)['copy']['rev'])
        self.close_sheet(self.card)

        self.assertGreater(revs[1], revs[0])
        force = self.state(force_id)
        self.assertEqual(force['name'], f"{card['name']}〔整卡同步〕")
        meta = force['copy']
        self.assertEqual((meta['kind'], meta['pred'], meta['level'], meta['cardId']), ('force', None, card['level'], self.card))
        forces = [copy_id for copy_id in self.p1.evaluate(COPIES_OF, self.card) if self.state(copy_id)['copy']['kind'] == 'force']
        self.assertEqual(forces, [force_id])
        self.assertEqual(self.state(copy7_id)['copy'], copy7_before)

        self.run_engine()
        listed = {copy['copyId']: copy for copy in self.read_published(f'copies/{self.card}.json')}
        self.assertEqual(listed.get(force_id), meta)

    def test_09_cleanup_deletes_only_processed_copies(self):
        # 降到 6 级交一份升级副本和一份整卡副本，再升回 7 级重交；主服套用 7 级后，6 级的升级副本低于主服、该删，
        # 同样是 6 级的整卡副本还在等 GM 决定、不删
        self.p1.evaluate(SET_LEVEL, [self.card, 6])
        copy6 = self.p1.evaluate(SUBMIT_LEVEL, [SUBMIT_URL, self.card])
        force = self.p1.evaluate(SUBMIT_FORCE, [SUBMIT_URL, self.card])
        self.p1.evaluate(SET_LEVEL, [self.card, 7])
        copy7 = self.p1.evaluate(SUBMIT_LEVEL, [SUBMIT_URL, self.card])
        # 主人卡的整卡副本被主服忽略，魔宠的整卡副本随它删
        master_force = self.p1.evaluate(SUBMIT_FORCE, [SUBMIT_URL, self.master])
        familiar_force = self.state(self.p1.evaluate(FORCE_COPY_ID, [CHAIN_URL, self.familiar]))['copy']
        applied = {'ref': copy_ref(copy7), 'at': int(time.time() * 1000), 'undoPath': 'e2e', 'level': 7}
        self.edit_main(UPDATE_MAIN, [
            [self.card, {'system.details.level.value': 7, 'flags.taka-charsync.applied': applied}],
            [self.master, {'flags.taka-charsync.undone': [ref_key(master_force)]}],
        ])
        # 引擎读过之后重交又撤回的新卡副本：撤回了，但这一版还没发布
        new_card_copy = self.p1.evaluate(SUBMIT_NEW, [SUBMIT_URL, self.new_card, CAMP_WORLD])
        self.p1.evaluate(WITHDRAW_COPY, [SUBMIT_URL, new_card_copy['copyId']])

        published = self.read_published('meta.json')['published']
        for copy in (copy6, force, copy7, master_force, familiar_force):
            self.assertIn(ref_key(copy), published)
        self.assertNotIn(ref_key(new_card_copy), published)
        self.assertEqual((force['copyId'], force['level'], force['withdrawn']), (self.force_id, 6, False))
        self.assertFalse(self.state(copy6['copyId'])['copy']['withdrawn'])
        self.assertTrue(self.state(new_card_copy['copyId'])['copy']['withdrawn'])
        self.assertEqual(familiar_force['companionOf'], master_force['copyId'])

        processed = [
            copy7['copyId'], self.copy_id(self.card, 8), copy6['copyId'], master_force['copyId'], familiar_force['copyId'],
        ]
        pending = [
            self.force_id,
            self.copy_id(self.same_level, 6),
            *(self.copy_id(companion_id, 6) for companion_id in self.same_level_companions),
            self.copy_id(self.master, self.master_level),
            self.copy_id(self.familiar, self.master_level),
            new_card_copy['copyId'],
        ]
        gm = self.login('Gamemaster')
        try:
            gm.evaluate(SHOW_ACTORS)
            gm.click('.taka-charsync-cleanup button')
            gm.locator(DIALOG).locator('button[data-action="yes"]').click()
            self.wait_notice(f'已删除 {len(processed)} 份副本', page=gm)
            remaining = gm.evaluate(EXISTING, processed + pending)
            again = gm.evaluate(CLEANUP, [SUBMIT_URL, PUBLISHED_URL])
        finally:
            gm.context.close()
        self.assertEqual(remaining, pending)
        self.assertEqual(again, {'deleted': []})

    def test_10_reset_withdraws_unapplied_copies(self):
        backups = self.p1.evaluate(BACKUP_COUNT, self.same_level)
        self.edit_main(SET_MAIN_LEVEL, [self.same_level, 7])
        latest = self.latest(self.same_level)

        self.login_p1()
        self.p1.wait_for_function(BASE_IS, arg=[self.same_level, latest['hash']], timeout=WAIT_MS)
        copies = [copy_id for card_id in (self.same_level, *self.same_level_companions)
                  for copy_id in self.p1.evaluate(COPIES_OF, card_id)]
        self.assertEqual(len(copies), 1 + len(self.same_level_companions))
        self.p1.wait_for_function(ALL_WITHDRAWN, arg=copies, timeout=WAIT_MS)
        self.assertEqual([self.state(copy_id)['role'] for copy_id in copies], ['withdrawn'] * len(copies))
        self.assertEqual(self.p1.evaluate(BACKUP_COUNT, self.same_level), backups + 1)

    def test_11_only_owners_submit_and_only_gm_cleans_up(self):
        copy_id = self.copy_id(self.master, self.master_level)
        calls = [SUBMIT_URL, self.master, copy_id, self.new_card, CAMP_WORLD]
        before = self.p1.evaluate(ALL_COPIES)
        player_bar = self.open_sheet(self.card)
        self.close_sheet(self.card)
        self.assertIn(FORCE, player_bar['buttons'])
        self.p1.evaluate(SHOW_ACTORS)
        self.assertFalse(self.p1.evaluate(HAS_CLEANUP_BUTTON))
        self.assertIn('只有 GM', self.p1.evaluate(CLEANUP, [SUBMIT_URL, PUBLISHED_URL])['error'])

        # P2 不是这些卡的卡主，也不是副本的提交者
        p2 = self.login('P2')
        try:
            p2_calls = p2.evaluate(NOT_OWNER_CALLS, calls)
        finally:
            p2.context.close()
        gm = self.login('Gamemaster')
        try:
            gm_bar = self.open_sheet(self.card, page=gm)
            gm_calls = gm.evaluate(NOT_OWNER_CALLS, calls)
            after = gm.evaluate(ALL_COPIES)
        finally:
            gm.context.close()
        nothing = {'level': None, 'force': None, 'withdraw': None, 'new': None}
        self.assertEqual(p2_calls, nothing)
        self.assertEqual(gm_calls, nothing)
        self.assertIn('刷新状态', gm_bar['buttons'])
        self.assertEqual(set(gm_bar['buttons']) - {'刷新状态', '查看提交', '打开主人卡', '打开同伴卡'}, set())
        self.assertEqual(after, before)

    def test_12_withdraw_button_withdraws_copy_with_its_companions(self):
        copy_id = self.copy_id(self.master, self.master_level)
        familiar_copy_id = self.copy_id(self.familiar, self.master_level)
        bar = self.open_sheet(copy_id, state='copy')
        self.assertIn(WITHDRAW, bar['buttons'])
        self.click_bar(copy_id, WITHDRAW)
        dialog = self.p1.locator(DIALOG)
        self.assertIn('这份副本、随它提交的同伴及本卡更高级的同类副本会一起标记撤回', dialog.inner_text())
        self.assertIn('撤回需要等同步引擎发布后主服才会看到', dialog.inner_text())
        self.assertIn('已经套用的结果不会自动撤销', dialog.inner_text())
        dialog.locator('button[data-action="yes"]').click()
        self.wait_notice('已标记撤回，等待同步引擎发布到主服')
        self.p1.wait_for_function(ALL_WITHDRAWN, arg=[copy_id, familiar_copy_id], timeout=WAIT_MS)
        self.p1.wait_for_function(BAR_IN_STATE, arg=[copy_id, 'copy'], timeout=WAIT_MS)
        self.assertIn('已撤回', self.p1.evaluate(BAR, copy_id)['text'])
        self.close_sheet(copy_id)
        self.assertEqual([self.state(i)['role'] for i in (copy_id, familiar_copy_id)], ['withdrawn', 'withdrawn'])

    def test_13_discarding_a_draft_keeps_submitted_copies(self):
        latest = self.latest(self.card)
        self.p1.wait_for_function(BASE_IS, arg=[self.card, latest['hash']], timeout=WAIT_MS)
        backups = self.p1.evaluate(BACKUP_COUNT, self.card)
        self.p1.evaluate(ADD_FEAT, [self.card, SUBMITTED_FEAT])
        bar = self.open_sheet(self.card, state='submitted')
        self.assertIn('当前草稿', bar['text'])
        self.submit(self.card, 7)
        self.p1.evaluate(ADD_FEAT, [self.card, DISCARDED_FEAT])
        self.click_bar(self.card, DISCARD)
        self.p1.locator(DIALOG).locator('button[data-action="yes"]').click()
        self.wait_notice('已放弃')
        self.close_sheet(self.card)

        card = self.state(self.card)
        self.assertNotIn(SUBMITTED_FEAT, card['itemNames'])
        self.assertNotIn(DISCARDED_FEAT, card['itemNames'])
        self.assertEqual(card['base']['hash'], latest['hash'])
        self.assertEqual(self.p1.evaluate(BACKUP_COUNT, self.card), backups + 1)
        copy7 = self.state(self.copy_id(self.card, 7))
        self.assertIn(SUBMITTED_FEAT, copy7['itemNames'])
        for copy in (copy7, self.state(self.force_id)):
            self.assertFalse(copy['copy']['withdrawn'])
            self.assertEqual(copy['role'], 'submitted')

    def test_14_entries_left_out_of_a_copy_are_reported(self):
        copy_id = self.copy_id(self.card, 7)
        feat = self.p1.evaluate(FIRST_FEAT, self.card)
        before = self.p1.evaluate(ITEM_LEVEL, [copy_id, feat['id']])
        rev = self.state(copy_id)['copy']['rev']
        self.p1.evaluate('() => ui.notifications.clear()')
        log_offsets = [len(logs) for logs in self.client_logs]
        copy = self.p1.evaluate(SUBMIT_WITH_INVALID_ITEM, [SUBMIT_URL, self.card, feat['id']])
        self.assertGreater(copy['rev'], rev)
        self.wait_notice('没有进副本')
        self.assertTrue(self.p1.evaluate(NOTICE_HAS, feat['name']))
        self.assertEqual(self.p1.evaluate(ITEM_LEVEL, [copy_id, feat['id']]), before)
        # 写入副本会触发团规重算，此时工作卡的源里仍是本测试注入的非法等级。
        errors = [(log_index, index, line) for log_index, logs in enumerate(self.client_logs)
                  for index, line in enumerate(logs) if index >= log_offsets[log_index] and is_module_error(line)]
        self.assertEqual(len(errors), 1, errors)
        log_index, index, line = errors[0]
        parts = line.splitlines()
        self.assertEqual(parts[:3], [
            f'error: {MODULE_ID} | 按更新后的团规重算角色失败 {self.card} '
            'DataModelValidationError: [FeatSystemData] validation errors: SchemaField#_validateRecursive',
            '  level: SchemaField#_validateRecursive',
            '    value: must be a number',
        ])
        self.assertTrue(parts[3:] and all(part.startswith('    at ') for part in parts[3:]), line)
        self.expected_module_errors[(log_index, index)] = line

    def test_15_no_module_errors(self):
        errors = [line for log_index, logs in enumerate(self.client_logs) for index, line in enumerate(logs)
                  if is_module_error(line) and self.expected_module_errors.get((log_index, index)) != line]
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
