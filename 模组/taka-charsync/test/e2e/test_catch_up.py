"""建卡服跟上主服与卡面状态条：两个本机实例种好夹具卡，主服 GM 改卡、引擎发布后，以 P1 登录建卡服，
验证三方合并、草稿备份 + 重置、打开卡面时跟上、放弃草稿、Actor 团规、GM 不跟上玩家的卡和规则版本守卫。

用法：python -X utf8 test_catch_up.py
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from harness import CAMP_WORLD, MODULE_ID, PUBLISH_DIR, READY_TIMEOUT_MS, Env, fixtures_dir

OVERRIDE_URL = f'/modules/{MODULE_ID}/scripts/char/rules-override.mjs'
MAIN_URL = f'/modules/{MODULE_ID}/scripts/main.mjs'
CATCH_UP_URL = f'/modules/{MODULE_ID}/scripts/char/catch-up.mjs'
PUBLISHED_URL = f'/modules/{MODULE_ID}/scripts/foundry/published.mjs'
CAMP_TITLE = 'E2E 测试团'
WAIT_MS = 60_000

DRAFT_FEAT = 'E2E 草稿专长'
DISCARDED_FEAT = 'E2E 放弃的专长'
GM_SEEN_FEAT = 'E2E GM 看到的草稿'
DISCARD = '放弃草稿，恢复成主服版'
LOOT = 'E2E 战利品'
OTHER_MASTER = 'e2eOtherMaster01'

# 建卡服默认只有助理 GM 以上能建 actor；线上 pf2creation 放开给了玩家，草稿备份要用到
GRANT_ACTOR_CREATE = """async () => {
  const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
  const {PLAYER, TRUSTED, ASSISTANT, GAMEMASTER} = CONST.USER_ROLES;
  permissions.ACTOR_CREATE = [PLAYER, TRUSTED, ASSISTANT, GAMEMASTER];
  await game.settings.set('core', 'permissions', permissions);
}"""

MAIN_PLAY_SESSION = """async ([cardId, familiarId, loot]) => {
  const actor = game.actors.get(cardId);
  const hp = actor._source.system.attributes.hp.value - 5;
  await actor.update({'system.attributes.hp.value': hp});
  const consumables = actor.items.filter(item => item.type === 'consumable');
  const potion = consumables.find(item => item._source.system.quantity >= 2) ?? consumables[0];
  const quantity = potion._source.system.quantity - 1;
  await potion.update({'system.quantity': quantity});
  const [treasure] = await actor.createEmbeddedDocuments('Item', [{name: loot, type: 'treasure', system: {quantity: 1}}]);
  const familiar = game.actors.get(familiarId);
  const familiarHp = familiar._source.system.attributes.hp.value - 2;
  await familiar.update({'system.attributes.hp.value': familiarHp});
  return {hp, potionId: potion.id, quantity, lootId: treasure.id, familiarHp};
}"""

DELETE_ITEM = """async ([actorId, itemId]) => { await game.actors.get(actorId).deleteEmbeddedDocuments('Item', [itemId]); }"""

UPDATE_ACTOR = """async ([actorId, changes]) => { await game.actors.get(actorId).update(changes); }"""

LOSE_HP = """async ([actorId, loss]) => {
  const actor = game.actors.get(actorId);
  const hp = actor._source.system.attributes.hp.value - loss;
  await actor.update({'system.attributes.hp.value': hp});
  return hp;
}"""

ADD_FEAT = """async ([actorId, name]) => {
  const [item] = await game.actors.get(actorId).createEmbeddedDocuments('Item', [
    {name, type: 'feat', system: {category: 'general', level: {value: 1}}},
  ]);
  return item.id;
}"""

CARD_STATE = """id => {
  const actor = game.actors.get(id);
  const own = actor._source.flags['taka-charsync'] ?? {};
  return {
    name: actor.name,
    hp: actor._source.system.attributes.hp.value,
    level: actor._source.system.details?.level?.value ?? null,
    master: actor._source.system.master?.id ?? null,
    base: own.base ?? null,
    conflicts: own.conflicts ?? null,
    items: Object.fromEntries(actor._source.items.map(item => [item._id, {
      name: item.name, quantity: item.system.quantity ?? null, location: item.system.location ?? null,
    }])),
  };
}"""

BASE_IS = """([id, hash]) => game.actors.get(id)?._source.flags['taka-charsync']?.base?.hash === hash"""

BACKUPS = """id => game.actors
  .filter(actor => actor._source.flags['taka-charsync']?.backupOf === id)
  .map(actor => ({
    name: actor.name,
    role: actor.folder?.getFlag('taka-charsync', 'role') ?? null,
    itemNames: actor._source.items.map(item => item.name),
  }))"""

DISCARD_DRAFT = """async ([catchUpUrl, publishedUrl, id]) => {
  const [{discardDraft}, {getMeta}] = await Promise.all([import(catchUpUrl), import(publishedUrl)]);
  return discardDraft(game.actors.get(id), {meta: await getMeta()});
}"""

OWNERSHIP_OF_CURRENT_USER = """id => game.actors.get(id).ownership[game.user.id] ?? null"""

OPEN_SHEET = """id => { game.actors.get(id).sheet.render(true); }"""
RERENDER_SHEET = """id => { game.actors.get(id).sheet.render(false); }"""
CLOSE_SHEET = """async id => { await game.actors.get(id).sheet.close(); }"""

BAR_SELECTOR = "game.actors.get(id).sheet.element?.[0]?.querySelector('.window-content > .taka-charsync-bar')"
BAR_SETTLED = f"""id => {{
  const bar = {BAR_SELECTOR};
  return Boolean(bar) && bar.dataset.state !== 'loading';
}}"""
BAR_IN_STATE = f"""([id, state]) => {BAR_SELECTOR}?.dataset.state === state"""
BAR_COUNT = """id => game.actors.get(id).sheet.element?.[0]?.querySelectorAll('.taka-charsync-bar').length ?? 0"""
BAR = f"""id => {{
  const bar = {BAR_SELECTOR};
  return bar && {{state: bar.dataset.state, text: bar.innerText, buttons: [...bar.querySelectorAll('button')].map(b => b.textContent)}};
}}"""

# 模组脚本顶层装规则覆盖时抛错：重新执行一遍 main.mjs（带查询串就是新的模块实例），这次一读 ClientSettings 就抛错
PROBE_TOP_LEVEL_FAILURE = """async url => {
  const errors = [];
  const originalError = console.error;
  console.error = (...args) => { errors.push(args.map(String).join(' ')); };
  const helpers = foundry.helpers;
  foundry.helpers = new Proxy(helpers, {get: (target, key) => {
    if (key === 'ClientSettings') throw new Error('e2e：ClientSettings 不可用');
    return target[key];
  }});
  try {
    await import(`${url}?probe=${Date.now()}`);
    return {loaded: true, errors};
  } catch (error) {
    return {loaded: false, message: String(error), errors};
  } finally {
    foundry.helpers = helpers;
    console.error = originalError;
  }
}"""

# Actor 的团规不依赖 game.user 或 game.data.users 的初始化时机
MISSING_USER_ROW = """async ([url, actorId]) => {
  const {actorRuleState, withActorRules} = await import(url);
  const actor = game.actors.get(actorId);
  const rows = game.data.users;
  Object.defineProperty(game, 'user', {value: null, configurable: true});
  try {
    const withRow = actorRuleState(actor);
    game.data.users = rows.filter(row => row._id !== game.userId);
    return {
      withRow, withoutRow: actorRuleState(actor),
      actorFa: withActorRules(actor, () => game.pf2e.settings.variants.fa),
      worldFa: game.pf2e.settings.variants.fa,
    };
  } finally {
    delete game.user;
    game.data.users = rows;
  }
}"""

BACKUP_NAME = re.compile(r'（草稿备份 \d\d-\d\d \d\d:\d\d）$')


def is_module_error(line: str) -> bool:
    return line.startswith(('error:', 'pageerror:')) and MODULE_ID in line


def plain_feats(doc: dict) -> list[str]:
    """不授予、也不被授予、没有规则元素、占着专长槽的专长：删掉或挪槽位不会牵连别的条目。"""
    feats = []
    for item in doc['items']:
        pf2e = item.get('flags', {}).get('pf2e', {})
        if (item['type'] == 'feat' and not pf2e.get('grantedBy') and not pf2e.get('itemGrants')
                and not item['system'].get('rules') and item['system'].get('location')):
            feats.append(item['_id'])
    return feats


class CatchUpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-e2e-')), fixtures_dir())
        cls.addClassCleanup(env.close)
        actors = env.actors

        def casts(card_id):
            return any(item['type'] == 'spellcastingEntry' for item in actors[card_id]['items'])

        def level(card_id):
            return actors[card_id]['system'].get('details', {}).get('level', {}).get('value')

        cls.card = next(c for c in env.main_cards if level(c) == 6 and not casts(c))
        cls.familiar = next(actor_id for actor_id, doc in actors.items() if doc['type'] == 'familiar')
        cls.master = actors[cls.familiar]['system']['master']['id']
        cls.deleted_feat, cls.contested_feat = plain_feats(actors[cls.card])[:2]

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

        cls.player_logs = []
        cls.p1 = None
        cls.login_p1()
        meta = cls.read_published('meta.json')
        for actor_id in (cls.card, cls.familiar):
            bootstrap = meta['bootstrap'][actor_id]
            cls.p1.wait_for_function(BASE_IS, arg=[actor_id, bootstrap['hash']], timeout=WAIT_MS)
            if cls.p1.evaluate(CARD_STATE, actor_id)['base'] != bootstrap:
                raise AssertionError(f'{actor_id} 第一次登录写下的 base 不是起步基线')
        cls.draft_feat = cls.p1.evaluate(ADD_FEAT, [cls.card, DRAFT_FEAT])

    @classmethod
    def run_engine(cls):
        report = cls.env.engine_once()
        if report['errors']:
            raise AssertionError(f"引擎一轮有错误：{report['errors']}")

    @classmethod
    def login_p1(cls):
        """关掉 P1 之前的页面，重新登录建卡服：登录时跟上主服在 ready 里跑。"""
        if cls.p1 is not None:
            cls.p1.context.close()
        cls.p1, logs = cls.env.login('char', 'P1')
        cls.player_logs.append(logs)
        cls.p1_logs = logs

    @classmethod
    def read_published(cls, rel):
        return json.loads((cls.env.data_dir('char') / PUBLISH_DIR / rel).read_text(encoding='utf-8'))

    @classmethod
    def latest(cls, actor_id):
        return cls.read_published(f'main/{CAMP_WORLD}/{actor_id}.json')

    def edit_main(self, script, arg):
        """起主服世界，以 GM 跑 `script`，再停掉世界、跑引擎一轮。返回脚本的结果。"""
        self.env.start('main')
        page, _ = self.env.login('main', 'Gamemaster')
        try:
            result = page.evaluate(script, arg)
        finally:
            page.context.close()
        self.env.stop('main')
        self.run_engine()
        return result

    def state(self, actor_id):
        return self.p1.evaluate(CARD_STATE, actor_id)

    def wait_base(self, actor_id, hash_):
        self.p1.wait_for_function(BASE_IS, arg=[actor_id, hash_], timeout=WAIT_MS)

    def open_sheet(self, state=None):
        self.p1.evaluate(OPEN_SHEET, self.card)
        if state:
            self.p1.wait_for_function(BAR_IN_STATE, arg=[self.card, state], timeout=WAIT_MS)
        else:
            self.p1.wait_for_function(BAR_SETTLED, arg=self.card, timeout=WAIT_MS)
        return self.p1.evaluate(BAR, self.card)

    def test_01_login_merges_main_changes_into_draft(self):
        played = self.edit_main(MAIN_PLAY_SESSION, [self.card, self.familiar, LOOT])
        latest = self.latest(self.card)
        familiar_latest = self.latest(self.familiar)

        self.login_p1()
        self.wait_base(self.card, latest['hash'])
        self.wait_base(self.familiar, familiar_latest['hash'])

        card = self.state(self.card)
        self.assertEqual(card['base'], {'kind': 'main', 'hash': latest['hash']})
        self.assertEqual(card['hp'], played['hp'])
        self.assertEqual(card['items'][played['potionId']]['quantity'], played['quantity'])
        self.assertEqual(card['items'][played['lootId']]['name'], LOOT)
        self.assertEqual(card['items'][self.draft_feat]['name'], DRAFT_FEAT)
        self.assertIsNone(card['conflicts'])
        self.p1.wait_for_function(
            'name => document.querySelector("#notifications")?.innerText.includes(`${name}已跟上主服`)',
            arg=card['name'], timeout=WAIT_MS)

        familiar = self.state(self.familiar)
        self.assertEqual(familiar['hp'], played['familiarHp'])
        self.assertEqual(familiar['master'], self.master)

    def test_02_main_review_deletion_and_conflict(self):
        self.edit_main(DELETE_ITEM, [self.card, self.deleted_feat])
        latest = self.latest(self.card)
        self.login_p1()
        self.wait_base(self.card, latest['hash'])
        card = self.state(self.card)
        self.assertNotIn(self.deleted_feat, card['items'])
        self.assertIn(self.contested_feat, card['items'])
        self.assertIn(self.draft_feat, card['items'])

        # 草稿里挪了 D2 的槽位，主服又删了它：专长归建卡服，留建卡服的版本并报冲突
        self.p1.evaluate("""async ([actorId, itemId]) => {
          await game.actors.get(actorId).items.get(itemId).update({'system.location': null});
        }""", [self.card, self.contested_feat])
        self.edit_main(DELETE_ITEM, [self.card, self.contested_feat])
        latest = self.latest(self.card)
        self.login_p1()
        self.wait_base(self.card, latest['hash'])
        card = self.state(self.card)
        self.assertIn(self.contested_feat, card['items'])
        self.assertIsNone(card['items'][self.contested_feat]['location'])
        self.assertEqual([entry['id'] for entry in card['conflicts']['list']], [self.contested_feat])
        self.assertEqual(card['conflicts']['mainHash'], latest['hash'])
        self.p1.wait_for_function(
            'name => document.querySelector("#notifications")?.innerText.includes(`${name}已跟上主服`)'
            ' && document.querySelector("#notifications").innerText.includes("有 1 处两边都改了，打开卡面看详情")',
            arg=card['name'], timeout=WAIT_MS)

        bar = self.open_sheet()
        self.assertEqual(bar['state'], 'conflict')
        self.assertIn('冲突', bar['text'])
        self.assertIn(card['items'][self.contested_feat]['name'], bar['text'])
        self.assertIn(DISCARD, bar['buttons'])
        self.p1.evaluate(CLOSE_SHEET, self.card)

    def test_03_main_level_up_backs_up_draft_and_resets(self):
        self.edit_main(UPDATE_ACTOR, [self.card, {'system.details.level.value': 7}])
        latest = self.latest(self.card)
        self.assertEqual(latest['level'], 7)

        self.login_p1()
        self.wait_base(self.card, latest['hash'])
        card = self.state(self.card)
        self.assertEqual(card['level'], 7)
        self.assertEqual(set(card['items']), {item['_id'] for item in latest['doc']['items']})
        self.assertNotIn(self.draft_feat, card['items'])
        self.assertIsNone(card['conflicts'])

        backups = self.p1.evaluate(BACKUPS, self.card)
        self.assertEqual(len(backups), 1)
        backup = backups[0]
        self.assertEqual(backup['role'], 'backup')
        self.assertTrue(backup['name'].startswith(f"{card['name']}（草稿备份 "), backup['name'])
        self.assertRegex(backup['name'], BACKUP_NAME)
        self.assertIn(DRAFT_FEAT, backup['itemNames'])

        warning = self.p1.locator('#notifications li.notification.warning.permanent')
        warning.first.wait_for(timeout=WAIT_MS)
        self.assertIn('恢复成主服', warning.first.inner_text())

    def test_04_open_sheet_keeps_card_until_reopened(self):
        bar = self.open_sheet()
        self.assertEqual(bar['state'], 'synced', bar)
        self.assertIn('已和主服一致', bar['text'])
        before = self.state(self.card)

        hp = self.edit_main(LOSE_HP, [self.card, 3])
        latest = self.latest(self.card)
        self.assertNotEqual(latest['hash'], before['base']['hash'])

        # 卡面开着：重绘也不跟上
        self.p1.evaluate(RERENDER_SHEET, self.card)
        self.p1.wait_for_timeout(3000)
        during = self.state(self.card)
        self.assertEqual(during['hp'], before['hp'])
        self.assertEqual(during['base'], before['base'])
        self.assertEqual(self.p1.evaluate(BAR_COUNT, self.card), 1)

        self.p1.evaluate(CLOSE_SHEET, self.card)
        self.p1.evaluate(OPEN_SHEET, self.card)
        self.wait_base(self.card, latest['hash'])
        self.assertEqual(self.state(self.card)['hp'], hp)
        self.p1.wait_for_function(BAR_IN_STATE, arg=[self.card, 'synced'], timeout=WAIT_MS)

        self.p1.evaluate(OPEN_SHEET, self.familiar)
        self.p1.wait_for_function(BAR_SETTLED, arg=self.familiar, timeout=WAIT_MS)
        self.assertEqual(self.p1.evaluate(BAR, self.familiar)['state'], 'synced')
        self.p1.evaluate(CLOSE_SHEET, self.familiar)

    def test_05_discard_draft_restores_main_version(self):
        before = self.state(self.card)
        feat = self.p1.evaluate(ADD_FEAT, [self.card, DISCARDED_FEAT])
        self.p1.wait_for_function(BAR_IN_STATE, arg=[self.card, 'draft'], timeout=WAIT_MS)
        # 用例 3 的常驻警告压在卡面按钮上
        self.p1.evaluate('() => ui.notifications.clear()')
        self.p1.locator('.taka-charsync-details > summary').click()
        bar = self.p1.evaluate(BAR, self.card)
        self.assertIn(DISCARDED_FEAT, bar['text'])

        self.p1.locator('.taka-charsync-secondary > summary').click()
        self.p1.click('.taka-charsync-bar button:has-text("放弃草稿")')
        self.p1.click('.application.dialog button[data-action="yes"]')
        self.p1.wait_for_function('([actorId, itemId]) => !game.actors.get(actorId).items.has(itemId)',
                                  arg=[self.card, feat], timeout=WAIT_MS)
        after = self.state(self.card)
        self.assertEqual(set(after['items']), set(before['items']))
        self.assertEqual(after['base'], before['base'])
        backups = self.p1.evaluate(BACKUPS, self.card)
        self.assertEqual(len(backups), 2)
        self.assertTrue(any(DISCARDED_FEAT in backup['itemNames'] for backup in backups))
        self.p1.wait_for_function(BAR_IN_STATE, arg=[self.card, 'synced'], timeout=WAIT_MS)
        self.p1.evaluate(CLOSE_SHEET, self.card)

    def test_06_legacy_browser_campaign_does_not_change_actor_rules(self):
        self.p1.evaluate("() => localStorage.setItem('taka-charsync.campaign', 'e2e-other')")
        self.p1.reload()
        self.p1.wait_for_function('() => window.game?.ready === true', timeout=READY_TIMEOUT_MS)
        bar = self.open_sheet()
        self.assertIn(f'按{CAMP_TITLE}团规', bar['text'])
        self.assertNotIn('e2e-other', bar['text'])
        self.assertFalse(any('规则显示' in label or '团规' in label for label in bar['buttons']), bar)
        state = self.p1.evaluate("""async ([url, actorId]) => {
          const {actorRuleState, selfCheck} = await import(url);
          const actor = game.actors.get(actorId);
          return {rules: actorRuleState(actor), errors: selfCheck(actor)};
        }""", [OVERRIDE_URL, self.card])
        self.assertTrue(state['rules']['ok'], state)
        self.assertEqual(state['rules']['world'], CAMP_WORLD)
        self.assertEqual(state['errors'], [])
        self.p1.evaluate(CLOSE_SHEET, self.card)

    def test_07_missing_user_row_keeps_actor_rules(self):
        state = self.p1.evaluate(MISSING_USER_ROW, [OVERRIDE_URL, self.card])
        self.assertTrue(state['withRow']['ok'], state)
        self.assertEqual(state['withRow']['world'], CAMP_WORLD)
        self.assertEqual(state['withRow'], state['withoutRow'])
        self.assertIs(state['actorFa'], True)
        self.assertIs(state['worldFa'], False)

    def test_08_override_failure_at_top_level_keeps_module_loading(self):
        p2, _ = self.env.login('char', 'P2')
        try:
            result = p2.evaluate(PROBE_TOP_LEVEL_FAILURE, MAIN_URL)
        finally:
            p2.context.close()
        self.assertTrue(result['loaded'], result)
        self.assertTrue(any(MODULE_ID in line for line in result['errors']), result)

    def test_09_gm_leaves_player_cards_alone(self):
        before = self.state(self.card)
        backups_before = len(self.p1.evaluate(BACKUPS, self.card))
        hp = self.edit_main(LOSE_HP, [self.card, 2])
        latest = self.latest(self.card)

        gm, _ = self.env.login('char', 'Gamemaster')
        try:
            # Foundry 建卡时给创建者补显式 OWNER：夹具卡是 GM 建的，GM 在 ownership 里也是所有者
            self.assertEqual(gm.evaluate(OWNERSHIP_OF_CURRENT_USER, self.card), 3)
            gm.evaluate(OPEN_SHEET, self.card)
            gm.wait_for_function(BAR_SETTLED, arg=self.card, timeout=WAIT_MS)
            bar = gm.evaluate(BAR, self.card)
            gm.wait_for_timeout(3000)
            after = gm.evaluate(CARD_STATE, self.card)
            backups = gm.evaluate(BACKUPS, self.card)
        finally:
            gm.context.close()
        self.assertEqual(bar['state'], 'paused', bar)
        self.assertEqual(after['base'], before['base'])
        self.assertEqual(after['hp'], before['hp'])
        self.assertEqual(len(backups), backups_before)

        self.login_p1()
        self.wait_base(self.card, latest['hash'])
        self.assertEqual(self.state(self.card)['hp'], hp)

        # 卡上有草稿时 GM 打开卡面：看得到草稿，但没有放弃草稿的按钮，直接调用也不动卡
        feat = self.p1.evaluate(ADD_FEAT, [self.card, GM_SEEN_FEAT])
        gm, _ = self.env.login('char', 'Gamemaster')
        try:
            gm.evaluate(OPEN_SHEET, self.card)
            gm.wait_for_function(BAR_IN_STATE, arg=[self.card, 'draft'], timeout=WAIT_MS)
            gm.locator('.taka-charsync-details > summary').click()
            bar = gm.evaluate(BAR, self.card)
            discarded = gm.evaluate(DISCARD_DRAFT, [CATCH_UP_URL, PUBLISHED_URL, self.card])
            gm.wait_for_timeout(3000)
            after = gm.evaluate(CARD_STATE, self.card)
            backups = gm.evaluate(BACKUPS, self.card)
        finally:
            gm.context.close()
        self.assertIn(GM_SEEN_FEAT, bar['text'])
        self.assertNotIn(DISCARD, bar['buttons'])
        self.assertIsNone(discarded)
        self.assertIn(feat, after['items'])
        self.assertEqual(len(backups), backups_before)

    def test_10_familiar_keeps_char_master_when_main_differs(self):
        # 线上主服和建卡服的主人卡 id 可以不同；测试台两边同 id，把发布的主服魔宠改成指向别的主人
        published = self.env.data_dir('char') / PUBLISH_DIR / 'main' / CAMP_WORLD / f'{self.familiar}.json'
        original = published.read_text(encoding='utf-8')
        entry = json.loads(original)
        entry['doc']['system']['master']['id'] = OTHER_MASTER
        published.write_text(json.dumps(entry, ensure_ascii=False), encoding='utf-8')
        try:
            self.p1.evaluate(DISCARD_DRAFT, [CATCH_UP_URL, PUBLISHED_URL, self.familiar])
        finally:
            published.write_text(original, encoding='utf-8')
        self.assertEqual(self.state(self.familiar)['master'], self.master)

    def test_11_schema_mismatch_pauses_sync(self):
        self.edit_main(LOSE_HP, [self.card, 1])
        meta_file = self.env.data_dir('char') / PUBLISH_DIR / 'meta.json'
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        meta['char']['schema'] = 'e2e-mismatch'
        meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        before = self.state(self.card)

        self.login_p1()
        bar = self.open_sheet(state='paused')
        self.assertIn('暂停', bar['text'])
        after = self.state(self.card)
        self.assertEqual(after['hp'], before['hp'])
        self.assertEqual(after['base'], before['base'])

    def test_12_no_module_errors(self):
        errors = [line for logs in self.player_logs for line in logs if is_module_error(line)]
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
