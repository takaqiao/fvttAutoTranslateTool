"""建卡服按 Actor 所属团准备规则、整卡导入：两个本机实例种好夹具卡，引擎跑一轮写出规则表后，在建卡服上以 P1、P2、GM 验证。

用法：python -X utf8 test_rules_and_import.py
"""
import json
import tempfile
import unittest
from pathlib import Path

from harness import CAMP_WORLD, MODULE_ID, PUBLISH_DIR, Env, fixtures_dir

OVERRIDE_URL = f'/modules/{MODULE_ID}/scripts/char/rules-override.mjs'
IMPORT_URL = f'/modules/{MODULE_ID}/scripts/foundry/import.mjs'
PUBLISHED_URL = f'/modules/{MODULE_ID}/scripts/foundry/published.mjs'
RULES_URL = f'/modules/{MODULE_ID}/generated/rules.mjs'
WAIT_MS = 30_000

RULES_STATE = """async ([url, cardId]) => {
  const m = await import(url);
  const actor = game.actors.get(cardId);
  const doc = game.settings.get('pf2e', 'freeArchetypeVariant', {document: true});
  const group = actor.feats.get('archetype');
  const scoped = m.withActorRules(actor, () => ({
    fa: game.pf2e.settings.variants.fa,
    stamina: game.pf2e.settings.variants.stamina,
    settingFa: game.settings.get('pf2e', 'freeArchetypeVariant'),
    documentFa: game.settings.get('pf2e', 'freeArchetypeVariant', {document: true}).value,
  }));
  return {
    ruleState: m.actorRuleState(actor),
    ...scoped,
    explicitFa: m.ruleValue(actor, 'pf2e', 'freeArchetypeVariant'),
    worldFa: game.pf2e.settings.variants.fa,
    worldStamina: game.pf2e.settings.variants.stamina,
    archetypeSlots: group ? Object.values(group.slots).map(slot => slot.level) : null,
    settingIsDocument: doc instanceof CONFIG.Setting.documentClass,
    settingValue: doc.value,
    selfCheck: m.selfCheck(actor),
  };
}"""

# 绕过本客户端的缓存，向服务器查世界设置的原始 JSON；不存在时为 null
SERVER_SETTINGS = """async keys => {
  const out = {};
  for (const key of keys) {
    const docs = await CONFIG.DatabaseBackend.get(CONFIG.Setting.documentClass, {query: {key}});
    out[key] = docs.find(doc => !doc._source.user)?._source.value ?? null;
  }
  return out;
}"""

SET_SETTING = """async ([namespace, key, value]) => { await game.settings.set(namespace, key, value); }"""

# 记下本客户端每次收到某个设置的建、改之后（在模组的处理之后）PF2e 缓存里的值
RECORD_SETTING_CHANGES = """key => {
  window.e2eSeen = [];
  const record = setting => {
    if (setting.key !== key) return;
    const {variants} = game.pf2e.settings;
    window.e2eSeen.push({value: setting.value, fa: variants.fa, stamina: variants.stamina});
  };
  Hooks.on('createSetting', record);
  Hooks.on('updateSetting', record);
}"""

SAVE_VARIANT_FORM = """async () => {
  const app = new (game.settings.menus.get('pf2e.variantRules').type)();
  await app.render({force: true});
  app.element.querySelector('button[type=submit]').click();
  for (let i = 0; i < 100 && app.rendered; i++) await new Promise(resolve => setTimeout(resolve, 100));
  return app.rendered;
}"""

MODULE_STATE = f"""() => ({{
  active: game.modules.get('{MODULE_ID}')?.active === true,
  publishDirRegistered: game.settings.settings.has('{MODULE_ID}.publishDir'),
}})"""

IMPORT_PRELUDE = """
  const {importFull} = await import(url);
  const actor = game.actors.get(cardId);
  const cardState = () => foundry.utils.deepClone({
    items: actor._source.items, effects: actor._source.effects, flags: actor._source.flags['taka-charsync'] ?? null,
  });
  const notifications = [];
  const watchNotifications = async fn => {
    const original = ui.notifications.notify;
    ui.notifications.notify = function (message, type, options) {
      notifications.push({message: String(message), type});
      return original.call(this, message, type, options);
    };
    try {
      return await fn();
    } finally {
      ui.notifications.notify = original;
    }
  };
"""

IMPORT_WITH_INVALID_ENTRIES = """async ([url, cardId]) => {
""" + IMPORT_PRELUDE + """
  const template = new CONFIG.Item.documentClass({name: 'E2E 新专长', type: 'feat', system: {category: 'general', level: {value: 1}}}).toObject();
  const valid = {...foundry.utils.deepClone(template), _id: foundry.utils.randomID()};
  const invalid = {...foundry.utils.deepClone(template), _id: foundry.utils.randomID(), name: 'E2E 非法新专长'};
  invalid.system.level.value = 'abc';
  const doc = actor.toObject();
  const edited = doc.items.find(item => item.type === 'feat');
  const editedOriginal = foundry.utils.deepClone(edited);
  edited.system.level.value = 'abc';
  doc.items.push(invalid, valid);
  const flags = {base: {kind: 'main', hash: 'e2e-new-items'}};
  const sent = foundry.utils.deepClone(doc);
  const result = await watchNotifications(() => importFull(actor, doc, {flags}));
  return {...result, notifications, sentFlags: flags, validId: valid._id, invalidId: invalid._id, editedOriginal,
    sent: sent.items, sentEffects: sent.effects, ...cardState()};
}"""

IMPORT_OVER_INVALID = """async ([url, cardId, invalidId]) => {
""" + IMPORT_PRELUDE + """
  const invalidBefore = actor.items.invalidDocumentIds.has(invalidId);
  const doc = actor.toObject();
  const flags = {base: {kind: 'main', hash: 'e2e-existing-invalid'}};
  const sent = foundry.utils.deepClone(doc);
  const result = await watchNotifications(() => importFull(actor, doc, {flags}));
  const state = cardState();
  // 同一会话里再导入一次：Foundry 不清 invalidDocumentIds，已经删掉的条目不能再报一次
  const again = await watchNotifications(() => importFull(actor, actor.toObject(), {flags}));
  return {...result, notifications, sentFlags: flags, invalidBefore, again, sent: sent.items, sentEffects: sent.effects, ...state};
}"""

READ_PUBLISHED = """async ([url, world, cardId]) => {
  const m = await import(url);
  const meta = await m.getMeta();
  const latest = await m.getMainLatest(world, cardId);
  const history = latest && await m.getHistoryDoc({kind: 'main', hash: latest.hash});
  return {
    dir: m.getPublishDir(),
    card: meta?.cards?.[cardId] ?? null,
    level: latest?.level ?? null,
    historyId: history?._id ?? null,
    copies: await m.getCopies(cardId),
    missing: await m.fetchPublished('missing.json'),
  };
}"""

IMPORT_DROPPED = """async ([url, cardId]) => {
  const {importFull} = await import(url);
  const actor = game.actors.get(cardId);
  actor.update = async () => undefined;
  try {
    await importFull(actor, actor.toObject(), {flags: {}});
    return {threw: false};
  } catch (error) {
    return {threw: true, message: error.message};
  } finally {
    delete actor.update;
  }
}"""

# 主服 GM 新建的条目，ownership 里带着建卡服没有的用户 id；玩家把它写回已有条目会被服务器整次拒绝
IMPORT_FOREIGN_OWNERSHIP = """async ([url, cardId]) => {
  const {importFull} = await import(url);
  const actor = game.actors.get(cardId);
  const doc = actor.toObject();
  const target = doc.items[0];
  const before = foundry.utils.deepClone(actor._source.items.find(item => item._id === target._id));
  target.ownership = {default: 0, [foundry.utils.randomID()]: 3};
  target.sort = before.sort + 1000;
  target.name = `${before.name} E2E`;
  try {
    await importFull(actor, doc, {flags: actor._source.flags['taka-charsync'] ?? {}});
  } catch (error) {
    return {threw: true, message: error.message};
  }
  const after = actor._source.items.find(item => item._id === target._id);
  return {
    threw: false, sentName: target.name,
    before: {ownership: before.ownership, sort: before.sort},
    after: {name: after.name, ownership: after.ownership, sort: after.sort},
  };
}"""


def without_stats(value):
    """去掉每一层的 `_stats`：服务器写入时会刷新时间戳。"""
    if isinstance(value, list):
        return [without_stats(item) for item in value]
    if isinstance(value, dict):
        return {key: without_stats(child) for key, child in value.items() if key != '_stats'}
    return value


def is_module_error(line: str) -> bool:
    return line.startswith(('error:', 'pageerror:')) and MODULE_ID in line


class RulesAndImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = fixtures_dir()
        worlds = {card_id: entry['world'] for card_id, entry in
                  json.loads((fixtures / 'cards.json').read_text(encoding='utf-8'))['actors'].items()}
        env = cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-e2e-')), fixtures)
        cls.addClassCleanup(env.close)
        env.build()
        env.start('main')
        env.start('char')
        env.seed()

        def level(card_id):
            return env.actors[card_id]['system'].get('details', {}).get('level', {}).get('value')

        def casts(card_id):
            return any(item['type'] == 'spellcastingEntry' for item in env.actors[card_id]['items'])

        # 夹具里肆季那张 L6 非施法的主卡（同伴不在 main_cards 里）
        cls.sog_card = next(c for c in env.main_cards if worlds[c] == 'sog' and level(c) == 6 and not casts(c))
        others = [c for c in env.main_cards if c != cls.sog_card]
        cls.import_card, cls.invalid_card = others[0], others[1]

        env.stop('main')
        env.stop('char')
        cls.invalid_item = env.inject_invalid_item('char', cls.invalid_card)
        report = env.engine_once()
        if report['errors']:
            raise AssertionError(f"引擎一轮有错误：{report['errors']}")
        env.start('char')

        cls.gm, cls.gm_logs = env.login('char', 'Gamemaster')
        cls.server_before = cls.gm.evaluate(SERVER_SETTINGS, ['pf2e.freeArchetypeVariant', 'pf2e.campaignFeatSections'])
        cls.p1, cls.p1_logs = env.login('char', 'P1')

    def wait_seen(self, count):
        self.p1.wait_for_function('n => window.e2eSeen.length >= n', arg=count, timeout=WAIT_MS)
        return self.p1.evaluate('() => window.e2eSeen')

    def assert_campaign_rules(self, state):
        self.assertTrue(state['ruleState']['ok'], state)
        self.assertEqual(state['ruleState']['world'], CAMP_WORLD)
        self.assertRegex(state['ruleState']['hash'], r'^[a-f0-9]{64}$')
        self.assertIs(state['fa'], True)
        self.assertIs(state['settingFa'], True)
        self.assertIs(state['explicitFa'], True)
        self.assertIs(state['stamina'], True)
        self.assertEqual(sorted(state['archetypeSlots'] or []), [2, 4, 6])
        self.assertTrue(state['settingIsDocument'])
        self.assertIs(state['settingValue'], False)
        self.assertIs(state['documentFa'], False)
        self.assertIs(state['worldFa'], False)
        self.assertIs(state['worldStamina'], False)
        self.assertEqual(state['selfCheck'], [])

    def test_01_player_reads_actor_campaign_rules(self):
        self.assert_campaign_rules(self.p1.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card]))

    def test_02_gm_reads_same_actor_campaign_rules(self):
        state = self.gm.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card])
        self.assert_campaign_rules(state)
        self.assertEqual(state, self.p1.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card]))

    def test_03_first_world_write_keeps_actor_rules(self):
        snapshot = self.p1.evaluate('async url => (await import(url)).SNAPSHOT', RULES_URL)
        self.assertEqual(snapshot['version'], 2)
        self.assertIs(snapshot['rules'][CAMP_WORLD].get('pf2e.staminaVariant'), True)
        self.assertEqual(snapshot['cardWorld'][self.sog_card], CAMP_WORLD)
        self.assertIsNone(self.gm.evaluate(SERVER_SETTINGS, ['pf2e.staminaVariant'])['pf2e.staminaVariant'])
        self.assert_campaign_rules(self.p1.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card]))

        self.p1.evaluate(RECORD_SETTING_CHANGES, 'pf2e.staminaVariant')
        self.gm.evaluate(SET_SETTING, ['pf2e', 'staminaVariant', False])
        seen = self.wait_seen(1)

        self.assertEqual(self.gm.evaluate(SERVER_SETTINGS, ['pf2e.staminaVariant'])['pf2e.staminaVariant'], 'false')
        self.assertEqual([entry['value'] for entry in seen], [False])
        self.assertEqual([entry['stamina'] for entry in seen], [False])
        self.assert_campaign_rules(self.p1.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card]))

    def test_04_gm_saving_variant_form_keeps_world_rules(self):
        self.assertFalse(self.gm.evaluate(SAVE_VARIANT_FORM))
        after = self.gm.evaluate(SERVER_SETTINGS, ['pf2e.freeArchetypeVariant', 'pf2e.campaignFeatSections'])
        self.assertEqual(after, self.server_before)

    def test_05_world_rule_changes_stay_outside_actor_scope(self):
        self.p1.evaluate(RECORD_SETTING_CHANGES, 'pf2e.freeArchetypeVariant')
        self.gm.evaluate(SET_SETTING, ['pf2e', 'freeArchetypeVariant', True])
        self.wait_seen(1)
        self.gm.evaluate(SET_SETTING, ['pf2e', 'freeArchetypeVariant', False])
        seen = self.wait_seen(2)

        self.assertEqual([entry['value'] for entry in seen], [True, False])
        self.assertEqual([entry['fa'] for entry in seen], [True, False])
        self.assert_campaign_rules(self.p1.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card]))

    def test_06_player_without_assigned_character_reads_actor_rules(self):
        p2, _ = self.env.login('char', 'P2')
        try:
            self.assertIsNone(p2.evaluate('() => game.user.character?.id ?? null'))
            state = p2.evaluate(RULES_STATE, [OVERRIDE_URL, self.sog_card])
        finally:
            p2.context.close()
        self.assert_campaign_rules(state)

    def assert_items_match(self, result, excluded_id, kept=None):
        """卡上的条目逐条等于传入的，`excluded_id` 那条不在卡上；`kept` 是保留卡上原版本的条目。"""
        expected = {item['_id']: without_stats(item) for item in result['sent'] if item['_id'] != excluded_id}
        if kept:
            expected[kept['_id']] = without_stats(kept)
        actual = {item['_id']: without_stats(item) for item in result['items']}
        self.assertEqual(set(actual), set(expected))
        for item_id, item in expected.items():
            self.assertEqual(actual[item_id], item, f"条目 {item_id}（{item.get('name')}）与传入的不一致")
        self.assertEqual(without_stats(result['effects']), without_stats(result['sentEffects']))
        self.assertEqual(result['flags'], result['sentFlags'])

    def test_07_import_skips_invalid_entries(self):
        result = self.p1.evaluate(IMPORT_WITH_INVALID_ENTRIES, [IMPORT_URL, self.import_card])
        edited = result['editedOriginal']
        self.assertEqual([entry['id'] for entry in result['skipped']], [edited['_id'], result['invalidId']])
        self.assertTrue(all(entry['error'] for entry in result['skipped']))
        self.assertEqual(result['removed'], [])
        self.assertIn(result['validId'], {item['_id'] for item in result['items']})
        # 新增的非法条目不加；已有条目的非法改动不套用，卡上留原版本
        self.assert_items_match(result, result['invalidId'], kept=edited)
        self.assertEqual(result['notifications'], [])

    def test_08_import_removes_existing_invalid_item(self):
        result = self.p1.evaluate(IMPORT_OVER_INVALID, [IMPORT_URL, self.invalid_card, self.invalid_item])
        self.assertTrue(result['invalidBefore'])
        self.assertEqual([entry['id'] for entry in result['removed']], [self.invalid_item])
        self.assertEqual(result['skipped'], [])
        self.assertNotIn(self.invalid_item, {item['_id'] for item in result['items']})
        self.assert_items_match(result, self.invalid_item)
        self.assertEqual(result['notifications'], [])
        self.assertEqual(result['again'], {'skipped': [], 'removed': []})

    def test_09_import_fails_when_update_is_dropped(self):
        result = self.p1.evaluate(IMPORT_DROPPED, [IMPORT_URL, self.import_card])
        self.assertTrue(result['threw'], result)
        self.assertIn('丢弃', result['message'])

    def test_10_reads_published_files(self):
        state = self.p1.evaluate(READ_PUBLISHED, [PUBLISHED_URL, CAMP_WORLD, self.sog_card])
        self.assertEqual(state['dir'], PUBLISH_DIR)
        self.assertEqual((state['card'] or {}).get('world'), CAMP_WORLD)
        self.assertEqual(state['level'], 6)
        self.assertEqual(state['historyId'], self.sog_card)
        self.assertEqual(state['copies'], [])
        self.assertIsNone(state['missing'])

    def test_11_import_keeps_local_keys_of_existing_entries(self):
        result = self.p1.evaluate(IMPORT_FOREIGN_OWNERSHIP, [IMPORT_URL, self.import_card])
        self.assertFalse(result['threw'], result)
        self.assertEqual(result['after']['name'], result['sentName'])
        self.assertEqual(result['after']['ownership'], result['before']['ownership'])
        self.assertEqual(result['after']['sort'], result['before']['sort'])

    def test_12_module_loads_without_errors(self):
        for page in (self.p1, self.gm):
            self.assertEqual(page.evaluate(MODULE_STATE), {'active': True, 'publishDirRegistered': True})
        errors = [line for line in [*self.p1_logs, *self.gm_logs] if is_module_error(line)]
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
