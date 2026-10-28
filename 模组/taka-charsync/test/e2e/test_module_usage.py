"""模块使用证据的冻结、实际导入范围和原生模块管理。按顺序运行完整类。"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from harness import CHAR_CAMPAIGN_FOLDER, CAMP_WORLD, MODULE_DIR, MODULE_ID, PUBLISH_DIR, Env, PageLogOnFailure, fixtures_dir


SUBMIT = f'/modules/{MODULE_ID}/scripts/char/submit.mjs'
APPLY = f'/modules/{MODULE_ID}/scripts/main/apply.mjs'
API = f"game.modules.get('{MODULE_ID}').api"
WAIT = 60_000
RULE = 'charsync-e2e-rule'
SOURCE = 'charsync-e2e-source'
FLAG = 'charsync-e2e-flag'
GEAR = 'charsync-e2e-gear'
PET = 'charsync-e2e-pet'
TITLES = {RULE: 'E2E 规则包', SOURCE: 'E2E 内容来源包', FLAG: 'E2E 编辑数据包',
          GEAR: 'E2E 装备规则包', PET: 'E2E 同伴包'}
DOC = 'id => game.actors.get(id).toObject()'


def uuid(module_id):
    return f'Compendium.{module_id}.features.Item.ModuleUsage00001'


def grant(module_id):
    # 保留合法的字面量引用，但不让 PF2e 在测试中实际生成条目。
    return {'key': 'GrantItem', 'uuid': uuid(module_id), 'predicate': ['charsync-e2e:never-grant']}


def equipment():
    return {'name': 'E2E 模块装备', 'type': 'equipment', 'system': {'rules': [
        {'key': 'EphemeralEffect', 'uuid': uuid(GEAR), 'selectors': ['strike-damage'],
         'predicate': ['charsync-e2e:never-grant']},
    ]}}


def usage_by_id(usage):
    return {entry['id']: entry for entry in usage}


def source_hashes():
    paths = [*MODULE_DIR.glob('scripts/**/*.mjs'), *MODULE_DIR.glob('engine/**/*.mjs'),
             *MODULE_DIR.glob('styles/*.css'), MODULE_DIR / 'module.json',
             Path(__file__), MODULE_DIR / 'test/e2e/harness.py']
    return {path.relative_to(MODULE_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


class ModuleUsageTest(PageLogOnFailure, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-module-usage-e2e-')), fixtures_dir())
        cls.addClassCleanup(cls.env.close)
        cls.artifacts = Path(os.environ.get('CHARSYNC_E2E_ARTIFACTS', str(cls.env.root / 'screenshots')))
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        cls.evidence = {'root': str(cls.env.root), 'sourceHashes': source_hashes()}
        cls.addClassCleanup(cls.save_evidence)
        cls.env.build()
        # 本套只测模块提示，私有夹具中的 Tokenizer 原图不随卡片 JSON 分发。
        for doc in cls.env.actors.values():
            doc['img'] = 'icons/svg/mystery-man.svg'
            doc['prototypeToken']['texture']['src'] = 'icons/svg/mystery-man.svg'
            doc['prototypeToken']['ring']['subject']['texture'] = ''
            for source in (doc, doc['prototypeToken']):
                source.get('flags', {}).pop('tokenizer-2', None)
                source.get('flags', {}).pop('taka-charsync-media', None)
            for item in doc['items']:
                item['img'] = 'icons/svg/item-bag.svg'
        for side in ('main', 'char'):
            for module_id, title in TITLES.items():
                if side == 'main' and module_id != RULE:
                    continue
                folder = cls.env.data_dir(side) / 'modules' / module_id
                folder.mkdir()
                (folder / 'module.json').write_text(json.dumps({
                    'id': module_id, 'title': title, 'version': '1.0.0',
                    'compatibility': {'minimum': '14', 'verified': '14'},
                    'description': '角色同步隔离测试夹具，无脚本和内容包。',
                }, ensure_ascii=False, indent=2), encoding='utf-8')
        cls.familiar = next(actor_id for actor_id, doc in cls.env.actors.items() if doc['type'] == 'familiar')
        cls.master = cls.env.actors[cls.familiar]['system']['master']['id']
        independent = [actor_id for actor_id in cls.env.main_cards if actor_id != cls.master]
        cls.card, cls.auto_card = independent[:2]
        cls.env.start('main')
        cls.env.start('char')
        cls.env.seed()
        gm, _ = cls.env.login('char', 'Gamemaster')
        gm.evaluate("""async () => {
          const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
          permissions.ACTOR_CREATE = Object.values(CONST.USER_ROLES).filter(role => role > 0);
          await game.settings.set('core', 'permissions', permissions);
        }""")
        gm.context.close()
        cls.env.stop('main')
        cls.run_engine()
        cls.p1, _ = cls.env.login('char', 'P1')
        cls.p1.wait_for_function("""ids => ids.every(id => game.actors.get(id)?._source.flags['taka-charsync']?.base)""",
                                 arg=[cls.card, cls.auto_card, cls.master, cls.familiar], timeout=WAIT)
        cls.env.start('main')
        cls.gm, _ = cls.env.login('main', 'Gamemaster')
        cls.gm.wait_for_function(f'() => Boolean({API}.lastCheck())', timeout=WAIT)
        cls.evidence['versions'] = cls.gm.evaluate('() => ({foundry: game.version, pf2e: game.system.version})')

    @classmethod
    def save_evidence(cls):
        cls.evidence['runtimeHashes'] = {
            side: {relative: hashlib.sha256((cls.env.data_dir(side) / 'modules' / MODULE_ID / relative).read_bytes()).hexdigest()
                   for relative in cls.evidence['sourceHashes'] if not relative.startswith('test/')}
            for side in ('main', 'char')
        }
        (cls.artifacts / 'module-usage.json').write_text(
            json.dumps(cls.evidence, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def run_engine(cls):
        report = cls.env.engine_once()
        cls.evidence.setdefault('engine', []).append(report)
        if report['errors']:
            raise AssertionError(report['errors'])

    def submit(self, actor_id, method='submitLevel'):
        copy = self.p1.evaluate('async ([url,id,method]) => (await import(url))[method](game.actors.get(id))',
                                [SUBMIT, actor_id, method])
        self.assertIsNotNone(copy)
        return copy

    def check_main(self):
        return self.gm.evaluate(f"async () => {{ await {API}.runApplyCheck({{trigger:'tick'}}); return {API}.lastCheck(); }}")

    def pending(self, copy):
        check = self.check_main()
        item = next(item for item in check['pending'] if item.get('copy', {}).get('copyId') == copy['copyId']
                    and item['copy']['rev'] == copy['rev'])
        self.assertNotEqual(item['reason'], 'error', item)
        return item

    def open_panel(self):
        self.gm.evaluate(f'() => {API}.openPanel()')
        panel = self.gm.locator('#taka-charsync-panel')
        panel.wait_for(timeout=WAIT)
        return panel

    def usage_section(self):
        return self.open_panel().get_by_role('heading', name='同步角色的模组提示', exact=True).locator('..')

    def preview(self, item, screenshot):
        actor_id = item['actorId']
        before = self.gm.evaluate(DOC, actor_id) if actor_id else None
        panel = self.open_panel()
        selector = f'[data-actor-id="{actor_id}"]' if actor_id else f'[data-card-id="{item["copy"]["cardId"]}"]'
        panel.locator(selector).get_by_role('button', name='查看变动', exact=True).click()
        dialog = self.gm.locator('.application.dialog').last
        dialog.wait_for(timeout=WAIT)
        reports = dialog.locator('.taka-charsync-module-usage')
        reports.locator('details').evaluate_all('elements => elements.forEach(element => element.open = true)')
        text = reports.all_text_contents()
        self.gm.evaluate('() => ui.notifications.clear()')
        dialog.screenshot(path=str(self.artifacts / screenshot))
        if actor_id:
            self.assertEqual(before, self.gm.evaluate(DOC, actor_id), '预览不能写卡')
        dialog.get_by_role('button', name='关闭', exact=True).click()
        self.evidence[screenshot] = {'copy': item['copy'], 'reports': text}
        return '\n'.join(text)

    def apply(self, item):
        self.gm.evaluate('async ([url,item]) => (await import(url)).applyPending(item)', [APPLY, item])
        doc = self.gm.evaluate(DOC, item['actorId'])
        return usage_by_id(doc['flags'][MODULE_ID]['applied']['moduleUsage'])

    def assert_report(self, text, module_id):
        self.assertIn(TITLES[module_id], text)
        self.assertIn(f'({module_id})', text)
        self.assertIn('未启用' if module_id == RULE else '本世界未找到（未安装或不兼容）', text)

    def tearDown(self):
        result = self._outcome.result
        if any(test is self for test, _ in result.failures + result.errors):
            for name in ('p1', 'gm'):
                page = getattr(self, name, None)
                if page and not page.is_closed():
                    page.screenshot(path=str(self.artifacts / f'{self._testMethodName}-{name}-failed.png'))

    def test_01_freeze_publish_preview_and_apply_use_imported_scope(self):
        items = [
            {'name': 'E2E 规则专长', 'type': 'feat', 'system': {
                'category': 'general', 'level': {'value': 1}, 'rules': [grant(RULE)]}},
            {'name': 'E2E 来源专长', 'type': 'feat', 'system': {'category': 'general', 'level': {'value': 1}},
             '_stats': {'compendiumSource': uuid(SOURCE)}, 'flags': {'core': {'sourceId': uuid(SOURCE)}}},
            {'name': 'E2E 编辑记录专长', 'type': 'feat', 'system': {'category': 'general', 'level': {'value': 1}},
             'flags': {FLAG: {'edited': True}}},
            equipment(),
        ]
        ids = self.p1.evaluate("""async ([id,items]) => (await game.actors.get(id)
          .createEmbeddedDocuments('Item', items)).map(item => item.id)""", [self.card, items])
        copy = self.submit(self.card)
        frozen = usage_by_id(copy['moduleUsage'])
        for module_id, kind in ((RULE, 'reference'), (SOURCE, 'source'), (FLAG, 'flag'), (GEAR, 'reference')):
            self.assertEqual(frozen[module_id]['title'], TITLES[module_id])
            self.assertEqual({entry['kind'] for entry in frozen[module_id]['evidence']}, {kind})
        self.assertTrue(any(entry['path'].endswith('_stats.compendiumSource') for entry in frozen[SOURCE]['evidence']))
        self.assertTrue(any(entry['path'].endswith('system.rules.0.uuid') for entry in frozen[RULE]['evidence']))
        self.p1.evaluate("""async ([id,itemId]) => game.actors.get(id).items.get(itemId)
          .update({'system.rules': []})""", [self.card, ids[0]])
        self.assertEqual(copy['moduleUsage'], self.p1.evaluate(
            "id => game.actors.get(id)._source.flags['taka-charsync'].copy.moduleUsage", copy['copyId']))
        self.run_engine()
        for side in ('main', 'char'):
            root = self.env.data_dir(side) / PUBLISH_DIR
            published = json.loads((root / 'copies' / f'{self.card}.json').read_text(encoding='utf-8'))
            entry = next(entry for entry in published if entry['copyId'] == copy['copyId'])
            self.assertEqual(entry['moduleUsage'], copy['moduleUsage'])
            history = json.loads((root / 'history' / f'copy-{copy["copyId"]}-{copy["rev"]}.json').read_text(encoding='utf-8'))
            self.assertEqual(history['flags'][MODULE_ID]['copy']['moduleUsage'], copy['moduleUsage'])
            feat = next(item for item in history['items'] if item['_id'] == ids[0])
            self.assertEqual(feat['system']['rules'][0]['uuid'], uuid(RULE))
        item = self.pending(copy)
        self.assertEqual(item['reason'], 'same-level')
        report = self.preview(item, 'normal-preview.png')
        for module_id in (RULE, SOURCE, FLAG):
            self.assert_report(report, module_id)
        self.assertIn('规则引用的模组', report)
        self.assertIn('模组数据，需核对', report)
        self.assertIn('内容来源（仅供参考）', report)
        self.assertNotIn(GEAR, report)
        applied = self.apply(item)
        self.assertTrue({RULE, SOURCE, FLAG}.issubset(applied))
        self.assertNotIn(GEAR, applied)
        self.assertFalse(self.gm.evaluate("id => game.actors.get(id).items.some(item => item.name === 'E2E 模块装备')", self.card))
        whisper = self.gm.evaluate("""moduleId => {
          const message = game.messages.contents.findLast(message => message.content.includes(moduleId));
          return {content: message?.content, whisper: message?.whisper,
            gmIds: game.users.filter(user => user.isGM).map(user => user.id)};
        }""", RULE)
        self.assertEqual(set(whisper['whisper']), set(whisper['gmIds']))
        self.assertNotIn(GEAR, whisper['content'])
        self.evidence['normal'] = {'frozen': copy['moduleUsage'], 'applied': list(applied.values()), 'whisper': whisper}

    def test_02_auto_apply_and_reopen_keep_module_hints(self):
        level = self.gm.evaluate("""async id => {
          const actor = game.actors.get(id);
          await actor.update({'system.details.xp.value': actor.system.details.xp.max});
          return actor.level;
        }""", self.auto_card)
        self.p1.evaluate("""async ([id,level,rule]) => {
          const actor = game.actors.get(id);
          await actor.update({'system.details.level.value': level + 1});
          await actor.createEmbeddedDocuments('Item', [{name: 'E2E 自动升级规则', type: 'feat',
            system: {category: 'general', level: {value: 1}, rules: [rule]}}]);
        }""", [self.auto_card, level, grant(RULE)])
        copy = self.submit(self.auto_card)
        self.run_engine()
        check = self.check_main()
        self.gm.wait_for_function("""([id,copyId,rev]) => {
          const ref = game.actors.get(id)._source.flags['taka-charsync']?.applied?.ref;
          return ref?.copyId === copyId && ref.rev === rev;
        }""", arg=[self.auto_card, copy['copyId'], copy['rev']], timeout=WAIT)
        self.assertFalse(any(item.get('actorId') == self.auto_card for item in check['pending']))
        self.assertEqual(self.gm.evaluate('id => game.actors.get(id).level', self.auto_card), level + 1)
        self.assertIn(RULE, usage_by_id(self.gm.evaluate(DOC, self.auto_card)['flags'][MODULE_ID]['applied']['moduleUsage']))
        self.assert_report(self.usage_section().text_content(), RULE)
        self.open_panel().locator('button[data-action="close"]').click()
        self.assert_report(self.usage_section().text_content(), RULE)
        self.gm.reload()
        self.gm.wait_for_function('() => game.ready === true', timeout=WAIT)
        section = self.usage_section()
        self.assert_report(section.text_content(), RULE)
        section.screenshot(path=str(self.artifacts / 'auto-applied-reopened.png'))
        self.evidence['automatic'] = {'copy': copy, 'check': check, 'panel': section.text_content()}

    def test_03_only_companion_usage_is_visible(self):
        self.p1.evaluate("""async ([id,moduleId,source]) => game.actors.get(id).createEmbeddedDocuments('Item', [{
          name: 'E2E 同伴模块能力', type: 'action', system: {actionType: {value: 'passive'}},
          flags: {[moduleId]: {edited: true}, core: {sourceId: source}},
        }])""", [self.familiar, PET, uuid(PET)])
        copy = self.submit(self.master)
        self.assertNotIn(PET, usage_by_id(copy['moduleUsage']))
        pet_copy = self.p1.evaluate("id => game.actors.get(id)._source.flags['taka-charsync'].copy", copy['companions'][0]['copyId'])
        self.assertIn(PET, usage_by_id(pet_copy['moduleUsage']))
        self.run_engine()
        item = self.pending(copy)
        self.assert_report(self.preview(item, 'companion-only-preview.png'), PET)
        self.assertNotIn(PET, self.apply(item))
        applied = self.gm.evaluate(DOC, self.familiar)['flags'][MODULE_ID]['applied']['moduleUsage']
        self.assertIn(PET, usage_by_id(applied))
        section = self.usage_section()
        self.assert_report(section.text_content(), PET)
        section.locator('details').evaluate_all('elements => elements.forEach(element => element.open = true)')
        section.screenshot(path=str(self.artifacts / 'companion-applied.png'))
        self.evidence['companion'] = {'copy': pet_copy, 'applied': applied, 'panel': section.text_content()}

    def test_04_force_sync_includes_equipment_usage(self):
        copy = self.submit(self.card, 'submitForce')
        self.run_engine()
        item = self.pending(copy)
        self.assertEqual(item['reason'], 'force')
        self.assert_report(self.preview(item, 'force-preview.png'), GEAR)
        applied = self.apply(item)
        self.assertIn(GEAR, applied)
        self.assertTrue(self.gm.evaluate("id => game.actors.get(id).items.some(item => item.name === 'E2E 模块装备')", self.card))
        self.evidence['force'] = {'copy': copy, 'applied': list(applied.values())}

    def test_05_new_card_import_includes_equipment_usage(self):
        copy = self.p1.evaluate("""async ([url,folder,world,gear]) => {
          const actor = await Actor.create({name: 'E2E 模块新卡', type: 'character', folder, items: [gear]});
          return (await import(url)).submitNew(actor, {targetWorld: world});
        }""", [SUBMIT, CHAR_CAMPAIGN_FOLDER, CAMP_WORLD, equipment()])
        self.assertIsNotNone(copy)
        self.run_engine()
        item = self.pending(copy)
        self.assertEqual(item['reason'], 'new')
        self.assert_report(self.preview(item, 'new-card-preview.png'), GEAR)
        actor_id = self.gm.evaluate('async ([url,item]) => (await import(url)).applyPending(item)', [APPLY, item])
        doc = self.gm.evaluate(DOC, actor_id)
        self.assertIn(GEAR, usage_by_id(doc['flags'][MODULE_ID]['applied']['moduleUsage']))
        self.evidence['newCard'] = {'copy': copy, 'applied': doc['flags'][MODULE_ID]['applied']['moduleUsage']}

    def test_06_native_manager_opens_without_changes_and_enabled_hint_disappears(self):
        before = self.gm.evaluate("() => game.settings.get('core', 'moduleConfiguration')")
        section = self.usage_section()
        self.assert_report(section.text_content(), RULE)
        section.get_by_role('button', name='打开模组管理', exact=True).click()
        manager = self.gm.locator('#module-management')
        manager.wait_for(timeout=WAIT)
        self.assertEqual(before, self.gm.evaluate("() => game.settings.get('core', 'moduleConfiguration')"))
        self.assertFalse(manager.locator(f'input[name="{RULE}"]').is_checked())
        self.assertFalse(self.gm.evaluate('id => game.modules.get(id).active', RULE))
        manager.screenshot(path=str(self.artifacts / 'native-module-management.png'))
        manager.locator(f'input[name="{RULE}"]').check()
        manager.locator('button[type="submit"]').click()
        dialog = self.gm.locator('.application.dialog').last
        dialog.locator('button[data-action="yes"]').wait_for(timeout=WAIT)
        dialog.screenshot(path=str(self.artifacts / 'native-reload-confirmation.png'))
        with self.gm.expect_navigation(wait_until='domcontentloaded', timeout=WAIT):
            dialog.locator('button[data-action="yes"]').click()
        self.gm.wait_for_function('() => game.ready === true', timeout=WAIT)
        self.assertTrue(self.gm.evaluate('id => game.modules.get(id).active', RULE))
        after = self.gm.evaluate("() => game.settings.get('core', 'moduleConfiguration')")
        self.assertTrue(after[RULE])
        self.assertTrue(after[MODULE_ID])
        section = self.usage_section()
        self.assertNotIn(f'({RULE})', section.text_content())
        self.assert_report(section.text_content(), GEAR)
        section.screenshot(path=str(self.artifacts / 'enabled-rule-hidden.png'))
        self.evidence['moduleManagement'] = {'before': before, 'after': after, 'panel': section.text_content()}


if __name__ == '__main__':
    unittest.main(verbosity=2)
