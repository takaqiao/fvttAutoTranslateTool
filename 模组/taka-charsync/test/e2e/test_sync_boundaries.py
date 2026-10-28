"""原生魔宠主人选择、外观提交与普通同步的库存和消耗边界。运行完整类。"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from harness import CHAR_CAMPAIGN_FOLDER, CAMP_WORLD, MODULE_ID, PUBLISH_DIR, Env, PageLogOnFailure, fixtures_dir, pf2e_dir


SUBMIT = f'/modules/{MODULE_ID}/scripts/char/submit.mjs'
API = f"game.modules.get('{MODULE_ID}').api"
OTHER_WORLD = 'e2e-other'
OTHER_FOLDER = 'CharsyncOther001'
GRANT_FEAT = 'E2EGrantFeat0001'
GRANT_GEAR = 'E2EGrantGear0001'
WAIT = 60_000
PHYSICAL = {'armor', 'backpack', 'book', 'consumable', 'equipment', 'treasure', 'weapon'}
DOC = 'id => game.actors.get(id).toObject()'


def inventory(doc):
    return sorted(({key: item.get(key) for key in ('_id', 'name', 'type', 'img', 'system', 'flags')}
                   for item in doc['items'] if item['type'] in PHYSICAL), key=lambda item: item['_id'])


def item_by_id(doc, item_id):
    return next(item for item in doc['items'] if item['_id'] == item_id)


class SyncBoundariesTest(PageLogOnFailure, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-boundaries-e2e-')), fixtures_dir())
        cls.addClassCleanup(cls.env.close)
        cls.artifacts = Path(os.environ.get('CHARSYNC_E2E_ARTIFACTS', str(cls.env.root / 'screenshots')))
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        cls.evidence = {'root': str(cls.env.root)}
        cls.addClassCleanup(lambda: (cls.artifacts / 'sync-boundaries.json').write_text(
            json.dumps(cls.evidence, ensure_ascii=False, indent=2), encoding='utf-8'))
        cls.env.build()
        cls.env.start('main')
        cls.env.start('char')
        cls.familiar = next(actor_id for actor_id, doc in cls.env.actors.items() if doc['type'] == 'familiar')
        cls.master = cls.env.actors[cls.familiar]['system']['master']['id']
        cls.caster = next(actor_id for actor_id in cls.env.main_cards if actor_id != cls.master and any(
            item['type'] == 'spellcastingEntry' and item['system']['prepared']['value'] == 'spontaneous'
            for item in cls.env.actors[actor_id]['items']))
        cls.visual = next(actor_id for actor_id in cls.env.main_cards if actor_id not in (cls.master, cls.caster))
        cls.env.actors[cls.caster]['items'].extend([
            {'_id': GRANT_FEAT, 'name': 'E2E 授予装备专长', 'type': 'feat',
             'system': {'category': 'general', 'level': {'value': 1}},
             'flags': {'pf2e': {'itemGrants': {'gift': {'id': GRANT_GEAR, 'onDelete': 'cascade'}}}}},
            {'_id': GRANT_GEAR, 'name': 'E2E 专长授予装备', 'type': 'equipment',
             'flags': {'pf2e': {'grantedBy': {'id': GRANT_FEAT, 'onDelete': 'cascade'}}}},
        ])
        cls.env.seed()
        gm, _ = cls.env.login('char', 'Gamemaster')
        cls.candidates = gm.evaluate("""async ([folder, otherFolder, backupOf]) => {
          const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
          permissions.ACTOR_CREATE = Object.values(CONST.USER_ROLES).filter(role => role > 0);
          await game.settings.set('core', 'permissions', permissions);
          await Folder.create({_id: otherFolder, name: 'E2E 另一团', type: 'Actor'}, {keepId: true});
          const ownership = {default: 0, [game.users.getName('P1').id]: 3};
          const create = async (name, data) => (await Actor.create({name, type: 'character', ownership, ...data})).id;
          return {
            working: await create('E2E 可改绑工作卡', {folder}),
            other: await create('E2E 跨团工作卡', {folder: otherFolder}),
            backup: await create('E2E 草稿备份', {folder, flags: {'taka-charsync': {backupOf}}}),
            unassigned: await create('E2E 未归属工作卡', {}),
          };
        }""", [CHAR_CAMPAIGN_FOLDER, OTHER_FOLDER, cls.visual])
        gm.context.close()
        cls.env.stop('main')

        # 第二个团由 Foundry 自己初始化，供引擎发布真实的跨团规则来源。
        other = cls.env.data_dir('main') / 'worlds' / OTHER_WORLD
        other.mkdir()
        manifest = json.loads((cls.env.data_dir('main') / 'worlds' / CAMP_WORLD / 'world.json').read_text(encoding='utf-8'))
        manifest.update(id=OTHER_WORLD, title='E2E 另一团')
        (other / 'world.json').write_text(json.dumps(manifest), encoding='utf-8')
        cls.env._launch('main', OTHER_WORLD)
        cls.env._wait_status('main', active=True)
        other_gm, _ = cls.env.login('main', 'Gamemaster')
        other_gm.context.close()
        cls.env.stop('main')
        config_path = cls.env.root / 'engine.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['worlds'].append(OTHER_WORLD)
        config['campaignFolders'][OTHER_FOLDER] = OTHER_WORLD
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        cls.run_engine()
        cls.p1, _ = cls.env.login('char', 'P1')
        cls.p1.wait_for_function("""ids => ids.every(id => game.actors.get(id)?._source.flags['taka-charsync']?.base)""",
                                 arg=[cls.visual, cls.caster, cls.master, cls.familiar], timeout=WAIT)
        cls.env.start('main')
        cls.gm, _ = cls.env.login('main', 'Gamemaster')
        cls.main_player, _ = cls.env.login('main', 'P1')
        cls.gm.wait_for_function(f'() => Boolean({API}.lastCheck())', timeout=WAIT)
        cls.evidence['versions'] = cls.gm.evaluate('() => ({foundry: game.version, pf2e: game.system.version})')

    @classmethod
    def run_engine(cls):
        report = cls.env.engine_once()
        if report['errors']:
            raise AssertionError(report['errors'])

    def open_sheet(self, actor_id):
        self.p1.evaluate('id => game.actors.get(id).sheet.render(true)', actor_id)
        self.p1.wait_for_function("""id => {
          const bar = game.actors.get(id).sheet.element?.[0]?.querySelector('.taka-charsync-bar');
          return bar && bar.dataset.state !== 'loading';
        }""", arg=actor_id, timeout=WAIT)
        sheet_id = self.p1.evaluate('id => game.actors.get(id).sheet.id', actor_id)
        return self.p1.locator(f'#{sheet_id}')

    def submit(self, actor_id, label):
        sheet = self.open_sheet(actor_id)
        expected = self.p1.evaluate("""async id => {
          const {copyIdFor}=await import('/modules/taka-charsync/scripts/lib/chain.mjs');
          const copyId=await copyIdFor(id,game.actors.get(id).level);
          return {copyId,rev:game.actors.get(copyId)?._source.flags['taka-charsync']?.copy?.rev ?? null};
        }""", actor_id)
        self.p1.evaluate('() => ui.notifications.clear()')
        sheet.get_by_role('button', name=label, exact=True).click()
        self.p1.wait_for_function("""({copyId,rev}) => {
          const copy=game.actors.get(copyId)?._source.flags['taka-charsync']?.copy;
          return copy?.kind==='level' && !copy.withdrawn && copy.rev!==rev;
        }""", arg=expected, timeout=WAIT)
        return self.p1.evaluate("id => game.actors.get(id)._source.flags['taka-charsync'].copy", expected['copyId'])

    def preview_and_apply(self, actor_id, copy, screenshot, included=(), excluded=()):
        self.run_engine()
        check = self.gm.evaluate(f"async () => {{ await {API}.runApplyCheck({{trigger:'tick'}}); return {API}.lastCheck(); }}")
        entries = [item for item in check['pending'] if item.get('actorId') == actor_id]
        pending = next(item for item in entries if item.get('copy', {}).get('copyId') == copy['copyId']
                       and item['copy']['rev'] == copy['rev'])
        self.assertNotEqual(pending['reason'], 'error', pending)
        before = self.gm.evaluate(DOC, actor_id)
        self.gm.evaluate(f'() => {API}.openPanel()')
        row = self.gm.locator(f'#taka-charsync-panel [data-actor-id="{actor_id}"] li.taka-charsync-item').nth(entries.index(pending))
        self.assertEqual(row.get_attribute('data-reason'), pending['reason'])
        old_dialogs = self.gm.locator('.application.dialog').count()
        row.get_by_role('button', name='查看变动', exact=True).click()
        self.gm.wait_for_function("count => document.querySelectorAll('.application.dialog').length > count",
                                  arg=old_dialogs, timeout=WAIT)
        dialog = self.gm.locator('.application.dialog').last
        dialog.wait_for(timeout=WAIT)
        preview = dialog.inner_text()
        self.gm.evaluate('() => ui.notifications.clear()')
        dialog.screenshot(path=str(self.artifacts / screenshot))
        self.evidence[screenshot] = {'copy': copy, 'preview': preview, 'before': before}
        try:
            for text in included:
                self.assertIn(text, preview)
            for text in excluded:
                self.assertNotIn(text, preview)
            self.assertEqual(before, self.gm.evaluate(DOC, actor_id), '预览不能写卡')
        finally:
            dialog.get_by_role('button', name='关闭', exact=True).click()
        row.get_by_role('button', name='套用', exact=True).click()
        self.gm.wait_for_function("""([id,copyId,rev]) => {
          const ref=game.actors.get(id)._source.flags['taka-charsync']?.applied?.ref;
          return ref?.copyId === copyId && ref.rev === rev;
        }""", arg=[actor_id, copy['copyId'], copy['rev']], timeout=WAIT)
        after = self.gm.evaluate(DOC, actor_id)
        self.evidence[screenshot]['after'] = after
        return before, after, preview

    def tearDown(self):
        result = self._outcome.result
        if any(test is self for test, _ in result.failures + result.errors):
            for name in ('p1', 'gm'):
                page = getattr(self, name, None)
                if page and not page.is_closed():
                    page.screenshot(path=str(self.artifacts / f'{self._testMethodName}-{name}-failed.png'))
                    self.evidence[f'{self._testMethodName}-{name}-failed'] = page.evaluate(
                        '() => game.actors.map(actor => actor.toObject())')

    def test_01_native_familiar_master_select_persists_and_filters(self):
        self.assertFalse(self.p1.evaluate('() => game.user.isGM'))
        copy = self.p1.evaluate('async ([url,id]) => (await import(url)).submitForce(game.actors.get(id))', [SUBMIT, self.visual])
        pet = self.p1.evaluate("""async folder => (await Actor.create({
          name:'E2E 原生主人绑定', type:'familiar', folder,
        })).id""", CHAR_CAMPAIGN_FOLDER)
        sheet = self.open_sheet(pet)
        select = sheet.locator('select[name="system.master.id"]')
        self.assertTrue(select.is_enabled())
        options = select.locator('option').evaluate_all('(options) => options.map(option => ({value:option.value, text:option.textContent}))')
        ids = [option['value'] for option in options]
        self.assertIn(self.visual, ids)
        self.assertIn(self.candidates['working'], ids)
        for rejected in (copy['copyId'], self.candidates['backup'], self.candidates['other'], self.candidates['unassigned']):
            self.assertNotIn(rejected, ids)
        select.select_option(self.visual)
        self.p1.wait_for_function('([id,master]) => game.actors.get(id)._source.system.master.id === master',
                                 arg=[pet, self.visual], timeout=WAIT)
        self.p1.reload()
        self.env._wait_ready(self.p1)
        self.assertEqual(self.p1.evaluate('id => game.actors.get(id)._source.system.master.id', pet), self.visual)
        sheet = self.open_sheet(pet)
        self.p1.evaluate('() => ui.notifications.clear()')
        sheet.screenshot(path=str(self.artifacts / 'native-familiar-master.png'))
        sheet.locator('select[name="system.master.id"]').select_option(self.candidates['working'])
        self.p1.wait_for_function('([id,master]) => game.actors.get(id)._source.system.master.id === master',
                                 arg=[pet, self.candidates['working']], timeout=WAIT)
        sheet.locator('select[name="system.master.id"]').select_option('')
        self.p1.wait_for_function('id => !game.actors.get(id)._source.system.master.id', arg=pet, timeout=WAIT)
        self.p1.reload()
        self.env._wait_ready(self.p1)
        self.assertFalse(self.p1.evaluate('id => game.actors.get(id)._source.system.master.id', pet))
        self.assertTrue(self.open_sheet(pet).locator('select[name="system.master.id"]').is_enabled())
        self.evidence['native-master'] = {'pet': pet, 'options': options, 'bound': self.visual,
                                          'rebound': self.candidates['working'], 'unboundPersisted': True}
        self.p1.evaluate('async ([url,id]) => (await import(url)).withdraw(game.actors.get(id))', [SUBMIT, copy['copyId']])
        self.p1.evaluate('async id => { await game.actors.get(id).sheet.close(); }', pet)

    def test_02_only_appearance_changes_submit_and_keep_main_resources(self):
        sheet = self.open_sheet(self.visual)
        original = self.p1.evaluate(DOC, self.visual)
        paths = [
            ('img', 'systems/pf2e/icons/default-icons/familiar.svg'),
            ('prototypeToken.texture.src', 'systems/pf2e/icons/default-icons/npc.svg'),
            ('prototypeToken.ring.subject.scale', 0.8),
        ]
        for path, value in paths:
            with self.subTest(path=path):
                old = original
                for key in path.split('.'):
                    old = old[key]
                self.p1.evaluate('async ([id,path,value]) => { await game.actors.get(id).update({[path]:value}); }',
                                 [self.visual, path, value])
                sheet.get_by_role('button', name='提交本级修改', exact=True).wait_for(timeout=WAIT)
                self.p1.evaluate('async ([id,path,value]) => { await game.actors.get(id).update({[path]:value}); }',
                                 [self.visual, path, old])
                sheet.get_by_role('button', name='提交本级修改', exact=True).wait_for(state='hidden', timeout=WAIT)

        texture = 'e2e-media/ring-only.svg'
        target = self.env.data_dir('char') / texture
        target.parent.mkdir()
        shutil.copyfile(pf2e_dir() / 'icons/default-icons/familiar.svg', target)
        self.p1.evaluate("""async ([id,texture]) => { await game.actors.get(id).update({
          'prototypeToken.ring.subject.texture':texture,
        }); }""", [self.visual, texture])
        sheet.get_by_role('button', name='提交本级修改', exact=True).wait_for(timeout=WAIT)
        self.p1.evaluate('() => ui.notifications.clear()')
        sheet.locator('.taka-charsync-bar').screenshot(path=str(self.artifacts / 'ring-only-submit.png'))
        self.gm.evaluate("""async id => {
          const actor=game.actors.get(id);
          const physical=actor.items.find(item => item.isOfType('physical') && item.system.quantity > 1);
          await physical.update({'system.quantity':1});
          await actor.update({'system.attributes.hp.value':1, 'system.resources.heroPoints.value':2, 'system.details.xp.value':200});
        }""", self.visual)
        copy = self.submit(self.visual, '提交本级修改')
        before, after, _ = self.preview_and_apply(self.visual, copy, 'gm-ring-only-preview.png', included=('ring.subject.texture',))
        self.assertEqual(inventory(after), inventory(before))
        self.assertEqual(after['system']['resources'], before['system']['resources'])
        self.assertEqual(after['system']['attributes']['hp'], before['system']['attributes']['hp'])
        self.assertEqual(after['system']['details']['xp'], before['system']['details']['xp'])
        self.assertEqual(after['img'], before['img'])
        self.assertEqual(after['prototypeToken']['texture'], before['prototypeToken']['texture'])
        self.assertEqual(after['prototypeToken']['ring']['subject']['texture'], texture)
        self.assertEqual((self.env.data_dir('main') / texture).read_bytes(), target.read_bytes())
        response = self.gm.request.get(f'{self.env.url("main")}/{texture}')
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body(), target.read_bytes())

    def test_03_level_preserves_inventory_and_expended_slots(self):
        self.open_sheet(self.caster)
        source = self.p1.evaluate(DOC, self.caster)
        quantity = next(item for item in source['items'] if item['type'] in PHYSICAL and item['system']['quantity'] > 1)
        deleted = next(item for item in source['items'] if item['type'] in PHYSICAL and item['_id'] != quantity['_id']
                       and item['type'] != 'backpack')
        entry = next(item for item in source['items'] if item['type'] == 'spellcastingEntry'
                     and item['system']['prepared']['value'] == 'spontaneous')
        self.assertEqual(entry['system']['slots']['slot4']['max'], 0)
        self.assertEqual(entry['system']['slots']['slot4']['value'], 0)
        level = source['system']['details']['level']['value'] + 1
        self.gm.evaluate("""async ([id,quantity,entry]) => {
          const actor=game.actors.get(id);
          await actor.items.get(quantity).update({'system.quantity':1, 'system.equipped.carryType':'worn'});
          await actor.items.get(entry).update({'system.slots.slot1.value':0});
          await actor.update({'system.resources.focus.value':0, 'system.resources.heroPoints.value':2,
            'system.details.xp.value':actor.system.details.xp.max+37});
          await actor.createEmbeddedDocuments('Item', [{name:'E2E 主服战利品',type:'equipment'}]);
        }""", [self.caster, quantity['_id'], entry['_id']])
        added = self.p1.evaluate("""async ([id,quantity,deleted,entry,level,grantFeat,grantGear]) => {
          const actor=game.actors.get(id);
          await actor.items.get(quantity).update({'system.quantity':88, 'system.equipped.carryType':'dropped'});
          await actor.deleteEmbeddedDocuments('Item',[deleted]);
          await actor.deleteEmbeddedDocuments('Item',[grantFeat,grantGear]);
          await actor.items.get(entry).update({'system.slots.slot1.value':2, 'system.slots.slot1.max':3,
            'system.slots.slot4.value':1, 'system.slots.slot4.max':1});
          await actor.update({'system.details.level.value':level, 'system.resources.focus.value':1});
          const items=await actor.createEmbeddedDocuments('Item', [
            {name:'E2E 车卡试算装备',type:'equipment'},
            {name:'E2E 升级构筑专长',type:'feat',system:{category:'general',level:{value:1}}},
          ]);
          return items[0].id;
        }""", [self.caster, quantity['_id'], deleted['_id'], entry['_id'], level, GRANT_FEAT, GRANT_GEAR])
        self.assertFalse(self.p1.evaluate('([id,items]) => items.some(item => game.actors.get(id).items.has(item))',
                                         [self.caster, [GRANT_FEAT, GRANT_GEAR]]))
        copy = self.submit(self.caster, f'提交 {level} 级升级')
        before, after, _ = self.preview_and_apply(self.caster, copy, 'gm-level-inventory-preview.png',
                                                included=('E2E 升级构筑专长',),
                                                excluded=('E2E 车卡试算装备', 'E2E 主服战利品', deleted['name'], quantity['name']))
        self.assertEqual(inventory(after), inventory(before))
        self.assertNotIn(GRANT_FEAT, [item['_id'] for item in after['items']])
        # Foundry 整卡写回会重建所有 Item 的 _stats，物品其余字段仍须逐项一致。
        self.assertEqual({key: value for key, value in item_by_id(after, GRANT_GEAR).items() if key != '_stats'},
                         {key: value for key, value in item_by_id(before, GRANT_GEAR).items() if key != '_stats'})
        self.assertEqual(after['system']['resources'], before['system']['resources'])
        self.assertEqual(after['system']['details']['level']['value'], level)
        self.assertEqual(after['system']['details']['xp']['value'], 37)
        self.assertEqual(item_by_id(after, entry['_id'])['system']['slots']['slot1']['value'], 0)
        self.assertEqual(item_by_id(after, entry['_id'])['system']['slots']['slot1']['max'], 3)
        self.assertEqual(item_by_id(after, entry['_id'])['system']['slots']['slot4']['max'], 1)
        self.assertEqual(item_by_id(after, entry['_id'])['system']['slots']['slot4']['value'], 1)
        self.assertNotIn(added, [item['_id'] for item in after['items']])
        type(self).inventory_case = {'quantity': quantity['_id'], 'deleted': deleted['_id'], 'entry': entry['_id'], 'added': added}

    def test_04_force_still_replaces_inventory_and_spell_consumption(self):
        source = self.p1.evaluate(DOC, self.caster)
        copy = self.p1.evaluate('async ([url,id]) => (await import(url)).submitForce(game.actors.get(id))', [SUBMIT, self.caster])
        before, after, _ = self.preview_and_apply(self.caster, copy, 'gm-force-inventory-preview.png',
                                                included=('E2E 车卡试算装备', 'E2E 主服战利品'))
        self.assertNotEqual(inventory(after), inventory(before))
        self.assertEqual(inventory(after), inventory(source))
        self.assertEqual(after['system']['details']['xp'], before['system']['details']['xp'])
        entry = self.inventory_case['entry']
        self.assertEqual(item_by_id(after, entry)['system']['slots']['slot1']['value'], 2)

    def test_05_existing_familiar_changes_follow_master(self):
        self.p1.evaluate("""async id => { await game.actors.get(id).createEmbeddedDocuments('Item', [
          {name:'E2E 同伴新能力',type:'action',system:{actionType:{value:'passive'}}},
        ]); }""", self.familiar)
        copy = self.submit(self.master, '提交本级修改')
        self.assertEqual([entry['sourceId'] for entry in copy['companions']], [self.familiar])
        pet_before = self.gm.evaluate(DOC, self.familiar)
        before, after, _ = self.preview_and_apply(self.master, copy, 'gm-companion-boundary-preview.png',
                                                included=('E2E 同伴新能力',), excluded=('当前血量',))
        pet_after = self.gm.evaluate(DOC, self.familiar)
        self.assertEqual(inventory(before), inventory(after))
        self.assertEqual(after['system']['attributes']['hp'], before['system']['attributes']['hp'])
        self.assertEqual(pet_after['system']['master']['id'], self.master)
        self.assertEqual(pet_after['system']['attributes']['hp'], pet_before['system']['attributes']['hp'])
        self.assertIn('E2E 同伴新能力', [item['name'] for item in pet_after['items']])

    def test_06_catch_up_refreshes_main_inventory_and_keeps_build_drafts(self):
        case = self.inventory_case
        self.gm.evaluate("""async ([id,quantity,entry]) => {
          const actor=game.actors.get(id);
          await actor.items.get(quantity).update({'system.quantity':5});
          await actor.items.get(entry).update({'system.slots.slot1.value':0});
          await actor.update({'system.resources.focus.value':0});
          await actor.createEmbeddedDocuments('Item', [{name:'E2E 主服新战利品',type:'equipment'}]);
        }""", [self.caster, case['quantity'], case['entry']])
        expected = self.gm.evaluate(DOC, self.caster)
        self.p1.evaluate("""async ([id,quantity]) => {
          const actor=game.actors.get(id);
          await actor.items.get(quantity).update({'system.quantity':99});
          await actor.createEmbeddedDocuments('Item', [
            {name:'E2E 尚未提交构筑',type:'feat',system:{category:'general',level:{value:1}}},
          ]);
          await actor.sheet.close();
        }""", [self.caster, case['quantity']])
        self.main_player.context.close()
        self.gm.context.close()
        self.env.stop('main')
        self.run_engine()
        latest = json.loads((self.env.data_dir('char') / PUBLISH_DIR / 'main' / CAMP_WORLD / f'{self.caster}.json')
                            .read_text(encoding='utf-8'))
        sheet = self.open_sheet(self.caster)
        self.p1.wait_for_function("""([id,hash]) => game.actors.get(id)._source.flags['taka-charsync']?.base?.hash===hash""",
                                 arg=[self.caster, latest['hash']], timeout=WAIT)
        refreshed = self.p1.evaluate(DOC, self.caster)
        self.assertEqual(inventory(refreshed), inventory(expected))
        self.assertEqual(refreshed['system']['resources'], expected['system']['resources'])
        self.assertEqual(item_by_id(refreshed, case['entry'])['system']['slots']['slot1']['value'], 0)
        self.assertIn('E2E 尚未提交构筑', [item['name'] for item in refreshed['items']])
        base_hash = refreshed['flags'][MODULE_ID]['base']['hash']

        self.p1.evaluate("""async ([id,quantity]) => {
          const actor=game.actors.get(id);
          await actor.items.get(quantity).update({'system.quantity':42});
          await actor.createEmbeddedDocuments('Item', [
            {name:'E2E 同基线试算物品',type:'equipment'},
            {name:'E2E 同基线构筑',type:'feat',system:{category:'general',level:{value:1}}},
          ]);
        }""", [self.caster, case['quantity']])
        draft = self.p1.evaluate(DOC, self.caster)
        sheet.get_by_role('button', name='刷新状态', exact=True).click()
        self.p1.wait_for_function("""id => !game.actors.get(id).sheet.element[0]
          .querySelector('.taka-charsync-bar button[disabled]')""", arg=self.caster, timeout=WAIT)
        self.assertEqual(self.p1.evaluate(DOC, self.caster), draft, '只刷新状态不应写卡')
        self.p1.evaluate('async id => { await game.actors.get(id).sheet.close(); }', self.caster)
        sheet = self.open_sheet(self.caster)
        self.p1.wait_for_function("""([id,item,quantity]) => game.actors.get(id).items.get(item)._source.system.quantity===quantity""",
                                 arg=[self.caster, case['quantity'], item_by_id(expected, case['quantity'])['system']['quantity']],
                                 timeout=WAIT)
        restored = self.p1.evaluate(DOC, self.caster)
        self.assertEqual(restored['flags'][MODULE_ID]['base']['hash'], base_hash)
        self.assertEqual(inventory(restored), inventory(expected))
        self.assertIn('E2E 尚未提交构筑', [item['name'] for item in restored['items']])
        self.assertIn('E2E 同基线构筑', [item['name'] for item in restored['items']])
        self.p1.evaluate('() => ui.notifications.clear()')
        sheet.locator('.taka-charsync-bar').screenshot(path=str(self.artifacts / 'main-inventory-catch-up.png'))
        self.evidence['catch-up'] = {'main': expected, 'first': refreshed, 'sameHashDraft': draft, 'restored': restored}


if __name__ == '__main__':
    unittest.main(verbosity=2)
