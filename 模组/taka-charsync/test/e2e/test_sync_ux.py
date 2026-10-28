"""同级修改、关联魔宠、首次成组导入及撤回旧操作的真实 Foundry 验收。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from harness import CHAR_CAMPAIGN_FOLDER, CAMP_WORLD, MODULE_ID, PUBLISH_DIR, Env, PageLogOnFailure, fixtures_dir


SUBMIT = f'/modules/{MODULE_ID}/scripts/char/submit.mjs'
APPLY = f'/modules/{MODULE_ID}/scripts/main/apply.mjs'
STATUS = f'/modules/{MODULE_ID}/scripts/char/status-bar.mjs'
API = f"game.modules.get('{MODULE_ID}').api"
WAIT = 60_000


class SyncUxTest(PageLogOnFailure, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-ux-e2e-')), fixtures_dir())
        cls.addClassCleanup(cls.env.close)
        cls.artifacts = Path(os.environ.get('CHARSYNC_E2E_ARTIFACTS', str(cls.env.root / 'screenshots')))
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        cls.env.build()
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
        cls.familiar = next(actor_id for actor_id, doc in cls.env.actors.items() if doc['type'] == 'familiar')
        cls.master = cls.env.actors[cls.familiar]['system']['master']['id']
        cls.card = next(card_id for card_id in cls.env.main_cards if card_id != cls.master)
        cls.p1.wait_for_function("""ids => ids.every(id => game.actors.get(id)?._source.flags['taka-charsync']?.base)""",
                                 arg=[cls.card, cls.master, cls.familiar], timeout=WAIT)
        cls.env.start('main')
        cls.gm, _ = cls.env.login('main', 'Gamemaster')
        cls.gm.wait_for_function(f'() => Boolean({API}.lastCheck())', timeout=WAIT)

    @classmethod
    def run_engine(cls):
        report = cls.env.engine_once()
        if report['errors']:
            raise AssertionError(report['errors'])

    def open_sheet(self, actor_id):
        self.p1.evaluate("id => game.actors.get(id).sheet.render(true)", actor_id)
        self.p1.wait_for_function("""id => {
          const bar = game.actors.get(id).sheet.element?.[0]?.querySelector('.taka-charsync-bar');
          return bar && bar.dataset.state !== 'loading';
        }""", arg=actor_id, timeout=WAIT)
        sheet_id = self.p1.evaluate('id => game.actors.get(id).sheet.id', actor_id)
        return self.p1.locator(f'#{sheet_id} .taka-charsync-bar')

    def check_main(self):
        return self.gm.evaluate(f"async () => {{ await {API}.runApplyCheck({{trigger: 'tick'}}); return {API}.lastCheck(); }}")

    def test_01_same_level_uses_gm_review_and_compact_details(self):
        bar = self.open_sheet(self.card)
        self.p1.evaluate("""async id => {
          await game.actors.get(id).createEmbeddedDocuments('Item', [
            {name: 'E2E 本级新专长', type: 'feat', system: {category: 'general', level: {value: 1}}},
          ]);
        }""", self.card)
        bar.get_by_role('button', name='提交本级修改', exact=True).wait_for(timeout=WAIT)
        details = bar.locator('details.taka-charsync-details')
        self.assertFalse(details.evaluate('(element) => element.open'))
        self.p1.evaluate('() => ui.notifications.clear()')
        bar.screenshot(path=str(self.artifacts / 'same-level-collapsed.png'))
        details.locator('summary').click()
        self.assertIn('E2E 本级新专长', details.inner_text())
        self.p1.evaluate("""async id => {
          await game.actors.get(id).update({'system.details.biography.appearance': 'E2E 新外观'});
        }""", self.card)
        self.p1.wait_for_function("""id => game.actors.get(id).sheet.element[0]
          .querySelector('details.taka-charsync-details')?.open""", arg=self.card, timeout=WAIT)
        self.p1.evaluate('() => ui.notifications.clear()')
        bar.screenshot(path=str(self.artifacts / 'same-level-expanded.png'))
        self.p1.evaluate('() => ui.notifications.clear()')
        bar.get_by_role('button', name='提交本级修改', exact=True).click()
        self.p1.wait_for_function("""id => game.actors.get(id).sheet.element[0]
          .querySelector('.taka-charsync-bar').innerText.includes('待 GM 确认')""", arg=self.card, timeout=WAIT)
        self.run_engine()
        check = self.check_main()
        item = next(item for item in check['pending'] if item.get('actorId') == self.card and item['reason'] == 'same-level')
        before = self.gm.evaluate("""async id => {
          const actor = game.actors.get(id);
          await actor.update({'system.attributes.hp.value': 1, 'system.details.xp.value': 200});
          return {id: actor.id, level: actor.level, hp: actor._source.system.attributes.hp.value,
            xp: actor._source.system.details.xp.value, names: actor.items.map(item => item.name)};
        }""", self.card)
        self.assertNotIn('E2E 本级新专长', before['names'])
        self.gm.evaluate(f'() => {API}.openPanel()')
        row = self.gm.locator(f'#taka-charsync-panel [data-actor-id="{self.card}"]')
        row.get_by_role('button', name='查看变动', exact=True).click()
        dialog = self.gm.locator('.application.dialog').last
        self.assertIn('E2E 本级新专长', dialog.inner_text())
        dialog.screenshot(path=str(self.artifacts / 'gm-same-level-preview.png'))
        dialog.get_by_role('button', name='关闭', exact=True).click()
        row.get_by_role('button', name='套用', exact=True).click()
        self.gm.wait_for_function("""id => game.actors.get(id).items.some(item => item.name === 'E2E 本级新专长')""",
                                  arg=self.card, timeout=WAIT)
        after = self.gm.evaluate("""id => {const a=game.actors.get(id);return {
          id:a.id, level:a.level, hp:a._source.system.attributes.hp.value, xp:a._source.system.details.xp.value};}""", self.card)
        self.assertEqual(after, {key: before[key] for key in ('id', 'level', 'hp', 'xp')})
        self.assertEqual(item['copy']['level'], before['level'])

    def test_02_only_familiar_changes_can_be_submitted_from_master(self):
        self.p1.evaluate("""async id => {
          await game.actors.get(id).createEmbeddedDocuments('Item', [
            {name: 'E2E 魔宠能力', type: 'action', system: {actionType: {value: 'passive'}}},
          ]);
        }""", self.familiar)
        bar = self.open_sheet(self.master)
        bar.get_by_role('button', name='提交本级修改', exact=True).wait_for(timeout=WAIT)
        self.assertIn(self.p1.evaluate('id => game.actors.get(id).name', self.familiar), bar.inner_text())
        bar.screenshot(path=str(self.artifacts / 'familiar-only-changes.png'))
        self.p1.evaluate('() => ui.notifications.clear()')
        bar.get_by_role('button', name='提交本级修改', exact=True).click()
        self.p1.wait_for_function("""id => game.actors.some(actor =>
          actor._source.flags['taka-charsync']?.copy?.cardId === id)""", arg=self.master, timeout=WAIT)
        self.run_engine()
        check = self.check_main()
        item = next(item for item in check['pending'] if item.get('actorId') == self.master)
        self.assertEqual(len(item['copy']['companions']), 1)
        preview = self.gm.evaluate('async ([url,item]) => (await import(url)).previewPending(item)', [APPLY, item])
        self.assertIn('E2E 魔宠能力', preview)
        self.gm.evaluate('async ([url,item]) => (await import(url)).applyPending(item)', [APPLY, item])
        self.assertTrue(self.gm.evaluate("id => game.actors.get(id).items.some(item => item.name === 'E2E 魔宠能力')", self.familiar))

    def test_03_new_character_and_familiar_are_imported_together(self):
        pair = self.p1.evaluate("""async folder => {
          const actor = await Actor.create({name: 'E2E 新主人', type: 'character', folder});
          const pet = await Actor.create({name: 'E2E 新魔宠', type: 'familiar', system: {master: {id: actor.id}}});
          return {actor: actor.id, pet: pet.id};
        }""", CHAR_CAMPAIGN_FOLDER)
        type(self).new_pair = pair
        bar = self.open_sheet(pair['actor'])
        self.p1.evaluate('() => ui.notifications.clear()')
        bar.get_by_role('button', name='作为新卡提交', exact=True).click()
        dialog = self.p1.locator('.application.dialog').last
        self.assertIn('E2E 新魔宠', dialog.inner_text())
        dialog.locator('select[name="world"]').select_option(CAMP_WORLD)
        dialog.locator('button[data-action="ok"]').click()
        self.p1.wait_for_function("""id => game.actors.get(id).sheet.element[0]
          .querySelector('.taka-charsync-bar').innerText.includes('待 GM 确认')""", arg=pair['actor'], timeout=WAIT)
        self.run_engine()
        check = self.check_main()
        item = next(item for item in check['pending'] if item.get('copy', {}).get('cardId') == pair['actor'])
        self.assertEqual(len(item['copy']['companions']), 1)
        self.assertFalse(any(entry.get('copy', {}).get('companionOf') for entry in check['pending']))
        self.gm.evaluate(f'() => {API}.openPanel()')
        row = self.gm.locator(f'#taka-charsync-panel [data-card-id="{pair["actor"]}"]')
        row.get_by_role('button', name='新增', exact=True).click()
        dialog = self.gm.locator('.application.dialog').last
        self.assertIn('E2E 新魔宠', dialog.inner_text())
        dialog.screenshot(path=str(self.artifacts / 'new-group-confirmation.png'))
        dialog.locator('button[data-action="yes"]').click()
        self.gm.wait_for_function("id => game.actors.some(a => a._source.flags['taka-charsync']?.charId === id)",
                                  arg=pair['pet'], timeout=WAIT)
        result = self.gm.evaluate("""pair => {
          const find = id => game.actors.find(a => a._source.flags['taka-charsync']?.charId === id);
          const actor=find(pair.actor), pet=find(pair.pet);
          return {actor:actor.id, pet:pet.id, master:pet._source.system.master.id,
            actorOwner:actor.ownership, petOwner:pet.ownership};
        }""", pair)
        self.assertEqual(result['master'], result['actor'])
        self.assertEqual(result['actorOwner'], result['petOwner'])
        type(self).new_main_pair = result

    def test_04_familiar_upgrade_preview_matches_applied_hp(self):
        before = self.gm.evaluate("""async ([ownerId, petId]) => {
          const owner=game.actors.get(ownerId), pet=game.actors.get(petId);
          await pet.update({'system.attributes.hp.value': pet.system.attributes.hp.max - 7});
          return {level:owner.level, hp:pet.system.attributes.hp.value, max:pet.system.attributes.hp.max};
        }""", [self.master, self.familiar])
        copy = self.p1.evaluate("""async ([url,id,level]) => {
          const actor=game.actors.get(id);
          await actor.update({'system.details.level.value':level});
          return (await import(url)).submitLevel(actor);
        }""", [SUBMIT, self.master, before['level'] + 1])
        self.run_engine()
        item = {'actorId': self.master, 'reason': 'manual', 'copy': copy}
        original = self.gm.evaluate('ids => ids.map(id => game.actors.get(id).toObject())', [self.master, self.familiar])
        preview = self.gm.evaluate('async ([url,item]) => (await import(url)).previewPending(item)', [APPLY, item])
        self.assertEqual(original, self.gm.evaluate('ids => ids.map(id => game.actors.get(id).toObject())', [self.master, self.familiar]))
        self.gm.evaluate('async ([url,item]) => (await import(url)).applyPending(item)', [APPLY, item])
        after = self.gm.evaluate("""id => {const pet=game.actors.get(id);return {
          hp:pet.system.attributes.hp.value, max:pet.system.attributes.hp.max};}""", self.familiar)
        self.assertGreater(after['max'], before['max'])
        self.assertEqual(after['hp'], before['hp'] + after['max'] - before['max'])
        self.assertIn(f"当前血量：{before['hp']} → {after['hp']}", preview)

    def test_05_withdrawn_copy_rejects_old_action_without_changes(self):
        copy = self.p1.evaluate('async ([url,id]) => (await import(url)).submitForce(game.actors.get(id))', [SUBMIT, self.card])
        self.run_engine()
        stale = {'actorId': self.card, 'reason': 'force', 'copy': copy}
        before = self.gm.evaluate('id => game.actors.get(id).toObject()', self.card)
        self.p1.evaluate('async ([url,id]) => (await import(url)).withdraw(game.actors.get(id))', [SUBMIT, copy['copyId']])
        self.run_engine()
        result = self.gm.evaluate("""async ([url,item]) => {
          try {await (await import(url)).applyPending(item); return {applied:true};}
          catch(error) {return {applied:false, error:error.message};}
        }""", [APPLY, stale])
        self.assertFalse(result['applied'])
        self.assertIn('撤回', result['error'])
        self.assertEqual(before, self.gm.evaluate('id => game.actors.get(id).toObject()', self.card))
        self.gm.context.close()
        self.env.stop('main')
        self.run_engine()
        meta = json.loads((self.env.data_dir('char') / PUBLISH_DIR / 'meta.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['cards'][self.new_pair['pet']]['companionOf'], self.new_pair['actor'])
        self.assertEqual(meta['cards'][self.new_pair['actor']]['mainId'], self.new_main_pair['actor'])
        self.assertEqual(meta['cards'][self.new_pair['pet']]['mainId'], self.new_main_pair['pet'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
