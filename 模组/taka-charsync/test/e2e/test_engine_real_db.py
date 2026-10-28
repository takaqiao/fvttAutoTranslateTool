"""引擎读 Foundry 自己写出来的库：两个本机实例种好夹具卡，主服 GM 改一张卡的 hp，关掉团世界后跑引擎一轮。

用法：python -X utf8 test_engine_real_db.py
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness import CAMP_WORLD, MODULE_ID, PUBLISH_DIR, Env, fixtures_dir

HP_LOSS = 5

LOSE_HP = """async ([id, loss]) => {
  const actor = game.actors.get(id);
  const before = actor._source.system.attributes.hp.value;
  await actor.update({'system.attributes.hp.value': before - loss});
  return {before, after: actor._source.system.attributes.hp.value};
}"""


def read_rules(rules_file: Path) -> dict:
    """引擎写出的 `generated/rules.mjs` 里的 RULES 表。"""
    script = 'import(process.argv[1]).then(m => process.stdout.write(JSON.stringify(m.RULES)))'
    result = subprocess.run(['node', '-e', script, rules_file.as_uri()], capture_output=True, text=True, encoding='utf-8', check=True)
    return json.loads(result.stdout)


class EngineRealDbTest(unittest.TestCase):
    def test_publishes_main_card_and_rules_from_real_worlds(self):
        fixtures = fixtures_dir()
        with Env(Path(tempfile.mkdtemp(prefix='charsync-e2e-')), fixtures) as env:
            env.build()
            env.start('main')
            env.start('char')
            env.seed()

            card_id = env.main_cards[0]
            page, _ = env.login('main', 'Gamemaster')
            hp = page.evaluate(LOSE_HP, [card_id, HP_LOSS])
            page.context.close()
            self.assertEqual(hp['after'], hp['before'] - HP_LOSS)

            env.stop('main')
            report = env.engine_once()

            self.assertEqual(report['errors'], [])
            self.assertEqual(report['worldsRead'], [CAMP_WORLD])
            self.assertTrue(report['charRead'])

            char_data = env.data_dir('char')
            entry_file = char_data / PUBLISH_DIR / 'main' / CAMP_WORLD / f'{card_id}.json'
            self.assertTrue(entry_file.is_file(), entry_file)
            entry = json.loads(entry_file.read_text(encoding='utf-8'))
            self.assertEqual(entry['level'], env.actors[card_id]['system']['details']['level']['value'])
            self.assertEqual(entry['doc']['system']['attributes']['hp']['value'], hp['after'])

            rules = read_rules(char_data / 'modules' / MODULE_ID / 'generated' / 'rules.mjs')
            self.assertIs(rules[CAMP_WORLD]['pf2e.freeArchetypeVariant'], True)

            # 起步基线：种子卡和它们的同伴（幻灵列在种子里，魔宠不进种子、靠 system.master.id 认）；
            # seed() 之后只有主服那张卡动过
            seed_ids = {card['charId'] for card in env.seed_cards}
            familiars = [aid for aid, doc in env.actors.items()
                         if doc['type'] == 'familiar' and doc['system'].get('master', {}).get('id') in seed_ids]
            self.assertTrue(familiars)
            targets = [cid for card in env.seed_cards for cid in [card['charId'], *card.get('companions', [])]] + familiars
            bases = json.loads((env.root / 'state' / 'bootstrap.json').read_text(encoding='utf-8'))
            self.assertEqual(
                {cid: base['case'] for cid, base in bases.items()},
                {cid: 'char-unchanged' if cid == card_id else 'main-unchanged' for cid in targets},
            )
            meta = json.loads((char_data / PUBLISH_DIR / 'meta.json').read_text(encoding='utf-8'))
            self.assertEqual(set(meta['bootstrap']), set(targets))


if __name__ == '__main__':
    unittest.main()
