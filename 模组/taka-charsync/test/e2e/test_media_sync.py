"""真实 Tokenizer 保存、媒体冻结、发布闭包及配方追平。按顺序运行完整类。"""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageDraw

import test_sync_boundaries as boundaries
from harness import CAMP_WORLD, MODULE_DIR, MODULE_ID, PUBLISH_DIR, Env, PageLogOnFailure, fixtures_dir


WAIT = 90_000
MEDIA = 'assets/charsync-media-e2e'
DOC = 'id => game.actors.get(id).toObject()'
TOKENIZER = 'tokenizer-2'
API = f"game.modules.get('{MODULE_ID}').api"
AVATAR_FIELDS = ('img', 'flags.tokenizer-2.avatarLayerStack', 'flags.tokenizer-2.avatarAspect', 'flags.tokenizer-2.avatarCanvasSize')
TOKEN_FIELDS = ('prototypeToken.texture', 'prototypeToken.ring.enabled', 'prototypeToken.ring.subject',
                'prototypeToken.flags.pf2e.autoscale', 'prototypeToken.flags.tokenizer-2',
                *('flags.tokenizer-2.' + key for key in ('layerStack', 'isDynamicRingMode', 'isOversized',
                  'oversizedScale', 'oversizedRingFill', 'modeIsOversized', 'modeOversizedScale', 'modeOversizedRingFill')))

OPEN_EDITOR = """async ({id, mode, offset, masked}) => {
  await game.settings.set('tokenizer-2', 'tour-shown', true);
  await game.settings.set('tokenizer-2', 'tour-version', game.modules.get('tokenizer-2').version);
  const actor = game.actors.get(id);
  const editor = window.mediaEditor = new Tokenizer2.TokenImageEditor({actor});
  await editor.render({force: true});
  if (mode === 'avatar') await editor.modeSwitcher.setMode('avatar');
  const layer = editor.layerManager.layers.find(layer => layer.type === 'image' && !layer.isFrame && !layer.isDynamicRingPreview);
  if (!layer) throw new Error('原生编辑器没有图像层');
  layer.transform.x += offset;
  if (masked) {
    layer.masks = [{id: foundry.utils.randomID(), type: 'alpha', src: 'assets/charsync-media-e2e/alpha.png', ringConfig: null}];
    layer.paintMask = {src: 'assets/charsync-media-e2e/paint.png'};
    await editor.canvasEngine.preloadMasks(editor.layerManager.layers, editor.maskManager);
  }
  editor._scheduleRender();
  await editor.render();
  return {mode: editor._editorMode, layers: editor.layerManager.serialise()};
}"""

SAVE_EDITOR = """async ({method, mode, folder}) => {
  const editor = window.mediaEditor;
  const overrides = {folder, filename: 'same-name.webp'};
  let result;
  if (method === 'saveBoth') {
    const other = editor._editorMode === 'avatar' ? 'token' : 'avatar';
    if (!editor.save.getModeLayers(other)?.length) {
      const active = editor._editorMode;
      await editor.modeSwitcher.setMode(other);
      await editor.modeSwitcher.setMode(active);
    }
    result = await editor.save.saveBoth({overrides: {token: overrides, avatar: {...overrides}}});
  } else if (method === 'saveHeadless') {
    result = await editor.save.saveHeadless(mode, editor.layerManager.serialise(), {overrides});
    await editor.close();
  } else result = await editor.save[method]({overrides});
  if (!result) throw new Error(`原生 ${method} 未成功`);
  return editor.actor.toObject();
}"""


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media_path(src):
    return unquote(urlsplit(src).path).lstrip('/')


def fields(doc, paths):
    result = {}
    for path in paths:
        value = doc
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            result[path] = value
    return result


def source_hashes():
    paths = [*MODULE_DIR.glob('scripts/**/*.mjs'), *MODULE_DIR.glob('engine/**/*.mjs'),
             Path(__file__), MODULE_DIR / 'test/e2e/harness.py']
    return {path.relative_to(MODULE_DIR).as_posix(): sha256(path) for path in sorted(paths)}


def create_assets(data):
    folder = data / MEDIA
    folder.mkdir(parents=True)
    (folder / 'exports').mkdir()
    image = Image.new('RGBA', (256, 256), '#172947')
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 24, 232, 232), fill='#e7a847')
    draw.rectangle((112, 40, 144, 216), fill='#e8f0f6')
    image.save(folder / 'source.png')
    alpha = Image.new('RGBA', (256, 256), 'white')
    ImageDraw.Draw(alpha).rectangle((0, 0, 48, 256), fill=(0, 0, 0, 0))
    alpha.save(folder / 'alpha.png')
    paint = Image.new('RGBA', (256, 256), 'white')
    ImageDraw.Draw(paint).ellipse((160, 160, 230, 230), fill=(0, 0, 0, 0))
    paint.save(folder / 'paint.png')


class MediaSyncTest(PageLogOnFailure, unittest.TestCase):
    open_sheet = boundaries.SyncBoundariesTest.open_sheet
    preview_and_apply = boundaries.SyncBoundariesTest.preview_and_apply

    @classmethod
    def setUpClass(cls):
        bundle = Path(os.environ['CHARSYNC_TOKENIZER_DIR'])
        cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-media-e2e-')), fixtures_dir())
        cls.addClassCleanup(cls.env.close)
        cls.artifacts = Path(os.environ.get('CHARSYNC_E2E_ARTIFACTS', str(cls.env.root / 'screenshots')))
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        cls.evidence = {'root': str(cls.env.root), 'sourceHashes': {'initial': source_hashes()}}
        cls.addClassCleanup(cls.save_evidence)
        cls.env.build()
        cls.evidence['managedBeforeEngine'] = {side: (cls.env.data_dir(side) / 'assets/taka-charsync-media').exists()
                                              for side in ('main', 'char')}
        if any(cls.evidence['managedBeforeEngine'].values()):
            raise AssertionError('受管目录必须由真实引擎初始化')
        for side in ('main', 'char'):
            shutil.copytree(bundle, cls.env.data_dir(side) / 'modules' / TOKENIZER)
            create_assets(cls.env.data_dir(side))
        for doc in cls.env.actors.values():
            doc['img'] = f'{MEDIA}/source.png'
            doc['prototypeToken']['texture']['src'] = f'{MEDIA}/source.png'
            doc['prototypeToken']['ring']['enabled'] = False
            doc['prototypeToken']['ring']['subject']['texture'] = f'{MEDIA}/source.png'
            doc.get('flags', {}).pop(TOKENIZER, None)
            for item in doc['items']:
                item['img'] = 'icons/svg/item-bag.svg'
        cls.familiar = next(id for id, doc in cls.env.actors.items() if doc['type'] == 'familiar')
        cls.master = cls.env.actors[cls.familiar]['system']['master']['id']
        cls.visual = next(id for id in cls.env.main_cards if id != cls.master)
        cls.env.start('main')
        cls.env.start('char')
        cls.env.seed()
        for side in ('main', 'char'):
            gm, _ = cls.env.login(side, 'Gamemaster')
            gm.evaluate("""async side => {
              await game.settings.set('core', 'language', 'en');
              await game.settings.set('core', 'moduleConfiguration', {...game.settings.get('core', 'moduleConfiguration'), 'tokenizer-2': true});
              const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
              permissions.FILES_UPLOAD = permissions.FILES_BROWSE = Object.values(CONST.USER_ROLES).filter(role => role > 0);
              if (side === 'char') permissions.ACTOR_CREATE = [...new Set([...permissions.ACTOR_CREATE, CONST.USER_ROLES.PLAYER])];
              await game.settings.set('core', 'permissions', permissions);
            }""", side)
            gm.reload()
            cls.env._wait_ready(gm)
            gm.wait_for_function('() => Boolean(window.Tokenizer2)', timeout=WAIT)
            gm.evaluate("async () => { await game.settings.set('tokenizer-2', 'tour-shown', true); await game.settings.set('tokenizer-2', 'tour-version', game.modules.get('tokenizer-2').version); }")
            gm.context.close()
        cls.env._write_seed()
        cls.env.stop('main')
        cls.run_engine()
        cls.evidence['managedAfterEngine'] = {side: (cls.env.data_dir(side) / 'assets/taka-charsync-media').is_dir()
                                             for side in ('main', 'char')}
        if not all(cls.evidence['managedAfterEngine'].values()):
            raise AssertionError('引擎没有初始化双端受管媒体目录')
        cls.p1, _ = cls.env.login('char', 'P1')
        cls.p1.wait_for_function("ids => ids.every(id => game.actors.get(id)?._source.flags['taka-charsync']?.base)",
                                 arg=[cls.master, cls.familiar, cls.visual], timeout=WAIT)
        cls.env.start('main')
        cls.gm, _ = cls.env.login('main', 'Gamemaster')
        cls.main_player, _ = cls.env.login('main', 'P1')
        cls.evidence['versions'] = cls.gm.evaluate("() => ({foundry: game.version, pf2e: game.system.version, tokenizer: game.modules.get('tokenizer-2').version})")
        cls.save_evidence()

    @classmethod
    def save_evidence(cls):
        write_json(cls.artifacts / 'media-sync.json', cls.evidence)

    @classmethod
    def run_engine(cls):
        report = cls.env.engine_once()
        cls.evidence.setdefault('engine', []).append(report)
        cls.save_evidence()
        if report['errors']:
            raise AssertionError(report['errors'])
        return report

    def native_save(self, page, actor_id, method, mode='token', offset=0, masked=True, screenshot=None):
        prepared = page.evaluate(OPEN_EDITOR, {'id': actor_id, 'mode': mode, 'offset': offset, 'masked': masked})
        if screenshot:
            page.evaluate('() => ui.notifications.clear()')
            page.screenshot(path=str(self.artifacts / screenshot))
        saved = page.evaluate(SAVE_EDITOR, {'method': method, 'mode': mode, 'folder': f'{MEDIA}/exports'})
        return {'prepared': prepared, 'doc': saved}

    def submit(self, actor_id, label):
        sheet = self.open_sheet(actor_id)
        expected = self.p1.evaluate("""async id => {
          const {copyIdFor} = await import('/modules/taka-charsync/scripts/lib/chain.mjs');
          const copyId = await copyIdFor(id, game.actors.get(id).level);
          window.mediaSubmissionError = null;
          window.mediaOriginalError = ui.notifications.error;
          ui.notifications.error = function(message, ...args) {
            window.mediaSubmissionError = String(message);
            return window.mediaOriginalError.call(this, message, ...args);
          };
          return {copyId, rev: game.actors.get(copyId)?._source.flags['taka-charsync']?.copy?.rev ?? null};
        }""", actor_id)
        try:
            self.p1.evaluate('() => ui.notifications.clear()')
            sheet.get_by_role('button', name=label, exact=True).click()
            self.p1.wait_for_function("""({copyId, rev}) => {
              const copy = game.actors.get(copyId)?._source.flags['taka-charsync']?.copy;
              return window.mediaSubmissionError || (copy?.kind === 'level' && !copy.withdrawn && copy.rev !== rev);
            }""", arg=expected, timeout=WAIT)
            error = self.p1.evaluate('() => window.mediaSubmissionError')
            self.assertIsNone(error, error)
            return self.p1.evaluate("id => game.actors.get(id)._source.flags['taka-charsync'].copy", expected['copyId'])
        finally:
            self.p1.evaluate('() => { ui.notifications.error = window.mediaOriginalError; }')

    def tearDown(self):
        result = self._outcome.result
        if any(test is self for test, _ in result.failures + result.errors):
            for name in ('p1', 'gm', 'main_player'):
                page = getattr(self, name, None)
                if page and not page.is_closed():
                    page.screenshot(path=str(self.artifacts / f'{self._testMethodName}-{name}-failed.png'))
                    self.evidence[f'{self._testMethodName}-{name}-failed'] = page.evaluate('() => game.actors.map(actor => actor.toObject())')
        self.save_evidence()

    def test_01_native_save_entry_points_create_immutable_files_on_both_servers(self):
        outputs = []
        self.evidence['nativeSaves'] = outputs
        for side, page in (('main', self.main_player), ('char', self.p1)):
            self.assertFalse(page.evaluate('() => game.user.isGM'))
            for method, mode in [('saveToken', 'token'), ('saveAvatar', 'avatar'), ('saveBoth', 'token'),
                                 ('saveHeadless', 'token'), ('saveHeadless', 'avatar')]:
                previous = None
                for iteration in (1, 2):
                    saved = self.native_save(page, self.visual, method, mode, offset=iteration * 7,
                                             screenshot=f'{side}-{method}-{mode}.png' if iteration == 1 else None)
                    doc = saved['doc']
                    paths = [doc['img']] if mode == 'avatar' else [doc['prototypeToken']['texture']['src']]
                    if method == 'saveBoth':
                        paths.append(doc['img'])
                        self.assertNotEqual(media_path(paths[0]), media_path(paths[1]))
                        self.assertEqual(Path(media_path(paths[0])).stem.rsplit('-token-', 1)[1],
                                         Path(media_path(paths[1])).stem.rsplit('-avatar-', 1)[1])
                    files = {media_path(src): sha256(self.env.data_dir(side) / media_path(src)) for src in paths}
                    seen = {path for prior in outputs if prior['side'] == side for path in prior['files']}
                    self.assertTrue(seen.isdisjoint(files), '每个保存入口都须分配新路径')
                    outputs.append({'side': side, 'method': method, 'mode': mode, 'iteration': iteration, 'files': files, **saved})
                    self.save_evidence()
                    if previous:
                        self.assertTrue(set(previous).isdisjoint(files), '连续保存必须采用不同文件路径')
                    previous = files
            for saved in outputs:
                for path, digest in saved['files'].items():
                    self.assertEqual(sha256(self.env.data_dir(saved['side']) / path), digest, '后续保存不能改写先前图片字节')

    def references(self, page, actor_id):
        return page.evaluate("""async id => {
          const {collectMediaReferences} = await import('/modules/taka-charsync/scripts/lib/media.mjs');
          return collectMediaReferences(game.actors.get(id).toObject());
        }""", actor_id)

    def test_02_submitted_group_freezes_bytes_and_requires_complete_masks_before_apply(self):
        self.p1.reload()
        self.env._wait_ready(self.p1)
        case = self.evidence['freeze'] = {}
        before_refs = {}
        for actor_id in (self.master, self.familiar):
            self.native_save(self.p1, actor_id, 'saveBoth', offset=11,
                             screenshot=f'char-{actor_id}-submitted-editor.png')
            before_refs[actor_id] = {tuple(ref['path']): sha256(self.env.data_dir('char') / media_path(ref['src']))
                                    for ref in self.references(self.p1, actor_id) if ref['src'].startswith(MEDIA)}
        working_before = {id: self.p1.evaluate(DOC, id) for id in (self.master, self.familiar)}
        copy = self.submit(self.master, '提交本级修改')
        self.assertEqual([ref['sourceId'] for ref in copy['companions']], [self.familiar])
        copied = {self.master: copy['copyId'], self.familiar: copy['companions'][0]['copyId']}
        frozen = {id: self.p1.evaluate(DOC, copy_id) for id, copy_id in copied.items()}
        case.update(copy=copy, workingBefore=working_before, frozen=frozen)
        self.save_evidence()
        frozen_files = {}
        for actor_id, copy_id in copied.items():
            refs = {tuple(ref['path']): ref['src'] for ref in self.references(self.p1, copy_id)}
            for path, digest in before_refs[actor_id].items():
                src = refs[path]
                self.assertTrue(src.startswith('assets/taka-charsync-media/'), src)
                self.assertEqual(Path(src).stem, digest)
                self.assertEqual(sha256(self.env.data_dir('char') / src), digest)
                frozen_files[src] = digest
            working_after = self.p1.evaluate(DOC, actor_id)
            self.assertEqual(fields(working_after, AVATAR_FIELDS + TOKEN_FIELDS), fields(working_before[actor_id], AVATAR_FIELDS + TOKEN_FIELDS))
        self.assertTrue(any('masks' in path for refs in before_refs.values() for path in refs))
        self.assertTrue(any('paintMask' in path for refs in before_refs.values() for path in refs))

        pet_refs = self.references(self.p1, copied[self.familiar])
        mask = next(ref['src'] for ref in pet_refs if 'paintMask' in ref['path'])
        source_mask = self.env.data_dir('char') / mask
        held_mask = source_mask.with_suffix(source_mask.suffix + '.withheld')
        self.assertFalse((self.env.data_dir('main') / mask).exists())
        source_mask.rename(held_mask)
        try:
            blocked = self.env.engine_once()
            case['missingMask'] = {'path': mask, 'report': blocked}
            self.save_evidence()
            self.assertTrue(any(error['scope'] == 'images' and mask in error['message'] for error in blocked['errors']), blocked)
            for side in ('main', 'char'):
                published = self.env.data_dir(side) / PUBLISH_DIR
                meta = json.loads((published / 'meta.json').read_text(encoding='utf-8'))
                for copy_id in copied.values():
                    self.assertNotIn(f"copy:{copy_id}:{copy['rev']}", meta['published'])
                    self.assertFalse((published / 'history' / f"copy-{copy_id}-{copy['rev']}.json").exists())
            for actor_id, copy_id in copied.items():
                self.assertEqual(self.p1.evaluate(DOC, copy_id), frozen[actor_id], '缺媒体不得改副本或另造版本')
        finally:
            held_mask.rename(source_mask)
        case['repairedReport'] = self.run_engine()
        for side in ('main', 'char'):
            published = self.env.data_dir(side) / PUBLISH_DIR
            meta = json.loads((published / 'meta.json').read_text(encoding='utf-8'))
            for copy_id in copied.values():
                self.assertIn(f"copy:{copy_id}:{copy['rev']}", meta['published'])
                self.assertTrue((published / 'history' / f"copy-{copy_id}-{copy['rev']}.json").is_file())
            for src, digest in frozen_files.items():
                self.assertEqual(sha256(self.env.data_dir(side) / src), digest)

        # 工作图和原始素材随后改变，先前提交仍须按当时的像素与配方套用。
        Image.new('RGBA', (256, 256), '#d1356f').save(self.env.data_dir('char') / MEDIA / 'source.png')
        for actor_id in (self.master, self.familiar):
            self.native_save(self.p1, actor_id, 'saveBoth', offset=29)
        case['laterDrafts'] = {id: self.p1.evaluate(DOC, id) for id in copied}
        gear = self.gm.evaluate("""async id => {
          const actor = game.actors.get(id);
          const item = actor.items.find(item => item.type === 'equipment');
          if (!item) throw new Error('主卡夹具缺少物理装备');
          await item.update({'system.quantity': 4});
          return item.id;
        }""", self.master)
        self.p1.evaluate("async ([id, gear]) => { await game.actors.get(id).items.get(gear).update({'system.quantity': 99}); }", [self.master, gear])
        pet_before = self.gm.evaluate(DOC, self.familiar)
        before, after, preview = self.preview_and_apply(self.master, copy, 'gm-frozen-media-preview.png')
        pet_after = self.gm.evaluate(DOC, self.familiar)
        case.update(before=before, after=after, petBefore=pet_before, petAfter=pet_after, preview=preview, files=frozen_files)
        self.save_evidence()
        self.assertEqual(boundaries.inventory(before), boundaries.inventory(after))
        self.assertEqual(before['system']['resources'], after['system']['resources'])
        for actor_id, actual in ((self.master, after), (self.familiar, pet_after)):
            self.assertEqual(fields(actual, AVATAR_FIELDS + TOKEN_FIELDS), fields(frozen[actor_id], AVATAR_FIELDS + TOKEN_FIELDS))
            for ref in self.references(self.gm, actor_id):
                if ref['src'].startswith('assets/taka-charsync-media/'):
                    self.assertEqual(sha256(self.env.data_dir('main') / ref['src']), frozen_files[ref['src']])
        self.assertEqual(pet_after['system']['master']['id'], self.master)
        self.gm.evaluate('id => game.actors.get(id).sheet.render(true)', self.master)
        self.gm.screenshot(path=str(self.artifacts / 'main-frozen-media-applied.png'))

    def publish_main_and_catch_up(self, expected):
        self.main_player.context.close()
        self.gm.context.close()
        self.env.stop('main')
        self.run_engine()
        latest = json.loads((self.env.data_dir('char') / PUBLISH_DIR / 'main' / CAMP_WORLD / f'{self.master}.json').read_text(encoding='utf-8'))
        self.p1.evaluate('async id => { await game.actors.get(id).sheet.close(); }', self.master)
        self.open_sheet(self.master)
        self.p1.wait_for_function("([id, hash]) => game.actors.get(id)._source.flags['taka-charsync']?.base?.hash === hash",
                                 arg=[self.master, latest['hash']], timeout=WAIT)
        doc = self.p1.evaluate(DOC, self.master)
        self.assertEqual(boundaries.inventory(doc), boundaries.inventory(expected))
        return doc

    def test_03_catch_up_keeps_atomic_drafts_and_reopens_matching_recipes(self):
        case = self.evidence['catchUp'] = {}
        main = self.gm.evaluate(DOC, self.master)
        draft = self.p1.evaluate(DOC, self.master)
        first = self.publish_main_and_catch_up(main)
        case.update(main=main, draft=draft, first=first)
        self.save_evidence()
        self.assertEqual(fields(first, AVATAR_FIELDS + TOKEN_FIELDS), fields(draft, AVATAR_FIELDS + TOKEN_FIELDS))

        # 回到当前主服头像，再独立编辑 token；下一次主服头像更新应整组进入。
        self.p1.evaluate("""async ({id, avatar}) => { await game.actors.get(id).update(avatar); }""",
                         {'id': self.master, 'avatar': fields(main, AVATAR_FIELDS)})
        token_draft = self.p1.evaluate(DOC, self.master)
        self.env.start('main')
        type(self).gm, _ = self.env.login('main', 'Gamemaster')
        type(self).main_player, _ = self.env.login('main', 'P1')
        saved_avatar = self.native_save(self.main_player, self.master, 'saveAvatar', mode='avatar', offset=41,
                                       screenshot='main-avatar-independent-edit.png')
        self.gm.wait_for_function('([id, img]) => game.actors.get(id)._source.img === img',
                                  arg=[self.master, saved_avatar['doc']['img']], timeout=WAIT)
        changed_main = self.gm.evaluate(DOC, self.master)
        merged = self.publish_main_and_catch_up(changed_main)
        case.update(changedMain=changed_main, tokenDraft=token_draft, merged=merged)
        self.save_evidence()
        self.assertEqual(fields(merged, AVATAR_FIELDS), fields(changed_main, AVATAR_FIELDS))
        self.assertEqual(fields(merged, TOKEN_FIELDS), fields(token_draft, TOKEN_FIELDS))

        for mode, stack in (('token', 'layerStack'), ('avatar', 'avatarLayerStack')):
            reopened = self.p1.evaluate(OPEN_EDITOR, {'id': self.master, 'mode': mode, 'offset': 0, 'masked': False})
            self.p1.screenshot(path=str(self.artifacts / f'char-catch-up-{mode}-editor.png'))
            stored = merged['flags'][TOKENIZER][stack]
            for field in ('src', 'transform', 'adjustments', 'masks', 'paintMask'):
                self.assertEqual([layer.get(field) for layer in reopened['layers']], [layer.get(field) for layer in stored], field)
            self.p1.evaluate('async () => { await window.mediaEditor.close(); }')
            case[f'reopened{mode}'] = reopened
        before_second = self.p1.evaluate(DOC, self.master)
        self.p1.evaluate('async id => { await game.actors.get(id).sheet.close(); }', self.master)
        self.open_sheet(self.master)
        # 原生状态栏读取完成后再核对同 hash 分支；旧 DOM 不能作为完成信号。
        self.p1.wait_for_function("id => !game.actors.get(id).sheet.element[0].querySelector('.taka-charsync-bar button[disabled]')", arg=self.master, timeout=WAIT)
        self.p1.wait_for_timeout(1000)
        second = self.p1.evaluate(DOC, self.master)
        case.update(beforeSameHash=before_second, sameHash=second)
        self.save_evidence()
        self.assertEqual(fields(second, AVATAR_FIELDS + TOKEN_FIELDS), fields(before_second, AVATAR_FIELDS + TOKEN_FIELDS))
        self.assertEqual(boundaries.inventory(second), boundaries.inventory(changed_main))
        # PF2e 关闭卡面会提交 prepared HP/SP/resolve；先等追平校正，再独立观察同版本重复调用。
        repeat = self.p1.evaluate("""async id => {
          const actor = game.actors.get(id);
          const {catchUpCard} = await import('/modules/taka-charsync/scripts/char/catch-up.mjs');
          const {getMeta} = await import('/modules/taka-charsync/scripts/foundry/published.mjs');
          const meta = await getMeta();
          await catchUpCard(actor, {meta});
          const before = actor.toObject();
          const writes = [];
          const hook = Hooks.on('updateActor', (updated, changes) => {
            if (updated.id === id) writes.push(foundry.utils.deepClone(changes));
          });
          try {
            const result = await catchUpCard(actor, {meta});
            return {before, after: actor.toObject(), result, writes};
          } finally {
            Hooks.off('updateActor', hook);
          }
        }""", self.master)
        case['sameHashRepeat'] = repeat
        self.save_evidence()
        self.assertEqual(repeat['result'], {'action': 'none'})
        self.assertEqual(repeat['writes'], [], '同 hash 第二次追平不能重复改写或丢配方')
        self.assertEqual(repeat['after'], repeat['before'])

    def test_04_final_storage_uploads_new_bytes_without_browse_permission(self):
        gm, _ = self.env.login('char', 'Gamemaster')
        try:
            gm.evaluate("""async () => {
              const permissions = foundry.utils.deepClone(game.settings.get('core', 'permissions'));
              permissions.FILES_BROWSE = permissions.FILES_BROWSE.filter(role => role !== CONST.USER_ROLES.PLAYER);
              await game.settings.set('core', 'permissions', permissions);
            }""")
        finally:
            gm.context.close()
        self.p1.reload()
        self.env._wait_ready(self.p1)
        case = self.evidence['uploadWithoutBrowse'] = {}
        case['permissions'] = self.p1.evaluate("() => ({isGM:game.user.isGM, upload:game.user.can('FILES_UPLOAD'), browse:game.user.can('FILES_BROWSE'), actorCreate:game.user.can('ACTOR_CREATE')})")
        self.assertEqual(case['permissions'], {'isGM': False, 'upload': True, 'browse': False, 'actorCreate': True})
        source = self.env.data_dir('char') / MEDIA / 'upload-only-new.png'
        Image.new('RGBA', (41, 37), '#186f4d').save(source)
        digest = sha256(source)
        managed = f'assets/taka-charsync-media/{digest}.png'
        case.update(source=f'{MEDIA}/upload-only-new.png', managed=managed, sha256=digest,
                    existedBefore=(self.env.data_dir('char') / managed).exists())
        self.save_evidence()
        self.assertFalse(case['existedBefore'], '必须上传此前不存在的新字节，不能只走已有文件回读')
        self.p1.evaluate("""async ({id, src}) => {
          await game.actors.get(id).update({img: src, 'flags.tokenizer-2.-=avatarLayerStack': null,
            'flags.tokenizer-2.-=avatarAspect': null, 'flags.tokenizer-2.-=avatarCanvasSize': null});
          const picker = foundry.applications.apps.FilePicker.implementation;
          window.mediaOriginalUpload = picker.upload;
          window.mediaUploadReceipts = [];
          picker.upload = async function(...args) {
            const receipt = await window.mediaOriginalUpload.apply(this, args);
            window.mediaUploadReceipts.push({source:args[0], target:args[1], filename:args[2].name, receipt});
            return receipt;
          };
        }""", {'id': self.master, 'src': case['source']})
        try:
            case['copy'] = self.submit(self.master, '提交本级修改')
        finally:
            case['receipts'] = self.p1.evaluate("""() => {
              foundry.applications.apps.FilePicker.implementation.upload = window.mediaOriginalUpload;
              return window.mediaUploadReceipts;
            }""")
            self.save_evidence()
        frozen = self.p1.evaluate(DOC, case['copy']['copyId'])
        case['frozen'] = frozen
        case['httpSha256'] = self.p1.evaluate("""async src => {
          const response = await fetch(src, {cache:'no-store'});
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const digest = await crypto.subtle.digest('SHA-256', await response.arrayBuffer());
          return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
        }""", '/' + managed)
        self.p1.screenshot(path=str(self.artifacts / 'player-upload-without-browse.png'))
        self.save_evidence()
        self.assertEqual(frozen['img'], managed)
        self.assertEqual(sha256(self.env.data_dir('char') / managed), digest)
        self.assertEqual(case['httpSha256'], digest)
        self.assertTrue(any(receipt['receipt'].get('path') == managed for receipt in case['receipts']))
        refs = frozen['flags']['taka-charsync-media']['references']
        self.assertTrue(any(ref['path'] == ['img'] and ref['frozen'] == managed and ref['sha256'] == digest for ref in refs))


if __name__ == '__main__':
    unittest.main(verbosity=2)
