"""部署工具实跑：enable-worlds、reset-xp 在本机两个 Foundry 实例的真库上走 dry 与 apply，install.sh 在临时目录里装，
再查按 HEAD 用 git archive 打出来的 install.sh 是 LF。

两个世界种好夹具卡、不启用模组。工具离线写库时实例停着，写完再起实例以 GM 登录读回。建卡服的进程状态由一个假的
pm2（经 `PM2_BIN` 传入）报告，控制器的状态地址默认指向没人监听的端口；要它有应答、或主服状态接口返回坏数据时，
在随机端口上临时起一个 HTTP 服务。

用法：python -X utf8 test_tools.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness import CAMP_WORLD, CHAR_WORLD, FOUNDRY_APP, MODULE_DIR, MODULE_ID, Env, PageLogOnFailure, fixtures_dir

TOOLS = MODULE_DIR / 'tools'
PUBLISH_DIR = 'charsync-tools-e2e-0000'
EXTRA_MODULE = 'e2e-extra-module'
# 种子时已经在主服的启用项里，启用本模组后要原样留着
OTHER_MODULE = 'e2e-other-module'
CONTROL_PORT = 30183
ROLE_NAMES = {'submitted': '已提交', 'withdrawn': '已撤回', 'backup': '草稿备份'}

PM2_STOPPED = [
    {'name': 'fvtt-v14-char', 'pm2_env': {'status': 'stopped'}},
    {'name': 'fvtt-v14-char-control', 'pm2_env': {'status': 'stopped'}},
]
PM2_CORE_ONLINE = [
    {'name': 'fvtt-v14-char', 'pm2_env': {'status': 'online'}},
    {'name': 'fvtt-v14-char-control', 'pm2_env': {'status': 'stopped'}},
]

SEED_MAIN = """async ({actors, xp, otherModule}) => {
  const modules = game.settings.get('core', 'moduleConfiguration');
  await game.settings.set('core', 'moduleConfiguration', {...modules, [otherModule]: true});
  for (const data of actors) {
    data.ownership = {default: 0};
    data.folder = null;
    await Actor.create(data, {keepId: true});
  }
  for (const [id, value] of Object.entries(xp)) await game.actors.get(id).update({'system.details.xp.value': value});
}"""

MODULE_STATE = """moduleId => ({
  active: game.modules.get(moduleId)?.active === true,
  publishDir: game.settings.get(moduleId, 'publishDir'),
  modules: game.settings.get('core', 'moduleConfiguration'),
  folders: game.folders.filter(folder => folder.type === 'Actor' && folder.getFlag(moduleId, 'role'))
    .map(folder => [folder.getFlag(moduleId, 'role'), folder.name]),
})"""

ACTOR_STATE = """ids => Object.fromEntries(ids.map(id => {
  const source = game.actors.get(id)._source;
  return [id, {name: source.name, level: source.system.details.level?.value ?? null, xp: source.system.details.xp?.value ?? null}];
}))"""

# 读一个库目录的全部行（原样字符串），调用方先把目录拷出来
READ_ROWS = """
const [appDir, dir] = process.argv.slice(1);
const {createRequire} = await import('node:module');
const path = await import('node:path');
const {ClassicLevel} = createRequire(path.join(appDir, 'package.json'))('classic-level');
const db = new ClassicLevel(dir, {keyEncoding: 'utf8', valueEncoding: 'utf8', createIfMissing: false});
await db.open();
const rows = {};
for await (const [key, value] of db.iterator()) rows[key] = value;
await db.close();
process.stdout.write(JSON.stringify(rows));
"""

# 打开库占住 LOCK，直到 stdin 关掉
HOLD_LOCK = """
const [appDir, dir] = process.argv.slice(1);
const {createRequire} = await import('node:module');
const path = await import('node:path');
const {ClassicLevel} = createRequire(path.join(appDir, 'package.json'))('classic-level');
const db = new ClassicLevel(dir);
await db.open();
console.log('locked');
process.stdin.resume();
process.stdin.on('end', async () => { await db.close(); process.exit(0); });
"""


def file_state(dirs) -> dict:
    """目录里每个文件的大小、修改时间和 sha256。"""
    state = {}
    for directory in dirs:
        for file in sorted(directory.iterdir()):
            if file.is_file():
                stat = file.stat()
                state[str(file)] = (stat.st_size, stat.st_mtime_ns, hashlib.sha256(file.read_bytes()).hexdigest())
    return state


def write_fake_pm2(file: Path, processes: list) -> None:
    file.write_text(f'@echo off\r\necho {json.dumps(processes, separators=(",", ":"))}\r\n', encoding='ascii')


def git_bash() -> str:
    """Git 自带的 bash：Windows 上 PATH 里的 bash 可能是 WSL 的。`CHARSYNC_BASH` 可以指定。"""
    if os.environ.get('CHARSYNC_BASH'):
        return os.environ['CHARSYNC_BASH']
    if os.name != 'nt':
        return 'bash'
    # git.exe 可能是 <Git>/cmd/git.exe，也可能是 <Git>/mingw64/bin/git.exe
    git = shutil.which('git')
    for parent in Path(git).resolve().parents if git else []:
        if (parent / 'bin' / 'bash.exe').is_file():
            return str(parent / 'bin' / 'bash.exe')
    raise FileNotFoundError('找不到 Git 自带的 bash，设 CHARSYNC_BASH 指向 bash.exe')


@contextmanager
def http_server(body: bytes):
    """在随机端口上起一个对任何 GET 都回 200 和 `body` 的 HTTP 服务，产出它的地址。"""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/api/status'
    finally:
        server.shutdown()
        server.server_close()


class DeployToolsTest(PageLogOnFailure, unittest.TestCase):
    env: Env

    @classmethod
    def setUpClass(cls):
        cls.env = Env(Path(tempfile.mkdtemp(prefix='charsync-e2e-')), fixtures_dir())
        try:
            cls.env.build()
            cls.env.start('main')
            cls.env.start('char')
            cls.cards = cls.env.main_cards[:3]
            cls.xp = {cls.cards[0]: 1000, cls.cards[1]: 640, cls.cards[2]: 0}
            page, _ = cls.env.login('main', 'Gamemaster')
            page.evaluate(SEED_MAIN, {'actors': list(cls.env.actors.values()), 'xp': cls.xp, 'otherModule': OTHER_MODULE})
            page.context.close()
            # 建卡服只要 GM 进过一次，世界设置和用户表就齐了
            page, _ = cls.env.login('char', 'Gamemaster')
            page.context.close()
            cls.tmp_root = cls.env.root / 'tool-tmp'
            cls.tmp_root.mkdir()
            cls.pm2_stopped = cls.env.root / 'pm2-stopped.cmd'
            write_fake_pm2(cls.pm2_stopped, PM2_STOPPED)
            cls.pm2_core_online = cls.env.root / 'pm2-core-online.cmd'
            write_fake_pm2(cls.pm2_core_online, PM2_CORE_ONLINE)
        except BaseException:
            cls.env.close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def run_tool(self, name: str, **variables) -> tuple[int, dict]:
        env = {
            **os.environ,
            'FOUNDRY_APP': str(FOUNDRY_APP),
            'MAIN_DATA': str(self.env.data_dir('main')),
            'CHAR_DATA': str(self.env.data_dir('char')),
            'MAIN_STATUS_URL': f"{self.env.url('main')}/api/status",
            'CHAR_STATUS_URL': f"{self.env.url('char')}/api/status",
            'CONTROL_URL': f'http://127.0.0.1:{CONTROL_PORT}/control/status',
            'PM2_BIN': str(self.pm2_stopped),
            'TMP_ROOT': str(self.tmp_root),
            **variables,
        }
        result = subprocess.run(['node', str(TOOLS / name)], env=env, capture_output=True, text=True, encoding='utf-8')
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.fail(f'{name} 没有输出 JSON 报告（退出码 {result.returncode}）：\n{result.stdout}\n{result.stderr}')
        self.assertEqual(list(self.tmp_root.iterdir()), [], f'{name} 没删掉临时副本')
        return result.returncode, report

    def db_dir(self, which: str, name: str) -> Path:
        world = CAMP_WORLD if which == 'main' else CHAR_WORLD
        return self.env.data_dir(which) / 'worlds' / world / 'data' / name

    def world_dbs(self) -> list[Path]:
        return [self.db_dir('main', 'settings'), self.db_dir('main', 'users'), self.db_dir('main', 'actors'),
                self.db_dir('char', 'settings'), self.db_dir('char', 'folders'), self.db_dir('char', 'users')]

    def read_rows(self, db: Path) -> dict:
        copy = self.env.root / 'rows-copy'
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(db, copy, ignore=shutil.ignore_patterns('LOCK'))
        try:
            result = subprocess.run(['node', '--input-type=module', '-e', READ_ROWS, str(FOUNDRY_APP), str(copy)],
                                    capture_output=True, text=True, encoding='utf-8', check=True)
            return json.loads(result.stdout)
        finally:
            shutil.rmtree(copy)

    def enable(self, mode: str, worlds: str, **variables) -> tuple[int, dict]:
        return self.run_tool('enable-worlds.mjs', MODE=mode, WORLDS=worlds, PUBLISH_DIR=PUBLISH_DIR,
                             CHAR_EXTRA_MODULES=EXTRA_MODULE, **variables)

    def test_1_apply_refuses_while_worlds_are_open_or_locked(self):
        both = f'{CAMP_WORLD},{CHAR_WORLD}'
        backup = self.env.root / 'backup-refused'

        code, report = self.enable('apply', both, BACKUP=str(backup))
        self.assertEqual(code, 1)
        self.assertFalse(report['ok'])
        self.assertTrue(any(f'主服正开着 {CAMP_WORLD}' in p for p in report['problems']), report['problems'])
        self.assertTrue(any('建卡服没有完全停下' in p for p in report['problems']), report['problems'])
        self.assertFalse(backup.exists())

        code, report = self.run_tool('reset-xp.mjs', MODE='apply', WORLD=CAMP_WORLD, ACTORS=self.cards[0], BACKUP=str(backup))
        self.assertEqual(code, 1)
        self.assertTrue(any(f'主服正开着 {CAMP_WORLD}' in p for p in report['problems']), report['problems'])
        self.assertFalse(backup.exists())

        self.env.stop('main')
        self.env.stop('char')

        # 状态接口说世界没开，但库被别的进程占着
        with subprocess.Popen(
            ['node', '--input-type=module', '-e', HOLD_LOCK, str(FOUNDRY_APP), str(self.db_dir('main', 'settings'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8',
        ) as holder:
            try:
                self.assertEqual(holder.stdout.readline().strip(), 'locked')
                code, report = self.enable('apply', CAMP_WORLD, BACKUP=str(backup))
            finally:
                holder.stdin.close()
                holder.wait(timeout=30)
        self.assertEqual(code, 1)
        self.assertTrue(any('打不开' in p for p in report['problems']), report['problems'])
        self.assertFalse(backup.exists())

        # 占锁的进程打开过库，库文件可能被整理过；从这里起什么都不该再动它们
        before = file_state(self.world_dbs())

        # pm2 说建卡服核心还在跑，哪怕端口都拒连
        code, report = self.enable('apply', CHAR_WORLD, BACKUP=str(backup), PM2_BIN=str(self.pm2_core_online))
        self.assertEqual(code, 1)
        self.assertTrue(any('建卡服没有完全停下' in p for p in report['problems']), report['problems'])
        self.assertFalse(backup.exists())

        # 控制器端口有人应答，哪怕 pm2 说两个进程都停了
        with http_server(b'{"state":"offline"}') as url:
            code, report = self.enable('apply', CHAR_WORLD, BACKUP=str(backup), CONTROL_URL=url)
        self.assertEqual(code, 1)
        self.assertTrue(any('建卡服没有完全停下' in p for p in report['problems']), report['problems'])
        self.assertFalse(backup.exists())

        # 主服状态接口连不上、返回的不是 JSON、active 不是布尔值：判断不了团世界开没开
        for label, server in (('连不上', nullcontext(f'http://127.0.0.1:{CONTROL_PORT}/api/status')),
                              ('不是 JSON', http_server(b'not json')),
                              ('active 不是布尔值', http_server(b'{}'))):
            with self.subTest(label), server as url:
                code, report = self.enable('apply', CAMP_WORLD, BACKUP=str(backup), MAIN_STATUS_URL=url)
                self.assertEqual(code, 1)
                self.assertTrue(any('主服状态接口' in p and '读不到' in p for p in report['problems']), report['problems'])
                self.assertFalse(backup.exists())

        # BACKUP 没给，或者已经存在
        code, report = self.enable('apply', CAMP_WORLD)
        self.assertEqual(code, 1)
        self.assertIn('MODE=apply 要给 BACKUP', report['problems'])
        backup.mkdir()
        try:
            code, report = self.run_tool('reset-xp.mjs', MODE='apply', WORLD=CAMP_WORLD, ACTORS=self.cards[0], BACKUP=str(backup))
            self.assertEqual(code, 1)
            self.assertTrue(any('BACKUP 已存在' in p for p in report['problems']), report['problems'])
            self.assertEqual(list(backup.iterdir()), [])
        finally:
            backup.rmdir()

        self.assertEqual(file_state(self.world_dbs()), before, '被拒绝的 apply 改到了库')

    def test_2_enable_worlds_dry_runs_on_copies(self):
        before = file_state(self.world_dbs())
        code, report = self.enable('dry', f'{CAMP_WORLD},{CHAR_WORLD}')
        self.assertEqual(code, 0, report)
        self.assertEqual(report['problems'], [])

        main, char = report['worlds'][CAMP_WORLD], report['worlds'][CHAR_WORLD]
        self.assertEqual(main['moduleConfiguration']['status'], 'update')
        self.assertEqual(main['moduleConfiguration']['enable'], [MODULE_ID])
        self.assertEqual(main['publishDir']['status'], 'create')
        self.assertEqual(char['moduleConfiguration']['enable'], [MODULE_ID, EXTRA_MODULE])
        self.assertEqual({role: entry['status'] for role, entry in char['folders'].items()},
                         {role: 'create' for role in ROLE_NAMES})
        self.assertEqual(report['written'][CAMP_WORLD]['settings']['put'], 2)
        self.assertEqual(report['written'][CHAR_WORLD]['folders']['put'], 3)
        for dbs in report['written'].values():
            for result in dbs.values():
                self.assertEqual(result['problems'], [])
        self.assertEqual(file_state(self.world_dbs()), before, 'dry 改到了真库')

    def test_3_enable_worlds_apply_shows_up_in_foundry(self):
        backup = self.env.root / 'backup-enable'
        code, report = self.enable('apply', f'{CAMP_WORLD},{CHAR_WORLD}', BACKUP=str(backup))
        self.assertEqual(code, 0, report)
        self.assertEqual(set(report['backup']['dbs']), {f'{CAMP_WORLD}/settings', f'{CHAR_WORLD}/settings', f'{CHAR_WORLD}/folders'})
        for name in report['backup']['dbs']:
            self.assertTrue(any((backup / name).iterdir()), name)

        # 写进去的行记在该世界 GM 名下
        for which, world, names in (('main', CAMP_WORLD, ['settings']), ('char', CHAR_WORLD, ['settings', 'folders'])):
            gm = report['worlds'][world]['gmUser']
            users = [json.loads(raw) for key, raw in self.read_rows(self.db_dir(which, 'users')).items() if key.startswith('!users!')]
            self.assertIn(gm, {user['_id'] for user in users if user['role'] == 4}, world)
            written = []
            for name in names:
                for key, raw in self.read_rows(self.db_dir(which, name)).items():
                    doc = json.loads(raw)
                    if doc.get('key') in ('core.moduleConfiguration', f'{MODULE_ID}.publishDir') or MODULE_ID in (doc.get('flags') or {}):
                        written.append(key)
                        self.assertEqual(doc['_stats']['lastModifiedBy'], gm, key)
            self.assertEqual(len(written), 2 if which == 'main' else 5, written)

        self.env.start('main')
        self.env.start('char')
        states = {}
        for which in ('main', 'char'):
            page, _ = self.env.login(which, 'Gamemaster')
            states[which] = page.evaluate(MODULE_STATE, MODULE_ID)
            page.context.close()
        self.env.stop('main')
        self.env.stop('char')

        for which, state in states.items():
            self.assertTrue(state['active'], which)
            self.assertEqual(state['publishDir'], PUBLISH_DIR, which)
        self.assertIs(states['main']['modules'].get(OTHER_MODULE), True)
        self.assertNotIn(EXTRA_MODULE, states['main']['modules'])
        self.assertIs(states['char']['modules'].get(EXTRA_MODULE), True)
        self.assertEqual(sorted(states['char']['folders']), sorted([role, name] for role, name in ROLE_NAMES.items()))

    def test_4_enable_worlds_apply_again_writes_nothing(self):
        before = file_state(self.world_dbs())
        backup = self.env.root / 'backup-enable-again'
        code, report = self.enable('apply', f'{CAMP_WORLD},{CHAR_WORLD}', BACKUP=str(backup))
        self.assertEqual(code, 0, report)
        self.assertEqual(report['result'], '全部已是目标状态，没有写入')
        for world in (CAMP_WORLD, CHAR_WORLD):
            self.assertEqual(report['worlds'][world]['moduleConfiguration']['status'], 'already')
            self.assertEqual(report['worlds'][world]['publishDir']['status'], 'already')
        self.assertEqual({entry['status'] for entry in report['worlds'][CHAR_WORLD]['folders'].values()}, {'already'})
        self.assertFalse(backup.exists())
        self.assertEqual(file_state(self.world_dbs()), before)

        # 已有不同的发布目录名时不覆盖
        code, report = self.run_tool('enable-worlds.mjs', MODE='dry', WORLDS=CAMP_WORLD, PUBLISH_DIR='charsync-another-name-00')
        self.assertEqual(code, 1)
        self.assertTrue(any('已经是另一个值' in p for p in report['problems']), report['problems'])

    def test_5_reset_xp_changes_only_the_xp_value(self):
        actors = ','.join(self.cards)
        rows_before = self.read_rows(self.db_dir('main', 'actors'))
        files_before = file_state([self.db_dir('main', 'actors')])

        code, report = self.run_tool('reset-xp.mjs', MODE='dry', WORLD=CAMP_WORLD, ACTORS=actors)
        self.assertEqual(code, 0, report)
        self.assertEqual([(a['id'], a['xpBefore'], a['status']) for a in report['actors']],
                         [(self.cards[0], 1000, 'reset'), (self.cards[1], 640, 'reset'), (self.cards[2], 0, 'already')])
        self.assertEqual(report['written'][CAMP_WORLD]['actors']['put'], 2)
        self.assertEqual(file_state([self.db_dir('main', 'actors')]), files_before, 'dry 改到了真库')

        backup = self.env.root / 'backup-xp'
        code, report = self.run_tool('reset-xp.mjs', MODE='apply', WORLD=CAMP_WORLD, ACTORS=actors, BACKUP=str(backup))
        self.assertEqual(code, 0, report)
        self.assertEqual(report['written'][CAMP_WORLD]['actors']['problems'], [])
        self.assertTrue(any((backup / CAMP_WORLD / 'actors').iterdir()))

        rows_after = self.read_rows(self.db_dir('main', 'actors'))
        self.assertEqual(set(rows_after), set(rows_before))
        changed = {f'!actors!{self.cards[0]}': 1000, f'!actors!{self.cards[1]}': 640}
        for key, raw in rows_before.items():
            if key not in changed:
                self.assertEqual(rows_after[key], raw, key)
                continue
            old = f'"xp":{{"value":{changed[key]},'
            self.assertEqual(raw.count(old), 1, key)
            self.assertEqual(rows_after[key], raw.replace(old, '"xp":{"value":0,'), key)

        self.env.start('main')
        page, _ = self.env.login('main', 'Gamemaster')
        state = page.evaluate(ACTOR_STATE, self.cards)
        page.context.close()
        self.env.stop('main')
        for actor_id in self.cards:
            doc = self.env.actors[actor_id]
            self.assertEqual(state[actor_id], {'name': doc['name'], 'level': doc['system']['details']['level']['value'], 'xp': 0})

        backup_again = self.env.root / 'backup-xp-again'
        code, report = self.run_tool('reset-xp.mjs', MODE='apply', WORLD=CAMP_WORLD, ACTORS=actors, BACKUP=str(backup_again))
        self.assertEqual(code, 0, report)
        self.assertEqual(report['result'], '全部已是目标状态，没有写入')
        self.assertFalse(backup_again.exists())


class InstallScriptTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='charsync-install-'))
        self.addCleanup(shutil.rmtree, self.root)
        self.main = self.root / 'main'
        self.char = self.root / 'char'
        self.home = self.root / 'home'
        (self.main / 'modules' / 'pf2e-eidolon-helper').mkdir(parents=True)
        (self.main / 'modules' / 'pf2e-eidolon-helper' / 'module.json').write_text('{"id": "pf2e-eidolon-helper"}', encoding='utf-8')
        (self.char / 'modules').mkdir(parents=True)

    def run_install(self, *args, source: Path = MODULE_DIR) -> subprocess.CompletedProcess:
        env = {**os.environ, 'MAIN_DATA': self.main.as_posix(), 'CHAR_DATA': self.char.as_posix(), 'CHARSYNC_HOME': self.home.as_posix()}
        return subprocess.run([git_bash(), (TOOLS / 'install.sh').as_posix(), *args, source.as_posix()],
                              env=env, capture_output=True, text=True, encoding='utf-8')

    def install(self, *args) -> list[str]:
        result = self.run_install(*args)
        self.assertEqual(result.returncode, 0, f'{result.stdout}\n{result.stderr}')
        return result.stdout.splitlines()

    def staging_dirs(self) -> list[str]:
        return sorted(p.name for data in (self.main, self.char) for p in (data / 'modules').iterdir() if p.name.startswith('.'))

    def test_dry_lists_only_client_files_for_data(self):
        lines = self.install('--dry')
        copies = [line.split()[1:] for line in lines if line.startswith('+ cp ')]
        self.assertTrue(copies)
        data_roots = (self.main.as_posix(), self.char.as_posix())
        allowed = {f'{MODULE_DIR.as_posix()}/{part}' for part in ('module.json', 'scripts', 'styles', 'generated/rules.mjs')}
        allowed.add(f'{self.main.as_posix()}/modules/pf2e-eidolon-helper')
        for args in copies:
            source, target = args[-2], args[-1]
            if target.startswith(data_roots):
                self.assertIn(source, allowed, args)
                for part in ('engine', 'test', 'tools'):
                    self.assertNotIn(f'{MODULE_DIR.as_posix()}/{part}', source)
        self.assertEqual(sorted(p.name for p in (self.main / 'modules').iterdir()), ['pf2e-eidolon-helper'])
        self.assertEqual(list((self.char / 'modules').iterdir()), [])
        self.assertFalse(self.home.exists())

    def test_install_twice_keeps_generated_rules_and_moves_old_release(self):
        self.install()
        for data in (self.main, self.char):
            installed = data / 'modules' / MODULE_ID
            self.assertEqual(sorted(p.name for p in installed.iterdir()), ['generated', 'module.json', 'scripts', 'styles'])
            self.assertEqual([p.name for p in (installed / 'generated').iterdir()], ['rules.mjs'])
            self.assertEqual(sorted(p.name for p in data.joinpath('modules').iterdir() if p.name.startswith('.')), [])
        self.assertTrue((self.home / 'app' / 'engine' / 'main.mjs').is_file())
        self.assertTrue((self.home / 'app' / 'tools' / 'install.sh').is_file())
        eidolon = self.char / 'modules' / 'pf2e-eidolon-helper' / 'module.json'
        self.assertTrue(eidolon.is_file())

        rules = self.char / 'modules' / MODULE_ID / 'generated' / 'rules.mjs'
        rules.write_text('// 引擎写的\n', encoding='utf-8')
        main_rules = self.main / 'modules' / MODULE_ID / 'generated' / 'rules.mjs'
        main_rules.write_text('// 不该留下\n', encoding='utf-8')
        eidolon.write_text('{"id": "pf2e-eidolon-helper", "marker": true}', encoding='utf-8')
        # 旧版按秒挪进 releases/<时间>，同一秒里再装会被拒绝
        time.sleep(1.1)
        self.install()
        self.assertEqual(rules.read_text(encoding='utf-8'), '// 引擎写的\n')
        self.assertEqual(main_rules.read_bytes(), (MODULE_DIR / 'generated' / 'rules.mjs').read_bytes())
        self.assertIn('marker', eidolon.read_text(encoding='utf-8'))
        self.assertEqual(self.staging_dirs(), [])
        releases = list((self.home / 'releases').iterdir())
        self.assertEqual(len(releases), 1)
        self.assertEqual(sorted(p.name for p in releases[0].iterdir()), ['app', 'char-module', 'main-module'])

    def test_wrong_directory_fails_without_changes(self):
        not_module = self.root / 'not-a-module'
        not_module.mkdir()
        result = self.run_install(source=not_module)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('module.json', result.stderr)
        self.assertEqual(sorted(p.name for p in (self.main / 'modules').iterdir()), ['pf2e-eidolon-helper'])
        self.assertEqual(list((self.char / 'modules').iterdir()), [])
        self.assertFalse(self.home.exists())

    def test_failed_copy_removes_staging(self):
        # generated/rules.mjs 是目录时 cp 会失败，失败点在主服的暂存目录拷到一半
        source = self.root / 'source'
        shutil.copytree(MODULE_DIR, source, ignore=shutil.ignore_patterns('__pycache__'))
        (source / 'generated' / 'rules.mjs').unlink()
        (source / 'generated' / 'rules.mjs').mkdir()
        result = self.run_install(source=source)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('已删掉暂存目录', result.stderr)
        self.assertEqual(self.staging_dirs(), [])
        self.assertFalse((self.main / 'modules' / MODULE_ID).exists())
        self.assertFalse(self.home.exists())


class PackageTest(unittest.TestCase):
    def test_git_archive_keeps_install_sh_lf(self):
        # CN 上的 bash 不认 CRLF；Windows 上 core.autocrlf=true 时 git archive 默认会把文本转成 CRLF。按 HEAD、用默认配置打包
        toplevel = subprocess.run(['git', '-C', str(MODULE_DIR), 'rev-parse', '--show-toplevel'],
                                  capture_output=True, text=True, encoding='utf-8', check=True).stdout.strip()
        prefix = MODULE_DIR.resolve().relative_to(Path(toplevel).resolve()).as_posix()
        archive = subprocess.run(['git', '-C', toplevel, 'archive', '--format=tar', f'HEAD:{prefix}', 'tools/install.sh'],
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            data = tar.extractfile('tools/install.sh').read()
        self.assertEqual(data.count(b'\r'), 0)


if __name__ == '__main__':
    unittest.main()
