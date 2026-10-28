"""本机双实例测试台：主服团世界和建卡服各起一个真实的 Foundry 14，数据目录都在 `root` 下，用完整个删掉。

两个实例共用本机的 Foundry 程序和许可证；pf2e 系统用目录联接挂进两边的 Data，模组从工作树拷一份进去
（引擎会重写建卡服那份的 `generated/rules.mjs`，不能联接回工作树）。
"""
from __future__ import annotations

import _winapi
import json
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

MODULE_DIR = Path(__file__).resolve().parents[2]
MODULE_ID = 'taka-charsync'
FOUNDRY_APP = Path(os.environ.get('CHARSYNC_FOUNDRY_APP', 'C:/Program Files/Foundry Virtual Tabletop/resources/app'))
LICENSE_FILE = Path(os.environ['LOCALAPPDATA']) / 'FoundryVTT' / 'Config' / 'license.json'

PUBLISH_DIR = 'charsync-e2e-0000000000'
CAMP_WORLD = 'e2e-camp'
CHAR_WORLD = 'pf2creation'
CHAR_CAMPAIGN_FOLDER = 'CharsyncCamp0001'
INSTANCES = {
    'main': {'world': CAMP_WORLD, 'port': 30181},
    'char': {'world': CHAR_WORLD, 'port': 30182},
}

START_TIMEOUT_S = 300
READY_TIMEOUT_MS = 180_000
# 用例失败时打出的页面日志行数，只算下面这几种记录：控制台的 warning/error、页面错误、崩溃、主框架导航
PAGE_LOG_TAIL = 60
NOTABLE_PAGE_EVENTS = {'warning', 'error', 'pageerror', 'crash', 'navigated'}
# 用例失败时整份页面日志拷到这里，删测试临时目录时不删
FAILED_LOG_DIR = Path(tempfile.gettempdir()) / 'charsync-e2e-logs'

ANCESTRY_PARAGON = {
    'id': 'pf2e-sf2e-extra-feat-slots-ancestry-paragon', 'label': '族裔典范', 'supported': ['ancestry'],
    'slots': [1, 3, 7, 11, 15, 19],
}
# 形状照建卡服现状的 Team+ 无限制+包（职业专长 1、2、4…20 级，通用专长 1 级）；分节 id 是测试自己起的
TEAM_PLUS_SECTIONS = [
    {'id': 'team-plus-class', 'label': 'Team+ 职业专长', 'supported': ['class'], 'slots': [1, *range(2, 21, 2)]},
    {'id': 'team-plus-general', 'label': 'Team+ 通用专长', 'supported': ['general'], 'slots': [1]},
]
SEED_SETTINGS = {
    'main': {
        'pf2e.freeArchetypeVariant': True,
        'pf2e.campaignFeatSections': [ANCESTRY_PARAGON],
        'pf2e.campaignFeats': False,
        'pf2e.staminaVariant': True,
    },
    # 建卡服现状；`pf2e.staminaVariant` 故意不设，留给「第一次设置从未落盘的键」的场景
    'char': {
        'pf2e.freeArchetypeVariant': False,
        'pf2e.gradualBoostsVariant': True,
        'pf2e.campaignFeats': True,
        'pf2e.campaignFeatSections': TEAM_PLUS_SECTIONS,
    },
}
SEED_USERS = {
    'main': [('P1', 'PLAYER'), ('P2', 'PLAYER'), ('GM2', 'GAMEMASTER')],
    'char': [('P1', 'PLAYER'), ('P2', 'PLAYER')],
}
CHAR_FOLDERS = [('已提交', 'submitted'), ('已撤回', 'withdrawn'), ('草稿备份', 'backup')]

SEED_BEFORE_RELOAD = """async ({users, settings, moduleId}) => {
  for (const [name, role] of users) await User.create({name, role: CONST.USER_ROLES[role]});
  for (const [key, value] of settings) {
    const dot = key.indexOf('.');
    await game.settings.set(key.slice(0, dot), key.slice(dot + 1), value);
  }
  const modules = game.settings.get('core', 'moduleConfiguration');
  await game.settings.set('core', 'moduleConfiguration', {...modules, [moduleId]: true});
}"""

SEED_AFTER_RELOAD = """async ({moduleId, publishDir, actors, characterId, folders, campaignFolder}) => {
  if (!game.modules.get(moduleId)?.active) throw new Error(`${moduleId} 没有启用`);
  await game.settings.set(moduleId, 'publishDir', publishDir);
  const p1 = game.users.getName('P1');
  if (campaignFolder) await Folder.create({_id: campaignFolder, name: 'E2E 测试团', type: 'Actor'}, {keepId: true});
  for (const data of actors) {
    data.ownership = {default: 0, [p1.id]: 3};
    data.folder = campaignFolder ?? null;
    await Actor.create(data, {keepId: true});
  }
  await p1.update({character: characterId});
  for (const [name, role] of folders) await Folder.create({name, type: 'Actor', flags: {[moduleId]: {role}}});
}"""

# Foundry 在 ready 前给没有指定角色的玩家异步弹出用户配置窗，等它渲染出来再关
CLOSE_USER_CONFIG = """async () => {
  if (game.user.isGM || game.user.character) return;
  const sheet = game.user.sheet;
  for (let i = 0; i < 50 && !sheet.rendered; i++) await new Promise(resolve => setTimeout(resolve, 100));
  await sheet.close();
}"""

INJECT_ITEM = """
const [leveldbUrl, foundryAppDir, dbDir, actorId, itemJson] = process.argv.slice(1);
const {loadClassicLevel} = await import(leveldbUrl);
const ClassicLevel = loadClassicLevel(foundryAppDir);
const db = new ClassicLevel(dbDir, {keyEncoding: 'utf8', valueEncoding: 'utf8', createIfMissing: false});
await db.open();
try {
  const actorKey = `!actors!${actorId}`;
  const raw = await db.get(actorKey);
  if (raw == null) throw new Error(`库里没有 actor ${actorId}`);
  const actor = JSON.parse(raw);
  const item = JSON.parse(itemJson);
  await db.batch([
    {type: 'put', key: `!actors.items!${actorId}.${item._id}`, value: itemJson},
    {type: 'put', key: actorKey, value: JSON.stringify({...actor, items: [...actor.items, item._id]})},
  ]);
} finally {
  await db.close();
}
"""

# 不走环境变量里的代理：状态接口都在本机
_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _private_dir() -> Path:
    """主检出里放私有夹具和 pf2e 缓存的目录。工作树和主检出共用一个 .git，从 git 的公共目录推回主检出。"""
    common = subprocess.run(
        ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        cwd=MODULE_DIR, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent / 'output' / 'charsync-20261016'


def _located(var: str, default, marker: str) -> Path:
    if os.environ.get(var):
        path = Path(os.environ[var])
    else:
        try:
            path = default()
        except (OSError, subprocess.CalledProcessError) as error:
            raise FileNotFoundError(f'推不出默认位置（{error}），设环境变量 {var} 指向含 {marker} 的目录') from error
    if not (path / marker).is_file():
        raise FileNotFoundError(f'{path} 下没有 {marker}，设环境变量 {var} 指向含 {marker} 的目录')
    return path


def pf2e_dir() -> Path:
    """`CHARSYNC_PF2E_DIR`：含 `system.json` 的 pf2e 系统目录。"""
    return _located('CHARSYNC_PF2E_DIR', lambda: _private_dir() / 'e2e-cache' / 'systems' / 'pf2e', 'system.json')


def fixtures_dir() -> Path:
    """`CHARSYNC_FIXTURES`：含 `cards.json` 的夹具目录。"""
    return _located('CHARSYNC_FIXTURES', lambda: _private_dir() / 'fixtures', 'cards.json')


def _write_json(file: Path, value) -> None:
    file.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _foundry_id() -> str:
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))


def seed_cards(actors: dict) -> list[dict]:
    """种子表的卡：夹具里除同伴外的每张卡，按夹具顺序。幻灵按 `pf2e-eidolon-helper` 的 `summoner` 记号认出主人，
    记进主人的 `companions`；魔宠由引擎按 `system.master.id` 自己认，不进种子。"""
    eidolons = {}
    for actor_id, doc in actors.items():
        summoner = doc.get('flags', {}).get('pf2e-eidolon-helper', {}).get('summoner')
        if summoner in actors:
            eidolons.setdefault(summoner, []).append(actor_id)
    companions = {actor_id for ids in eidolons.values() for actor_id in ids}
    cards = []
    for actor_id, doc in actors.items():
        master = (doc['system'].get('master') or {}).get('id') if doc['type'] == 'familiar' else None
        if actor_id in companions or master in actors:
            continue
        card = {'charId': actor_id, 'world': CAMP_WORLD, 'mainId': actor_id}
        if actor_id in eidolons:
            card['companions'] = eidolons[actor_id]
        cards.append(card)
    return cards


def _listening_pids(port: int) -> set[int]:
    # 只看本地端口和「对端为 :0」，不认状态那一列的文字（中文系统上会本地化）
    output = subprocess.run(['netstat', '-ano', '-p', 'TCP'], capture_output=True).stdout.decode('ascii', 'replace')
    output += subprocess.run(['netstat', '-ano', '-p', 'TCPv6'], capture_output=True).stdout.decode('ascii', 'replace')
    pids = set()
    for line in output.splitlines():
        match = re.match(r'\s*TCP\s+\S+:(\d+)\s+\S+:0\s+\S+\s+(\d+)\s*$', line)
        if match and int(match[1]) == port:
            pids.add(int(match[2]))
    return pids


class Env:
    """一次测试的两个实例。`fixtures` 是含 `cards.json`（`{exportedAt, actors: {id: {world, doc}}}`）的目录。

    用 `with Env(root, fixtures) as env:`，退出时关浏览器、停两个实例、删掉 `build()` 建的 `root`，测试失败也一样；
    环境变量 `CHARSYNC_E2E_KEEP=1` 时保留 `root` 供排查。
    """

    def __init__(self, root: Path, fixtures: Path):
        self.root = Path(root)
        exported = json.loads((Path(fixtures) / 'cards.json').read_text(encoding='utf-8'))
        self.actors = {actor_id: entry['doc'] for actor_id, entry in exported['actors'].items()}
        self.seed_cards = seed_cards(self.actors)
        # 主卡 id，按夹具顺序；第一张是 P1 的指定角色
        self.main_cards = [card['charId'] for card in self.seed_cards]
        self._procs = {}
        self._playwright = None
        self._browser = None
        # 只删自己建的 root
        self._owns_root = False
        # 所有页面的主框架导航、控制台和页面错误，按时间写进 root/pages.log
        self._page_log = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def data_dir(self, which: str) -> Path:
        return self.root / which / 'Data'

    def url(self, which: str) -> str:
        return f"http://127.0.0.1:{INSTANCES[which]['port']}"

    def build(self) -> None:
        """两个数据目录、引擎配置和种子表。`root` 要么不存在、要么是空目录。种子表的 `bootstrapSince` 由 `seed()` 结束时改写。"""
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f'{self.root} 不是空目录，测试台只在新目录里建')
        self.root.mkdir(parents=True, exist_ok=True)
        self._owns_root = True
        pf2e = pf2e_dir()
        release = json.loads((FOUNDRY_APP / 'package.json').read_text(encoding='utf-8'))['release']
        system_version = json.loads((pf2e / 'system.json').read_text(encoding='utf-8'))['version']
        for which, instance in INSTANCES.items():
            (self.root / which / 'Config').mkdir(parents=True)
            shutil.copyfile(LICENSE_FILE, self.root / which / 'Config' / 'license.json')
            data = self.data_dir(which)
            (data / 'systems').mkdir(parents=True)
            _winapi.CreateJunction(str(pf2e), str(data / 'systems' / 'pf2e'))
            shutil.copytree(MODULE_DIR, data / 'modules' / MODULE_ID, ignore=shutil.ignore_patterns('test', '__pycache__'))
            world = data / 'worlds' / instance['world']
            world.mkdir(parents=True)
            _write_json(world / 'world.json', {
                'id': instance['world'],
                'title': instance['world'],
                'system': 'pf2e',
                'coreVersion': f"{release['generation']}.{release['build']}",
                'systemVersion': system_version,
                'compatibility': {'minimum': str(release['generation']), 'verified': str(release['generation'])},
            })
        (self.root / 'tmp').mkdir()
        _write_json(self.root / 'engine.json', {
            'foundryAppDir': str(FOUNDRY_APP),
            'main': {'dataDir': str(self.data_dir('main')), 'statusUrl': f"{self.url('main')}/api/status"},
            'char': {'dataDir': str(self.data_dir('char')), 'world': CHAR_WORLD, 'statusUrl': f"{self.url('char')}/api/status"},
            'publishDir': PUBLISH_DIR,
            'worlds': [CAMP_WORLD],
            'titles': {CAMP_WORLD: 'E2E 测试团'},
            'campaignFolders': {CHAR_CAMPAIGN_FOLDER: CAMP_WORLD},
            'stateDir': str(self.root / 'state'),
            'tmpRoot': str(self.root / 'tmp'),
            'seedPath': str(self.root / 'seed.json'),
        })
        self._write_seed()

    def _write_seed(self) -> None:
        """种子表，`bootstrapSince` 取现在。账号不写进种子，两边同名的由引擎配对。"""
        _write_json(self.root / 'seed.json', {
            'cards': self.seed_cards,
            'users': [],
            'bootstrapSince': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
        })

    def start(self, which: str) -> None:
        """起实例并直接进它的世界，等到状态接口报告世界已开。先停掉本 Env 持有的旧实例；端口上的未知进程会报错。"""
        self._launch(which, INSTANCES[which]['world'])
        self._wait_status(which, active=True)

    def stop(self, which: str) -> None:
        """停掉实例。主服随后以不开世界的状态重新起来：线上主服核心常驻、世界不开时状态接口报 `active: false`，
        引擎要连得上它才会去读团世界。"""
        self._kill(which)
        if which == 'main':
            self._launch('main', None)
            self._wait_status('main', active=False)

    def engine_once(self) -> dict:
        """跑引擎一轮，返回它最后一行打印的 TickReport。"""
        result = subprocess.run(
            ['node', str(MODULE_DIR / 'engine' / 'main.mjs'), '--config', str(self.root / 'engine.json'), '--once'],
            capture_output=True, text=True, encoding='utf-8',
        )
        lines = result.stdout.strip().splitlines()
        if result.returncode == 0 and lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        last = lines[-1] if lines else '（空）'
        stderr_tail = '\n'.join(result.stderr.splitlines()[-40:])
        raise RuntimeError(f'引擎没有给出 TickReport（退出码 {result.returncode}，stdout 最后一行：{last}）。stderr 末尾：\n{stderr_tail}')

    def login(self, which: str, user: str):
        """新开一个浏览器上下文（登录状态互不共享）以 `user` 登进实例，等到 `game.ready`。返回 `(page, logs)`，
        `logs` 随页面的控制台输出增长。"""
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        page = self._browser.new_context(viewport={'width': 1600, 'height': 900}).new_page()
        logs = []
        page.on('console', lambda message: logs.append(f'{message.type}: {message.text}'))
        page.on('pageerror', lambda error: logs.append(f'pageerror: {error}'))
        label = f'{which}-{user}'
        page.on('console', lambda message: self._log_page(label, message.type, message.text))
        page.on('pageerror', lambda error: self._log_page(label, 'pageerror', str(error)))
        page.on('crash', lambda _: self._log_page(label, 'crash', '页面崩溃'))
        page.on('framenavigated', lambda frame: frame.parent_frame is None and self._log_page(label, 'navigated', frame.url))
        page.goto(f'{self.url(which)}/join')
        page.wait_for_selector('#join-username', timeout=60_000)
        page.fill('#join-username', user)
        page.click('button[name=join]')
        self._wait_ready(page)
        page.evaluate(CLOSE_USER_CONFIG)
        return page, logs

    def seed(self) -> None:
        """两边以 Gamemaster 建账号、改规则设置、启用模组、写发布目录名，按夹具原 id 建卡并只归 P1，
        P1 的指定角色设为第一张主卡；建卡服另建三个带 `role` 记号的文件夹。

        结束时把种子表的 `bootstrapSince` 改成此刻：两边的卡到这里内容相同，对应线上一次性同步写完卡的时刻。
        Foundry 建卡时由服务端写 `_stats.modifiedTime`，取更早的时刻会让每张卡都算两边动过、认不出起步基线。"""
        for which in INSTANCES:
            page, _ = self.login(which, 'Gamemaster')
            try:
                page.evaluate(SEED_BEFORE_RELOAD, {
                    'users': SEED_USERS[which], 'settings': list(SEED_SETTINGS[which].items()), 'moduleId': MODULE_ID,
                })
                page.reload()
                self._wait_ready(page)
                page.evaluate(SEED_AFTER_RELOAD, {
                    'moduleId': MODULE_ID,
                    'publishDir': PUBLISH_DIR,
                    'actors': list(self.actors.values()),
                    'characterId': self.main_cards[0],
                    'folders': CHAR_FOLDERS if which == 'char' else [],
                    'campaignFolder': CHAR_CAMPAIGN_FOLDER if which == 'char' else None,
                })
            finally:
                page.context.close()
        self._write_seed()

    def inject_invalid_item(self, which: str, actor_id: str) -> str:
        """离线往该世界 actors 库加一条 `system.level.value` 为 "abc" 的专长行，id 追加进 actor 行的 `items`，
        模拟线上过不了校验的旧数据。世界开着时拒绝。返回新条目的 id。"""
        if self._world_active(which):
            raise RuntimeError(f'{which} 的世界开着，不能离线写库')
        item = {
            '_id': _foundry_id(),
            'name': 'E2E 非法专长',
            'type': 'feat',
            'img': 'systems/pf2e/icons/default-icons/feat.svg',
            'system': {'level': {'value': 'abc'}, 'category': 'general'},
            'effects': [],
            'flags': {},
            'folder': None,
            'sort': 0,
            'ownership': {'default': 0},
        }
        db_dir = self.data_dir(which) / 'worlds' / INSTANCES[which]['world'] / 'data' / 'actors'
        leveldb_url = (MODULE_DIR / 'engine' / 'leveldb.mjs').as_uri()
        result = subprocess.run(
            ['node', '--input-type=module', '-e', INJECT_ITEM, leveldb_url, str(FOUNDRY_APP), str(db_dir), actor_id, json.dumps(item)],
            capture_output=True, text=True, encoding='utf-8',
        )
        if result.returncode != 0:
            raise RuntimeError(f'离线写库失败：\n{result.stderr}')
        return item['_id']

    def notable_page_log(self, lines: int = PAGE_LOG_TAIL) -> str:
        """页面日志里 `NOTABLE_PAGE_EVENTS` 那几种记录的最后 `lines` 行。"""
        log = self.root / 'pages.log'
        if not log.exists():
            return '（没有页面日志）'
        kept, keep = [], False
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            # 缩进的是上一条记录的后续行，跟着那条记录取舍
            if not line.startswith(' '):
                parts = line.split(' ', 3)
                keep = len(parts) > 2 and parts[2].rstrip(':') in NOTABLE_PAGE_EVENTS
            if keep:
                kept.append(line)
        return '\n'.join(kept[-lines:]) or '（没有 warning/error 或导航记录）'

    def keep_page_log(self, name: str) -> Path | None:
        """把整份页面日志拷到 `FAILED_LOG_DIR/<时间>-<name>.log`，返回拷贝的路径；还没有日志时返回 None。"""
        log = self.root / 'pages.log'
        if not log.exists():
            return None
        FAILED_LOG_DIR.mkdir(exist_ok=True)
        target = FAILED_LOG_DIR / f'{datetime.now():%Y%m%d-%H%M%S}-{name}.log'
        shutil.copyfile(log, target)
        return target

    def _log_page(self, label: str, event: str, text: str) -> None:
        """一条记录一行：`时间 页面 种类: 内容`；内容有多行时，后续行缩进四格。"""
        if self._page_log is None:
            self._page_log = open(self.root / 'pages.log', 'a', encoding='utf-8')
        body = text.replace('\n', '\n    ')
        self._page_log.write(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} {label} {event}: {body}\n")
        self._page_log.flush()

    def _close_page_log(self) -> None:
        if self._page_log is not None:
            self._page_log.close()
            self._page_log = None

    def close(self) -> None:
        """关浏览器、停两个实例、删 `root`。某一步出错也接着做后面的，最后抛出第一个错误。"""
        steps = [
            self._close_browser, *[lambda which=which: self._kill(which) for which in INSTANCES], self._close_page_log,
            self._remove_root,
        ]
        first_error = None
        for step in steps:
            try:
                step()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _close_browser(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._browser = self._playwright = None

    def _launch(self, which: str, world: str | None) -> None:
        self._kill(which)
        args = [
            'node', str(FOUNDRY_APP / 'main.mjs'), f'--dataPath={self.root / which}', f"--port={INSTANCES[which]['port']}",
            '--noupnp', '--noupdate',
        ]
        if world:
            args.append(f'--world={world}')
        with open(self.root / f'{which}.log', 'ab') as log:
            self._procs[which] = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

    def _kill(self, which: str) -> None:
        port = INSTANCES[which]['port']
        proc = self._procs.pop(which, None)
        if proc and proc.poll() is None:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
        if proc:
            proc.wait(timeout=30)
        listeners = _listening_pids(port)
        if listeners:
            raise RuntimeError(f'端口 {port} 被非本次测试进程占用（PID {sorted(listeners)}），未终止该进程')
        # Foundry 用 Config 下的目录锁占住数据目录，强杀后锁还在，10 秒不更新才算过期；进程已经停了，直接删
        lock = self.root / which / 'Config' / 'options.json.lock'
        if lock.is_dir():
            lock.rmdir()

    def _status(self, which: str) -> dict | None:
        try:
            with _http.open(f'{self.url(which)}/api/status', timeout=5) as response:
                return json.load(response)
        except (OSError, ValueError):
            return None

    def _world_active(self, which: str) -> bool:
        return bool((self._status(which) or {}).get('active'))

    def _wait_status(self, which: str, active: bool) -> None:
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            proc = self._procs[which]
            if proc.poll() is not None:
                raise RuntimeError(f'{which} 实例退出了（退出码 {proc.returncode}）：\n{self._log_tail(which)}')
            status = self._status(which)
            if status is not None and bool(status.get('active')) == active:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f'{which} 实例 {START_TIMEOUT_S} 秒内没有就绪：\n{self._log_tail(which)}')
            time.sleep(1)

    def _wait_ready(self, page) -> None:
        page.wait_for_function('() => window.game?.ready === true', timeout=READY_TIMEOUT_MS)

    def _log_tail(self, which: str, lines: int = 40) -> str:
        log = self.root / f'{which}.log'
        if not log.exists():
            return ''
        return '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])

    def _remove_root(self) -> None:
        if not self._owns_root or not self.root.exists():
            return
        # 先拆联接本身，再删其余；联接指向的 pf2e 缓存要留着给下一次用。保留 root 时也拆，免得日后手删目录删进缓存
        for which in INSTANCES:
            link = self.data_dir(which) / 'systems' / 'pf2e'
            if link.is_junction():
                os.rmdir(link)
        if os.environ.get('CHARSYNC_E2E_KEEP') == '1':
            print(f'CHARSYNC_E2E_KEEP=1，保留测试目录：{self.root}', file=sys.stderr)
            return
        # 进程刚被杀时文件句柄可能还没放完，目录会删不掉或删不空
        for attempt in range(10):
            try:
                shutil.rmtree(self.root)
                return
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(1)


class PageLogOnFailure:
    """测试类的混入：用例失败或出错时，整份页面日志拷到 `FAILED_LOG_DIR` 保留，stderr 打出拷贝的路径和
    最后几十行 warning/error 与导航记录，偶发的页面重新导航之类能看到前因。要求类上有 `env`（`Env`）。"""

    def run(self, result=None):
        if result is None:
            return super().run(result)
        before = len(result.failures) + len(result.errors)
        super().run(result)
        if len(result.failures) + len(result.errors) > before:
            # 拷不出来也要接着跑后面的用例和收尾（停实例、删临时目录），只把原因打出来
            try:
                kept = self.env.keep_page_log(self._testMethodName) or '（还没有页面日志）'
            except OSError as error:
                kept = f'（没能拷出来：{error}）'
            print(f'--- 页面日志整份：{kept}\n--- 最后 {PAGE_LOG_TAIL} 行 warning/error 与导航 ---\n'
                  f'{self.env.notable_page_log()}', file=sys.stderr)
        return result
