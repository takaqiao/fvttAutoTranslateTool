#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""探针：两个插件的 lang/*.json 有没有**覆盖别人命名空间**的键。

同一类问题的 i18n 版本：为了修自己的一处显示，往**全局共享字典**里写键，
把 Foundry 核心 / 系统 / 别的模块的译文一起改掉。

判据：
  H1  插件 lang 里的键，如果 Foundry 核心 public/lang/en.json 也有同名键 → 覆盖核心
  H2  如果 crucible / ember 上游 lang 也有同名键 → 覆盖上游（可能是有意的，需人工看）
  H3  babele-register.js / register.js 里直接对 game.i18n.translations.X 赋值的键
      —— 这一类连 manifest 都不经过，任何其它模块提供的同名键都会被它盖掉

只读。假阳性模式：
  - Foundry 的合并顺序是 核心 → 系统 → 模块（按加载序），模块之间互相覆盖取决于顺序，
    脚本无法判定实际生效方；所以 H1/H2 报的是「有覆盖能力」，不是「一定覆盖了」。
  - 覆盖上游 crucible/ember 的键**正是本项目的目的**，H2 只在键属于**核心命名空间**
    （不带自己前缀）时才值得看。
"""
import json, os, re, collections

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
CORE = r'C:\Program Files\Foundry Virtual Tabletop\resources\app\public\lang\en.json'
UP = {
    'crucible': r'C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\lang\en.json',
    'ember':    r'C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\lang\en.json',
}
PLUGINS = {
    'ember_cn_unofficial': os.path.join(ROOT, '1-Ember汉化插件'),
    'crucible-cn':         os.path.join(ROOT, '2-Crucible汉化插件'),
}

def load(p):
    try:
        return json.load(open(p, encoding='utf-8-sig'))
    except Exception as e:
        print(f'  !! 读不了 {p}: {e}'); return {}

core = load(CORE)
up = {k: load(v) for k, v in UP.items()}
print(f'核心 en.json 键 {len(core)}；crucible {len(up["crucible"])}；ember {len(up["ember"])}')

for mid, repo in PLUGINS.items():
    print(f'\n===== {mid}')
    for fn in ('cn.json', 'en.json'):
        p = os.path.join(repo, 'lang', fn)
        if not os.path.exists(p):
            continue
        d = load(p)
        hit_core = sorted(k for k in d if k in core)
        # 上游没有、核心也没有的「孤儿键」：写了没人读
        orphan = [k for k in d if k not in core and k not in up['crucible'] and k not in up['ember']]
        print(f'  {fn}: {len(d)} 键 | 与核心同名 {len(hit_core)} | 上游三方都没有 {len(orphan)}')
        for k in hit_core[:40]:
            print(f'      覆盖核心 {k!r}: 核心={core[k]!r}  本模块={d[k]!r}')
        if orphan:
            print(f'      孤儿键样本: {orphan[:12]}')

# H3：源码里直接写 game.i18n.translations
RX = re.compile(r'game\.i18n\.translations(?:\.([\w$]+)|\[([\'"])([^\'"]+)\2\])\s*=\s*(.+?);')
print('\n===== 源码直写 game.i18n.translations')
for mid, repo in PLUGINS.items():
    for root, _dirs, files in os.walk(repo):
        if '.git' in root or 'compendium' in root:
            continue
        for f in files:
            if not f.endswith(('.js', '.mjs')):
                continue
            fp = os.path.join(root, f)
            for i, ln in enumerate(open(fp, encoding='utf-8', errors='replace')):
                m = RX.search(ln)
                if m:
                    key = m.group(1) or m.group(3)
                    where = 'core-en.json 有此键' if key in core else 'core-en.json 无此键'
                    inup = [n for n, dd in up.items() if key in dd]
                    print(f'  {mid} {os.path.relpath(fp, repo)}:{i+1}  key={key!r} value={m.group(4)}  [{where}; 上游有: {inup or "无"}]')
