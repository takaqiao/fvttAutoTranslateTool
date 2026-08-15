#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""探针：插件里用**子串/正则嗅探别人身份**当闸门的地方，实际会命中多少不该命中的目标。

同一类问题的「闸门」版本：为一个窄目标写的处理，闸门写成子串匹配，
于是被套到一大批别人的对象上。

本脚本枚举**真实存在的全集**（已装模块 + 系统 + Foundry v14 核心），
拿插件里的两道闸各跑一遍：
  G1  /ember/i.test(root.className)                     —— ember-hardcoded-cn.mjs:453
  G3  /attunement|language|knowledge|.../i.test(pattern) —— ember-hardcoded-cn.mjs:368
全集来源：源码里 `classes: [...]`（ApplicationV2 DEFAULT_OPTIONS）与
`CONFIG.TextEditor.enrichers.push({... pattern: /.../ ...})`。

只读。假阳性模式：
  - 静态扫源码，取不到运行时动态拼出来的 className；
  - 模块未启用时不会注册，命中数是**上界**；
  - `classes:` 也可能出现在非 ApplicationV2 的对象里。
"""
import os
import re
import collections

DATA = r'C:\Users\Taka\AppData\Local\FoundryVTT\Data'
CORE = r'C:\Program Files\Foundry Virtual Tabletop\resources\app'
SELF = {'ember', 'ember_cn_unofficial'}

G1 = re.compile('ember', re.I)
G3 = re.compile('attunement|language|knowledge|soundscape|ancestry|culture|path'
                '|eventState|outcome|Advantage|Critical|date', re.I)

CLASSES = re.compile(r'classes\s*:\s*\[([^\]]{0,300})\]')
TOKEN = re.compile(r'["\']([\w\-\s]+)["\']')
ENRICHER = re.compile(r'pattern\s*:\s*(/.+?/[gimsuy]*)\s*,')


def walk_js(root, limit_mb=25):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns
                  if d not in ('.git', 'node_modules', 'assets', 'packs', 'icons', 'audio', 'fonts')]
        for fn in fns:
            if fn.endswith(('.mjs', '.js', '.hbs')):
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) > limit_mb * 1024 * 1024:
                        continue
                    yield p, open(p, encoding='utf-8', errors='replace').read()
                except OSError:
                    pass


pkgs = []
for base, kind in ((os.path.join(DATA, 'modules'), 'module'),
                   (os.path.join(DATA, 'systems'), 'system')):
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isdir(p):
                pkgs.append((kind, name, p))
pkgs.append(('core', 'foundry-v14-client', os.path.join(CORE, 'client')))
pkgs.append(('core', 'foundry-v14-templates', os.path.join(CORE, 'templates')))
print(f'全集：{len(pkgs)} 个包（模块/系统/核心）')

g1_hits = collections.defaultdict(set)
g3_hits = collections.defaultdict(list)
g3_all = collections.defaultdict(int)
scanned = 0
for kind, name, p in pkgs:
    for fp, src in walk_js(p):
        scanned += 1
        for m in CLASSES.finditer(src):
            for t in TOKEN.findall(m.group(1)):
                for tok in t.split():
                    if G1.search(tok) and name not in SELF:
                        g1_hits[name].add(tok)
        for m in ENRICHER.finditer(src):
            pat = m.group(1)
            g3_all[name] += 1
            if name not in SELF and G3.search(pat):
                g3_hits[name].append((os.path.basename(fp), pat[:100]))

print(f'扫了 {scanned} 个 js/mjs/hbs 文件\n')
print('--- G1  /ember/i 命中的**非 ember** class token：')
if not g1_hits:
    print('    （无）')
for k, v in sorted(g1_hits.items()):
    print(f'    {k}: {sorted(v)}')

print(f'\n--- G3  全集里共找到 {sum(g3_all.values())} 条 `pattern: /.../` 声明'
      f'（分布 {len(g3_all)} 个包）')
print('--- 其中被 ember 增强器闸命中的**非 ember** pattern：')
tot = 0
for k, v in sorted(g3_hits.items()):
    tot += len(v)
    print(f'    {k} ({len(v)} 条 / 该包共 {g3_all[k]} 条)')
    for f, pat in v[:10]:
        print(f'        {f}: {pat}')
print(f'    合计 {tot} 条非 ember 的增强器会被包住')
