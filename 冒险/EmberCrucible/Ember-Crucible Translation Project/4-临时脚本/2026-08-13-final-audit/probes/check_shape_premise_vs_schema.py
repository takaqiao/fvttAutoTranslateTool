#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""探针：插件里「把数据强转成某个形状」的补救，其**前提**在上游 schema 里成立吗？

这是「窄场景兜底被无条件套到全集」这一类里最要命的一支：
补救本身没有 bug，错的是它假定的「正确形状」。前提一错，
判据（typeof x === 'string'）就从「找出坏数据」变成了「找出好数据」。

两项机械检查
------------
S1  形状前提 —— 拿 crucible system.json 的 documentTypes.<Doc>.<type>.htmlFields
    反推每个 Item 子类型的 system.description 是标量还是 {public,...}，
    与插件里 normalizeDescriptionValue 的假定（永远是对象）对照。
S2  守卫可用性 —— 源码里对 `game.world` 调 getFlag/setFlag。
    game.world 是 foundry.packages.World -> BasePackage -> DataModel，
    **没有** flags API（getFlag 只定义在 common/abstract/document.mjs 的 Document 上）。
    写成 `world?.getFlag?.(...)` 时可选调用返回 undefined，
    于是「只跑一次」的闸永远不生效，而 setFlag 的 try/catch 也永远不报错。

只读。假阳性模式：
  - S1 依赖 system.json 的 htmlFields 声明；上游若漏声明某个字段会漏报
    （已用 crucible-compiled.mjs 的 defineSchema 逐条复核过，两者一致）。
  - S2 是纯文本匹配，若将来 game.world 换成 Document 就会变成假阳性。
"""
import json
import os
import re

CRUCIBLE_SYSJSON = r'C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\system.json'
CORE_DOC = r'C:\Program Files\Foundry Virtual Tabletop\resources\app\common\abstract\document.mjs'
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
PLUGINS = [r'1-Ember汉化插件', r'2-Crucible汉化插件']

print('===== S1  system.description 的真实形状（crucible 0.10.1）')
sysj = json.load(open(CRUCIBLE_SYSJSON, encoding='utf-8'))
scalar, nested = [], []
for t, cfg in sorted(sysj['documentTypes']['Item'].items()):
    hf = cfg.get('htmlFields', [])
    if 'description' in hf:
        scalar.append(t)
    elif any(f.startswith('description.') for f in hf):
        nested.append(t)
print(f'  标量（HTMLField / 字符串）{len(scalar)} 个子类型: {scalar}')
print(f'  对象（SchemaField）      {len(nested)} 个子类型: {nested}')
print('  → register.js 的 normalizeDescriptionValue 假定「新 schema 一律是 {public,private}」，'
      f'对上面 {len(scalar)} 个子类型是**反的**。')
print('  → 这些子类型的 description 字段是 StringField 子类，'
      'StringField._cast(value)=String(value)，传对象进去落库变成 "[object Object]"。')

print('\n===== S2  对 game.world 调 flags API 的地方')
has = re.search(r'^\s{2}getFlag\(scope, key\)', open(CORE_DOC, encoding='utf-8').read(), re.M)
print(f'  核心里 getFlag 定义在 common/abstract/document.mjs（Document）: {bool(has)}；'
      'BasePackage extends DataModel，无 flags API')
RX = re.compile(r'\b(world|game\.world)\s*(?:\?\.|\.)\s*(get|set)Flag\s*(?:\?\.)?\s*\(')
n = 0
for repo in PLUGINS:
    base = os.path.join(ROOT, repo)
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d not in ('.git', 'compendium', 'release', '__pycache__')]
        for fn in fns:
            if not fn.endswith(('.js', '.mjs')):
                continue
            fp = os.path.join(dp, fn)
            for i, ln in enumerate(open(fp, encoding='utf-8', errors='replace'), 1):
                if RX.search(ln):
                    n += 1
                    print(f'  {os.path.relpath(fp, ROOT)}:{i}  {ln.strip()}')
print(f'  合计 {n} 处 —— 每一处都恒为 undefined / 静默无操作')
