# -*- coding: utf-8 -*-
"""补第二批：补扫字段挑出来的 2 条 + 运行时拼接的那句 + 观景点名。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
A = '  "Chamber of Agaseros - Reservoir": "阿加瑟罗斯之室 - 蓄水池 Chamber of Agaseros - Reservoir",'
assert s.count(A) == 1
ADD = nl + nl.join([
'',
'  // ── 第二批：第一版扫描的字段名清单里漏了 `header` / `prompt` 这一族，补扫又捞出 24 条',
'  //    （日志页的分节标题等），其中 22 条已被现有表盖住，只剩下面两条。',
'  "Terrain": "地形",                    // EmberTerrainLegend 的图例标题',
'  // ⚠ glossary_ec 给的是「调谐特性」，**是陈旧值**：项目 2026-08-06 已裁 `Attunement`＝同调，',
'  //   `unify_rules.2026-08-06c` 把「调谐」明确列为已废变体（当时 UI 与角色卡都已改成同调，',
'  //   战役包里还剩 418 处旧译）。这里取**裁决**而不是词表。',
'  "Attunement Features": "同调特性",     // ember.mjs:138591 创角向导的分节标题',
'',
'  // ── 那句 hint 是**运行时拼接**的（ember.mjs:138592-138593 两段字面量 `+` 起来），',
'  //    所以键必须写拼完的全串 —— 只登记前半截的话上屏那一整句一个字都不会变。',
'  "You will begin your journey at Rank 1 of this attunement. You may further progress this and other attunements during the course of your adventure.":',
'    "你将从该同调的阶位 1 开始你的旅程。在冒险过程中，你可以继续提升这一同调以及其他同调。",',
'',
'  // ── 观景点名（ember.mjs:135282 `vantagePoints.aedirSignalpost.label`）',
'  "Aedir Signalpost Telescope": "艾迪尔信号哨站望远镜",',
])
s = s.replace(A, A + ADD)
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('ok')
