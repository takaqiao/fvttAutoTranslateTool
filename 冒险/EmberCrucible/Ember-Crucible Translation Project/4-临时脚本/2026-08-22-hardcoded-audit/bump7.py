# -*- coding: utf-8 -*-
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
p = '1-Ember汉化插件/module.json'
s = io.open(p, encoding='utf-8', newline='').read()
assert s.count('1.1.30') >= 1
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace('1.1.30', '1.1.31'))
j = json.loads(io.open(p, encoding='utf-8-sig').read())
assert j['version'] == '1.1.31' and '1.1.31' in j['download']
print('  module.json → 1.1.31')

P = 'PROJECT.md'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60])
    s = s.replace(a, b)
rep1('（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～六轮），均已发版。' + nl
     + '当前已发布 `crucible-cn 0.9.18` / `ember_cn_unofficial v1.1.30`。**',
     '（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～七轮），均已发版。' + nl
     + '当前已发布 `crucible-cn 0.9.18` / `ember_cn_unofficial v1.1.31`。**')
rep1('| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.30**（2026-08-22；',
     '| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.31**（2026-08-22；')

ROW = (
 '| `v1.1.31` | 08-22 | **ember 侧玩家可见硬编码 40 条**（审计口径下的收尾）＋ 一条导入崩溃的定性。<br>'
 '· **掷骰卡上的加值/减值来源名 7 条**（混沌折射 / 混沌继承者 / 湮解印记 / 尼尔艾抗性 / 钻研反制 / '
 '屠龙 / 余烬之火花）。⚠ 机制上要紧：渲染它们的是 **crucible 的** '
 '`standard-check-details.hbs:6/18`（`{{localize boon.label}}`），那是 crucible 自己的掷骰卡、'
 '**不带 `flags.ember`** —— 本文件原有的聊天钩子第一句就是旗标闸，**一条都接不住**。'
 '⇒ 新增 `BOON_BANE_LABELS`，按结构选择器 `.boon-details .boon > .label` 认归属，'
 '跑在旗标闸**之前**。与 `crucible-cn 0.9.18` 的同名表刻意不重叠（那边 11 条是 crucible 自己的），'
 '两个模块在同一棵 DOM 上各翻各的。<br>'
 '· **日志页/创角/日历/派系/旅行事件/区域切片 33 条**：事件页 5 · 生物群系页 5 · 血统页 2 · '
 '创角 3（含一句**运行时 `+` 拼接**的 hint —— 键必须写拼完的全串，只登记前半截的话整句不变）· '
 '日历 2 · 派系 5 · 旅行事件 4 · 区域切片与图例 3（余烬地表 / 通路 / 屏障）· 层名兜底 4 · 其余。<br>'
 '　译名全部落到既定裁决上；⚠ 其中 `Attunement Features` glossary_ec 给的是「调谐特性」，'
 '**是陈旧值** —— 项目 2026-08-06 已裁 `Attunement`＝同调、`unify_rules.2026-08-06c` 把「调谐」'
 '列为已废变体，这里取**裁决**而非词表。<br>'
 '· ⚠ **审计口径本身修了两处**：① 第一版覆盖判据只喂了 `translateText` 那条通道，把 12 个 '
 '`patch*` 在**数据侧**改掉的那批全判成未覆盖 —— **天气 26 条**就是这么被误报的'
 '（`patchWeatherLabels` 改 `slices[*].config.weather[*].label`，`WEATHER` 表早就全译好了）；'
 '项目所有者一眼看出「我以为天气已经翻过了」，**他对我错**。'
 '② 第一版扫描的字段名清单漏了 `header`/`prompt` 一族，补扫又捞出 24 条（日志页分节标题 + '
 '三条升降梯/升降机的操控面板提示），其中 22 条已被现有表盖住。<br>'
 '· 判据 73 → 74：`R-ember-boon-labels`（钉「规则在、且在旗标闸之前」）。'
 '新表同时登记进 `SELFCHECK_TABLES`（41 → 42 张）—— 不登记的话面板对它是瞎的。<br>'
 '⚠ **明确不做**（项目所有者裁定「只有 GM 能看到的都不用做」）：Vista 建图器资产/图层名约 3080 条、'
 '区域地图场景定义 199 条、GM 配置/编辑器 40 条。另有 3 条经核实是 GM 项：'
 '`CONFIG.DND5E.armorClasses` 的 `Cor\'ak Leviathan Hide`（只在 5e 世界生效且走 CONFIG 通道）、'
 '两条 sheet 注册名。<br>'
 '主闸 74/0/0 · `--selftest` 357/357 · 灵敏度回测 13/13。 |')
anchor = '| `0.9.18` | 08-22 |'
i = s.index(anchor); j = s.index(nl + nl, i)
s = s[:j] + nl + nl + ROW + s[j:]

ROW2 = (
 '| — | 08-22 | **导入崩溃的定性（不发版，仅记录）**：项目所有者重导冒险时报 '
 '`Cannot create property \'public\' on string \'<中文>\'`，卡在 27%。<br>'
 '· **根因是上游改了字段形状，不是汉化**：ember **0.6.0** 里 `Potion of Climbing` / '
 '`Growing Thorns` 的 `system.description` 是**纯字符串**（349 / 1360 字符，见 '
 '`english-baseline/ember-0.6.0`），**0.6.1** 改成了 `{public, private}` 且 `type` 变 `consumable`。'
 '世界里存的是 0.6.0 那次导入的字符串，再导入时 Foundry 的 `SchemaField._updateDiff`'
 '（`common/data/fields.mjs:1231`）执行 `const source = (state.source[key] ||= {})` —— '
 '旧值是非空字符串 ⇒ `||=` 不替换 ⇒ 随后 `state.source["public"] = …` 往字符串上建属性 ⇒ TypeError。'
 '**报错里是中文纯属巧合**：不装汉化时存的是那 349 字符的英文原文，同样会崩。<br>'
 '· **不伤存档**：崩在写盘之前（`updateSource` 是在克隆上预校验），失败的文档被跳过、'
 '其余照常更新；真正的风险只有「半截导入」，重跑一次即可。<br>'
 '· 影响面精确到 **2 条**（用「世界里是字符串 ∧ 冒险里是对象」逐条比对算出，'
 '不是按类型猜），就地补成对象形状后导入通过。<br>'
 '· **我们这一侧清白**：`crucibleDescription` 拿 ember 5 包 + crucible 12 包共 **2769 条 item** '
 '回测，对象被压成字符串 **0**、形状变化 **0**。<br>'
 '· ⚠ 但顺手**收紧了一处**：`value === undefined`（源字段不存在）旧写法会返回 '
 '`translation.public` 这个**裸字符串** —— 那是在猜形状，而猜错的代价就是上面这种崩溃。'
 '改为按**译文自己的形状**还回去。新增 16 格形状回测（含一格回归对照：旧写法在那一格会吐字符串），'
 '这一支此前**从未被覆盖**（2769 条全包回测的采集器只收 `system` 里已有 `description` 的条目）。<br>'
 '⚠⚠ **两个探针写错的教训**：① 第一个按**纯英文名**精确匹配，而我们的译名是**双语**的（`卷轴匣 Scroll Case`）'
 '⇒ 目标就在眼前却报「不在」；② 第二个把 `type:"base"` 归进「schema 要字符串·别动」，'
 '可 `base` 恰恰是**取不到 schema** 的那一档 —— 真正的嫌疑全在里面。'
 '两次都是**判据写错**，不是数据的问题。 |')
i2 = s.index('| `v1.1.31` | 08-22 |'); j2 = s.index(nl + nl, i2)
s = s[:j2] + nl + nl + ROW2 + s[j2:]
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('PROJECT.md 已更新')
