# -*- coding: utf-8 -*-
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
p = '2-Crucible汉化插件/module.json'
s = io.open(p, encoding='utf-8', newline='').read()
assert s.count('0.9.17') >= 1
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace('0.9.17', '0.9.18'))
j = json.loads(io.open(p, encoding='utf-8-sig').read())
assert j['version'] == '0.9.18' and '0.9.18' in j['download']
print('  module.json → 0.9.18')

P = 'PROJECT.md'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60])
    s = s.replace(a, b)
rep1('（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～五轮），均已发版。' + nl
     + '当前已发布 `crucible-cn 0.9.17` / `ember_cn_unofficial v1.1.30`。**',
     '（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～六轮），均已发版。' + nl
     + '当前已发布 `crucible-cn 0.9.18` / `ember_cn_unofficial v1.1.30`。**')
rep1('| crucible-cn | 汉化模块（本项目） | **0.9.17**（2026-08-22 发布；',
     '| crucible-cn | 汉化模块（本项目） | **0.9.18**（2026-08-22 发布；')

ROW = (
 '| `0.9.18` | 08-22 | **crucible 侧第一个硬编码汉化通道：18 条，此前覆盖 0**。<br>'
 '· 起因是一次**上游硬编码串的全量审计**（`4-临时脚本/2026-08-22-hardcoded-audit/`）：'
 '扫模板文本节点 / 展示属性 / JS 的 11 个展示字段 / `ui.notifications` 实参 / 反引号 HTML 五个通道。'
 '结论 —— crucible 的 99 份模板**几乎全走 `{{localize}}`**，JS 展示字段里 i18n 键 437 个、'
 '硬编码只有 **11** 个，模板另有 7 条硬编码 placeholder/aria-label ⇒ **合计 18 条**，'
 '而本仓此前**根本没有硬编码翻译脚本**（只有 babele 注册 + 抢回器 + 自检面板），覆盖 0。'
 '量虽小，那 11 条全在**掷骰面板与聊天卡**上：`Special` / `Reserved Action` / `Slow Weaponry` / '
 '`Bulky Armor` / `Elite` / `Boss`（加值减值来源）与 `Strikes` / `Reload` / `Weapon Tags` / '
 '`Spell Tags` / `Skill Tags`（上下文标签 tooltip）—— **每次攻击掷骰都会看到**。<br>'
 '· **为什么不塞进 `lang/cn.json`**（那 11 条其实能走：`{{localize boon.label}}` 与 Foundry 的 '
 '`if ( game.i18n.has(text) ) _loc(text)` 都会拿字面量当键查）：① `R-lang-parity` 钉死 '
 '`len(cn)===len(en)`，加上游没有的键会当场红，而那条闸是 1.1.0「77% 键静默失效」换来的；'
 '② 裸键是**全局**的 —— 加一个 `"Reload"` 之后**任何**模块的 `data-tooltip="Reload"` 都会变中文，'
 '我们刚被 `foundry_chn` 的裸串顶掉 42 条，不该转头自己当同一种肇事者。<br>'
 '· 译名全部对齐既有定译，不新造：`ACTION.TAG_CATEGORIES.Special`＝特殊 · '
 '`ACTOR.ADVERSARY.THREAT_RANKS.{Elite,Boss}`＝精英/首领 · `ARMOR.PROPERTIES.Bulky`＝笨重 · '
 '`ACTIVE_EFFECT.STATUSES.Slowed`＝迟缓 · `ACTION.DEFAULT_ACTIONS.Strike.Name`＝打击 · '
 '`ACTION.TAG.Reload`＝装填 · 三条 `* Tags` 与 `ACTION.TAGS.Action`＝动作标签 同构 · '
 '`TYPES.ActiveEffect.affix`＝词缀 · `ACTOR.GROUP.*`＝团队。<br>'
 '· 🔑 **规则分两档**，这是本轮的设计核心：`STRUCTURAL` 档的选择器是 crucible 模板独有的结构，'
 '命中即归属、**不要求** `.crucible` 祖先 —— 必须如此，因为掷骰聊天卡的根是 '
 '`<div class="{{cssClass}} line-item">`（动态、不保证含 crucible），要求祖先反而会漏掉最值钱的两条；'
 '`SCOPED` 档的词太通用（`Item Name`/`Actor Name`/`Add one …`），**必须**落在 `.crucible` 内 —— '
 '否则 `renderApplicationV2` 对所有窗口触发，我们会去改别的模块的输入框。<br>'
 '· **真浏览器验** `scope.html`（真 DOM / 真选择器引擎 / 真身函数，代码从发布文件原文内联）：'
 '**8/8** —— 18 处全译、负例区（别的模块窗口里摆的同名 placeholder/tooltip/.label/aria-label）'
 '**一处未动**、第二遍全零（幂等）。<br>'
 '· 判据 +2（71 → 73）：`R-crucible-hardcoded-wired`（接线，钉整行不钉名字）与 '
 '`R-crucible-hardcoded-scope`（两档作用域不许合并）。<br>'
 '⚠⚠ **审计本身有一处大错，一并记下来**：第一版覆盖判据只喂了 `translateText` 那条通道，'
 '于是把 **12 个 `patch*` 函数在数据侧改掉**的那批全判成「未覆盖」—— '
 '**天气 26 条**就是这么被误报的（它由 `patchWeatherLabels()` 改 '
 '`slices[*].config.weather[*].label`，压根不经过 DOM 查表；`WEATHER` 表 :1818 早就全译好了）。'
 '项目所有者一眼看出「我以为天气已经翻过了」——**他对我错**。'
 '⇒ 本项目有**两条**汉化通道（DOM 查表 / 数据侧 patch），量覆盖率必须两条都算；'
 '并入之后 ember 的候选玩家可见缺口从 107 降到 **74**，再剔 GM 项约 **40 条**（下一轮做）。<br>'
 '主闸 73/0/0 · `--selftest` 357/357 · 真浏览器作用域验 8/8。 |')

anchor = '| `0.9.17` | 08-22 |'
i = s.index(anchor)
j = s.index(nl + nl, i)
s = s[:j] + nl + nl + ROW + s[j:]
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('PROJECT.md 已更新')
