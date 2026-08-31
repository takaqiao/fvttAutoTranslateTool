# -*- coding: utf-8 -*-
"""发版：ember v1.1.29 → v1.1.30 · crucible-cn 0.9.15 → 0.9.16，并写 PROJECT.md。"""
import io, json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

PAIRS = [('1-Ember汉化插件/module.json', '1.1.29', '1.1.30'),
         ('2-Crucible汉化插件/module.json', '0.9.15', '0.9.16')]
for p, old, new in PAIRS:
    s = io.open(p, encoding='utf-8', newline='').read()
    n = s.count(old)
    assert n >= 1, (p, n)
    io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
    print(f'  {p}: {old} → {new}（替换 {n} 处）')
    j = json.loads(io.open(p, encoding='utf-8-sig').read())
    assert j['version'] == new and new in j.get('download', ''), j.get('download')

P = 'PROJECT.md'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60])
    s = s.replace(a, b)

rep1('**发版状态（2026-08-22）：第三十二轮（⛔ 收官轮）之后又做了跟版与 UI 补漏共四轮'
     + nl + '（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～三轮），均已发版。'
     + nl + '当前已发布 `crucible-cn 0.9.15` / `ember_cn_unofficial v1.1.29`。**',
     '**发版状态（2026-08-22）：第三十二轮（⛔ 收官轮）之后又做了跟版与 UI 补漏共五轮'
     + nl + '（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～四轮），均已发版。'
     + nl + '当前已发布 `crucible-cn 0.9.16` / `ember_cn_unofficial v1.1.30`。**')
rep1('| crucible-cn | 汉化模块（本项目） | **0.9.15**（2026-08-22 发布；',
     '| crucible-cn | 汉化模块（本项目） | **0.9.16**（2026-08-22 发布；')
rep1('| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.29**（2026-08-22；',
     '| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.30**（2026-08-22；')

ROW = (
 '| `0.9.16` / `v1.1.30` | 08-22 | **装备族部件名 611 条补齐 + 给「抢回器」上判据与面板出口**。<br>'
 '· **部件显示名做完了**：先把**分母重算**了一遍 —— 上一轮拿随包图集当全集（1512），'
 '而图集是贴图，真正决定「进不进选择器」的是 `templateLayer.parts`。'
 '整份 ember.mjs 装不进 Node（六层 stub 之后卡在 `HEXES[…].terrain`，与部件毫无关系），'
 '改用**切片求值**：只取 53648~61260 行那一段，用真身 `foundry.utils` 建出 22 个模板。'
 '三重自证（切片首尾行逐字符 · 模板 id 集合 · 2579 个 id 三段形状并**逐个回图集交叉核**）'
 '⇒ 真全集 **1458** 条。<br>'
 '　本轮之前盖住 767（53%），补完 **1378 / 1458（95%）**；余下 80 条**全是纯数字**'
 '（symbol 族 10..82，上屏就是「10」）⇒ **可译部分 100%**。<br>'
 '　译法三层：品质四档直接取 crucible 系统定译（Shoddy 粗糙 / Standard 标准 / Fine 精良 / '
 'Superior 卓越，出现 87 次）· 本表已定过的同名段 · glossary_ec；'
 '⚠ **词表有六条套了就错**（`Shield`→护盾术是法术、`Point`→岬是地名裁决、`Split`→分裂、'
 '`Sticks`→专名、`Water`→水域、`Alchemist`→串行脏数据），逐条改判 —— '
 '**「词表里有」不等于「这条能用」**。<br>'
 '　拼串只出初稿，611 条**逐行读过一遍**才定稿；上表前三道机器核（互撞 0 / 撞现表 0 / '
 '同图层同名 0）＋ 611 键逐个回上游查字面量（**查不到 0 个** ⇒ 面板 miss 侧一条没涨）。<br>'
 '· **传送框下拉的 `Surface` / `Pathways` 两个分组名**补上了。上一轮记成「未登记」，'
 '其实够得到 —— `<optgroup label>` 早在第十四轮就进了属性白名单，纯粹是这两个词没进表。<br>'
 '· 🔥 **判据侧：`lang-reclaim.js` 从「零判据覆盖」变成 3 条**（`R-lang-reclaim-wired` / '
 '`R-lang-reclaim-mechanism` / `R-lang-squat-panel`，71 条 / 24 种 kind）。'
 '这套机制**全程静默**（抢回成功是中文、没被顶也是中文、根本没跑是英文且不报错），'
 '没装 `foundry_chn` 的维护者永远复现不了 ⇒ 任何一次「顺手清理」都能悄悄拆掉它。<br>'
 '　⚠ **第一版判据是弱的，灵敏度回测当场打回来**：require 只写了名字，于是把 '
 '`registerLangReclaim();` 整行删掉之后 `import { registerLangReclaim }` 照样满足它 —— '
 '而「文件还在包里、就是没人调」正是最可能的坏法。拆成两条、改钉整行之后，'
 '**9 格灵敏度回测每格只咬中该咬的那一条**（删调用 / 注释掉 / 删 import / 同步 XHR 改异步 / '
 '去掉 `enumerable:false` / 钩子挪到 ready / 摘掉账本出口 / 摘掉面板那一节 / 改探针键）。<br>'
 '· **自检面板新增「命名空间被顶 · 抢回」一节**：替人区分「没人顶」「顶了但抢回来了」'
 '「抢回器根本没跑」三态，并且**不信账本** —— 把抢回的每条丢回 `game.i18n.localize()` 当场复验。'
 '离线验六种情形 19 条断言全绿（用真身 `mergeObject` + 真实 `foundry_chn` 语料）。'
 '`I18N_PROBES` 另补两条被顶过的键。<br>'
 '⚠ **仍会看到英文**：GM 叠加层 155 条（PIXI `PreciseText`，DOM 遍历结构上够不到）。<br>'
 '⚠ **未了项**（登记在 `R-lang-reclaim-wired` 的 why 里）：抢回器只有**字面量闸**，'
 '钉不住**上游语义漂移**（Foundry 改 `mergeObject` 递归条件或 `getProperty` 顶层快路径）。'
 '要一道现跑的活闸，得新开一种 kind ＋ 一整套正反例，本轮没做 —— '
 '别下一轮读到那条就以为已经有活闸了。<br>'
 '主闸 71/0/0 · `--selftest` 357/357 · 灵敏度回测 9/9。 |')

anchor = '| `0.9.15` / `v1.1.29` | 08-22 |'
i = s.index(anchor)
j = s.index(nl + nl, i)
s = s[:j] + nl + nl + ROW + s[j:]
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('PROJECT.md 已更新')
