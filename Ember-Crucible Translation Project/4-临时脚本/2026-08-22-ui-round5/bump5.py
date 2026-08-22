# -*- coding: utf-8 -*-
"""发 crucible-cn 0.9.17（ember 本轮无改动，不动版本）。"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
p = '2-Crucible汉化插件/module.json'
s = io.open(p, encoding='utf-8', newline='').read()
n = s.count('0.9.16'); assert n >= 1
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace('0.9.16', '0.9.17'))
j = json.loads(io.open(p, encoding='utf-8-sig').read())
assert j['version'] == '0.9.17' and '0.9.17' in j['download']
print(f'  module.json 0.9.16 → 0.9.17（{n} 处）')

P = 'PROJECT.md'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60])
    s = s.replace(a, b)

rep1('（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～四轮），均已发版。' + nl
     + '当前已发布 `crucible-cn 0.9.16` / `ember_cn_unofficial v1.1.30`。**',
     '（crucible 0.10.2 跟版 · ember 0.6.1 跟版 · UI 补漏第一～五轮），均已发版。' + nl
     + '当前已发布 `crucible-cn 0.9.17` / `ember_cn_unofficial v1.1.30`。**')
rep1('| crucible-cn | 汉化模块（本项目） | **0.9.16**（2026-08-22 发布；',
     '| crucible-cn | 汉化模块（本项目） | **0.9.17**（2026-08-22 发布；')

ROW = (
 '| `0.9.17` | 08-22 | 🔥 **急修：0.9.15/0.9.16 的译文抢回器在真实环境里一次都没生效过**。<br>'
 '· **病象**：项目所有者在 VPS 上实测，「强制移动」的 `Forced` 与可视化夹击的 '
 '`{allies} Allies / {enemies} Enemies / Engagement {n}` 仍是英文 —— 正是抢回器该管的那 42 条。<br>'
 '· **病因**：`languages[].path` 在**已安装**的包上，运行时**已经是** '
 '`modules/crucible-cn/lang/cn.json`，不是清单里写的 `lang/cn.json`。'
 '成因在 Foundry 服务端 `dist/packages/package.mjs`：`languages.element.fields.path = new PackageAssetField`，'
 '而 `PackageAssetField.initialize()` 对 `installed` 的包把值解析成 **clientPath**'
 '（`relativeToPackage` 默认真 ⇒ 已含 `modules/<id>/`）。我又拼了一次前缀 ⇒ '
 '`/modules/crucible-cn/modules/crucible-cn/lang/cn.json` **404**，`loadOwnTranslationsSync` 整段抛，'
 '抢回一条都没写。修法：新增纯函数 `toClientPath()`，两种形态都接住。<br>'
 '· ✅ **面板那一节当场立功**：一次粘贴就打出 '
 '`phase:"error" · HTTP 404 for /modules/crucible-cn/modules/crucible-cn/lang/cn.json`，'
 '病因**直接写在报文里**，没有第二轮猜。这正是它存在的理由 —— '
 '抢回失败在屏幕上只是英文、控制台只有一条 warn。<br>'
 '⚠⚠ **两处教训，都要记死**：<br>'
 '　① **判据钉在了「会坏的那一行的隔壁」**。`R-lang-reclaim-mechanism` 钉的是 '
 '`xhr.open(\'GET\', url, false);`，而坏的是 **`url` 怎么来的**。同轮的离线复刻器同样脱靶：'
 '它验 `reclaimTranslations`（纯函数），表**直接从磁盘读** ⇒ `ownLanguagePath()` / '
 '`loadOwnTranslationsSync()` **一次都没被执行过**。这是空转形态 (h) 的又一变体：'
 '**验的对象不是会坏的那个对象**。⇒ 现在两行都钉死（require 4→6），并新增端到端离线验 '
 '`verify_reclaim_e2e.mjs`：从 `registerLangReclaim()` 真入口进，走完'
 '钩子→取路径→取值→回写全链，16 条断言；第 ③ 格是**回归对照**（断言旧拼法的 URL 在假服务器上 404）'
 '⇒ 这一验若在 0.9.16 上跑必红。灵敏度回测 9→11 格，新增的 ⓪⑩ 两格正是这次的事故与其归一分支。<br>'
 '　② **第二次拿本机语料推 VPS 的病因**。第三轮判定「肇事者是 `foundry_chn` 顶层裸串」'
 '用的是**本机**那份 `foundry_chn/cn.json`，离线复刻器也拿它跑出 42/42 —— 而项目所有者'
 '**早就明确说过**「汉化都装在 VPS 上，本地可能是旧版」。VPS 实测的真实形状是：'
 '顶层 `TOKEN` 是**对象**（内容是别的模块的 `TABS`，我们的 LABELS/MOVEMENT 已被冲掉）、'
 '`WARNING` 是裸串「警告」—— 比本机那份多一步「后来的模块又把 TOKEN 建成对象」。'
 '结论侥幸没变（抢回逻辑对两种形状都成立），但**推导过程当时并不成立**。'
 '⇒ 本轮 e2e 验的语料形状一律取自出问题的那台机器。<br>'
 '主闸 71/0/0 · `--selftest` 357/357 · 灵敏度回测 11/11 · e2e 16/16。 |')

anchor = '| `0.9.16` / `v1.1.30` | 08-22 |'
i = s.index(anchor)
j = s.index(nl + nl, i)
s = s[:j] + nl + nl + ROW + s[j:]
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('PROJECT.md 已更新')
