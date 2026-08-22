# -*- coding: utf-8 -*-
"""把 R-lang-reclaim-wired 拆成两条，并把 require 从「名字出现过」换成「那一行真的在」。

为什么要改：灵敏度回测（sensitivity.py）当场抓到两处漏 ——
  ① 把 `registerLangReclaim();` 整行删掉，判据**没响**：因为 require 只写了名字
     `registerLangReclaim`，而同文件的 `import { registerLangReclaim } from ...` 一行
     照样含这个子串 ⇒ 「文件还在包里、但没人调」这个最可能的坏法**恰好钉不住**；
  ③ 把同步 XHR 换成异步，判据**没响**：那几条实现细节压根没进 require
     （source_literal 的 require 是**逐文件全量套用**的，两个文件没有共同的强字面量 ⇒
      只能拆成两条，各钉各的）。
这正是「新判据必须做灵敏度回测」的用处：不做的话，这两条会以「已经钉住了」的名义躺在库里。
"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(P, encoding='utf-8'))
olds = [a for a in d['assertions'] if a['id'] == 'R-lang-reclaim-wired']
assert len(olds) == 1
old = olds[0]
d['assertions'].remove(old)

WHY_TAIL = (
  "｜⚠ **判据边界（不许读成钉住了全部）**：本条只钉**我们这一侧的源码行**。"
  "钉不住的是**上游语义漂移** —— 哪天 Foundry 改了 `mergeObject` 的递归条件或 `getProperty` 的"
  "顶层快路径（`if ( key in object ) return object[key]`），这些行原样留着、机制却已经失效，"
  "本条照样全绿。那需要一道**现跑**的活闸（形如 `R-selfcheck-d-liveness`：把离线复刻器提升进 "
  "`3-常用脚本/qa/`，每跑一次闸就用真身 helpers 重放一次装载顺序）。本轮**没做**，"
  "理由是它要新开一种 kind 连带一整套正反例；登记在此当未了项，"
  "别下一轮读到这条就以为已经有活闸了。"
)
BG = (
  "`crucible-cn/lang-reclaim.js` 修的是一个**跨模块**故障：`foundry_chn/cn.json` 顶层有 98 个"
  "**裸字符串**，其中 `\"TOKEN\"` 与 `\"WARNING\"` 正好压在我们建好的两个命名空间上；"
  "Foundry 逐份 `expandObject` 后按模块顺序 `mergeObject`，而 `_mergeUpdate` **只在两边都是对象时递归**，"
  "否则整块覆盖 ⇒ 我们那 42 条（夹击标签 5 · 移动方式与「强制」27 · 报错提示 10）被一个字符串盖掉，"
  "`localize()` 退回英文 fallback。"
  "｜这套东西**全程静默**：抢回成功是正常中文，没被顶也是正常中文，抢回器根本没跑则是英文**且不报错**。"
  "维护者只要没装 `foundry_chn` 就**永远复现不了**。本轮之前 `lang-reclaim` 在整个判据侧出现 **0 次**。"
)

NEW = [
{
 "id": "R-lang-reclaim-wired",
 "title": "抢回器必须**真的被调**：`registerLangReclaim();` 那一行不许消失，也不许被注释掉",
 "decision": "2026-08-22（第三十六轮 / UI 补漏第四轮；同轮灵敏度回测后加强）",
 "why": (
   BG +
   "｜⚠ **本条第一版是弱的，灵敏度回测当场把它打回来了**：原来 require 只写了名字 "
   "`registerLangReclaim`，于是把 `registerLangReclaim();` 整行删掉之后，"
   "同文件的 `import { registerLangReclaim } from './lang-reclaim.js';` **照样满足了它** —— "
   "而「文件还在包里、就是没人调」正是这套机制最可能的坏法（`.js` 静静躺在 zip 里，"
   "两仓 CI、主闸、面板没有一样会响）。现在 require 钉的是**两行各自的完整字面量**："
   "import 那一行 + 调用那一行。forbid_re 是第二道保险，专抓「注释掉但没删」这种半吊子回退"
   "（回测②实测会响）。" + WHY_TAIL
 ),
 "kind": "source_literal",
 "min_files": 1,
 "min_checks": 4,
 "files": [{"repo": "crucible", "path": "babele-register.js"}],
 "require": [
   "import { registerLangReclaim } from './lang-reclaim.js';",
   "registerLangReclaim();"
 ],
 "forbid_re": [
   "//\s*registerLangReclaim\s*\(\s*\)\s*;",
   "/\*[^*]*registerLangReclaim\s*\(\s*\)\s*;"
 ]
},
{
 "id": "R-lang-reclaim-mechanism",
 "title": "抢回器那三处**实测逼出来的**实现细节不许被顺手改掉（同步 XHR / 非枚举 / i18nInit）",
 "decision": "2026-08-22（第三十六轮 / UI 补漏第四轮；同轮灵敏度回测后新增）",
 "why": (
   BG +
   "｜这三处每一处都是**实测**逼出来的，不是风格偏好，而且**改坏之后的样子全都是「英文，不报错」**："
   "① `xhr.open('GET', url, false)` —— 必须**同步**。实测 top-level await **不推迟 "
   "`DOMContentLoaded`**（探针序列：mod:before-await | DOMContentLoaded | load | mod:after-await），"
   "而 Foundry 正是在 `DOMContentLoaded` 里 `await game.initialize()`、`i18nInit` 就在其中 ⇒ "
   "改回 `fetch`+`await` 必然赶不上。"
   "② `enumerable: false` —— 否则 `#hotReloadJSON` 的 `mergeObject` 会因 `expanded.TOKEN` 已是字符串"
   "而在严格模式抛 `TypeError`（离线复刻器用**真身** `mergeObject` 正反两侧实测过）。"
   "③ `Hooks.once('i18nInit', …)` —— 早于它取不到 `translations`，晚于它那 42 条已经被查过了。"
   "④ `pkg.api.getReclaimState = getReclaimState;` —— 账本的出口，摘掉它面板那一节就只能报"
   "「装了但没挂账本」。"
   "｜⚠ 本条与 `R-lang-reclaim-wired` **必须分成两条**：`source_literal` 的 require 是"
   "**逐文件全量套用**的，而这两个文件没有共同的强字面量 —— 硬塞进一条就只能退回到"
   "「钉名字」那种弱写法，那正是回测打回来的东西。" + WHY_TAIL
 ),
 "kind": "source_literal",
 "min_files": 1,
 "min_checks": 4,
 "files": [{"repo": "crucible", "path": "lang-reclaim.js"}],
 "require": [
   "xhr.open('GET', url, false);",
   "      enumerable: false,",
   "Hooks.once('i18nInit', () => {",
   "pkg.api.getReclaimState = getReclaimState;"
 ]
}
]
# 插回原来的位置，保持规则集里相邻
i = next(k for k, a in enumerate(d['assertions']) if a['id'] == 'R-lang-squat-panel')
d['assertions'][i:i] = NEW
d['meta']['updated'] = "2026-08-22（第三十六轮 / UI 补漏第四轮：+3 判据、面板 D 档 4 个覆盖侧登记值跟涨）"
io.open(P, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print('断言总数 =', len(d['assertions']))
