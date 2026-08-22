# -*- coding: utf-8 -*-
"""往规则集加两条 source_literal 判据，把「抢回器」这套机制钉在源码里。

为什么是 source_literal 而不是新开一种 kind：
  这套机制**我们这一侧**能坏的方式全是「某一行被删/被改」——
  取值改回 fetch、enumerable:false 被去掉、registerLangReclaim 不再被调、
  面板那一节被从 checkI18n 里摘掉。逐字符钉行就能全部咬住，
  而 require 不会因为正则写坏而静默失效（空转形态 (a)/(f) 在它身上不成立）。
  真正钉不住的只有一种：**上游 Foundry 把 mergeObject / getProperty 的语义改了**，
  那要靠现跑的活闸。这一条**明写在 why 里当未了项**，不假装钉住了。
"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')

P = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(P, encoding='utf-8'))
ids = {a['id'] for a in d['assertions']}
assert 'R-lang-reclaim-wired' not in ids and 'R-lang-squat-panel' not in ids, '已经加过了'
before = len(d['assertions'])

NEW = [
{
 "id": "R-lang-reclaim-wired",
 "title": "译文抢回器必须**接在管线上**，且三处实测逼出来的实现细节不许被顺手改掉",
 "decision": "2026-08-22（第三十六轮 / UI 补漏第四轮）",
 "why": (
   "`crucible-cn/lang-reclaim.js` 修的是一个**跨模块**故障：`foundry_chn/cn.json` 顶层有 98 个"
   "**裸字符串**，其中 `\"TOKEN\"` 与 `\"WARNING\"` 正好压在我们建好的两个命名空间上；"
   "Foundry 逐份 `expandObject` 后按模块顺序 `mergeObject`，而 `_mergeUpdate` **只在两边都是对象时递归**，"
   "否则整块覆盖 ⇒ 我们那 42 条（夹击标签 5 · 移动方式与「强制」27 · 报错提示 10）被一个字符串盖掉，"
   "`localize()` 退回英文 fallback。"
   "｜这套东西**全程静默**：修好了是正常中文，没被顶也是正常中文，抢回器根本没跑还是……英文，而且不报错。"
   "维护者自己那台机器只要没装 `foundry_chn` 就**永远复现不了**。⇒ 没有判据的话，"
   "下一轮任何一次「顺手清理」都能把它悄悄拆掉，而两个汉化仓的 CI、主闸、面板**没有一样会响**"
   "（复核过：本轮之前 `lang-reclaim` 在整个判据侧出现 **0 次**）。"
   "｜本条钉四件，全部是**实测逼出来的**、不是风格偏好："
   "① `registerLangReclaim()` 必须真的在 `babele-register.js` 里被调 —— 不调等于文件白进包；"
   "② 取值必须是**同步 XHR**（`xhr.open('GET', url, false)`）—— 实测 top-level await **不推迟 "
   "`DOMContentLoaded`**，而 Foundry 正是在 `DOMContentLoaded` 里 `await game.initialize()`"
   "（`i18nInit` 在其中），改回 `fetch`+`await` 必然赶不上，且**赶不上的样子就是英文，不报错**；"
   "③ 写入必须是 `enumerable: false` —— 否则 `#hotReloadJSON` 的 `mergeObject` 会因 `expanded.TOKEN` "
   "已是字符串而在严格模式抛 `TypeError`（离线复刻器用**真身** `mergeObject` 正反两侧实测过）；"
   "④ 必须挂在 `i18nInit`。"
   "｜⚠ **判据边界（不许读成钉住了全部）**：本条只钉**我们这一侧的源码行**。"
   "真正钉不住的是**上游语义漂移** —— 哪天 Foundry 改了 `mergeObject` 的递归条件或 `getProperty` 的"
   "顶层快路径（`if ( key in object ) return object[key]`），这四行原样留着、机制却已经失效，"
   "本条照样全绿。那需要一道**现跑**的活闸（形如 `R-selfcheck-d-liveness`：把离线复刻器提升进 "
   "`3-常用脚本/qa/`，每跑一次闸就用真身 helpers 重放一次装载顺序）。"
   "本轮**没做**，理由是它要新开一种 kind 连带一整套正反例；"
   "登记在此当未了项，别下一轮读到这条就以为已经有活闸了。"
 ),
 "kind": "source_literal",
 "min_files": 2,
 "min_checks": 6,
 "files": [
  {"repo": "crucible", "path": "lang-reclaim.js"},
  {"repo": "crucible", "path": "babele-register.js"}
 ],
 "require": ["registerLangReclaim"],
 "forbid_re": [
  "//\s*registerLangReclaim\s*\(",
  "/\*[^*]*registerLangReclaim"
 ]
},
{
 "id": "R-lang-squat-panel",
 "title": "自检面板必须留着「命名空间被顶 · 抢回」那一节 —— 它是这套静默机制唯一的人可见出口",
 "decision": "2026-08-22（第三十六轮 / UI 补漏第四轮）",
 "why": (
   "抢回是静默的（理由见 `R-lang-reclaim-wired`），于是面板这一节是**唯一**能替人区分三态的地方："
   "「没人顶」「顶了但抢回来了」「抢回器根本没跑」。它读的是抢回器自己记的账，"
   "并且**不信账本**、把抢回的每一条丢回 `game.i18n.localize()` 当场复验 —— "
   "账上写了、玩家通道上没生效，正是本项目登记的空转形态 (h)（探针自己捏输入）。"
   "｜要钉的是**接线**，不是实现：`checkLangSquat` 这个函数留着、却没人从 `checkI18n` 里调它，"
   "面板就会安安静静少一节，而 `R-selfcheck-twin` 只保证两份面板**彼此**相同 —— "
   "**一起摘掉它照样绿**（这正是 `R-selfcheck-d-section-name` 当初要新开 source_literal 的同一个理由）。"
   "所以 require 里同时钉「函数在」与「调用点在」两行。"
   "｜另钉两条探针键：`TOKEN.MOVEMENT.ACTIONS.walk.label` 与 `WARNING.NoParty`。"
   "它们各来自一个被顶过的命名空间，走的是 `localize()` 这条**玩家真正用的**通道；"
   "与上面那一节两边同时绿，才说明「账上抢回了」和「屏幕上真是中文」是同一件事。"
   "｜⚠ 判据边界：只钉这四行**在不在**，不判那一节报得对不对 —— "
   "报文对不对由离线验（`4-临时脚本/2026-08-22-ui-round4/verify_squat.mjs`，"
   "六种情形 19 条断言全绿）负责，而那是**一次性探针、不入主闸**，同 `R-lang-reclaim-wired` 的未了项。"
 ),
 "kind": "source_literal",
 "min_files": 2,
 "min_checks": 8,
 "files": [
  {"repo": "ember", "path": "scripts/ember-cn-selfcheck.mjs"},
  {"repo": "crucible", "path": "selfcheck/cn-selfcheck.mjs"}
 ],
 "require": [
  "function checkLangSquat(S, table) {",
  "out.push(...checkLangSquat(S, table));",
  "\"TOKEN.MOVEMENT.ACTIONS.walk.label\"",
  "\"WARNING.NoParty\""
 ]
}
]
d['assertions'].extend(NEW)
d['meta']['updated'] = "2026-08-22（第三十六轮 / UI 补漏第四轮：+R-lang-reclaim-wired · +R-lang-squat-panel）"
io.open(P, 'w', encoding='utf-8', newline='\n').write(
    json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print(f"断言 {before} → {len(d['assertions'])}")
