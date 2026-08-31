# -*- coding: utf-8 -*-
"""给 crucible 的硬编码翻译器上两条判据（接线 + 作用域），并同步四张登记表。"""
import io, os, sys, json, importlib.util
sys.stdout.reconfigure(encoding='utf-8')

RJ = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(RJ, encoding='utf-8'))
have = {a['id'] for a in d['assertions']}
assert 'R-crucible-hardcoded-wired' not in have

BG = (
  "2026-08-22 做过一次**上游硬编码串的全量审计**（`4-临时脚本/2026-08-22-hardcoded-audit/`）："
  "crucible 的 99 份模板几乎全走 `{{localize}}`，JS 展示字段里 i18n 键 437 个、硬编码只有 11 个，"
  "模板另有 7 条硬编码 placeholder/aria-label ⇒ 合计 **18 条**，而本仓此前**根本没有硬编码翻译脚本**，"
  "覆盖 0。量虽小，那 11 条却全在**掷骰面板与聊天卡**上 —— 每次攻击掷骰都会看到。"
)
NEW = [
{
 "id": "R-crucible-hardcoded-wired",
 "title": "crucible 的硬编码翻译器必须真的被调（`registerCrucibleHardcoded();` 不许消失或被注释）",
 "decision": "2026-08-22（UI 补漏第六轮）",
 "why": (
   BG +
   "｜与 `R-lang-reclaim-wired` 同型、同理由：新文件在包里、却没人调它，"
   "两仓 CI、主闸、面板**没有一样会响**，而屏幕上只是英文。"
   "本条的 require 直接钉**整行**（import 行 + 调用行），不钉名字 —— "
   "0.9.16 那次已经证明只钉名字会被同文件的 import 行满足（见 `R-lang-reclaim-wired` 的 why）。"
   "｜⚠ 判据边界：只钉「被调」，不判它翻得对不对；作用域正确性归 `R-crucible-hardcoded-scope`，"
   "报文正确性归真浏览器验 `4-临时脚本/2026-08-22-crucible-hardcoded/scope.html`（8/8，**不入主闸**）。"
 ),
 "kind": "source_literal",
 "min_files": 1,
 "min_checks": 4,
 "files": [{"repo": "crucible", "path": "babele-register.js"}],
 "require": [
   "import { registerCrucibleHardcoded } from './crucible-hardcoded-cn.mjs';",
   "registerCrucibleHardcoded();",
 ],
 "forbid_re": [
   "//\s*registerCrucibleHardcoded\s*\(\s*\)\s*;",
   "/\*[^*]*registerCrucibleHardcoded\s*\(\s*\)\s*;",
 ],
},
{
 "id": "R-crucible-hardcoded-scope",
 "title": "硬编码翻译器的**两档作用域**不许被合并 —— 通用词必须留在 `.crucible` 里",
 "decision": "2026-08-22（UI 补漏第六轮）",
 "why": (
   "规则分两档，判据是「选择器本身够不够独特」：`STRUCTURAL` 档"
   "（`.boon-details .boon > .label` / `.context-tags .tag-icon`）是 crucible 模板独有的结构，"
   "命中即归属，**不要求** `.crucible` 祖先 —— 必须如此，因为掷骰聊天卡的根是 "
   "`<div class=\"{{cssClass}} line-item\">`（`standard-check-chat.hbs:1`），`cssClass` 是**动态**的、"
   "并不保证含 `crucible`，要求祖先反而会把最值钱的两条规则漏掉。"
   "`SCOPED` 档的词太通用（`Item Name` / `Actor Name` / `Add one …` 别的模块也会用），"
   "**必须**落在 `.crucible` 内才动。"
   "｜⚠ 把两档合并（或把 `apply(SCOPED, scopes)` 改成 `apply(SCOPED, [root])`）之后，"
   "`renderApplicationV2` 对**所有**窗口都触发 ⇒ 我们会去改别的模块输入框的 placeholder。"
   "**那是越界，而且是静默的**：受害的是别人的窗口，我们自己的世界里一切正常。"
   "本项目一贯把这类越界看得比漏译更重（EXACT 与作用域表之分就是为它设的）。"
   "｜另钉 `scopes.push(root)` 那一行：根**自己**带 `crucible` 类时 `querySelectorAll` 找不到它"
   "（只查后代），漏掉这一行会让所有卡的 placeholder 静默失效 —— 真浏览器验里 "
   "`#sheet` / `#sheet2` / `#sheet4` / `#sheet5` 四个根正是这种形态。"
   "｜实测依据：`scope.html` 在真浏览器里跑真身函数，负例区（一个 "
   "`class=\"application some-other-module\"` 的窗口，摆着同名 placeholder / tooltip / .label / aria-label）"
   "**一处未动**，计数 `{text:0,attr:0,aria:0}`；首遍共改 18 处，第二遍全零（幂等）。8/8 通过。"
 ),
 "kind": "source_literal",
 "min_files": 1,
 "min_checks": 5,
 "files": [{"repo": "crucible", "path": "crucible-hardcoded-cn.mjs"}],
 "require": [
   "const STRUCTURAL = [",
   "const SCOPED = [",
   "  if (root.classList?.contains?.(\"crucible\")) scopes.push(root);",
   "  apply(SCOPED, scopes);",
   "  for (const el of scopes.flatMap((r) => Array.from(r.querySelectorAll(\"[aria-label]\")))) {",
 ],
},
]
i = next(k for k, a in enumerate(d['assertions']) if a['id'] == 'R-lang-squat-panel')
d['assertions'][i:i] = NEW
d['meta']['updated'] = "2026-08-22（UI 补漏第六轮：crucible 硬编码翻译器 18 条 + 两条判据）"
io.open(RJ, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print('断言总数 =', len(d['assertions']))

PY = os.path.join('3-常用脚本', 'qa', 'assert_resolutions.py')
spec = importlib.util.spec_from_file_location('ar', os.path.abspath(PY))
M = importlib.util.module_from_spec(spec); sys.modules['ar'] = M; spec.loader.exec_module(M)
byid = {a['id']: a for a in d['assertions']}
der = M._derive_payload_floors([byid['R-crucible-hardcoded-wired'], byid['R-crucible-hardcoded-scope']])
for k, v in der.items(): print(' 现推地板', k, '=', v)

s = io.open(PY, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
fmt = lambda sp: '{' + ', '.join('"%s": ("%s", %r)' % (k, v[0], v[1]) for k, v in sorted(sp.items())) + '}'
rep1("    'R-lang-reclaim-mechanism': 'source_literal',",
     "    'R-lang-reclaim-mechanism': 'source_literal'," + nl +
     "    'R-crucible-hardcoded-wired': 'source_literal'," + nl +
     "    'R-crucible-hardcoded-scope': 'source_literal',")
line = [l for l in s.split(nl) if l.startswith('    "R-lang-reclaim-mechanism": {')]
assert len(line) == 1
rep1(line[0], line[0] + nl
     + '    "R-crucible-hardcoded-wired": %s,' % fmt(der['R-crucible-hardcoded-wired']) + nl
     + '    "R-crucible-hardcoded-scope": %s,' % fmt(der['R-crucible-hardcoded-scope']))
rep1('    "R-lang-reclaim-mechanism": 6,',
     '    "R-lang-reclaim-mechanism": 6,' + nl
     + '    "R-crucible-hardcoded-wired": 4,'   # 1 仓 ×（2 require + 2 forbid）
     + nl + '    "R-crucible-hardcoded-scope": 5,')  # 1 仓 × 5 require
i2 = s.index('RULESET_SHAPE = {'); j2 = s.index(nl + '}' + nl, i2)
blk = s[i2:j2]
assert blk.count('"source_literal": 4,') == 1
blk = blk.replace('"source_literal": 4,', '"source_literal": 6,')
assert blk.count('    "min_assertions": 71,') == 1
blk = blk.replace('    "min_assertions": 71,', '    "min_assertions": 73,')
s = s[:i2] + blk + s[j2:]
io.open(PY, 'w', encoding='utf-8', newline='').write(s)
print('四张登记表已同步')
