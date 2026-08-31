# -*- coding: utf-8 -*-
"""把规则分成两档：结构选择器够独特的直接跑；通用词必须落在 `.crucible` 作用域内。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '2-Crucible汉化插件/crucible-hardcoded-cn.mjs'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

OLD = nl.join([
'/** 每条规则：选择器 + 作用在哪儿 + 用哪张表。**纯数据**，好逐条回模板核。 */',
'const RULES = [',
'  { sel: ".boon-details .boon > .label, .bane-details .bane > .label", kind: "text", table: BOON_BANE_LABELS },',
'  { sel: ".context-tags .tag-icon[data-tooltip]", kind: "attr", attr: "data-tooltip", table: CONTEXT_TOOLTIPS },',
'  { sel: "input[placeholder], textarea[placeholder]", kind: "attr", attr: "placeholder", table: PLACEHOLDERS },',
'];',
])
NEW = nl.join([
'/**',
' * 规则**分两档**，判据是「这个选择器本身够不够独特」：',
' *',
' * `STRUCTURAL` —— 选择器是 crucible 模板独有的结构（`.boon-details .boon > .label`、',
' *   `.context-tags .tag-icon`），命中即归属，不再要求祖先带 `crucible` 类。',
' *   ⚠ **必须这样**：掷骰聊天卡的根是 `<div class="{{cssClass}} line-item">`',
' *   （`standard-check-chat.hbs:1`），`cssClass` 是**动态**的、并不保证含 `crucible` ——',
' *   要求 `.crucible` 祖先反而会把这两条最值钱的规则漏掉。',
' *',
' * `SCOPED` —— 词太通用（`Item Name` / `Actor Name` 这种别的模块也会用），',
' *   必须落在 `.crucible` 作用域内才动。crucible 的每一张卡都带这个类',
' *   （动作 :14148 / 角色 :14528 / 团队 :15862 / 全屏创建 :16470 / 效果 :17731 /',
' *    词缀 :17827 / 物品 :24351），所以够得着且只够得着自家窗口。',
' *   ⚠ 不这么分的话，`renderApplicationV2` 对**所有**窗口都触发，',
' *     我们会去改别的模块输入框的 placeholder —— 那正是本项目一贯最忌的越界。',
' */',
'const STRUCTURAL = [',
'  { sel: ".boon-details .boon > .label, .bane-details .bane > .label", kind: "text", table: BOON_BANE_LABELS },',
'  { sel: ".context-tags .tag-icon[data-tooltip]", kind: "attr", attr: "data-tooltip", table: CONTEXT_TOOLTIPS },',
'];',
'const SCOPED = [',
'  { sel: "input[placeholder], textarea[placeholder]", kind: "attr", attr: "placeholder", table: PLACEHOLDERS },',
'];',
])
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)

OLD2 = nl.join([
'export function translateCrucibleRoot(root) {',
'  const n = { text: 0, attr: 0, aria: 0 };',
'  if (!root || typeof root.querySelectorAll !== "function") return n;',
'  for (const rule of RULES) {',
'    for (const el of root.querySelectorAll(rule.sel)) {',
])
NEW2 = nl.join([
'export function translateCrucibleRoot(root) {',
'  const n = { text: 0, attr: 0, aria: 0 };',
'  if (!root || typeof root.querySelectorAll !== "function") return n;',
'',
'  // `.crucible` 作用域：根**自己**带这个类时 querySelectorAll 是找不到它的',
'  // （只查后代），所以要单独把根算进去。',
'  const scopes = [];',
'  if (root.classList?.contains?.("crucible")) scopes.push(root);',
'  for (const el of root.querySelectorAll(".crucible")) scopes.push(el);',
'',
'  const apply = (rules, roots) => {',
'  for (const rule of rules) {',
'    for (const el of roots.flatMap((r) => Array.from(r.querySelectorAll(rule.sel)))) {',
])
assert s.count(OLD2) == 1, s.count(OLD2)
s = s.replace(OLD2, NEW2)

OLD3 = nl.join([
'        if (cn && cn !== raw) { el.setAttribute(rule.attr, cn); n.attr += 1; }',
'      }',
'    }',
'  }',
'  for (const el of root.querySelectorAll("[aria-label]")) {',
])
NEW3 = nl.join([
'        if (cn && cn !== raw) { el.setAttribute(rule.attr, cn); n.attr += 1; }',
'      }',
'    }',
'  }',
'  };',
'  apply(STRUCTURAL, [root]);',
'  apply(SCOPED, scopes);',
'',
'  // 创建页的加减按钮同样只在 `.crucible` 作用域内动：`Add one …` 也是通用说法。',
'  for (const el of scopes.flatMap((r) => Array.from(r.querySelectorAll("[aria-label]")))) {',
])
assert s.count(OLD3) == 1, s.count(OLD3)
s = s.replace(OLD3, NEW3)

s = s.replace('    rules: RULES.length,', '    rules: STRUCTURAL.length + SCOPED.length,')
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已分档')
