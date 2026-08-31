# -*- coding: utf-8 -*-
"""给 ember 侧「crucible 掷骰卡上的 boon/bane 来源名」这条新机制上判据。"""
import io, os, sys, json, importlib.util
sys.stdout.reconfigure(encoding='utf-8')
RJ = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(RJ, encoding='utf-8'))
assert 'R-ember-boon-labels' not in {a['id'] for a in d['assertions']}
NEW = {
 "id": "R-ember-boon-labels",
 "title": "ember 塞进 crucible 掷骰卡的加值/减值来源名必须按**结构**认，不能靠 `flags.ember` 旗标",
 "decision": "2026-08-22（UI 补漏第七轮）",
 "why": (
   "ember 往 `usage.boons` / `usage.banes` 里塞了 7 条来源名（`ember.mjs:140804` 屠龙 / "
   "`:141167` 余烬之火花 / `:142148-142149` 混沌折射 …），渲染它们的是 **crucible 的** "
   "`templates/dice/partials/standard-check-details.hbs:6/18`（`{{localize boon.label}}`）。"
   "｜⚠ 关键：那是 **crucible 自己的掷骰卡，不带 `flags.ember`** —— 而本文件的 "
   "`renderChatMessageHTML` 钩子原本第一句就是 `if (!msg?.flags?.ember) return;`，"
   "**一条都接不住**。所以这条规则必须跑在旗标闸**之前**，并按结构选择器 "
   "`.boon-details .boon > .label` 认归属（那是 crucible 模板独有的形状）。"
   "把它挪到旗标闸之后、或改用旗标判归属，这 7 条会**静默失效**：屏幕上是英文，控制台一声不响。"
   "｜⚠ 与 `crucible-cn 0.9.18` 的同名表**刻意不重叠**：那边 11 条是 crucible 自己的"
   "（Special / Elite / Boss …），这边 7 条只有 ember 加的。两个模块在同一棵 DOM 上各跑各的，"
   "查不到的原样返回，所以互不影响 —— 但也意味着**谁都不能替对方兜底**，两边各自要有判据。"
   "｜⚠ 判据边界：只钉「规则在、且在旗标闸之前」，不判译名对不对（那 7 条译名取的是 "
   "glossary_ec 已定的中文段）。"
 ),
 "kind": "source_literal",
 "min_files": 1,
 "min_checks": 3,
 "files": [{"repo": "ember", "path": "scripts/ember-hardcoded-cn.mjs"}],
 "require": [
   "const BOON_BANE_LABELS = {",
   '      for (const el of chatRoot.querySelectorAll?.(".boon-details .boon > .label, .bane-details .bane > .label") ?? []) {',
   "      if (!msg?.flags?.ember) return;",
 ],
}
i = next(k for k, a in enumerate(d['assertions']) if a['id'] == 'R-crucible-hardcoded-scope')
d['assertions'].insert(i + 1, NEW)
d['meta']['updated'] = "2026-08-22（UI 补漏第七轮：ember 玩家可见硬编码 40 条 + 掷骰卡来源名判据）"
io.open(RJ, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print('断言总数 =', len(d['assertions']))

PY = os.path.join('3-常用脚本', 'qa', 'assert_resolutions.py')
spec = importlib.util.spec_from_file_location('ar', os.path.abspath(PY))
M = importlib.util.module_from_spec(spec); sys.modules['ar'] = M; spec.loader.exec_module(M)
der = M._derive_payload_floors([NEW])['R-ember-boon-labels']
print(' 现推地板 =', der)
s = io.open(PY, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60])
    s = s.replace(a, b)
fmt = lambda sp: '{' + ', '.join('"%s": ("%s", %r)' % (k, v[0], v[1]) for k, v in sorted(sp.items())) + '}'
rep1("    'R-crucible-hardcoded-scope': 'source_literal',",
     "    'R-crucible-hardcoded-scope': 'source_literal'," + nl + "    'R-ember-boon-labels': 'source_literal',")
line = [l for l in s.split(nl) if l.startswith('    "R-crucible-hardcoded-scope": {')]
assert len(line) == 1
rep1(line[0], line[0] + nl + '    "R-ember-boon-labels": %s,' % fmt(der))
rep1('    "R-crucible-hardcoded-scope": 5,',
     '    "R-crucible-hardcoded-scope": 5,' + nl + '    "R-ember-boon-labels": 3,')
i2 = s.index('RULESET_SHAPE = {'); j2 = s.index(nl + '}' + nl, i2)
blk = s[i2:j2]
blk = blk.replace('"source_literal": 6,', '"source_literal": 7,')
blk = blk.replace('    "min_assertions": 73,', '    "min_assertions": 74,')
s = s[:i2] + blk + s[j2:]
io.open(PY, 'w', encoding='utf-8', newline='').write(s)
print('登记表已同步')
