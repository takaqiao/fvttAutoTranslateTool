# -*- coding: utf-8 -*-
"""把两条新断言登记进四张表：REGISTERED_ASSERTIONS / PAYLOAD_FLOORS / JUDGED_UNITS / RULESET_SHAPE。

地板不手写：直接调判据自己的 `_derive_payload_floors` 现推 —— 手写等于给「登记表和推导器
对不上」留一处，而主闸的前置 C 正是拿现推值逐道比登记值。
"""
import io, os, sys, importlib.util
sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.abspath('.')
PY = os.path.join(ROOT, '3-常用脚本', 'qa', 'assert_resolutions.py')
NEW_IDS = ['R-lang-reclaim-wired', 'R-lang-squat-panel']
UNITS = {'R-lang-reclaim-wired': 3, 'R-lang-squat-panel': 8}

spec = importlib.util.spec_from_file_location('ar', PY)
M = importlib.util.module_from_spec(spec)
sys.modules['ar'] = M
spec.loader.exec_module(M)

import json
rules = json.load(io.open(os.path.join(ROOT, '5-其他内容', 'RESOLUTIONS.assertions.json'), encoding='utf-8'))
byid = {a['id']: a for a in rules['assertions']}
derived = M._derive_payload_floors([byid[i] for i in NEW_IDS])
for i in NEW_IDS:
    print(' 现推地板', i, '=', derived[i])

s = io.open(PY, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

def ins_after(text, anchor, addition):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:60])
    return text.replace(anchor, anchor + addition)

# ① REGISTERED_ASSERTIONS
anchor = "    'R-selfcheck-d-section-name': 'source_literal',"
add = nl + nl.join([
  "    # 第三十六轮（UI 补漏第四轮）：钉住「译文抢回器」这套**全程静默**的机制。",
  "    # 静默到什么程度：修好了是中文、没被顶也是中文、抢回器根本没跑则是英文**且不报错** ——",
  "    # 维护者只要没装 foundry_chn 就永远复现不了。本轮之前 `lang-reclaim` 在整个判据侧出现 0 次。",
  "    'R-lang-reclaim-wired': 'source_literal',",
  "    'R-lang-squat-panel': 'source_literal',",
])
s = ins_after(s, anchor, add)

# ② PAYLOAD_FLOORS
def fmt(spec_dict):
    inner = ', '.join('"%s": ("%s", %r)' % (k, v[0], v[1]) for k, v in sorted(spec_dict.items()))
    return '{' + inner + '}'
pf_anchor = [ln for ln in s.split(nl) if ln.startswith('    "R-selfcheck-d-section-name": {')]
assert len(pf_anchor) == 1, pf_anchor
add = nl + nl.join('    "%s": %s,' % (i, fmt(derived[i])) for i in NEW_IDS)
s = ins_after(s, pf_anchor[0] + ',' if not pf_anchor[0].rstrip().endswith(',') else pf_anchor[0], add)

# ③ JUDGED_UNITS
ju_anchor = [ln for ln in s.split(nl) if ln.startswith('    "R-selfcheck-d-section-name": ') and ln.rstrip().endswith(',') and '{' not in ln]
assert len(ju_anchor) == 1, ju_anchor
add = nl + nl.join([
  '    # source_literal 的规矩数 = 文件的**仓名**去重后 × (require + forbid_re) 条数：',
  '    # `_unit()` 的名字是 `require:{repo}:{字面量}`，两个文件同仓时会合成一条。',
  '    # R-lang-reclaim-wired：仓只有 crucible ⇒ 1×(1 require + 2 forbid) = 3',
  '    # R-lang-squat-panel  ：ember + crucible 两仓 ⇒ 2×4 require = 8',
] + ['    "%s": %d,' % (i, UNITS[i]) for i in NEW_IDS])
s = ins_after(s, ju_anchor[0], add)

# ④ RULESET_SHAPE：min_assertions 68→70、source_literal 1→3
i = s.index('RULESET_SHAPE = {')
j = s.index(nl + '}' + nl, i)
blk = s[i:j]
assert blk.count('"source_literal": 1,') == 1
blk = blk.replace('"source_literal": 1,', '"source_literal": 3,')
old = '    "min_assertions": 68,' + nl + '    "min_kinds": 24,'
assert blk.count(old) == 1
new = nl.join([
  '    # 第三十六轮（UI 补漏第四轮）：68 → 70，kind 数不变（两条都复用既有的 source_literal）。',
  '    # 同样走 §0.1 收官后「它会咬到诚实的维护者」那一支例外 —— 这次会咬到的动作是',
  '    # 「顺手清理一个看起来没人用的文件」：lang-reclaim.js 不被调就等于白进包，而没人会响。',
  '    "min_assertions": 70,',
  '    "min_kinds": 24,',
])
blk = blk.replace(old, new)
s = s[:i] + blk + s[j:]

io.open(PY, 'w', encoding='utf-8', newline='').write(s)
print('已登记', NEW_IDS)
