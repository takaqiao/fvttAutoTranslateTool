# -*- coding: utf-8 -*-
"""拆条之后重登记：REGISTERED_ASSERTIONS / PAYLOAD_FLOORS / JUDGED_UNITS / RULESET_SHAPE。"""
import io, os, sys, json, importlib.util
sys.stdout.reconfigure(encoding='utf-8')
PY = os.path.join('3-常用脚本', 'qa', 'assert_resolutions.py')
spec = importlib.util.spec_from_file_location('ar', os.path.abspath(PY))
M = importlib.util.module_from_spec(spec); sys.modules['ar'] = M; spec.loader.exec_module(M)
rules = json.load(io.open('5-其他内容/RESOLUTIONS.assertions.json', encoding='utf-8'))
byid = {a['id']: a for a in rules['assertions']}
NEW = 'R-lang-reclaim-mechanism'
derived = M._derive_payload_floors([byid['R-lang-reclaim-wired'], byid[NEW]])
print(' 现推地板 wired    =', derived['R-lang-reclaim-wired'])
print(' 现推地板 mechanism=', derived[NEW])

s = io.open(PY, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def rep1(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)

# ① REGISTERED_ASSERTIONS：在 wired 后面补 mechanism
rep1("    'R-lang-reclaim-wired': 'source_literal',",
     "    'R-lang-reclaim-wired': 'source_literal'," + nl +
     "    'R-lang-reclaim-mechanism': 'source_literal',")

# ② PAYLOAD_FLOORS：wired 的载荷变了（require 2 条、min_checks 4），mechanism 新增
def fmt(sp):
    return '{' + ', '.join('"%s": ("%s", %r)' % (k, v[0], v[1]) for k, v in sorted(sp.items())) + '}'
oldline = [l for l in s.split(nl) if l.startswith('    "R-lang-reclaim-wired": {')]
assert len(oldline) == 1, oldline
rep1(oldline[0],
     '    "R-lang-reclaim-wired": %s,' % fmt(derived['R-lang-reclaim-wired']) + nl +
     '    "%s": %s,' % (NEW, fmt(derived[NEW])))

# ③ JUDGED_UNITS：仓只有 crucible ⇒ 1 仓 ×（require + forbid_re）
rep1('    "R-lang-reclaim-wired": 3,',
     '    "R-lang-reclaim-wired": 4,' + nl +      # 1 仓 ×（2 require + 2 forbid）
     '    "R-lang-reclaim-mechanism": 4,')        # 1 仓 × 4 require

# ④ RULESET_SHAPE
i = s.index('RULESET_SHAPE = {'); j = s.index(nl + '}' + nl, i)
blk = s[i:j]
assert blk.count('"source_literal": 3,') == 1
blk = blk.replace('"source_literal": 3,', '"source_literal": 4,')
assert blk.count('    "min_assertions": 70,') == 1
blk = blk.replace('    "min_assertions": 70,', '    "min_assertions": 71,')
blk = blk.replace('    # 第三十六轮（UI 补漏第四轮）：68 → 70，kind 数不变（两条都复用既有的 source_literal）。',
                  '    # 第三十六轮（UI 补漏第四轮）：68 → 71，kind 数不变（三条都复用既有的 source_literal）。')
s = s[:i] + blk + s[j:]
io.open(PY, 'w', encoding='utf-8', newline='').write(s)
print('已重登记')
