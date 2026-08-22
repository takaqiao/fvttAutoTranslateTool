# -*- coding: utf-8 -*-
"""面板 D 档登记值跟涨（两层都改）。本轮新增一张表 ⇒ tableRows / tablesFedIn 也 +1。"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
NEW = {"checkedDistinct": 2261, "rawChecked": 2861, "registeredRaw": 3047,
       "registeredDistinct": 2425, "tableRows": 42, "tablesFedIn": 42}
OLD = {"checkedDistinct": 2217, "rawChecked": 2816, "registeredRaw": 3002,
       "registeredDistinct": 2381, "tableRows": 41, "tablesFedIn": 41}
P = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(P, encoding='utf-8'))
hit = 0
for a in d['assertions']:
    if a.get('kind') != 'panel_liveness': continue
    hit += 1
    for k, v in NEW.items():
        assert a['min'][k] == OLD[k], (k, a['min'][k], OLD[k])
        assert v > OLD[k]
        a['min'][k] = v
assert hit == 1
d['meta']['updated'] = "2026-08-22（UI 补漏第七轮：ember 玩家可见硬编码 40 条、新表进面板、+1 判据）"
io.open(P, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
PY_ = '3-常用脚本/qa/assert_resolutions.py'
s = io.open(PY_, encoding='utf-8', newline='').read()
for k in NEW:
    old = '"min.%s": ("eq", %d)' % (k, OLD[k]); new = '"min.%s": ("eq", %d)' % (k, NEW[k])
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)
io.open(PY_, 'w', encoding='utf-8', newline='').write(s)
print('两层已跟涨：', NEW)
