# -*- coding: utf-8 -*-
"""面板 D 档的四个覆盖侧登记值跟着装备族 611 条往上走（两层都改）。

方向：覆盖侧按「不得低于」记，表里加键这些数只会涨 —— 往上跟进是本设计要求的动作，
往下调才是作弊。miss 侧（missDistinct 4 / rawMiss 7）与没核侧（186 / 164）**一条没动**，
不需要碰天花板 —— 611 个键上表前逐个查过上游字面量，查不到的 0 个。
"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
NEW = {"checkedDistinct": 2217, "rawChecked": 2816, "registeredRaw": 3002, "registeredDistinct": 2381}
OLD = {"checkedDistinct": 1610, "rawChecked": 2205, "registeredRaw": 2391, "registeredDistinct": 1774}

# 第一层：规则集
P = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(P, encoding='utf-8'))
hit = 0
for a in d['assertions']:
    if a.get('kind') != 'panel_liveness':
        continue
    hit += 1
    for k, v in NEW.items():
        assert a['min'][k] == OLD[k], (k, a['min'][k], OLD[k])
        assert v > OLD[k], '只许往上'
        a['min'][k] = v
assert hit == 1
d['meta']['updated'] = ("2026-08-22（第三十六轮 / UI 补漏第四轮：+3 判据、"
                        "装备族部件名 611 条、面板 D 档 4 个覆盖侧登记值跟涨）")
io.open(P, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")

# 第二层：PAYLOAD_FLOORS 的钉死值
PY_ = '3-常用脚本/qa/assert_resolutions.py'
s = io.open(PY_, encoding='utf-8', newline='').read()
for k in NEW:
    old = '"min.%s": ("eq", %d)' % (k, OLD[k])
    new = '"min.%s": ("eq", %d)' % (k, NEW[k])
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)
io.open(PY_, 'w', encoding='utf-8', newline='').write(s)
print('两层都已跟涨：', NEW)
