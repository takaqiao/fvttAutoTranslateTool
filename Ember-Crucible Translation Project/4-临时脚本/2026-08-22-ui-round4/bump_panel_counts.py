# -*- coding: utf-8 -*-
"""面板 D 档的登记值跟着本轮新增的 DIALOG_UI 两条往上走。

方向说明（这条断言的设计）：覆盖侧按「不得低于」记，上游长东西、表里加键这些数只会涨；
掉下来才说明表被砍了或语料没抓着。所以**往上更新是正常跟进，往下调才是作弊**。
本轮加了 `Surface` / `Pathways` 两条到 DIALOG_UI：
  registeredRaw 2389 → 2391（+2，两条都是新登记行）
  registeredDistinct 1773 → 1774（+1，`Pathways` 作为 distinct 串在 ARRANGEMENTS 里已存在）
  rawChecked 2203 → 2205（+2）
  checkedDistinct 1609 → 1610（+1，同上）
⚠ miss 侧一条没动（missDistinct 4 / rawMiss 7），uncheckedRaw / uncheckedDistinct 也一条没动 ——
  也就是说这两条**在上游语料里逐字面量查得到**（76001 / 76010 / 76021），没往天花板上顶。
"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(P, encoding='utf-8'))
NEW = {"checkedDistinct": 1610, "rawChecked": 2205, "registeredRaw": 2391, "registeredDistinct": 1774}
OLD = {"checkedDistinct": 1609, "rawChecked": 2203, "registeredRaw": 2389, "registeredDistinct": 1773}
hit = 0
for a in d['assertions']:
    if a.get('kind') != 'panel_liveness':
        continue
    hit += 1
    for k, v in NEW.items():
        assert a['min'][k] == OLD[k], (k, a['min'][k], OLD[k])
        assert v > OLD[k], '只许往上'
        a['min'][k] = v
assert hit == 1, hit
d['meta']['updated'] = "2026-08-22（第三十六轮 / UI 补漏第四轮：+2 判据、面板 D 档 4 个覆盖侧登记值跟涨）"
io.open(P, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print('已更新', NEW)
