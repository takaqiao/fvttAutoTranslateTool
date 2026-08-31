# -*- coding: utf-8 -*-
"""第二层：PAYLOAD_FLOORS 里那份「钉死值」跟着规则集一起走。

两层是**故意的**：规则集里的 min 是判据用的阈值，PAYLOAD_FLOORS 里的 ("eq", N) 是
「这个阈值当初被记成多少」的存根，住在**另一个文件**里。想把阈值调松就必须同时改两处、
两处都在 diff 里 —— §3.7.2「强度参数与判据同层 = 没有强度」。
往上跟涨同样要改两处，这不是绕过，是这套设计要求的动作。
"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '3-常用脚本/qa/assert_resolutions.py'
s = io.open(P, encoding='utf-8', newline='').read()
PAIRS = [
    ('"min.checkedDistinct": ("eq", 1609)', '"min.checkedDistinct": ("eq", 1610)'),
    ('"min.rawChecked": ("eq", 2203)', '"min.rawChecked": ("eq", 2205)'),
    ('"min.registeredDistinct": ("eq", 1773)', '"min.registeredDistinct": ("eq", 1774)'),
    ('"min.registeredRaw": ("eq", 2389)', '"min.registeredRaw": ("eq", 2391)'),
]
for old, new in PAIRS:
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已跟涨', len(PAIRS), '道')
