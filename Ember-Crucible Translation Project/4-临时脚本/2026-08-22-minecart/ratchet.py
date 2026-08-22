import re, sys
sys.stdout.reconfigure(encoding='utf-8')
NEW = {"checkedDistinct":940, "rawChecked":1522, "registeredRaw":1708,
       "registeredDistinct":1104, "uncheckedDistinct":164}
p = "5-其他内容/RESOLUTIONS.assertions.json"
s = open(p, encoding='utf-8').read(); n = 0
for k, v in NEW.items():
    s2 = re.sub('("' + k + '":\s*)\d+', lambda m: m.group(1) + str(v), s, count=1)
    if s2 != s: n += 1; s = s2
open(p, 'w', encoding='utf-8', newline='').write(s)
print("规则侧改", n, "个")
q = "3-常用脚本/qa/assert_resolutions.py"
t = open(q, encoding='utf-8').read(); m = 0
for k, v in NEW.items():
    for pre in ("min", "max"):
        pat = '("' + pre + '\.' + k + '": \("eq", )\d+'
        t2 = re.sub(pat, lambda mm: mm.group(1) + str(v), t, count=1)
        if t2 != t: m += 1; t = t2
open(q, 'w', encoding='utf-8', newline='').write(t)
print("PAYLOAD_FLOORS 改", m, "处")
