# -*- coding: utf-8 -*-
"""把生成好的两块表塞进 assert_resolutions.py 的占位符。

⚠ 硬约束 3：**纯 str.replace，不走正则**，且生成物已断言无反斜杠。
⚠ 插入后当场自证：把插入的块原样再切掉，必须逐字节还原成插入前的内容。
"""
import os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TARGET = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")

payload_src, units_src = open(os.path.join(HERE, "gen_blocks.txt"),
                              encoding="utf-8").read().split("\n@@@\n")
assert "\\" not in payload_src and "\\" not in units_src, "生成物里有反斜杠，abort"

M1 = '    "__PAYLOAD_FLOORS_BODY__": None,\n'
M2 = '    "__JUDGED_UNITS_BODY__": None,\n'

src = open(TARGET, encoding="utf-8").read()
assert src.count(M1) == 1 and src.count(M2) == 1, "占位符不是各一处，abort"

new = src.replace(M1, payload_src + "\n").replace(M2, units_src + "\n")
# 自证：切回去必须逐字节还原
back = new.replace(payload_src + "\n", M1).replace(units_src + "\n", M2)
assert back == src, "切回去不等于原文 —— 插入动作不干净，abort"

open(TARGET, "w", encoding="utf-8", newline="").write(new)
print(f"插入 ok：{len(src)} B → {len(new)} B")
