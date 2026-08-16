# -*- coding: utf-8 -*-
"""重刷 PAYLOAD_FLOORS 的表体（按锚点定位，纯 str 操作，不走正则）。

⚠ 自证：锚点必须各出现一次；替换后把新块换回旧块必须逐字节还原。
"""
import os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TARGET = os.path.join(ROOT, "3-常用脚本", "qa", "assert_resolutions.py")

payload_src = open(os.path.join(HERE, "gen_blocks.txt"),
                   encoding="utf-8").read().split("\n@@@\n")[0]
assert "\\" not in payload_src, "生成物里有反斜杠，abort"

HEAD = "PAYLOAD_FLOORS = {\n"
TAIL = "\n}\n\n\n# ============ 「这次判了几条规矩」的地板"

src = open(TARGET, encoding="utf-8").read()
assert src.count(HEAD) == 1 and src.count(TAIL) == 1, "锚点不是各一处，abort"
i = src.index(HEAD) + len(HEAD)
j = src.index(TAIL)
old_body = src[i:j]
assert old_body.startswith('    "R-warlock-sorcerer"'), "切出来的表体开头不对，abort"
assert old_body.rstrip().endswith('},'), "切出来的表体结尾不对，abort"

new = src[:i] + payload_src + src[j:]
assert new.replace(payload_src, old_body) == src, "换回去不等于原文，abort"
open(TARGET, "w", encoding="utf-8", newline="").write(new)
print(f"重刷 ok：表体 {len(old_body)} B → {len(payload_src)} B")
