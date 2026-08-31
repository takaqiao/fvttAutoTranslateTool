# -*- coding: utf-8 -*-
"""调查 bestiary 中可疑 1 处 hit 的具体位置"""
import json

p = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\pf2e.fists-of-the-ruby-phoenix-bestiary.json"
with open(p, encoding="utf-8") as f:
    c = f.read()

for target in ["拉吉娜", "娜迦毒液", "锚定射击", "巨人摔角术", "君（Jun", "克兰基斯", "努莫里兹", "波尼玛", "拉贾娜"]:
    print(f"\n=== {target} ===")
    idx = 0
    n = 0
    while idx < len(c):
        i = c.find(target, idx)
        if i < 0: break
        n += 1
        before = c[max(0,i-80):i].replace("\n","\\n")
        after = c[i+len(target):i+len(target)+80].replace("\n","\\n")
        print(f"  [{n}] ...{before}【{target}】{after}...")
        idx = i + 1
