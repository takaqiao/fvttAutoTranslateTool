import sys
sys.stdout.reconfigure(encoding='utf-8')
p = "2-Crucible汉化插件/babele-register.js"
s = open(p, encoding='utf-8', newline='').read(); o = s
OLD = "所以补丁挂 `setup`（i18nInit 早于 init，太早）。"
NEW = ("所以补丁挂 `setup`。\n"
       " * ⚠ 本行原写「i18nInit 早于 init，太早」——**理由是反的**，2026-08-22 实测 v14.366 的真实顺序是\n"
       " *   `init`(game.mjs:652) → `i18nInit`(:663 内) → `setup`(:740) → `ready`(:779)，i18nInit **晚于** init。\n"
       " *   结论（挂 setup）仍然对，坏的只是理由。本项目靠注释传裁决，理由写反比没写更危险。")
assert OLD in s, "锚点没找到"
s = s.replace(OLD, NEW, 1)
open(p, 'w', encoding='utf-8', newline='').write(s)
print("已订正 babele-register.js 里反了的钩子顺序注释")
