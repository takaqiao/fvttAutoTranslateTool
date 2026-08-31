import sys
sys.stdout.reconfigure(encoding='utf-8')
p = "PROJECT.md"
s = open(p, encoding='utf-8').read(); o = s
OLD = "① 指示物制作器的**部件名**是 `getLayerChoicesV2()` 运行时从部件 id 拆词拼出来的（四千条量级、无字面量可查），槽位名与图层名已中文、部件名仍英文；"
NEW = ("① 指示物制作器的**部件名**是 `getLayerChoicesV2()` 运行时从部件 id 拆词拼出来的，槽位名与图层名已中文、部件名仍英文"
       "<br>　⚠⚠ **本格原写「四千条量级、无字面量可查」，第三十四轮 B 实测推翻**：拆词规则是确定的、id 可枚举，**能算出来**；"
       "真实唯一显示名 **≥1456**（既不是四千，也不是一度写下的 1356 —— `makeLegPoseParts()` 等运行时拼装出来的 id，字面量扫描天然抓不全，**须以随包图集为准**）。"
       "「做不到」这个结论让这块白搁了一轮 —— **判「做不到」之前先问一句「是不是只是还没算」**。"
       "<br>　⚠ 本格「已看到英文的两处」还归错了半边：`Cheeks`/`Hand Left`/`Hand Right`/`Tail`/`Ears` 当时是**图层名**不是部件名，"
       "成因是抽取器漏了 `Object.assign(cloneLayer(...), {label:\"…\"})` 这一整种形态，已在 v1.1.28 补上；")
assert OLD in s, "锚点没找到"
s = s.replace(OLD, NEW, 1)
open(p, 'w', encoding='utf-8', newline='').write(s)
print("已订正 v1.1.27 行里被推翻的两处说法")
