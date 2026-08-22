import sys
sys.stdout.reconfigure(encoding='utf-8')
p = "PROJECT.md"
s = open(p, encoding='utf-8').read(); o = s
old = "指示物制作器的**部件名**是 `getLayerChoicesV2()`（ember.mjs:64457）在运行时把部件 id 拆词拼出来的（`BeardWizard` → `Beard Wizard`），四千条量级、没有字面量可查。槽位名与图层名已中文、部件名仍英文；"
new = ("指示物制作器的**部件名**是 `getLayerChoicesV2()`（ember.mjs:64457）在运行时把部件 id 拆词拼出来的"
       "（`BeardWizard` → `Beard Wizard`）。<br>⚠⚠ **本行原写「四千条量级、没有字面量可查」，第三十四轮 B 实测推翻**："
       "拆词规则是确定的、id 可枚举，**能算出来**；真实唯一显示名 **≥1456**（不是四千，也不是一度写下的 1356 —— "
       "`makeLegPoseParts()` 等运行时拼装的 id 字面量扫描天然抓不全，**须以随包图集为准**）。"
       "「做不到」这个结论让这块白搁了一轮，**判「做不到」之前先问一句「是不是只是还没算」**。<br>"
       "⚠ 同行原写「已知仍会看到英文的两处」也归错了半边：当时 `Cheeks`/`Hand Left`/`Hand Right`/`Tail`/`Ears` "
       "是**图层名**（`Object.assign(cloneLayer(...), {label:\"…\"})` 这一形态被抽取器漏掉），不是部件名；"
       "已在 v1.1.28 补上。槽位名与图层名已中文、**部件名仍英文**；")
if old in s:
    s = s.replace(old, new, 1); print("已订正「四千条量级」那句")
else:
    print("⚠ 没匹配到原句，改用宽松定位")
    import re
    m = re.search(r'四千条量级、没有字面量可查', s)
    if m:
        s = s[:m.start()] + ("⚠⚠ **原写「四千条量级、没有字面量可查」，第三十四轮 B 实测推翻**："
             "拆词规则确定、id 可枚举，能算出来；真实唯一显示名 ≥1456，须以随包图集为准") + s[m.end():]
        print("  已按宽松定位订正")
if s != o:
    open(p, 'w', encoding='utf-8', newline='').write(s)
