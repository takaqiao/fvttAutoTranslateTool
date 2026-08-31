import sys
sys.stdout.reconfigure(encoding='utf-8')
for p, old, new, told, tnew in [
    ("1-Ember汉化插件/module.json", "1.1.28", "1.1.29", "v1.1.28", "v1.1.29"),
    ("2-Crucible汉化插件/module.json", "0.9.14", "0.9.15", "0.9.14", "0.9.15")]:
    s = open(p, encoding='utf-8').read()
    s2 = s.replace('"version": "' + old + '"', '"version": "' + new + '"', 1)
    s2 = s2.replace("/download/" + told + "/module.zip", "/download/" + tnew + "/module.zip")
    s2 = s2.replace("/releases/tag/" + told, "/releases/tag/" + tnew)
    assert s2 != s, p
    open(p, 'w', encoding='utf-8', newline='').write(s2)
    print(p.split('/')[0], old, "->", new)

q = "PROJECT.md"; t = open(q, encoding='utf-8').read(); o = t
t = t.replace("当前已发布 `crucible-cn 0.9.14` / `ember_cn_unofficial v1.1.28`。**",
              "当前已发布 `crucible-cn 0.9.15` / `ember_cn_unofficial v1.1.29`。**", 1)
t = t.replace("| crucible-cn | 汉化模块（本项目） | **0.9.14**", "| crucible-cn | 汉化模块（本项目） | **0.9.15**", 1)
t = t.replace("| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.28**",
              "| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.29**", 1)
row = ("| `0.9.15` / `v1.1.29` | 08-22 | **抢回被核心中文包顶掉的 42 条 + 互动面板/导入器/部件名**。"
 "<br>· 🔥 **crucible-cn 新增 `lang-reclaim.js`** —— `foundry_chn/cn.json` 顶层的裸字符串 `\"TOKEN\":\"指示物\"` / "
 "`\"WARNING\":\"警告\"` 在 `mergeObject` 时**整块盖掉**这两个命名空间（只在两边都是对象时才递归），"
 "打掉我们 **42 条**已译内容（夹击标签 5 · 移动方式与「强制」27 · 报错提示 10）。"
 "挂 `i18nInit` 就地抢回。**幂等**：没被顶掉时是 no-op（离线复刻实测写入 0 次、逐键快照零差异）。"
 "<br>　⚠ 实现上两处是被实测逼出来的、不是过度设计：① 用**同步 XHR** 而非 top-level await —— 实测 TLA "
 "**不推迟 DOMContentLoaded**，而 Foundry 正是在那里 `await game.initialize()`，必然赶不上；"
 "② 用 `enumerable:false` —— 否则 `#hotReloadJSON` 的 `mergeObject` 会因 `expanded.TOKEN` 已是字符串而抛 `TypeError`。"
 "<br>　⚠ **别读成「修好了核心的 WARNING/TOKEN」** —— 那 373 条核心叶仍走英文 fallback，本版没碰；我们只抢自己的键。"
 "<br>· **ember 互动面板 7 条 + 冒险导入器 4 条 + 部件显示名 763 条**（真实全集 **1454** 条，`templateLayer.parts` 口径，"
 "不是图集的 1535；**仍缺 687**，基本在装备族）。顺带修好两处既有错译：部件行 `Heavy`→粗壮、`Lithe`→柔韧。"
 "<br>⚠ **明确不做（本轮裁决）**：把 `Reveal Grayling` 那句宏提示塞进 `NOTIFICATIONS` —— 实测会让面板 "
 "`missDistinct` 4→5 顶破 `max` 天花板（那句话住在 LevelDB 的宏 `command` 里，不在面板语料内，按 literal 核必然报 miss）。"
 "抬天花板是本项目定性最重的动作，不为一条提示做。"
 "<br>⚠ **仍会看到英文**：GM 叠加层 155 条（PIXI `PreciseText`，DOM 遍历结构上够不到）· 传送框下拉的 "
 "`Surface`/`Pathways` 两个分组名（`<optgroup label>`，本轮未登记）· 装备族部件名 687 条。"
 "<br>主闸 68/0/0 · `--selftest` 357/357。 |\n\n")
old = "> ⚠⚠ **第二十四～二十六轮的产出尚未发版**"
assert old in t
t = t.replace(old, row + old, 1)
assert t != o
open(q, 'w', encoding='utf-8', newline='').write(t)
print("PROJECT.md 已更新")
