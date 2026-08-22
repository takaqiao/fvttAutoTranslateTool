import sys
sys.stdout.reconfigure(encoding='utf-8')
p = "1-Ember汉化插件/module.json"; s = open(p, encoding='utf-8').read()
s2 = s.replace('"version": "1.1.27"', '"version": "1.1.28"', 1)
s2 = s2.replace("/download/v1.1.27/module.zip", "/download/v1.1.28/module.zip")
s2 = s2.replace("/releases/tag/v1.1.27", "/releases/tag/v1.1.28")
assert s2 != s
open(p, 'w', encoding='utf-8', newline='').write(s2); print("module.json -> 1.1.28")

q = "PROJECT.md"; t = open(q, encoding='utf-8').read(); o = t
t = t.replace("`ember_cn_unofficial v1.1.27`。**", "`ember_cn_unofficial v1.1.28`。**", 1)
t = t.replace("| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.27**",
              "| ember_cn_unofficial | 汉化模块（本项目） | **v1.1.28**", 1)
row = ("| `v1.1.28` | 08-22 | **指示物制作器图层名 49 键**（项目所有者点名的 `Cheeks`/`Hand Left`/`Hand Right`/`Tail`/`Ears` 全在内）。"
 "<br>成因不是漏译，是上一轮抽取器**漏了一整种形态** —— `Object.assign(cloneLayer(...), {label:\"…\"})`，"
 "只抓了裸 `label:` 那一种。作用域实测干净（泄漏 0 / 近似串被吃 0），三张高危表一条没动。"
 "<br>⚠ **本轮查清、但未修的三件（都不是漏译）**："
 "<br>　① **token 上的夹击标签与「强制」仍是英文** —— crucible-cn 早就译好了，是 `foundry_chn/cn.json` 顶层的**裸字符串** "
 "`\"TOKEN\":\"指示物\"` / `\"WARNING\":\"警告\"` 在 `mergeObject` 时**整块盖掉**了这两个命名空间，"
 "连 Foundry 核心自己的 `WARNING.*` 中文一起打掉，**受害 42 条**。与第三十四轮 A 的 `EMBER.*` 事故**同型**，只是肇事者是第三方模块。"
 "<br>　② **地图上的地名不是翻译缺口** —— 那是 Note 的 text，唯一 295 条，我们**覆盖 295/295**（regions 310/310、levels 266/266 同样满覆盖）。"
 "<br>　③ **换新版汉化不会改已导入世界的内容** —— `babele/script/foundry/wrapper.js:50` 有 `if (!pack) return result;`，"
 "**世界文档没有 pack，永远不进翻译路径**。唯一通路是重新导入冒险，而 `Adventure#importContent` 用 "
 "`updateDocuments(data,{diff:false,recursive:false})` **整份替换**，世界侧改动会丢；可用导入器的 `importFields` 只勾场景缩小爆炸半径。"
 "<br>主闸 68/0/0 · `--selftest` 357/357。 |\n\n")
old = "> ⚠⚠ **第二十四～二十六轮的产出尚未发版**"
assert old in t
t = t.replace(old, row + old, 1)
assert t != o
open(q, 'w', encoding='utf-8', newline='').write(t); print("PROJECT.md 抬头/矩阵/发版一览已更新")
