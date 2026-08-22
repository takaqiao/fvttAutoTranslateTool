# -*- coding: utf-8 -*-
"""
C 块（GM 叠加层三处）的**通道**判定与全集枚举。

结论（逐条回 ember 0.6.1 源码核过）：**三处都不是 DOM，是 PIXI 画布文字**，
本项目的 translateNode / translateText 那一套（按文本节点查表）**结构上够不到**。
  · 生成区域标签   ember.mjs:78689  `new PreciseText(spawnConfig.label ?? location, style)`
                   挂进 `s = new PIXI.Container()`（#visualizeSpawn），由 spawns 工具开关画上去
  · Show Tracks    ember.mjs:129034 `new PreciseText(node.label, destinationStyle)`
                   挂进 `canvas.controls.addChildAt(c, 0)`（#drawTrackGraph）
    同族还有       ember.mjs:70332  `new PreciseText(backtick-${this.event.label} ${probLabel} backtick, style)`
                   六边格事件铭牌，`probLabel` 里那句 `[Complete]` 是字面量
  · Vista 调试     ember.mjs:68157/68162 `new PreciseText("Daylight Color"|"Darkness Color", style)`
                   在 `#drawDebug()` 里，挂进 `canvas.interface`（:68179）

⇒ 这不是「漏译」，是**少一条通道**。本项目已有的唯一一条 PIXI 通道是
  `createScrollingText` 的**类原型包裹**（ember-hardcoded-cn.mjs 的 SCROLLING_TEXT），
  那是包一个 Ember/核心的**具名方法**；要接住上面这三处得包 `PreciseText` 本身 ——
  那是**全局**面（核心自己的铭牌、别的模块的画布文字全要过一遍），爆炸半径与
  NOTIFICATION_PATTERNS 同级甚至更大。⇒ 判为**新增面，要主控裁**，本轮不做。

本脚本把「真要做的话得译多少」数清楚，免得下一轮再从零查一遍。
"""
import io, json, os, re

EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"
OUT = os.path.dirname(os.path.abspath(__file__))
s = io.open(EMBER, encoding="utf-8").read()

def block_labels(anchor):
    got = set()
    for m in re.finditer(anchor, s):
        i = m.end() - 1
        d, j = 0, i
        while j < len(s):
            if s[j] == "{": d += 1
            elif s[j] == "}":
                d -= 1
                if d == 0: break
            j += 1
        got |= {x.group(1) for x in re.finditer(r'label:\s*"([^"]+)"', s[i:j])}
    return sorted(got)

spawns = block_labels(r"\n  spawns: \{")
tracks = block_labels(r"\n    nodes: \{")
fixed = ["Daylight Color", "Darkness Color", "[Complete]"]

# 前置自证：三处上屏点必须逐条命中，且各恰好一处
for pat, n in ((r'PreciseText\(spawnConfig\.label \?\? location, style\)', 1),
               (r'PreciseText\(node\.label, destinationStyle\)', 1),
               (r'PreciseText\("Daylight Color", style\)', 1),
               (r'PreciseText\("Darkness Color", style\)', 1)):
    k = len(re.findall(pat, s))
    assert k == n, "上屏点 %s 命中 %d 处，期望 %d" % (pat, k, n)
print("前置 OK  四个 PIXI 上屏点逐条命中且各恰好 1 处")
print("生成区域标签（spawns[].label）唯一 %d 条" % len(spawns))
print("Show Tracks 目的地名（tracks.nodes[].label）唯一 %d 条" % len(tracks))
print("固定串 %d 条：%s" % (len(fixed), fixed))
print("合计 %d 条 —— 全部是 ember.mjs 里的字面量（枚举得出来），卡的是通道不是枚举" %
      (len(spawns) + len(tracks) + len(fixed)))
json.dump({"channel": "PIXI PreciseText（画布，不在 DOM 里）",
           "spawn_labels": spawns, "track_node_labels": tracks, "fixed_literals": fixed,
           "total": len(spawns) + len(tracks) + len(fixed)},
          io.open(os.path.join(OUT, "pixi_channel.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
