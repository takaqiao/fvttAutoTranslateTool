#!/usr/bin/env python3
"""把新手包里 6 张**文字烧死在图片里**的流程图重做成中文版。

  python make_charts.py [--only theskillroll] [--scale 2]

为什么必须重做而不是"翻译"
--------------------------
这 6 张是 .webp 位图，英文排版进像素里，Babele 与任何文本管线都够不着。
唯一的办法是照原样重画一版中文的，然后把 `<img src>` 指过去。

为什么产出 PNG 而不是只在 PDF 里替换
------------------------------------
替换 `src` 之后，**Foundry 里也是中文**——PDF 只是顺带。图片随
alien-evolved-starterset-cn 出货，不改上游模块一个字节。

版式取自原图实测
----------------
  奶白底 #fbf9f5 · 框内浅色 #edf2ee · 主青 #00675f · 深青文字 #125e59
  高亮框 #e9fffd · 标题栏右侧是网格纹 + 右上角切角
骰面符号复用系统自己的背景图（systems/alienrpg/ui/DsN/alien-dice-b6.png），
所以与正文里的 `.basesymbol` 完全一致，不是另画的。

⚠ 术语一律取**交付值**，不自造：近战/射击/机动/侦察/耐力/医疗/指挥/生命值/
   濒死/重伤/死亡检定/压力等级/压力反应/基础骰子/压力骰子/追骰/紧邻距离/节。
   改词表之后必须重跑本脚本，否则图与正文会对不上。
"""
import argparse
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data"
DICE = DATA + "/systems/alienrpg/ui/DsN/alien-dice-b6.png"
OUT_MOD = os.path.join(ROOT, "2-新手包汉化插件", "images", "journal")
CHROMES = [
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#fbf9f5;font-family:"Microsoft YaHei","微软雅黑","Source Han Sans SC",sans-serif;
     color:#125e59;font-size:15px;line-height:1.5}
.wrap{background:#fbf9f5;padding:0 0 10px}
/* ── 标题栏：左侧标题框 + 右侧网格纹 + 右上切角 ── */
.hdr{display:flex;align-items:stretch;height:52px;margin-bottom:0}
.hdr .t{border:2px solid #00675f;border-bottom:none;padding:10px 18px;font-size:21px;
        font-weight:700;letter-spacing:.5px;white-space:nowrap;background:#fbf9f5}
.hdr .grid{flex:1;border-top:2px solid #00675f;border-right:2px solid #00675f;
  background-image:linear-gradient(#cfe0dc 1px,transparent 1px),linear-gradient(90deg,#cfe0dc 1px,transparent 1px);
  background-size:9px 9px;position:relative;
  clip-path:polygon(0 0,calc(100% - 26px) 0,100% 26px,100% 100%,0 100%)}
.frame{border:2px solid #00675f;border-top:2px solid #00675f;padding:16px;background:#fbf9f5}
/* ── 盒子 ── */
.b{border:1px solid #00675f;background:#edf2ee;padding:9px 12px}
.b.hi{background:#e9fffd}
.b.solid{background:#00675f;color:#fbf9f5;font-weight:700;font-size:26px;
         display:flex;align-items:center;justify-content:center;min-width:52px}
.row{display:flex;gap:14px;align-items:stretch}
.row>*{flex:1}
.col{display:flex;flex-direction:column;gap:0}
b,strong{font-weight:700;color:#0a4f4a}
.sc{font-weight:700;letter-spacing:.5px}
/* ── 箭头 ── */
.ar{display:flex;align-items:center;justify-content:center;color:#00675f}
.ar.d{height:26px;font-size:19px;line-height:1;flex:none}
.ar.r{flex:0 0 34px !important;width:34px;font-size:19px}
.dice{display:inline-block;width:17px;height:17px;vertical-align:-3px;
      background-image:url("file:///__DICE__");background-size:17px;background-repeat:no-repeat}
/* ── 分叉连接线 ── */
/* 分叉：一条横线 + 三条下垂线（左 1/6、中 1/2、右 5/6），与三列盒子对齐 */
.fork{height:26px;position:relative}
.fork i{position:absolute;border-color:#00675f;border-style:solid;border-width:0}
.fork .h{left:16.6%;right:16.6%;top:12px;border-top-width:1.5px}
.fork .u{left:50%;top:0;height:12px;border-left-width:1.5px}
.fork .l{left:16.6%;top:12px;height:14px;border-left-width:1.5px}
.fork .m{left:50%;top:12px;height:14px;border-left-width:1.5px}
.fork .r{left:83.4%;top:12px;height:14px;border-left-width:1.5px}
/* 两向分叉 */
.fork2 .h{left:25%;right:25%}
.fork2 .l{left:25%}
.fork2 .r{left:75%}
.fork2 .m{display:none}
.step{display:flex;gap:0;border:1px solid #00675f;background:#edf2ee}
.step .n{background:#00675f;color:#fbf9f5;font-weight:700;font-size:26px;
         display:flex;align-items:center;justify-content:center;width:56px;flex:none}
.step .c{padding:10px 13px}
.sep{height:16px}

/* ── 无框小图（actions / resolvecalc）────────────────────────────
   实测原图：填充 #edf2ee，描边 #00675f，**左上角切角**（斜边上有描线），
   底边是一条粗青实边（约 6px，不是投影）。
   ⚠ 别用 ::after 画底边 + clip-path —— clip-path 会连伪元素一起裁掉。
   做法是外层青底 + 内层填充，两层各自切角，靠 padding 露出边宽。 */
.chips{display:flex;align-items:center;gap:12px}
.chip{background:#00675f;padding:2.5px 2.5px 7px;
      clip-path:polygon(13px 0,100% 0,100% 100%,0 100%,0 13px)}
.chip>span{display:block;background:#edf2ee;padding:7px 17px;
      font-size:24px;font-weight:700;white-space:nowrap;line-height:1.25;
      clip-path:polygon(11px 0,100% 0,100% 100%,0 100%,0 11px)}
.op{font-size:31px;font-weight:700;padding:0 2px}
.orx{font-size:23px;font-style:italic;padding:0 8px}
.stack{display:flex;flex-direction:column;gap:11px}

/* ── 距离段（zones，无框）──────────────────────────────────────
   实测：一条贯穿的青线，左端实心圆点、右端箭头；每段是「切角块 +
   右侧青色实心字母格」；块下有**点线引出**接到说明文字。
   字母 A/S/M/L/E 保持拉丁 —— 它们是角色卡与物品数据里用的射程代码，
   翻掉就对不上（T-FROZEN）。 */
.zrow{display:flex;align-items:flex-start}
.zdot{width:15px;height:15px;border-radius:50%;background:#00675f;margin-top:16px}
.zline{flex:1;height:0;border-top:2.5px solid #00675f;margin-top:22px}
.zarrow{margin-top:11px;font-size:25px;color:#00675f;line-height:1}
.ztick{width:0;height:30px;border-left:2.5px solid #00675f;margin:8px 5px 0}
.zcol{display:flex;flex-direction:column;align-items:center}
.zbox{display:flex;background:#00675f;padding:2.5px;
      clip-path:polygon(13px 0,100% 0,100% 100%,0 100%,0 13px)}
.zbox .nm{background:#edf2ee;padding:6px 14px;font-size:22px;font-weight:700;
      white-space:nowrap;clip-path:polygon(11px 0,100% 0,100% 100%,0 100%,0 11px)}
.zbox .lt{color:#fbf9f5;font-size:22px;font-weight:700;padding:6px 13px}
.zlead{width:0;border-left:2px dotted #00675f;height:20px;margin:4px 0}
.zcap{font-size:19px;white-space:nowrap}

/* ── 压力反应检定：公式行 + 两列表 ── */
.formula{display:flex;align-items:center;gap:10px;margin-bottom:14px;font-size:19px;font-weight:700}
.formula .k{border:2px solid #00675f;padding:6px 14px;background:#fbf9f5}
.formula .op{font-size:22px}
.thead{display:flex;font-weight:700;font-size:16px;letter-spacing:1px;padding:0 0 5px}
.thead .c1{flex:0 0 96px}
.trow{display:flex;padding:8px 0;align-items:baseline}
.trow.alt{background:#edf2ee}
.trow .c1{flex:0 0 96px;padding-left:8px;font-weight:700}
.trow .c2{flex:1;padding-right:8px}
""".replace("__DICE__", DICE.replace("\\", "/"))

D = '<span class="dice"></span>'


def page(title, body, width):
    # title=None → 无标题栏、无外框（上游有两张小图就是这个样子）
    if title is None:
        tpl = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>__CSS__
.wrap{width:__W__px;padding:6px 0 4px}</style></head><body>
<div class="wrap">__BODY__</div></body></html>"""
        return tpl.replace("__CSS__", CSS).replace("__W__", str(width)).replace("__BODY__", body)
    tpl = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>__CSS__
.wrap{width:__W__px}</style></head><body>
<div class="wrap"><div class="hdr"><div class="t">__TITLE__</div><div class="grid"></div></div>
<div class="frame">__BODY__</div></div></body></html>"""
    return (tpl.replace("__CSS__", CSS).replace("__W__", str(width))
               .replace("__TITLE__", title).replace("__BODY__", body))


CHARTS = {
    # ── 9. 距离段（无框）───────────────────────────────────────
    # 段名取系统交付值：紧邻距离/近距离/中距离/远距离/极远距离。
    "zones": (None, 1010, """
<div class="zrow">
  <div class="zdot"></div><div class="zline"></div>
  <div class="zcol"><div class="zbox"><div class="nm">紧邻距离</div><div class="lt">A</div></div>
    <div class="zlead"></div><div class="zcap">近在咫尺。</div></div>
  <div class="zline"></div>
  <div class="zcol"><div class="zbox"><div class="nm">近距离</div><div class="lt">S</div></div>
    <div class="zlead"></div><div class="zcap">在同一区域内。</div></div>
  <div class="zline"></div><div class="ztick"></div><div class="zline"></div>
  <div class="zcol"><div class="zbox"><div class="nm">中距离</div><div class="lt">M</div></div>
    <div class="zlead"></div><div class="zcap">在相邻区域。</div></div>
  <div class="zline"></div><div class="ztick"></div><div class="ztick"></div><div class="ztick"></div><div class="zline"></div>
  <div class="zcol"><div class="zbox"><div class="nm">远距离</div><div class="lt">L</div></div>
    <div class="zlead"></div><div class="zcap">最远四个区域外。</div></div>
  <div class="zline"></div><div class="ztick"></div><div class="zline"></div>
  <div class="zcol"><div class="zbox"><div class="nm">极远距离</div><div class="lt">E</div></div>
    <div class="zlead"></div><div class="zcap">更远的地方</div></div>
  <div class="zline"></div><div class="zarrow">➜</div>
</div>"""),

    # ── 7. 动作（无框）──────────────────────────────────────────
    # 用词取本书交付值：完整动作(45 处) / 快速动作(48 处)。
    "actions": (None, 470, """
<div class="chips">
  <div class="stack">
    <div class="chip"><span>完整动作</span></div>
    <div class="chip"><span>快速动作</span></div>
  </div>
  <div class="orx">或</div>
  <div class="stack">
    <div class="chip"><span>快速动作</span></div>
    <div class="chip"><span>快速动作</span></div>
  </div>
</div>"""),

    # ── 8. 精神强度算式（无框）──────────────────────────────────
    # 与第 6 张压力反应检定图顶部的公式是同一条，用词必须一致。
    "resolvecalc": (None, 430, """
<div class="chips">
  <div class="chip"><span>D6</span></div><div class="op">+</div>
  <div class="chip"><span>压力等级</span></div><div class="op">−</div>
  <div class="chip"><span>精神强度</span></div>
</div>"""),

    # ── 1. 技能检定 ──────────────────────────────────────────────
    "theskillroll": ("技能检定", 640, """
<div class="row">
  <div class="b">取来数量等于<b>属性</b>＋<b>技能</b>的基础骰子。</div>
  <div class="ar r">➜</div>
  <div class="b">按<b>难度</b>、<b>装备</b>与<b>协助</b>调整基础骰池。</div>
</div>
<div class="ar d">↓</div>
<div class="b" style="text-align:center">再加入数量等于你<b>压力等级</b>的<b>压力骰子</b>。</div>
<div class="ar d">↓</div>
<div class="b hi" style="text-align:center;max-width:200px;margin:0 auto">掷出所有骰子</div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row">
  <div class="b">掷不出 __D__，你的动作<b>失败</b>。你可以<b>追骰</b>——压力等级 +1 后重掷。</div>
  <div class="b">掷出一个或多个 __D__，你<b>成功</b>。想要更多 __D__，你可以<b>追骰</b>——压力等级 +1 后重掷。</div>
  <div class="b">若在一颗或多颗压力骰子上掷出压力符号，触发一次<b>压力反应</b>。</div>
</div>""".replace("__D__", D)),

    # ── 2. 潜行 ──────────────────────────────────────────────────
    "stealthflowchart": ("潜行流程", 480, """
<div class="step"><div class="n">1</div><div class="c">
<b>PC 移动一个区域。</b>若他们进入某个 NPC 的视线范围，进行一次被动开放对抗<span class="sc">侦察</span>检定。若是一群人，取最高的那次结果。</div></div>
<div class="fork fork2"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row">
  <div class="b"><b>若某一方胜出</b>，他们可以：</div>
  <div class="b hi"><b>若打平，</b><u>抽先攻牌。</u></div>
</div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row" style="gap:9px">
  <div class="b">现身</div>
  <div class="b">伏击敌人。</div>
  <div class="b">藏在本区域内。</div>
  <div class="b">原路退回。</div>
</div>
<div class="sep"></div>
<div class="step"><div class="n">2</div><div class="c">
<b>NPC 在 GM 的暗图上移动一个区域。</b>异形每点速度移动一个区域。若它们进入某个 PC 的视线范围，同上进行一次被动开放对抗<span class="sc">侦察</span>检定。</div></div>
<div class="sep"></div>
<div class="step"><div class="n">3</div><div class="c">
<b>新的一节开始。</b>PC 先行动，然后是 NPC，按上面第 1、2 步进行。</div></div>"""),

    # ── 3. 近战 ──────────────────────────────────────────────────
    "ccflowchart": ("近战攻击", 1000, """
<div class="row">
  <div class="b">声明你的<b>目标</b>、使用什么<b>武器</b>（如果有），以及是否进行<b>特殊攻击</b>。</div>
  <div class="ar r">➜</div>
  <div class="b">目标决定是否<b>⚠ 防御</b>（快速动作）。</div>
</div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row">
  <div class="col" style="gap:0">
    <div class="b"><b>目标防御：</b>进行一次<span class="sc">近战</span>对抗检定。用你掷出的 __D__ 数量减去目标掷出的 __D__ 数量。</div>
    <div class="ar d">↓</div>
    <div class="row" style="gap:9px">
      <div class="b"><b>若还有 __D__ 剩余，</b>你命中，造成武器基础伤害，且第一个之后每多剩一个 __D__ 再加 1 点。</div>
      <div class="b"><b>若没有 __D__ 剩余，</b>你失手。</div>
    </div>
  </div>
  <div class="col" style="gap:0">
    <div class="b"><b>目标不防御：</b><br>直接进行一次<span class="sc">近战</span>检定。</div>
    <div class="ar d">↓</div>
    <div class="row" style="gap:9px">
      <div class="b">若掷出<b>一个或多个</b> __D__，你命中，造成武器基础伤害，且第一个之后每多一个 __D__ 再加 1 点。</div>
      <div class="b"><b>若掷不出</b> __D__，你失手。</div>
    </div>
  </div>
</div>""".replace("__D__", D)),

    # ── 4. 射击 ──────────────────────────────────────────────────
    "rcflowchart": ("射击攻击", 1330, """
<div class="row">
  <div class="b" style="flex:0 0 40%">声明你的<b>目标</b>、使用什么<b>武器</b>。</div>
  <div class="ar r">➜</div>
  <div class="b">目标决定是否<b>⚠ 闪避</b>（快速动作）。</div>
</div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row">
  <div class="col" style="gap:0">
    <div class="b"><b>目标闪避：</b>进行一次<span class="sc">射击</span>对抗<span class="sc">机动</span>的检定。用你掷出的 __D__ 数量减去目标掷出的 __D__ 数量。</div>
    <div class="ar d">↓</div>
    <div class="row" style="gap:9px">
      <div class="b hi"><b>若还有 __D__ 剩余，</b>你命中，造成武器基础伤害，且第一个之后每多剩一个 __D__ 再加 1 点——或者额外花掉一个 __D__ 来节省弹药。</div>
      <div class="b"><b>若没有 __D__ 剩余，</b>你失手。</div>
    </div>
  </div>
  <div class="col" style="gap:0">
    <div class="b"><b>目标不闪避：</b><br>直接进行一次<span class="sc">射击</span>检定。</div>
    <div class="ar d">↓</div>
    <div class="row" style="gap:9px">
      <div class="b hi">若掷出<b>一个或多个</b> __D__，你命中，造成武器基础伤害，且第一个之后每多一个 __D__ 再加 1 点——或者额外花掉一个 __D__ 来节省弹药。</div>
      <div class="b"><b>若掷不出</b> __D__，你失手。</div>
    </div>
  </div>
</div>
<div class="ar d">↓</div>
<div class="b">进行一次弹药余量检定，除非攻击检定里额外的 __D__ 被用来节省弹药而不是提升伤害。</div>""".replace("__D__", D)),

    # ── 5. 伤害 ──────────────────────────────────────────────────
    "damageflowchart": ("伤害", 840, """
<div class="row">
  <div class="b">用护甲等级减免收到的<b>伤害</b>（按破甲与弱点修正）。</div>
  <div class="ar r">➜</div>
  <div class="b"><b>按减免后的伤害扣减你的生命值。</b></div>
</div>
<div class="ar d">↓</div>
<div class="b"><b>若你的生命值降到零</b>，你进入<b>濒死</b>，除每轮一个移动动作外无法进行任何动作。进入濒死时立刻掷一次<b>重伤</b>。<b>它致命吗？</b></div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row">
  <div class="b" style="flex:0 0 26%"><b>否：</b>施加该效果。</div>
  <div class="b"><b>是：</b>施加该效果，并在每次指定的时限过去之后进行一次<span class="sc">耐力</span><b>死亡检定</b>。该检定不能追骰，也不使用压力骰子。</div>
</div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row" style="gap:9px">
  <div class="b" style="flex:0 0 18%"><b>一次死亡检定失败：</b>你死亡。</div>
  <div class="b" style="flex:0 0 24%"><b>三次死亡检定成功：</b>停止进行死亡检定。</div>
  <div class="b">处于<b>紧邻距离</b>的其他人可以用一次<span class="sc">医疗</span>检定为你急救（完整动作）。成功则你稳定下来——停止死亡检定。若你处于濒死，每掷出一个 __D__ 还能恢复 1 点生命值。若检定失败，你的状况恶化——时限缩短一级。</div>
</div>
<div class="sep"></div>
<div class="b">处于濒死时，<b>紧邻距离</b>的人可以为你急救（<span class="sc">医疗</span>检定，完整动作），你所在区域内的任何人都可以尝试鼓舞你（<span class="sc">指挥</span>检定，快速动作）。</div>
<div class="fork"><i class="h"></i><i class="u"></i><i class="l"></i><i class="m"></i><i class="r"></i></div>
<div class="row">
  <div class="b" style="flex:0 0 26%"><b>失败：</b>无效果。</div>
  <div class="b"><b>成功：</b>每掷出一个 __D__ 恢复 1 点生命值。你不再处于濒死。</div>
</div>
<div class="sep"></div>
<div class="b"><b>每过一节恢复 1 点生命值</b>，即使身负重伤也一样。当你的生命值达到 1 或更高时，你不再处于濒死。</div>""".replace("__D__", D)),

    # ── 6. 压力反应检定 ────────────────────────────────────────
    # ⚠ 上游把这张图放在 **corerules** 下，而新手包的页面 src 指向
    #   alien-evolved-starterset/…，那个路径下没有文件 —— 上游的路径 bug。
    # ⚠⚠ 第一版我按「掷 1D6」做成六档，是**规则错误**：真正的判定式是
    #   D6 + 压力等级 − 精神强度，结果范围 ≤0 到 7+ 共八档，丢掉了
    #   「≤0 镇定」与「7+ 失误」两档。本版照真图重做，八行与包内
    #   Stress Response Table 的 range（0-0 … 7-10）一一对应。
    "stressresponseroll": ("压力反应检定", 560, """
<div class="formula">
  <span class="k">D6</span><span class="op">+</span>
  <span class="k">压力等级</span><span class="op">−</span>
  <span class="k">精神强度</span>
</div>
<div class="thead"><div class="c1">结果</div><div class="c2">反应</div></div>
<div class="trow alt"><div class="c1">≤0</div><div class="c2"><b>镇定。</b>无效果。</div></div>
<div class="trow"><div class="c1">1</div><div class="c2"><b>惊跳。</b>当你对技能检定进行追骰时，你获得的压力等级是 +2 而非 +1。</div></div>
<div class="trow alt"><div class="c1">2</div><div class="c2"><b>恍惚。</b>所有基于机智的技能检定，骰子 −2。</div></div>
<div class="trow"><div class="c1">3</div><div class="c2"><b>恼怒。</b>所有基于共情的技能检定，骰子 −2。</div></div>
<div class="trow alt"><div class="c1">4</div><div class="c2"><b>颤抖。</b>所有基于敏捷的技能检定，骰子 −2。</div></div>
<div class="trow"><div class="c1">5</div><div class="c2"><b>慌乱。</b>所有基于力量的技能检定，骰子 −2。</div></div>
<div class="trow alt"><div class="c1">6</div><div class="c2"><b>泄气。</b>你不能对任何技能检定进行追骰。如果你处于惊跳状态（上面的 #1），则移除该反应；在泄气期间，忽略之后掷出的任何惊跳结果。</div></div>
<div class="trow"><div class="c1">7+</div><div class="c2"><b>失误。</b>无论掷出什么结果，你的动作都失败，并且你的压力等级 +1。</div></div>"""),
}



# ------------------------------------------------------------------ wire --
UPSTREAM_DIR = "modules/alien-evolved-starterset/images/journal/"
OURS_DIR = "modules/alien-evolved-starterset-cn/images/journal/"


def cmd_wire():
    """把包内译文里那 6 张图的 src 指到我们自己的中文版。

    ⚠ 只改 src，不动任何标签结构 —— tag 多重集必须保持不变，否则
    markup 闸会红，而且那道闸是对的：图片替换不该改变文档结构。
    """
    import json
    p = os.path.join(ROOT, "2-新手包汉化插件", "compendium", "cn",
                     "alien-evolved-starterset.alien-evolved-starter-set.json")
    doc = json.load(io.open(p, encoding="utf-8"))
    adv = doc["entries"]["Alien Evolved Starter Set"]
    n = 0
    for j in adv.get("journals", {}).values():
        for pg in (j.get("pages") or {}).values():
            t = pg.get("text")
            if not t:
                continue
            before = t
            for name in CHARTS:
                t = t.replace(UPSTREAM_DIR + name + ".webp", OURS_DIR + name + ".png")
            if t != before:
                pg["text"] = t
                n += len(re.findall(re.escape(OURS_DIR), t))
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("  改写 %d 处 img src -> %s" % (n, OURS_DIR))
    missing = [c for c in CHARTS
               if not os.path.exists(os.path.join(OUT_MOD, c + ".png"))]
    if missing:
        print("  ⚠ 但这些图还没生成：%s" % missing)
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--scale", type=float, default=2.0, help="设备像素比，2 = 视网膜清晰度")
    ap.add_argument("--wire", action="store_true", help="把包内译文的 img src 指到中文图")
    a = ap.parse_args()

    if a.wire:
        return cmd_wire()

    chrome = next((c for c in CHROMES if os.path.exists(c)), None)
    if not chrome:
        sys.exit("找不到 Chrome/Edge")
    if not os.path.exists(DICE):
        sys.exit("骰面图缺失：%s" % DICE)
    os.makedirs(OUT_MOD, exist_ok=True)
    tmp = os.path.join(ROOT, "6-工作区", "charts")
    os.makedirs(tmp, exist_ok=True)

    todo = {a.only: CHARTS[a.only]} if a.only else CHARTS
    ok = 0
    for name, (title, width, body) in todo.items():
        html_path = os.path.join(tmp, name + ".html")
        png_path = os.path.join(OUT_MOD, name + ".png")
        io.open(html_path, "w", encoding="utf-8", newline="\n").write(page(title, body, width))
        if os.path.exists(png_path):
            os.remove(png_path)
        cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
               "--default-background-color=00000000",
               "--force-device-scale-factor=%s" % a.scale,
               "--window-size=%d,%d" % (width + 4, 2400),
               "--virtual-time-budget=8000",
               "--screenshot=" + png_path,
               "file:///" + html_path.replace("\\", "/")]
        subprocess.run(cmd, capture_output=True, timeout=120)
        if not os.path.exists(png_path):
            print("  ✗ %s 未产出" % name)
            continue
        # 截的是整窗，底部有大片空白 —— 裁掉
        try:
            from PIL import Image
            im = Image.open(png_path).convert("RGB")
            bg = im.getpixel((im.size[0] - 2, im.size[1] - 2))
            bbox = None
            for y in range(im.size[1] - 1, 0, -1):
                row = [im.getpixel((x, y)) for x in range(0, im.size[0], 7)]
                if any(abs(p[0] - bg[0]) + abs(p[1] - bg[1]) + abs(p[2] - bg[2]) > 24 for p in row):
                    bbox = (0, 0, im.size[0], min(im.size[1], y + int(12 * a.scale)))
                    break
            if bbox:
                im.crop(bbox).save(png_path)
                im = Image.open(png_path)
        except ImportError:
            pass
        print("  ✓ %-22s %s  %d KB" % (name, "x".join(map(str, Image.open(png_path).size)),
                                       os.path.getsize(png_path) // 1024))
        ok += 1
    print("\n产出 %d/%d -> %s" % (ok, len(todo), OUT_MOD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
