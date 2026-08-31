#!/usr/bin/env python3
"""把 mini-map.webp 上**烧死在像素里**的英文房间名换成中文。

  python make_minimap.py [--grid] [--only 停尸间]

为什么这张要单独处理
--------------------
新手包的 9 张流程图是纯矢量版式，重画即可（见 make_charts.py）。这张不一样：
它是一张**位图美术**（哈德利的希望两层站图），只有房间名是文字。重画整张图
不现实，也没必要——把英文标注抹掉、原位写中文就够了。

⚠ 房间名一个字都不许自造
------------------------
译名全部取自包内 **scenes 的 notes 映射**（Level 1/2 各 17 条图钉），
那是玩家在 Foundry 里点开场景真正看到的名字。图上写的要是另一套说法，
GM 对着图找房间就会对不上。两个例外，来源写在 LABELS 里各自的注释上。

抹字的做法
----------
中值滤波 + 轻微高斯模糊。**不要用 MinFilter**——它是腐蚀，会把亮部整体压暗，
停尸间那片青色辉光会明显发黑。中值能吃掉细笔画又基本保住色调。
补丁边缘用羽化 alpha 贴回，否则会看见一块方形的接缝。
"""
import argparse
import io
import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = (r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules"
       r"/alien-evolved-starterset/images/journal/mini-map.webp")
OUT = os.path.join(ROOT, "2-新手包汉化插件", "images", "journal", "mini-map.png")
FONT = "C:/Windows/Fonts/msyhbd.ttc"          # 微软雅黑 Bold

PALE = (208, 230, 238)      # 站图上标注的常见浅青白
BRIGHT = (232, 248, 252)    # 亮底（停尸间/手术室辉光）上用更白一点

# (中文, 断行, 字号, 中心x, 中心y, 抹除框 x0,y0,x1,y1, 旋转, 颜色)
LABELS = [
    # ── Level 1（左半张）──────────────────────────────────────
    ("卫生间",            1, 11, 237,  41, (226,  32, 249,  50), 0, PALE),
    ("储藏室",            1, 11, 349,  40, (340,   6, 358,  74), 90, PALE),
    ("韦兰德-汤谷|办公室", 2, 16, 139, 143, ( 82, 117, 198, 172), 0, PALE),
    ("资产/矿权|办公室",   2, 12, 483, 113, (456,  92, 510, 136), 0, PALE),
    ("助理主管|办公室",    2, 13, 390, 280, (350, 258, 432, 302), 0, PALE),
    ("储藏室",            1, 11, 505, 281, (496, 246, 514, 314), 90, PALE),
    ("勘测办公室",         1, 12, 148, 326, ( 98, 316, 198, 336), 0, PALE),
    ("西地质|实验室",      2, 16, 158, 461, (118, 441, 200, 480), 0, BRIGHT),
    ("东地质|实验室",      2, 16, 438, 423, (402, 402, 474, 443), 0, BRIGHT),
    ("无菌|实验室",        2, 11, 110, 560, ( 82, 543, 142, 578), 0, PALE),
    ("气闸",              1, 11, 192, 556, (170, 546, 216, 566), 0, PALE),
    ("储藏室",            1, 11, 196, 622, (170, 613, 224, 632), 0, PALE),
    ("储藏室",            1, 11, 379, 622, (352, 613, 406, 632), 0, PALE),
    ("地质主管|办公室",    2, 12, 493, 590, (450, 568, 538, 614), 0, PALE),
    # ⚠「南侧气闸」不在 notes 里 —— 取自冒险正文「PC 是从南边过来的，所以南侧气闸离得最近」
    ("南侧气闸",          1, 15, 293, 673, (260, 654, 328, 694), 0, PALE),
    # ── Level 2（右半张）──────────────────────────────────────
    ("通往赌场的天桥",     1, 11, 706, 104, (666,  88, 748, 121), 0, PALE),
    # ⚠「办公室」不在 notes 里 —— 图上就是个泛指的 OFFICE
    ("办公室",            1, 11, 1136, 113, (1110, 104, 1162, 122), 0, PALE),
    ("手术室",            1, 16, 783, 213, (742, 194, 824, 233), 0, BRIGHT),
    ("停尸间",            1, 17, 1071, 253, (1038, 241, 1106, 266), 0, BRIGHT),
    ("休眠室",            1, 12, 709, 318, (684, 300, 734, 336), 0, PALE),
    ("冷藏室",            1, 12, 1074, 341, (1028, 332, 1122, 352), 0, PALE),
    ("医疗实验室",         1, 16, 845, 379, (810, 367, 880, 392), 0, PALE),
    ("医疗实验室|办公室",  2, 11, 752, 486, (720, 468, 786, 504), 0, PALE),
    ("生命维持/电力",      1, 12, 1071, 446, (1024, 428, 1120, 464), 0, PALE),
    ("储藏室",            1, 11, 1185, 470, (1176, 436, 1194, 504), 90, PALE),
    ("站务维护|中心",      2, 12, 1140, 608, (1096, 584, 1184, 632), 0, PALE),
    ("指挥中心",          1, 16, 789, 661, (736, 650, 842, 672), 0, PALE),
]


def erase(im, box):
    """抹掉一块区域里的文字，保住底色与渐变。"""
    x0, y0, x1, y1 = box
    pad = 6
    ex = (max(0, x0 - pad), max(0, y0 - pad),
          min(im.width, x1 + pad), min(im.height, y1 + pad))
    patch = im.crop(ex)
    # 中值吃掉细笔画；核要比笔画粗，但别大到糊掉墙线
    patch = patch.filter(ImageFilter.MedianFilter(size=9))
    patch = patch.filter(ImageFilter.GaussianBlur(1.6))
    # 羽化：中间全不透明，边缘渐隐，避免方形接缝
    mask = Image.new("L", (ex[2] - ex[0], ex[3] - ex[1]), 0)
    ImageDraw.Draw(mask).rectangle(
        (pad // 2, pad // 2, mask.width - pad // 2, mask.height - pad // 2), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(pad / 1.6))
    im.paste(patch, ex[:2], mask)


def draw_label(im, text, nlines, size, cx, cy, rot, color):
    font = ImageFont.truetype(FONT, size)
    # rot=90 表示竖排。中文竖排是**字正立、逐字下排**，不是把整行转 90°——
    # 转出来的字既别扭又会读反方向（原图英文是从上往下的）。
    if rot:
        lines, rot = list(text), 0
    else:
        lines = text.split("|") if nlines > 1 else [text]
    lh = int(size * 1.28) if len(lines) <= 3 else int(size * 1.12)
    # 先画到透明层上，这样旋转的竖排标注也走同一条路径
    w = max(int(ImageDraw.Draw(im).textlength(s, font=font)) for s in lines) + 8
    h = lh * len(lines) + 6
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, s in enumerate(lines):
        tw = d.textlength(s, font=font)
        x, y = (w - tw) / 2, 3 + i * lh
        # 描一圈暗边，亮底暗底上都读得出来（原图靠辉光，我们靠描边）
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((x + dx, y + dy), s, font=font, fill=(4, 26, 32, 190))
        d.text((x, y), s, font=font, fill=color + (255,))
    if rot:
        layer = layer.rotate(rot, expand=True)
    im.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)), layer)


PACK = os.path.join(ROOT, "2-新手包汉化插件", "compendium", "cn",
                    "alien-evolved-starterset.alien-evolved-starter-set.json")


def wire():
    """只改 src，不动任何结构或文字。"""
    old = "modules/alien-evolved-starterset/images/journal/mini-map.webp"
    new = "modules/alien-evolved-starterset-cn/images/journal/mini-map.png"
    s = io.open(PACK, encoding="utf-8").read()
    n = s.count(old)
    if not n:
        print("  包里没有指向上游 mini-map 的引用（可能已接过）")
        return 0
    io.open(PACK, "w", encoding="utf-8", newline=chr(10)).write(s.replace(old, new))
    print("  改写 %d 处 img src -> %s" % (n, new))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="产出带坐标网格的图，用来核对位置")
    ap.add_argument("--only", help="只处理某一条标注（调位置用）")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--wire", action="store_true",
                    help="把 CN 包里这张图的 src 指到我们的 PNG（改完包，不改上游）")
    a = ap.parse_args()

    if a.wire:
        return wire()

    if not os.path.exists(SRC):
        sys.exit("找不到上游底图：%s" % SRC)
    im = Image.open(SRC).convert("RGB")

    todo = [l for l in LABELS if not a.only or l[0] == a.only]
    if a.only and not todo:
        sys.exit("没有这条标注：%s" % a.only)
    for text, nl, size, cx, cy, box, rot, color in todo:
        erase(im, box)
    for text, nl, size, cx, cy, box, rot, color in todo:
        draw_label(im, text, nl, size, cx, cy, rot, color)

    if a.grid:
        d = ImageDraw.Draw(im)
        f = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
        for x in range(0, im.width, 100):
            d.line([(x, 0), (x, im.height)], fill=(255, 60, 60))
            d.text((x + 2, 2), str(x), fill=(255, 220, 0), font=f)
        for y in range(0, im.height, 50):
            d.line([(0, y), (im.width, y)], fill=(255, 60, 60))
            d.text((2, y + 1), str(y), fill=(0, 255, 180), font=f)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    im.save(a.out)
    print("  ✓ %s  %dx%d  %d KB  （%d 条标注）"
          % (os.path.basename(a.out), im.width, im.height,
             os.path.getsize(a.out) // 1024, len(todo)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
