#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 TOKEN_MAKER_UI 表的 JS 源码片段（行号从上游现抠，不手写）。

前置自证：
  (A) 切对条数 —— 译名表的键集必须**逐个等于** tokenmaker_slots.py 抠出来的槽位集合（多一个少一个都 FAIL）；
  (B) 改对地方 —— 每个键都必须能在上游 ember.mjs 里找到 `label: "<键>"` 或已知的其它字面形态，
      找不到的当场列出来（那样的键进了表，自检面板 D 档会报 miss，主闸的 max 阈值会红）。
"""
import json, os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
MJS = os.path.join(U, "scripts", "ember.mjs")
SRC = open(MJS, encoding="utf-8").read()
SLOTS = json.load(open(os.path.join(HERE, "tokenmaker_slots.json"), encoding="utf-8"))

COLORS = {
    "Alloy": "合金", "Bark Accent": "树皮点缀色", "Bark Base": "树皮基色",
    "Birthmark Base": "胎记基色", "Birthmark Shimmer": "胎记微光",
    "Bloodstains": "血迹", "Bone": "骨色", "Claws": "爪", "Clay": "陶土",
    "Cloth 1": "布料 1", "Cloth 2": "布料 2", "Crystal": "水晶",
    "Embellishments": "缀饰", "Ethereal": "以太",
    "Eye Glow": "眼部辉光", "Eye Iris": "眼部虹膜", "Eye Sclera": "眼白",
    "Eyes": "眼睛", "Eyes Iris": "眼部虹膜", "Eyes Sclera": "眼白",
    "Fabric Accent": "织物点缀色", "Fabric Base": "织物基色",
    "Flame": "火焰", "Flora Detail": "草木细节",
    "Fur": "毛皮", "Fur Accent": "毛皮点缀色", "Fur Base": "毛皮基色",
    "Glow": "光晕",
    "Hair 1": "头发 1", "Hair Base": "头发基色", "Hair Glow": "头发辉光",
    "Hair Highlights": "头发挑染", "Hair Roots": "发根", "Hair Sparkle": "头发闪光",
    "Heraldry 1": "纹章 1", "Heraldry 2": "纹章 2", "Heraldry 3": "纹章 3",
    "Horns": "角", "Keratin": "角质", "Keratin Accent": "角质点缀色",
    "Keratin Base": "角质基色", "Keratin Detail": "角质细节",
    "Leather 1": "皮革 1", "Leather 2": "皮革 2",
    "Leather Accent": "皮革点缀色", "Leather Base": "皮革基色",
    "Leaves Accent": "叶片点缀色", "Leaves Base": "叶片基色",
    "Mask Base": "面具基色", "Mask Pattern A": "面具纹样 A", "Mask Pattern B": "面具纹样 B",
    "Metal 1": "金属 1", "Metal 2": "金属 2",
    "Metal Accent": "金属点缀色", "Metal Base": "金属基色",
    "Pyre Core": "焚焰核心", "Pyre Heat": "焚焰热光", "Radiance": "辉光",
    "Scales Accent": "鳞片点缀色", "Scales Base": "鳞片基色",
    "Skin": "皮肤", "Skin 1": "皮肤 1", "Skin 2": "皮肤 2",
    "Skin Accent": "皮肤点缀色", "Skin Base": "皮肤基色", "Skin Plates": "皮甲片",
    "Stone": "石料", "Straw": "干草", "Symbol 1": "徽记 1",
    "Tattoo": "刺青", "Teeth": "牙齿",
    "Wisps Base": "幽芒基色", "Wisps Glow": "幽芒辉光",
    "Wood": "木材", "Wood 1": "木材 1", "Wood 2": "木材 2",
}

LAYERS = {
    "Arms": "手臂", "Back Apparel": "背部衣饰", "Back Item": "背部物品",
    "Base": "基底", "Chest": "胸部", "Eyebrows": "眉毛", "Eyes": "眼睛",
    "Eyewear": "眼饰", "Face": "面部", "Feet": "脚", "Footwear": "鞋履",
    "Forearms": "前臂", "Front": "正面", "Hair": "头发", "Hand": "手",
    "Head": "头部", "Header": "顶饰", "Helm": "头盔", "Helm Lower": "头盔下部",
    "Horns": "角",
    "Item Hand (Left Lower)": "手持物（左下）", "Item Hand (Right Lower)": "手持物（右下）",
    "Item Waist": "腰间物品", "Leg": "腿", "Legs": "双腿", "Neck": "颈部",
    "Nipples": "乳首", "Pants": "裤装", "Pauldron": "护肩",
    "Pauldron Left": "左护肩", "Pauldron Right": "右护肩",
    "Pole": "旗杆", "Sleeve": "袖", "Sleeves": "双袖", "Symbol": "徽记",
    "Torso": "躯干", "Waist": "腰部", "Wrist": "腕部",
}

BUILDS = {"Lithe": "纤瘦", "Standard": "标准", "Heavy": "壮硕"}
STANCES = {"Action": "动作", "Land": "陆上", "Sitting": "坐姿",
           "Standing": "站立", "Walking": "行走", "Water": "水中"}

# 模板名＝血统名：与合集里的条目名**逐字一致**（合集用的是「中文 English」双语式，
# 玩家在创角向导里选的就是那个名字，下拉里换个写法会对不上）。
TEMPLATES = {
    "Altyra": "阿尔提拉 Altyra", "Ashka": "阿什卡 Ashka",
    "Construct": "构装体 Construct", "Cor'ak": "科拉克 Cor'ak",
    "Drakon": "龙裔 Drakon", "Fej": "费伊杰 Fej", "Hulg'run": "赫尔格伦 Hulg'run",
    "Human": "人类 Human", "Human (Abyssal)": "人类 Human（深渊裔）",
    "Human (Undead)": "人类 Human（不死）", "Jurtak": "尤尔塔克 Jurtak",
    "Keth": "凯思 Keth", "Kiska": "基斯卡 Kiska", "Kivahr": "基瓦尔 Kivahr",
    "Nir'ae": "尼尔艾 Nir'ae", "Party Banner": "队伍旗帜",
    "Undead Monster": "不死怪物", "Zeph": "泽夫 Zeph",
}

CHROME = {
    "None": "无",
    "Allow Restricted Parts": "允许受限部件",
    "Clear Custom Offsets": "清除自定义偏移",
    "Body Type": "身体类型",
    "Part Restrictions": "部件限制",
    "Color Restrictions": "颜色限制",
    "Choose Layer": "选择层",
    "Choose Color": "选择颜色",
    "Select the anatomy or equipment layer to constrain.": "选择要约束的身体或装备层。",
    "Select the color type to constrain.": "选择要约束的颜色类型。",
}

GROUPS = [("颜色槽（colors.hbs 的 <label>{{color.label}}</label>）", COLORS, SLOTS["colors"]),
          ("图层名（layers.hbs 的 {{layer.label}}）", LAYERS,
           {**SLOTS["layers"], **SLOTS["tpl_layer_override"]}),
          ("体格（layers.hbs 的 {{build.label}}）", BUILDS, SLOTS["builds"]),
          ("站姿（layers.hbs 的 {{stance.label}}）", STANCES, SLOTS["stances"]),
          ("模板名（body.hbs 的 <select name=\"template\">）", TEMPLATES, SLOTS["templates"])]

ok = True
print("== 前置自证 ==")
for name, tr, slots in GROUPS:
    extra = sorted(set(tr) - set(slots))
    missing = sorted(set(slots) - set(tr))
    good = not extra and not missing
    print(f"[A] {name}：译名 {len(tr)} 条 vs 上游槽位 {len(slots)} 条 -> " + ("OK" if good else "FAIL"))
    if extra: print("     多出来（上游没有这个槽）：", extra)
    if missing: print("     漏了（上游有、译名表没有）：", missing)
    ok = ok and good

# (B) 每个键都必须能在上游源码里找到字面量
print("[B] 上游字面量存在性（进表前先自查，避免自检面板 D 档 miss 涨）：")
allkeys = {}
for _, tr, _ in GROUPS:
    allkeys.update(tr)
allkeys.update(CHROME)
notfound = [k for k in allkeys if f'"{k}"' not in SRC]
print(f"     {len(allkeys) - len(notfound)}/{len(allkeys)} 在 ember.mjs 里有 \"…\" 字面量"
      + ("" if not notfound else f"  查无：{notfound}"))
ok = ok and not notfound

# 撞名自查：同一个英文键在两组里给了不同中文？
seen = {}
clash = []
for name, tr, _ in GROUPS + [("窗口/表单", CHROME, CHROME)]:
    for k, v in tr.items():
        if k in seen and seen[k][1] != v:
            clash.append((k, seen[k], (name, v)))
        seen[k] = (name, v)
print(f"[B] 组间同键异译：{len(clash)} 处" + ("" if not clash else f"  {clash}"))
ok = ok and not clash

if not ok:
    print("\n[STOP] 自证不过。"); sys.exit(1)


def lineno_of(key):
    m = re.search(r'label:\s*"' + re.escape(key).replace(r'\ ', ' ') + r'"', SRC)
    if m: return SRC.count("\n", 0, m.start()) + 1
    m = re.search(r'"' + re.escape(key) + r'"', SRC)
    return SRC.count("\n", 0, m.start()) + 1 if m else 0


out = []
for name, tr, _ in GROUPS:
    out.append(f"  // -- {name} --")
    for k in sorted(tr):
        out.append(f'  {json.dumps(k, ensure_ascii=False)}: {json.dumps(tr[k], ensure_ascii=False)},'
                   f'  // :{lineno_of(k)}')
out.append("  // -- 窗口头部控件 / 随机化配置表单 --")
for k in CHROME:
    out.append(f'  {json.dumps(k, ensure_ascii=False)}: {json.dumps(CHROME[k], ensure_ascii=False)},'
               f'  // :{lineno_of(k)}')

body = "\n".join(out)
open(os.path.join(HERE, "token_maker_ui.snippet.js"), "w", encoding="utf-8").write(body + "\n")
print(f"\n写出 token_maker_ui.snippet.js（{len(allkeys)} 键）；自证：OK")
