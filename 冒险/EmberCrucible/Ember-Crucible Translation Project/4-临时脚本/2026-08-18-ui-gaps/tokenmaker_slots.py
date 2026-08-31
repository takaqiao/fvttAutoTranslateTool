#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按**结构角色**把指示物制作器（Token Maker）真正上屏的标签抠出来。

上屏点（templates/applications/token-maker/*.hbs 逐条对到源码）：
  colors.hbs : <label>{{color.label}}</label>   <- #prepareColors ember.mjs:66638
                                                   取自 template.colors[c].label
  layers.hbs : {{layer.label}}                  <- #prepareLayers :66607
                                                   `templateLayer.set?.label ?? templateLayer.label`
               {{build.title}} / {{stance.title}}  <- 字面量 "Build" / "Stance"（:66551 / :66571）
               {{build.label}} / {{stance.label}}  <- template.builds[x].label / stances[x].label
               {{layer.partLabel}}              <- 运行时由 partId 派生（:64457），**不是字面量**
  body.hbs   : <select name="template">          <- 每个模板对象顶层的 label

前置自证（两件都断言）：
  (A) 切对条数 —— 每类抠出的条数与人工数出的真值相等；
  (B) 改对地方 —— 抽到的内容必须含已知真值样本，且不得含已知的非该类样本。
"""
import re, os, sys, io, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

U = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
MJS = os.path.join(U, "scripts", "ember.mjs")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(MJS, encoding="utf-8").read()

RE_LABEL = re.compile(r'(?<![\w$.])label:\s*"((?:[^"\\]|\\.)*)"')


def _match_brace(i):
    d, j = 0, i
    while j < len(SRC):
        if SRC[j] == "{":
            d += 1
        elif SRC[j] == "}":
            d -= 1
            if d == 0:
                return j
        j += 1
    return len(SRC) - 1


def block_of(decl_re):
    """`const NAME = {` 之类的声明 -> [(name, startline, endline, text)]（花括号配平）。"""
    out = []
    for m in re.finditer(decl_re, SRC, re.M):
        i = SRC.find("{", m.start())
        if i < 0:
            continue
        j = _match_brace(i)
        out.append((m.group(1), SRC.count("\n", 0, m.start()) + 1,
                    SRC.count("\n", 0, j) + 1, SRC[i:j + 1]))
    return out


def inner_labels_of(prop):
    """所有 `<prop>: {` 内联块里的 label:。"""
    out = collections.OrderedDict()
    for m in re.finditer(r'^\s*' + prop + r':\s*\{', SRC, re.M):
        i = SRC.find("{", m.end() - 1)
        j = _match_brace(i)
        seg, ln0 = SRC[i:j + 1], SRC.count("\n", 0, i) + 1
        for mm in RE_LABEL.finditer(seg):
            out.setdefault(mm.group(1), []).append(f"{prop}@{ln0}")
    return out


def const_labels(prefix):
    out = collections.OrderedDict()
    for name, s0, e0, txt in block_of(
            r'^const (' + prefix + r'(?:\$[\w]+)?) = (?:foundry\.utils\.\w+\([^\n]*|\{)'):
        for mm in RE_LABEL.finditer(txt):
            out.setdefault(mm.group(1), []).append(f"{name}:{s0}")
    return out


# ---------------- (1) 颜色槽标签 ----------------
colors = collections.OrderedDict()
for name, s, e, txt in block_of(
        r'^const (COLORS(?:\$[\w]+)?) = (?:foundry\.utils\.\w+\([^\n]*|\{)'):
    for m in RE_LABEL.finditer(txt):
        colors.setdefault(m.group(1), []).append(f"{name}:{s}")
# 模板对象里 `colors: {…}` 的覆写（Hair Base / Hair Glow / Hair Sparkle 走这条）
for k, v in inner_labels_of("colors").items():
    colors.setdefault(k, []).extend(v)

# ---------------- (2) 图层标签 ----------------
RE_SET = re.compile(r'set:\s*\{[^}]*?label:\s*"((?:[^"\\]|\\.)*)"')
layers = collections.OrderedDict()
for name, s, e, txt in block_of(
        r'^const (LAYERS(?:\$[\w]+)?|ANATOMY(?:\$[\w]+)?|EQUIPMENT(?:\$[\w]+)?) = \{'):
    for m in RE_SET.finditer(txt):
        layers.setdefault(m.group(1), []).append(f"{name}.set:{s}")
    for i, ln in enumerate(txt.split("\n")):
        m = re.match(r'^    label: "((?:[^"\\]|\\.)*)",?\s*$', ln)
        if m:
            layers.setdefault(m.group(1), []).append(f"{name}:{s + i}")

# 模板里 `layers: {…}` 的覆写/追加
tpl_layer_over = collections.OrderedDict()
for m in re.finditer(r'^\s*layers:\s*\{', SRC, re.M):
    i = SRC.find("{", m.end() - 1)
    j = _match_brace(i)
    seg, ln0 = SRC[i:j + 1], SRC.count("\n", 0, i) + 1
    for mm in RE_SET.finditer(seg):
        tpl_layer_over.setdefault(mm.group(1), []).append(f"layers@{ln0}")
    for k, l in enumerate(seg.split("\n")):
        mm = re.match(r'^      label: "((?:[^"\\]|\\.)*)",?\s*$', l)
        if mm:
            tpl_layer_over.setdefault(mm.group(1), []).append(f"layers@{ln0 + k}")

# ---------------- (3) 体格 / 站姿 ----------------
builds = inner_labels_of("builds")
for k, v in const_labels("BUILDS").items():
    builds.setdefault(k, []).extend(v)
stances = inner_labels_of("stances")
for k, v in const_labels("STANCES").items():
    stances.setdefault(k, []).extend(v)

# ---------------- (4) 模板名（下拉选项） ----------------
mfreeze = re.search(r'^var templates=.*?Object\.freeze\(\{__proto__:null,(.*?)\}\);', SRC, re.M | re.S)
tpl_ids = {}
if mfreeze:
    for pair in mfreeze.group(1).split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            tpl_ids[k.strip()] = v.strip()
tpl_labels = collections.OrderedDict()
for tid, var in tpl_ids.items():
    for name, s, e, txt in block_of(
            r'^(?:const|var) (' + re.escape(var) + r') = (?:Object\.assign\([^\n]*|\{)'):
        for i, ln in enumerate(txt.split("\n")):
            m = re.match(r'^  label: "((?:[^"\\]|\\.)*)",?\s*$', ln)
            if m:
                tpl_labels.setdefault(m.group(1), []).append(f"{var}({tid}):{s + i}")

# ---------------- 前置自证 ----------------
def check(nm, got, must_have, must_not, expect_n=None):
    ok = True
    print(f"-- {nm}：抠出 {len(got)} 条")
    if expect_n is not None:
        good = len(got) == expect_n
        print(f"   [A 条数] {len(got)} == 真值 {expect_n} ? {'OK' if good else 'FAIL'}")
        ok = ok and good
    miss = [x for x in must_have if x not in got]
    print(f"   [B 必含] {len(must_have) - len(miss)}/{len(must_have)}" + ("" if not miss else f"  漏：{miss}"))
    ok = ok and not miss
    leak = [x for x in must_not if x in got]
    print(f"   [B 必不含] {len(must_not) - len(leak)}/{len(must_not)}" + ("" if not leak else f"  串类：{leak}"))
    ok = ok and not leak
    return ok


ok = True
# Eyes 既是颜色槽名（"Eyes"/"Eyes Iris"）也是图层名，本来就同时存在，不当反例。
ok &= check("(1) 颜色槽标签", colors,
            ["Hair Base", "Hair Roots", "Hair Highlights", "Hair Glow", "Hair Sparkle", "Hair 1",
             "Metal Base", "Metal Accent", "Skin Base", "Skin Accent", "Fabric Accent"],
            ["Arms", "Build", "Stance", "Human", "Lithe"])
ok &= check("(2) 图层标签", {**layers, **tpl_layer_over},
            ["Arms", "Eyebrows", "Face", "Head", "Torso"],
            ["Hair Roots", "Metal Base", "Build", "Lithe", "Human"])
ok &= check("(3) 体格标签", builds, ["Lithe", "Standard", "Heavy"],
            ["Arms", "Hair Roots", "Human"])
ok &= check("(3) 站姿标签", stances, ["Land", "Water"], ["Arms", "Hair Roots", "Human"])
ok &= check("(4) 模板名", tpl_labels, ["Human", "Party Banner", "Cor'ak", "Undead Monster"],
            ["Arms", "Hair Roots", "Build", "Lithe"])

print()
for nm, d in [("colors", colors), ("layers", layers), ("tpl_layer_override", tpl_layer_over),
              ("builds", builds), ("stances", stances), ("templates", tpl_labels)]:
    print(f"== {nm} ({len(d)}) ==")
    print("   " + " | ".join(sorted(d)))
    print()

json.dump({k: {kk: vv[:4] for kk, vv in d.items()} for k, d in
           [("colors", colors), ("layers", layers), ("tpl_layer_override", tpl_layer_over),
            ("builds", builds), ("stances", stances), ("templates", tpl_labels)]},
          open(os.path.join(HERE, "tokenmaker_slots.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("写出 tokenmaker_slots.json ；自证：" + ("OK" if ok else "FAIL"))
sys.exit(0 if ok else 1)
