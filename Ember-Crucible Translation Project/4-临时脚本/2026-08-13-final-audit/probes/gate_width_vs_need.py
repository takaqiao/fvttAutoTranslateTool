#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gate_width_vs_need.py  —  第十三轮「举一反三」探针

抽象出来的判据（种子实例：DialogV2 例外分支只放行 .window-title）：

    汉化运行时里每一个「判别式」（闸 / 选择器 / 键白名单 / 匹配器）都定义了
    一个 ADMITTED 集合；上游实际需要替换的字符串构成 NEEDED 集合。
    凡 NEEDED \ ADMITTED 非空，就是同一类缺陷。

本脚本对 4 个判别式各算一次差集，**只读**，不改任何库文件：

  G1  patchEnrichers 的 pattern 词表闸
        ADMITTED = 正则 /attunement|language|...|date/i 命中 String(entry.pattern) 的 enricher
        NEEDED   = ember + crucible 注册的、输出里含硬编码英文字面量的 enricher
  G2  translateText 的匹配器（EXACT 全串相等 / PREFIXED 前缀 / 5 条 PATTERNS）
        ADMITTED = 能被三者之一匹配的字符串
        NEEDED   = enricher 实际可能吐出的 label / tooltip 形状（人工从上游源码抄出，见 SHAPES）
  G3  translateNode 的属性白名单
        ADMITTED = ["data-tooltip","data-tooltip-text","data-tooltip-html","title","aria-label"]
        NEEDED   = 闸放行的 Ember 应用模板里，值含英文字面量的全部属性
  G4  patchCrucibleConfig 的分组白名单 ["languages","knowledge"]
        NEEDED   = ember 往 *.CONFIG.<组> 写入的、带裸英文 label 的全部分组

已知假阳性模式（读结果时必须扣掉）：
  * G1/G2：上游用 _loc()/localize 取值的 label 不是硬编码，可由 lang/cn.json 覆盖 —— 本脚本
    对 enricher 只做「源码里出现字符串字面量拼接」的静态判断，_loc() 一律不算 NEEDED。
  * G3：alt/placeholder 里的非英文（如 "#FFFFFF"）、纯 {{handlebars}} 表达式要人工剔除。
  * G3：模板归属靠「哪个类的 PARTS/STEPS/template 引用了它」反查，反查不到的记 owner=?
  * G2：SHAPES 是人工从上游抄的，不是自动抽取；改上游版本要重抄。
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
EMBER_MOD = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")
CRUCIBLE = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs")
PLUGIN = ROOT / "1-Ember汉化插件"
HARD = PLUGIN / "scripts" / "ember-hardcoded-cn.mjs"

# ---------------------------------------------------------------- 读闸定义

src_hard = HARD.read_text(encoding="utf-8")

ENRICHER_GATE = re.search(r"if \(!/([^/]+)/i\.test\(src\)\) continue;", src_hard).group(1)
ATTR_ALLOW = re.search(r'for \(const attr of \[([^\]]+)\]\)', src_hard).group(1)
ATTR_ALLOW = [a.strip().strip('"') for a in ATTR_ALLOW.split(",")]
CONFIG_GROUPS = re.findall(r'\[\["languages", LANGUAGES\], \["knowledge", KNOWLEDGE\]\]', src_hard)
APP_GATE = re.search(r"if \(!/ember/i\.test\(cls\) && !/\^Ember/\.test\(id\)\)", src_hard) is not None


def table(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src_hard, re.S)
    if not m:
        return {}
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', m.group(1)))


EXACT = table("EXACT")
ATTUNEMENTS = table("ATTUNEMENTS")
LANGUAGES = table("LANGUAGES")
KNOWLEDGE = table("KNOWLEDGE")
MOODS = table("MOODS")
PREFIXES = re.findall(r'\{ en: "([^"]+)", cn:', src_hard)
PATTERNS = re.findall(r"\{ re: (/[^,]+/), cn:", src_hard)


def translate_text_can_match(s):
    """复刻 translateText 的判别顺序，只回答「能不能被匹配到」。"""
    raw = s.strip()
    if raw in EXACT:
        return "EXACT"
    for en in PREFIXES:
        if raw.startswith(en + ": "):
            return "PREFIXED:" + en
    for pat in (r"^Result of (.+)$", r"^Award Attunement: (.+)$", r"^Revoke Attunement: (.+)$",
                r"^Activate Attunement: (.+)$", r"^Day (\d+)\b(.*)$"):
        if re.match(pat, raw):
            return "PATTERN:" + pat
    return None


# ---------------------------------------------------------------- G1 enricher 闸

ENR_RE = re.compile(r"pattern:\s*(/(?:[^/\\]|\\.)+/[gimsuy]*)\s*,\s*\n?\s*enricher:\s*([\w$.]+)")


def collect_enrichers(path, tag):
    txt = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in ENR_RE.finditer(txt):
        pat, fn = m.group(1), m.group(2)
        line = txt[: m.start()].count("\n") + 1
        out.append({"src": tag, "line": line, "pattern": pat, "enricher": fn,
                    "gate_admits": bool(re.search(ENRICHER_GATE, pat, re.I))})
    return out


enrichers = collect_enrichers(EMBER_MOD / "scripts" / "ember.mjs", "ember.mjs")
enrichers += collect_enrichers(EMBER_MOD / "scripts" / "dnd5e-async.mjs", "dnd5e-async.mjs")
enrichers += collect_enrichers(CRUCIBLE, "crucible-compiled.mjs")

# enricher 函数体里是否存在「英文字面量拼进 innerHTML/innerText/dataset.tooltip」
LIT = re.compile(r'(innerHTML|innerText|dataset\.tooltip|setAttribute\("aria-label")\s*(?:=|\+=|,)\s*'
                 r'[`"\']([^`"\']*[A-Za-z]{3}[^`"\']*)')


def body_of(txt, fn):
    m = re.search(r"^(?:async )?function %s\(" % re.escape(fn.split(".")[-1]), txt, re.M)
    if not m:
        return ""
    i = txt.index("{", m.start())
    depth, j = 0, i
    while j < len(txt):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                return txt[i:j + 1]
        j += 1
    return txt[i:i + 4000]


bodies = {"ember.mjs": (EMBER_MOD / "scripts" / "ember.mjs").read_text(encoding="utf-8", errors="replace"),
          "dnd5e-async.mjs": (EMBER_MOD / "scripts" / "dnd5e-async.mjs").read_text(encoding="utf-8", errors="replace"),
          "crucible-compiled.mjs": CRUCIBLE.read_text(encoding="utf-8", errors="replace")}

g1 = []
for e in enrichers:
    b = body_of(bodies[e["src"]], e["enricher"])
    lits = [m.group(2).strip() for m in LIT.finditer(b) if re.search(r"[A-Za-z]{3}", m.group(2))]
    lits = [x for x in lits if not x.startswith("$")]
    e["hardcoded_english"] = sorted(set(lits))
    if lits and not e["gate_admits"]:
        g1.append(e)

# ---------------------------------------------------------------- G2 输出形状

# 人工从上游源码抄出的 enricher 输出形状（用 <X> 表示动态段）
SHAPES = [
    ("ember.mjs:16255", "soundscape reset", "Music: Reset"),
    ("ember.mjs:16266", "soundscape arrangement", "Music: Ankarist Theme"),
    ("ember.mjs:16267", "soundscape arrangement+mood", "Music: Ancient Ruins (Tension)"),
    ("ember.mjs:16271", "soundscape mood", "Music Mood: Calm"),
    ("ember.mjs:4134", "date tooltip", "After Shattering - 108 Years Ago"),
    ("ember.mjs:4134", "date tooltip current", "After Shattering - Current Year"),
    ("ember.mjs:4133", "date label", "AS1716"),
    ("ember.mjs:22890", "advantage boons", "+2 Boons"),
    ("ember.mjs:22890", "advantage banes out-of-table", "-6 Banes"),
    ("ember.mjs:23008", "attunement", "Attunement: 深渊 The Abyss"),
    ("ember.mjs:23012", "attunement award", "Attunement: 深渊 The Abyss (+1)"),
    ("ember.mjs:23955", "npc attunement option", "Abyss Rank 3"),
    ("crucible:46838", "talent", "Talent: 弑龙者"),
    ("crucible:46857", "spell tooltip", "Spell tooltips are still TO-DO."),
]
g2 = [{"where": w, "kind": k, "sample": s, "matcher": translate_text_can_match(s)} for w, k, s in SHAPES]

# ---------------------------------------------------------------- G3 属性白名单

TPL = sorted((EMBER_MOD / "templates").rglob("*.hbs"))
owner_map = {}
for name, txt in bodies.items():
    for m in re.finditer(r'template:\s*"modules/ember/(templates/[^"]+)"', txt):
        # 往上找最近的 class 名
        head = txt[: m.start()]
        c = re.findall(r"class (\w+) extends", head)
        owner_map.setdefault(m.group(1), c[-1] if c else "?")

ATTR_RE = re.compile(r'\b([a-zA-Z-]+)="([^"]*)"')
g3 = []
for t in TPL:
    rel = t.relative_to(EMBER_MOD).as_posix()
    txt = t.read_text(encoding="utf-8", errors="replace")
    for m in ATTR_RE.finditer(txt):
        attr, val = m.group(1), m.group(2)
        if attr in ATTR_ALLOW:
            continue
        if attr in ("class", "src", "href", "type", "name", "data-action", "data-tab", "data-group",
                    "data-application-part", "id", "style", "value", "for", "data-attunement",
                    "data-tooltip-direction", "data-inventory-section", "data-drop-behavior"):
            continue
        core = re.sub(r"\{\{[^}]*\}\}", "", val).strip()
        if not re.search(r"[A-Za-z]{3}", core):
            continue
        if core.startswith("fa-") or core.startswith("#"):
            continue
        g3.append({"tpl": rel, "attr": attr, "value": val, "owner": owner_map.get(rel, "?"),
                   "in_attr_allowlist": False})

# ---------------------------------------------------------------- G4 CONFIG 分组

g4 = []
for m in re.finditer(r"(crucible\.CONFIG\.(\w+))\.(\w+)\s*=\s*\{label:\s*\"([^\"]+)\"", bodies["ember.mjs"]):
    grp, key, lab = m.group(2), m.group(3), m.group(4)
    line = bodies["ember.mjs"][: m.start()].count("\n") + 1
    g4.append({"line": line, "group": grp, "key": key, "label": lab,
               "covered_by_patchCrucibleConfig": grp in ("languages", "knowledge")})

# ---------------------------------------------------------------- 输出

report = {
    "gates_read_from_plugin": {
        "enricher_pattern_gate": ENRICHER_GATE,
        "attr_allowlist": ATTR_ALLOW,
        "config_groups": ["languages", "knowledge"],
        "app_gate_present": APP_GATE,
        "n_EXACT": len(EXACT), "prefixes": PREFIXES, "n_patterns": len(PATTERNS),
    },
    "G1_enrichers_with_english_outside_gate": g1,
    "G1_all_enrichers": enrichers,
    "G2_output_shapes_vs_matcher": g2,
    "G3_attrs_outside_allowlist": g3,
    "G4_config_groups": g4,
}
out = Path(__file__).with_name("gate_width_vs_need.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

print("闸定义：", json.dumps(report["gates_read_from_plugin"], ensure_ascii=False))
print()
print("== G1 输出含硬编码英文但闸不放行的 enricher ==")
for e in g1:
    print("  %-22s %s:%s  %s" % (e["enricher"], e["src"], e["line"], e["hardcoded_english"]))
print()
print("== G2 输出形状 vs translateText 匹配器 ==")
for r in g2:
    print("  %-34s %-46r -> %s" % (r["kind"], r["sample"], r["matcher"]))
print()
print("== G3 闸外属性（前 40）==")
for r in g3[:40]:
    print("  %-52s %-14s %-28s %s" % (r["tpl"], r["attr"], r["value"][:28], r["owner"]))
print("  ... 共", len(g3))
print()
print("== G4 crucible.CONFIG 分组 ==")
for r in g4:
    print("  L%-7s %-18s %-12s %-22s covered=%s" % (r["line"], r["group"], r["key"], r["label"],
                                                    r["covered_by_patchCrucibleConfig"]))
print("\n写出 ->", out)
