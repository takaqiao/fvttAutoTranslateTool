# -*- coding: utf-8 -*-
"""把三个候选文件 + 库里既有译名，渲染成 GLOSSARY-061.md 的表格段。
散文段在 GLOSSARY-061.head.md / .tail.md 里手写，这里只负责表。"""
import json, os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f), encoding='utf-8'))
lib = L('lib_en2cn.json')
uniq = L('names_uniq.json')
c1, c2, c3 = L('candidates.json'), L('candidates_extra.json'), L('candidates_lang_hardcoded.json')

CJK = re.compile(r'[㐀-鿿]')


def best(en):
    v = lib.get(en) or {}
    return max(v.items(), key=lambda kv: kv[1])[0] if v else None


def form(cn, role):
    return cn if role == 'cn' else cn


def rows(d):
    out = []
    for k, v in d.items():
        if k.startswith('_'):
            continue
        en = re.sub(r'（.*?）$', '', k)
        out.append((k, en, v[0], v[1], v[2]))
    return out


BUCKETS = [
    ("人名（新角色）", ["Kyrban", "Maevren", "Verno Kreed", "Proctus Caylas"]),
    ("生物 / 敌手", ["Grayling", "Shadebranch", "Vespinoth", "Amerasp", "Amerasp Queen",
                  "Amerasp Swarm", "Casir Cat", "Doomsayer Acolyte", "Doomsayer Shadewright"]),
    ("地名 / 区域 / 结构名", ["Ruby Grove", "Amerasp Caves", "Kyrban’s Chancery", "Doomsayer Barracks",
                       "Vespinoth Pens", "Vespiary", "Atrium Floor", "Ground Atrium Balcony",
                       "Upper Atrium Balcony", "Storage Chamber", "Dungeon", "Corrupted Nest",
                       "Chapter 4 Events", "Patch 0.6.1"]),
    ("任务页 / 事件页", ["Ill Met in Talei", "Local Experts", "The Primordial Grove", "Visions of Doom",
                   "Where Shadows Lie", "Return to Talei", "Capricious Concerns", "The Ruby Trade",
                   "Just a Taste", "Family Feud", "Proving Your Metal", "Slipping Into Darkness",
                   "Traversing the Bloodwoods"]),
    ("物品 / 装备", ["Ambral Bronze", "Grayling Dust", "White Aspen Wand", "Shadebranch Missive",
                 "The Primordial Fragments", "Hodge's Heirloom", "Queen's Ruby",
                 "Kadra Zann Skeleton Key", "Kyrban's Chancery Key", "Vespine Domino",
                 "Amber Staff", "Amber Dart", "Amber Dirk", "Amber Sputum", "Silver Sword",
                 "Pitchvine Net", "Hunting Trap", "Doomsayer Trail Map (1)",
                 "Doomsayer Trail Map (2)", "Doomsayer Trail Map (3)",
                 "Doomsayer Trail Map (Complete)", "Nimaelle's Blessing"]),
]


def emit(title, keys, d, used):
    got = [r for r in rows(d) if r[0] in keys]
    for r in got:
        used.add(r[0])
    if not got:
        return ""
    s = f"\n### {title}\n\n| EN | CN（可直接照抄） | 依据 / 出处 |\n|---|---|---|\n"
    for k, en, cn, role, why in got:
        s += f"| `{en}` | **{cn}{'' if role=='cn' else ' ' + en}** | {why} |\n"
    return s


out = []
used = set()
for t, ks in BUCKETS:
    out.append(emit(t, ks, c1, used))

rest = [r for r in rows(c1) if r[0] not in used]
if rest:
    out.append("\n### 能力 / 攻击 / 法术 / 效果 / 宏 / outcome 标签\n\n"
               "| EN | CN（可直接照抄） | 依据 / 出处 |\n|---|---|---|\n")
    for k, en, cn, role, why in rest:
        out.append(f"| `{en}` | **{cn}{'' if role=='cn' else ' ' + en}** | {why} |\n")

out.append("\n### 1122 叶之外的名称位（folders / categories / levels / notes / sounds / encounterTokens，共 192 叶）\n\n"
           "| EN | CN（可直接照抄） | 形态 | 依据 / 出处 |\n|---|---|---|---|\n")
for k, en, cn, role, why in rows(c2):
    out.append(f"| `{k}` | **{cn}{'' if role=='cn' else ' ' + en}** | {'纯中文' if role=='cn' else '双语'} | {why} |\n")

out.append("\n### lang/cn.json 与 ember-hardcoded-cn.mjs（compendium 之外的两条通道）\n\n"
           "| EN | CN（可直接照抄） | 依据 / 出处 |\n|---|---|---|\n")
for k, en, cn, role, why in rows(c3):
    out.append(f"| `{en}` | **{cn}** | {why} |\n")

# 已有译名（直接沿用，不许重译）
known = [(x, best(x)) for x in uniq if x in lib]
out.append(f"\n## 附表 B：已有译名 —— 直接沿用，不许重译（{len(known)} 条）\n\n"
           "这 {n} 条英文在 0.6.0 的库里已经有译名。第二阶段十个译者**一律照抄**，"
           "任何一条被重译都会在主闸的 `same_en_split` 上炸。\n\n".format(n=len(known)))
out.append("| EN | 库内既定 CN |\n|---|---|\n")
for en, cn in known:
    out.append(f"| `{en}` | {cn} |\n")

open(os.path.join(HERE, 'GLOSSARY-061.tables.md'), 'w', encoding='utf-8').write(''.join(out))
print("tables written; 主表", len(rows(c1)), "额外", len(rows(c2)), "lang/hc", len(rows(c3)), "沿用", len(known))
