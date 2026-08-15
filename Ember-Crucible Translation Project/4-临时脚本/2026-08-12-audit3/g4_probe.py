# -*- coding: utf-8 -*-
"""G4 探针：看「叶级对照」这几个证据层各自能给出多少东西。

只读。用来决定 scan_cross_channel.py 的 A/C 段候选来源怎么定，不参与最终判据。
"""
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path("C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project")
spec = importlib.util.spec_from_file_location("scc", P / "3-常用脚本/qa/scan_cross_channel.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

leaves = m.load_leaves([str(P / "1-Ember汉化插件"), str(P / "2-Crucible汉化插件")], [])
print("leaves", len(leaves))

LATIN_TAIL = re.compile(r"[\s\u3000]*[（(\[]?[A-Za-z][A-Za-z0-9'’\-.,:;!?&/ ]*[)\]）]?\s*$")
LATIN_HEAD = re.compile(r"^\s*[A-Za-z][A-Za-z0-9'’\-.,:;!?&/ ]*[\s\u3000]+")
CJKW = re.compile(r"^[一-鿿]{1,4}(?:·[一-鿿]{1,6})*$")


def strip_tail(s):
    s = m.MARKUP.sub(" ", s).strip()
    s2 = LATIN_TAIL.sub("", s).strip()
    return s2 or s


# ---- 层 a: 整叶等值对照（英文叶去标记后 == 术语）
exact = defaultdict(Counter)
for r in leaves:
    en = m.MARKUP.sub(" ", r[3]).strip()
    if not r[4] or not en or len(en) > 48:
        continue
    cn = strip_tail(r[4])
    if cn:
        exact[en][cn] += 1

# ---- 层 c: 双语并列锚（中文里出现「中文 English」）
def anchor(term):
    rx = re.compile(r"([一-鿿·“”]{2,14})[\s\u3000]*[（(]?" + re.escape(term) + r"(?![A-Za-z])")
    c = Counter()
    for r in leaves:
        if not r[4]:
            continue
        for g in rx.findall(r[4]):
            c[g.strip("“”")] += 1
    return c

# ---- 层 b: UUID 标签对照
UU = re.compile(r"@(?:UUID|Embed)\[([^\]]*)\]\{([^}]*)\}")
uuid_pairs = defaultdict(Counter)
for r in leaves:
    if not r[4]:
        continue
    en_l = {t: lab for t, lab in UU.findall(r[3])}
    cn_l = {t: lab for t, lab in UU.findall(r[4])}
    for t, lab in en_l.items():
        if t in cn_l:
            uuid_pairs[lab.strip()][strip_tail(cn_l[t])] += 1

TESTS = ["Ward", "Aspect", "Shoddy", "Fan", "Blast", "Cone", "Ray", "Surge", "Conjure",
         "Social Event", "Reactive", "Total", "Currency", "Formula", "Stride", "Steading",
         "Temple Ward", "Blast Flask", "Begin Event", "Complete Event", "Choices",
         "Extend", "Pull", "Vocal", "Masterwork", "Superior", "Fine", "Standard"]
for t in TESTS:
    print(f"\n=== {t}")
    print("  exact-leaf :", dict(exact.get(t, Counter()).most_common(6)))
    print("  uuid-label :", dict(uuid_pairs.get(t, Counter()).most_common(6)))
    print("  bilingual  :", dict(anchor(t).most_common(6)))

# ---- 词表规模 & 关键反例是否在词表里
lex = Counter()
SEP = re.compile(r"[，,。；;：:！？!?…、（）()【】《》\[\]|/\\\n\r\t“”‘’\"'—–\-~]+")
for en, cs in exact.items():
    for cn, n in cs.items():
        for part in SEP.split(cn):
            part = part.strip()
            if 2 <= len(part) <= 12 and re.fullmatch(r"[一-鿿·]{2,12}", part):
                lex[part] += n
for lab, cs in uuid_pairs.items():
    for cn, n in cs.items():
        for part in SEP.split(cn):
            part = part.strip()
            if 2 <= len(part) <= 12 and re.fullmatch(r"[一-鿿·]{2,12}", part):
                lex[part] += n
print(f"\n词表条目（整叶对照 + UUID 标签，切分后）：{len(lex)}")
for w in ["会替", "品质", "神殿区", "爆炸瓶", "师级", "标准", "粗糙", "防护", "化相",
          "社交事件", "社交活动", "反应式", "反应的", "扇形", "扇子", "步幅", "跨步",
          "三道", "维持秩序", "配方", "区块中", "可以作", "尺锥", "裂的弧形", "奔涌",
          "庄园", "耕耘", "成该类型"]:
    print(f"  {w}: {lex.get(w, 0)}")
