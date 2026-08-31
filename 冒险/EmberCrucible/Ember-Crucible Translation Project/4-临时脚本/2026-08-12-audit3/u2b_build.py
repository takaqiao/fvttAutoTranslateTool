# -*- coding: utf-8 -*-
"""Build the U2 (findings 76..150) batches.

Two kinds of edit, both computed against the CURRENT compendium/cn (the audit-3
batches are already in the working tree) and both emitted as the *whole* leaf:

  LBL  inside `@UUID[<target ending in KEY>]{OLD}` only the {label} changes.
       The bracket body is copied byte for byte.
  STR  a plain string replacement inside a leaf, gated on the ENGLISH leaf
       matching a regex, so a Chinese form is only touched where the English
       really carries that word.

Every rule states the target form and the evidence for it; see findings/U2.md.
"""
import json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
OUT = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"
REPOS = ["1-Ember汉化插件"]
SKIP = {"_id", "path", "_variants", "_when"}
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\]\{([^{}]*)\}')

# --- label rules: key -> {old: new} -------------------------------------------------
LBL = {
    # Barrel of Blast Powder; CN name 爆炸火药桶. EN prose: "a locked @UUID[...]; an image
    # of a blast flask is painted on the barrel" — 爆炸瓶 is a *different* item (KdlUUFhAnpv6YIHT).
    "VCZEiWaAI3jHiDLU": {"爆炸瓶箱": "爆炸火药桶"},
    # Metal Crumbler; CN name 碎金者, and 8:4 in labels.
    "qvLD88iUi3FBdkdc": {"金属碎解剂": "碎金者"},
    # Rune-Marked Arrowhead; CN name 符文刻印箭头 (whole Rune-Marked family is 符文刻印).
    "w7jXkPu7MheM6bkw": {"符文标记箭头": "符文刻印箭头", "符文印记箭头": "符文刻印箭头"},
    # Rune-Marked Sash; CN name 符文刻印腰带.
    "OJRVuMrUlSAM3soe": {"符文标记绶带": "符文刻印腰带", "符文印记腰带": "符文刻印腰带"},
    # Nineteen Nights in Haxim; CN name 哈克西姆的十九个夜晚, prose 哈克西姆 6 leaves : 哈西姆 4.
    "wlTZjyyW8K69yf56": {"《哈西姆的十九夜》": "《哈克西姆的十九夜》"},
    # Sunalin deity page; CN name 苏纳林. 苏纳林斯 transliterates the plural -s.
    "6cMQOhdBouTFJlSl": {"苏纳林斯": "苏纳林", "苏纳林诸神": "苏纳林"},
    # Chasm Candle; CN name 裂隙蜡烛.
    "pCl2T09DRBETkkbZ": {"裂谷蜡烛": "裂隙蜡烛"},
}

# --- string rules: (cn_old, cn_new, english gate regex) -----------------------------
STR = [
    # Wandren is the family name (House Wandren 万德伦家族, Vitt/Juro/Hephiss Wandren,
    # Wandren Tracer 万德伦追踪者). 万德伦 379+437 : 旺德伦 6+5, 流浪 is "wandering".
    ("流浪注视者", "万德伦注视者", r"Wandren"),
    ("巡逻者旺德伦", "万德伦巡逻者", r"Wandren Patroller"),
    ("旺德伦", "万德伦", r"Wandren"),
    # Ordani: 奥尔达尼 647+684 : 奥达尼 10+10; every other Ordani* name is 奥尔达尼.
    ("奥达尼", "奥尔达尼", r"Ordani"),
    # Chiaroscuran Beast: label census 明暗野兽 9 : 明暗兽 6 : 基亚罗斯库兰野兽 1,
    # and the actor carried three spellings at once (name/tokenName/labels).
    # gate is `Chiaroscur` on purpose: upstream also spells it `Chiaroscurian` (tokenName).
    ("基亚罗斯库兰野兽", "明暗野兽", r"Chiaroscur"),
    ("基亚洛斯库里安野兽", "明暗野兽", r"Chiaroscur"),
    ("明暗兽", "明暗野兽", r"Chiaroscur"),
    # Rask Juvenile: 幼年拉斯克 14+14 : 拉斯克幼体 3+3, and "juvenile Rask" in prose is
    # already 幼年拉斯克. Same shape as PROJECT.md's Young Cheliceraeth ruling.
    ("拉斯克幼体", "幼年拉斯克", r"[Jj]uvenile"),
]


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else None))


def main():
    batches = {}
    log = Counter()
    samples = []
    for repo in REPOS:
        ed = os.path.join(ROOT, repo, "compendium", "en")
        for fn in sorted(os.listdir(ed)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(ed, fn), encoding="utf-8"))
            cp = os.path.join(ROOT, repo, "compendium", "cn", fn)
            if not os.path.isfile(cp):
                continue
            cn = json.load(open(cp, encoding="utf-8"))
            rows = []
            walk(en.get("entries", {}), cn.get("entries", {}), [], rows)
            for p, e, c in rows:
                if not c:
                    continue
                new = c
                # 1. label rules
                for key, table in LBL.items():
                    if key not in e:
                        continue

                    def sub(m, table=table, key=key):
                        tgt = (m.group(2) or "").split()[0].split("#")[0]
                        if tgt.split(".")[-1] != key:
                            return m.group(0)
                        lab = m.group(3)
                        if lab in table:
                            log[(key, lab, table[lab])] += 1
                            return f"{m.group(1)}[{m.group(2)}]{{{table[lab]}}}"
                        return m.group(0)
                    new = MARK.sub(sub, new)
                # 2. string rules
                for old, rep, gate in STR:
                    if old in new and re.search(gate, e):
                        n = new.count(old)
                        new = new.replace(old, rep)
                        log[("STR", old, rep)] += n
                if new != c:
                    batches.setdefault((repo, fn), {})[p] = new
                    samples.append((fn, p))
    os.makedirs(OUT, exist_ok=True)
    for (repo, fn), d in sorted(batches.items()):
        tag = "ember" if repo.startswith("1-") else "crucible"
        path = os.path.join(OUT, f"U2__{tag}__{fn}")
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("wrote", path, len(d), "leaves")
    print("---- replacements ----")
    for k, v in sorted(log.items(), key=lambda x: -x[1]):
        print(f"  {v:4}  {k[0][:20]:22} {k[1]!r} -> {k[2]!r}")


main()
