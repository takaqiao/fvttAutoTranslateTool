# -*- coding: utf-8 -*-
"""断言路探针：先把每条候选闸的命中数/违规数量出来，再决定 JSON 怎么写。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {"ember": "1-Ember汉化插件", "crucible": "2-Crucible汉化插件"}

def walk(n, p=""):
    if isinstance(n, dict):
        for k, v in n.items():
            yield from walk(v, f"{p}.{k}" if p else k)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            yield from walk(v, f"{p}[{i}]")
    elif isinstance(n, str):
        yield p, n

def pairs(repo_dir):
    en_dir = os.path.join(repo_dir, "compendium", "en")
    cn_dir = os.path.join(repo_dir, "compendium", "cn")
    for f in sorted(os.listdir(en_dir)):
        if not f.endswith(".json") or f == "_source.json":
            continue
        cp = os.path.join(cn_dir, f)
        if not os.path.exists(cp):
            continue
        en = dict(walk(json.load(open(os.path.join(en_dir, f), encoding="utf-8-sig"))))
        cn = dict(walk(json.load(open(cp, encoding="utf-8-sig"))))
        for p, ev in en.items():
            cv = cn.get(p)
            if cv is not None:
                yield f, p, ev, cv

ALL = []
for name, rel in REPOS.items():
    for row in pairs(os.path.join(ROOT, rel)):
        ALL.append((name,) + row)
print("叶对", len(ALL))

def gate(pat, req=None, forbid=(), flags=re.I, show=0, label=""):
    r = re.compile(pat, flags)
    hits = 0; bad = []
    for repo, pack, path, ev, cv in ALL:
        if not r.search(ev):
            continue
        hits += 1
        if req and req not in cv:
            bad.append((repo, pack, path, "缺" + req, cv[:70]))
        for f in forbid:
            if f in cv:
                bad.append((repo, pack, path, "含禁用" + f, cv[:70]))
    print(f"\n[{label or pat}] 命中 {hits} 叶，违规 {len(bad)}")
    for b in bad[:show]:
        print("   ", b[0], b[1], b[2][:70], "|", b[3], "|", b[4])
    return hits, bad

gate(r"\bTokens?\b", "指示物", ("令牌", "代币"), show=12, label="Token")
gate(r"\bCyclonic\b", "气旋", ("旋风的",), show=8, label="Cyclonic")
gate(r"\bOrb Destroyed\b", "法珠已摧毁", ("法珠被摧毁", "法珠已毁"), show=8, label="OrbDestroyed")
gate(r"\bLantyr", "兰提尔", ("兰蒂尔",), show=8, label="Lantyr")
gate(r"\bTemple Lunarium\b", "神殿月辉宫", ("月神殿",), show=8, label="TempleLunarium")
gate(r"\bhex(es)?\b", "六边格", ("六角格",), show=10, label="Hex")
gate(r"\bCorla\b", "科尔拉", (), show=8, label="Corla")
gate(r"\bCora\b", "科拉", (), show=8, label="Cora")
gate(r"\bObsidian Antiquar\w*", "黑曜石", ("黑曜古",), show=8, label="ObsidianAntiquar")
gate(r"\bShard God\b", "碎片之神", (), show=10, label="ShardGod")
gate(r"\bShard Gods\b", "碎片诸神", (), show=10, label="ShardGods")
gate(r"\bShard Goddess\b", "碎片女神", (), show=10, label="ShardGoddess")
gate(r"\b(Fear|Command) Aura\b", "灵气", (), show=10, label="Fear/CommandAura")
gate(r"\bAura of Life\b", "灵气", (), show=10, label="AuraOfLife")
gate(r"\bAuric\b", "奥拉的", (), show=10, label="Auric")
gate(r"\bWhirlwind\b", "旋风", (), show=6, label="Whirlwind")
