# -*- coding: utf-8 -*-
"""
dangling_id_probe.py  —— 只读探针

判据（把「Moiré/Borel/Kost 死键 + 裸标记外泄」抽象成一条可机械化规则）：

    汉化侧（合集正文 或 插件代码）引用了一个**上游注册表里不存在 / 形态对不上**
    的标识符 → 该引用永远不生效：
      A 类  合集正文里的富文本标记 id 查不到 → 增强器 `return new Text(match)`，
            玩家看到字面量 `[[/xxx yyy]]`
      B 类  运行时替换表的键与上游**实际拼出的字符串形态**对不上 → 键永不触发，
            界面/正文外泄英文

两类的共同点：现有判据（markup_drift / uuid_swap / prune_dead / lang_gap）
全都只在「中英两侧之间」比对，而这两类**中英两侧完全一致**（英文侧本来就这样），
所以任何双侧 diff 判据都看不见。

只读：不写库内任何文件。

假阳性模式（必须人工复核）
--------------------------
 A1  `<sub data-system="dnd5e">` 分支里的标记在 crucible 下会被
     `finalizeEnrichedHTML`（ember.mjs:23219）整块删掉，不显示 → 不算缺陷。
     本脚本按最近的 data-system 祖先做归属，但只做**字符串层面**的近似切分。
 A2  某些 id 走 alias（crucible.CONFIG.knowledge[*].aliases），已按 alias 放行。
 C1  C 段对**两个仓库的所有包**都用 ember.crucible-character 那份 identifier 白名单。
     dnd5e 侧（ember.adventure）真正该用的是 ember.character 的白名单（164 条）；
     单独用 ident_character.txt 复核过，dnd5e 孪生包同样是 41 处，结论不变，
     但要拿 C 段跑 dnd5e 侧时请换白名单文件。
 B1  上游可能用模板拼接，字面量搜不到不等于不存在；B 段只报**已逐行读过
     产出点源码**的条目，不做纯 grep 推断。
"""
from __future__ import annotations
import json, os, re, sys, io, collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_REPO = os.path.join(PROJ, "1-Ember\u6c49\u5316\u63d2\u4ef6")
CRUC_REPO = os.path.join(PROJ, "2-Crucible\u6c49\u5316\u63d2\u4ef6")
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
EMBER_MJS = os.path.join(DATA, "modules", "ember", "scripts", "ember.mjs")
DND_MJS = os.path.join(DATA, "modules", "ember", "scripts", "dnd5e-async.mjs")
CRUC_MJS = os.path.join(DATA, "systems", "crucible", "crucible-compiled.mjs")

# ---------------------------------------------------------------- registries
# crucible.CONFIG.languages = crucible LANGUAGES (common/sign) + ember 的 23 条
# 见 crucible/module/const/actor.mjs:392 与 ember.mjs:126690-126717
LANGUAGES = {"common", "sign", "arcden", "cascal", "forest", "hardac", "imperial",
             "solical", "mithia", "luma", "kaziric", "scripta", "wyrdic", "pathward",
             "scor", "towyr", "windclaw", "abyssal", "draconic", "druidic", "lunix",
             "caligon", "eonic", "harmos", "cant"}
# crucible DEFAULT_KNOWLEDGE 31 条 - outsiders + ember 4 条；outsiders 保留为 abyssals 的 alias
KNOWLEDGE = {"alchemy", "ancients", "artifacts", "arts", "beasts", "celestials",
             "cosmology", "crafts", "crime", "dragons", "elementals", "fey", "fiends",
             "forensics", "gods", "intrigue", "legends", "machines", "monsters",
             "plants", "politics", "rituals", "seafaring", "souls", "subterranea",
             "tracking", "trade", "undeath", "warfare", "weather",
             "abyssals", "aedir", "leviathans", "shent"}
KNOWLEDGE_ALIASES = {"outsiders"}

ENRICHERS = {
    # name -> (regex, registry or None, 上游失败时的行为)
    "language":  (re.compile(r"\[\[/language (\w+)\]\]", re.A), LANGUAGES,  "new Text(match)"),
    "knowledge": (re.compile(r"\[\[/knowledge (\w+)\]\]", re.A), KNOWLEDGE | KNOWLEDGE_ALIASES, "new Text(match)"),
}

# 需要「表键 vs 上游拼串形态」对表的 B 类产出点（逐条读过源码）
B_SITES = [
    # (标记正则, 上游产出的可见字符串模板, 产出点, 我们表里的键形态)
    (re.compile(r"\[\[/ancestry (\w+)\]\]", re.A), "Ancestry: {name}",  "ember.mjs:22932", 'EXACT["Ancestry"]'),
    (re.compile(r"\[\[/culture (\w+)\]\]", re.A),  "Culture: {name}",   "ember.mjs:22952", 'EXACT["Culture"]'),
    (re.compile(r"\[\[/path (\w+)\]\]", re.A),     "Path: {name}",      "ember.mjs:22984", 'EXACT["Path"]'),
    (re.compile(r"@Advantage\[(-?\d)\]", re.A),    "+{n} Boons / {n} Banes", "ember.mjs:22890", 'EXACT["±1..3 Boons/Banes"]'),
    (re.compile(r"\[\[/soundscape ([^\]]+)\]\]", re.A), "Music: {arrangement} / Music: Reset / Music Mood: {mood}",
     "ember.mjs:16252-16270", 'MOODS + PREFIXED["Music Mood"]'),
]


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(o, str):
        yield p, o


SWAP = re.compile(r'<(sub|div)\s[^>]*data-system="(\w+)"', re.I)


def system_branch(text, pos):
    """粗略判断 pos 处的标记落在哪个 data-system 分支里（假阳性模式 A1）。"""
    last = None
    for m in SWAP.finditer(text[:pos]):
        last = m.group(2)
        close = text.find("</%s>" % m.group(1), m.end())
        if close != -1 and close < pos:
            last = None
    return last


def scan(repo, side):
    d = os.path.join(repo, "compendium", side)
    rows = []
    if not os.path.isdir(d):
        return rows
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        for path, s in leaves(j):
            for kind, (rx, reg, fail) in ENRICHERS.items():
                for m in rx.finditer(s):
                    if m.group(1) in reg:
                        continue
                    rows.append({
                        "pack": fn, "side": side, "kind": kind, "id": m.group(1),
                        "branch": system_branch(s, m.start()),
                        "path": path, "snippet": s[max(0, m.start() - 70):m.end() + 40],
                    })
    return rows


def bscan(repo, side):
    out = collections.Counter()
    detail = collections.defaultdict(collections.Counter)
    d = os.path.join(repo, "compendium", side)
    if not os.path.isdir(d):
        return out, detail
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        for path, s in leaves(j):
            for rx, produced, site, key in B_SITES:
                for m in rx.finditer(s):
                    out[(fn, produced, site, key)] += 1
                    detail[(fn, site)][m.group(1)] += 1
    return out, detail




# ---------------------------------------------------------------- arm C
# C 段：[[/ancestry|culture|path ID]] 的 ID 必须等于 ember.crucible-character
# （crucible 下 ember.CONST.CHARACTER_OPTIONS_PACK，ember.mjs:123964）里某条
# Item 的 system.identifier。查不到时 enrichAncestry/Culture/Path 走
# `const name = ix?.name || ancestryId`（ember.mjs:22929/22949/22981），
# 把**原始英文 id 本身**当名字渲染出来。
# identifier 白名单由 dump_identifiers.mjs 从 LevelDB 直接读出（只读）。
IDENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ident_crucible_character.txt")
CHAR_OPTION_RX = {k: re.compile(r"\[\[/%s (\w+)\]\]" % k, re.A) for k in ("ancestry", "culture", "path")}


def cscan():
    if not os.path.exists(IDENT_FILE):
        print("  [SKIP] 先跑：node dump_identifiers.mjs <ember/packs/crucible-character> > ident_crucible_character.txt")
        return
    ids = set(open(IDENT_FILE, encoding="utf-8").read().split())
    print("  identifier 白名单 %d 条" % len(ids))
    for repo, tag in [(EMBER_REPO, "ember_cn"), (CRUC_REPO, "crucible-cn")]:
        d = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            bad = collections.Counter()
            where = {}
            for path, s in leaves(j):
                for k, rx in CHAR_OPTION_RX.items():
                    for m in rx.finditer(s):
                        if m.group(1) in ids:
                            continue
                        bad[(k, m.group(1))] += 1
                        where.setdefault((k, m.group(1)), (path, system_branch(s, m.start())))
            if not bad:
                continue
            print("  %s %s  共 %d 处" % (tag, fn, sum(bad.values())))
            for (k, i), n in sorted(bad.items(), key=lambda x: -x[1]):
                path, br = where[(k, i)]
                print("     %-9s %-26s x%-3d branch=%-7s %s" % (k, i, n, br, path))


def main():
    print("=" * 78)
    print("A 段：合集正文里 id 查不到注册表的富文本标记 → 渲染成字面量")
    print("=" * 78)
    allrows = []
    for repo, tag in [(EMBER_REPO, "ember_cn"), (CRUC_REPO, "crucible-cn")]:
        for side in ("cn", "en"):
            allrows += [dict(r, repo=tag) for r in scan(repo, side)]
    agg = collections.Counter((r["repo"], r["pack"], r["side"], r["kind"], r["id"], r["branch"]) for r in allrows)
    for k, n in sorted(agg.items()):
        print("  %-11s %-32s %-3s %-10s %-12s branch=%-7s x%d" % (k[0], k[1], k[2], k[3], k[4], k[5], n))
    print("\n  样本上下文：")
    seen = set()
    for r in allrows:
        key = (r["kind"], r["id"])
        if r["side"] != "cn" or key in seen:
            continue
        seen.add(key)
        print("   [%s %s] %s" % (r["kind"], r["id"], r["pack"]))
        print("     path=%s" % r["path"])
        print("     ...%s..." % r["snippet"].replace("\n", " "))

    print()
    print("=" * 78)
    print("B 段：上游拼出的可见串 vs 我们表里的键形态（逐条读过产出点源码）")
    print("=" * 78)
    for repo, tag in [(EMBER_REPO, "ember_cn"), (CRUC_REPO, "crucible-cn")]:
        cnt, det = bscan(repo, "cn")
        for (fn, produced, site, key), n in sorted(cnt.items()):
            print("  %-11s %-32s x%-5d %s" % (tag, fn, n, produced))
            print("               产出点 %-22s 表键 %s" % (site, key))
            print("               实参分布 %s" % dict(det[(fn, site)]))
    print()
    print("=" * 78)
    print("C 段：[[/ancestry|culture|path ID]] 在 ember.crucible-character 里查不到 identifier")
    print("=" * 78)
    cscan()


if __name__ == "__main__":
    main()
