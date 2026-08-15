"""Y2 探针：把「按 HTML 标签切块、再逐块对齐」套到三个覆盖洞上，先量再写断言。

三个洞（原文见 RESOLUTIONS.assertions.json 各条 why）：
  ① R-rank-sense-compendium —— 混合叶 8 / 无法分类叶 57 不查；纯 GAME 叶不上正向闸
  ② R-shard-god            —— 同叶单复数并存的 70 叶结构上就不进闸
  ③ R-arcturel-vs-arcturian —— 96% 的叶两个词都有，叶级判据抓不到叶内单处串行

判据形态照 `2026-08-15-round16/probes/split_dives.py`：标签是机械的、两侧逐字节相同，
所以 `TAG.split()` 的块数两侧应当相等；不等的会被**报出来**而不是静默跳过。

跑法：python block_align_probe.py --repo <ember> --repo <crucible> [--case <id>] [--show N]
"""
import argparse
import collections
import json
import os
import re
import sys

# ⚠ split_dives 按**所有**标签切块，本探针实测那样切太细：中文「定语在前」会把词搬过
# `<strong>` 边界（`Cora Attunement.description`：EN 块13「damage equal to 2 times your
# attunement rank」的「attunement rank」在中文里搬到了块11「你获得等同于同调阶位 2 倍的」），
# 于是正向闸报出成片假阳性。所以只按**块级标签**切，行内标签（strong/em/span/sup/a…）剥成空格。
BLOCK_TAG = re.compile(
    r"</?(?:p|div|li|ul|ol|tr|td|th|table|thead|tbody|tfoot|caption|h[1-6]|br|hr|"
    r"section|article|aside|header|footer|blockquote|figure|figcaption|dl|dt|dd|pre)\b[^>]*>",
    re.IGNORECASE)
INLINE_TAG = re.compile(r"<[^>]+>")
# ⚠ 富文本增强器连**标签**一起涂掉（`@UUID[…]{标签}` 的花括号部分），与 split_dives 不同。
# 理由是实测：`A Brush With Death` 的英文是**裸** `@UUID[…]`（没有标签，Foundry 渲染目标名），
# 而中文补了 `{阿克图里安}`；不涂标签就会得到「EN 空 / CN 有」的假阳性。
# 标签本身另有闸看着（R-arcturian-split 的 `\{Arcturians\}` 域 · R-arcturian-actor-card ·
# scan_uuid_swap），不是没人管的地方。
MASKP = [re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?"), re.compile(r"\[\[[^\]]*\]\]")]


def mask(s):
    for p in MASKP:
        s = p.sub(lambda m: " " * len(m.group()), s)
    return s


def blocks(s):
    return [INLINE_TAG.sub(" ", b) for b in BLOCK_TAG.split(mask(s))]


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def load_pairs(repo_dir):
    en_dir = os.path.join(repo_dir, "compendium", "en")
    cn_dir = os.path.join(repo_dir, "compendium", "cn")
    if not os.path.isdir(en_dir):
        return
    for fname in sorted(os.listdir(en_dir)):
        if not fname.endswith(".json") or fname == "_source.json":
            continue
        cn_path = os.path.join(cn_dir, fname)
        if not os.path.exists(cn_path):
            continue
        en = dict(walk(json.load(open(os.path.join(en_dir, fname), encoding="utf-8-sig"))))
        cn = dict(walk(json.load(open(cn_path, encoding="utf-8-sig"))))
        for p, ev in en.items():
            cv = cn.get(p)
            if cv is not None:
                yield fname, p, ev, cv


# ────────────────────────────────────────────────────────────── 三个洞的配置
CASES = {
    # ② 单复数：叶级负向先行只看得住「只含一种形态」的叶；切块后同叶两种形态可以分开判。
    "shard": dict(
        mode="count_ge",
        # ⚠ 第五次栽在大小写上：英文对白里大量写小写 `a shard god` / `of all the shard gods`
        # （Planting a Seed 一叶就有 4 处）。大小写敏感版把它们全报成「EN 空 CN 有」= 79 块假阳性。
        # 「shard god」没有第二个义项，所以这条闸**必须** IGNORECASE。
        leaf_gate=r"\bShard God",
        case_sensitive=False,
        # `Shards Gods`（Shard 上多一个 s）是上游拼写事故，Deities.pages.Shard Gods.text 里 2 处；
        # 不进 token 表就会变成「EN 空 CN 有」的假阳性。
        en_tokens=r"\bShards? Goddess(es)?\b|\bShards? Gods\b|\bShards? God\b",
        cn_tokens=r"碎片女神|碎片诸神|碎片之神",
        # ⚠ 大小写：`goddess` 小写形态真实存在（`the shard goddess Scoris`），
        # 用 `"Goddess" in x` 判会漏掉它并误判成复数 P。必须先 lower()。
        # ⚠⚠ **单／复数不进类**：实测按 S/P 逐位对齐，779 块里 46 块不对齐，逐条看过
        # **全部是合法中文** —— 中文不标复数，且惯于把 `the Shard God X` 译成「碎片诸神之一的 X」、
        # 把 `three Shard Gods of Fire and four of Battle` 拆成「三位火焰之碎片之神和四位战斗之
        # 碎片之神」。也就是说 S↔P 逐位对齐**判据本身不成立**，不是译文错。
        # 中文真正承载得住的区分是 **女神 vs 神**，那一支一个都不许错。
        en_cls=lambda x: "F" if "goddess" in x.lower() else "G",
        cn_cls={"碎片女神": "F", "碎片诸神": "G", "碎片之神": "G"},
        backward=["F"],   # CN 出现「碎片女神」而 EN 块内没有 Goddess → 违规
    ),
    # ③ 城名 vs 族名：`Arcturel Dives` 整体专名的中文「阿克图瑞尔矿渊」含「阿克图瑞尔」，
    #    两侧都出一个 E，天然对齐，不必单列一类。
    "arct": dict(
        mode="sequence",
        # ⚠ 上游有五种拼写事故／变体，全部实测存在，不进 token 表就是假阳性：
        #   Acturel（漏 r）· Arctural · Arcurel · Arturel —— 都是城名
        #   Acturian（漏 r，Golden Flats/Talei/Brevin 各 1）· Arcturelian（Chessman/Woven
        #   Construct 各 1，＝「Arcturel 的」的另一种构词）—— 都是族名／文化形容词那一支
        leaf_gate=r"(?i)\b(Arcturel|Arcturel?ians?|Arcturians?|Acturel|Acturians?|Arctural|Arcurel|Arturel)\b",
        case_sensitive=False,
        en_tokens=(r"\bArcturel?ians?\b|\bActurians?\b|\bArcturians?\b"
                   r"|\bArcturel\b|\bActurel\b|\bArctural\b|\bArcurel\b|\bArturel\b"),
        cn_tokens=r"阿克图里安人|阿克图里安|阿克图瑞尔",
        # 以 -ian/-ians 收尾的一律是族名／文化形容词那一支（含 Acturian / Arcturelian）
        en_cls=lambda x: "I" if x.lower().rstrip("s").endswith("ian") else "E",
        cn_cls={"阿克图里安人": "I", "阿克图里安": "I", "阿克图瑞尔": "E"},
    ),
    # ① rank：块内单一义项才上闸。窗口收缩到块内，所以分类更保守（UNKNOWN 变多），
    #    换来的是混合叶/无法分类叶里那些**块内单一**的部分终于能判。
    "rank": dict(
        mode="sense",
        leaf_gate=r"(?i)\branks?\b",
        case_sensitive=False,
        occ=r"\branks?\b",
        cn_required="阶位",
        # 强机制义：块级切分后 COMMON 的 `ranks of` 会咬到 `Ranks of attunement progression`
        # （叶级时那一叶别处还有 GAME、落进 MIX 桶所以没暴露）。同调／魂印／Rank N 是本系统
        # 的机制专名，优先级必须高于 COMMON 的泛化措辞。
        strong_game=r"(attunement|attuned|soulbound|soulmark|Rank\s*\d)",
        game=(r"(Novice|Journeyman|Adept|Master|Untrained|training|skill|Attunement|Attuned|"
              r"Soulbound|Soulmark|talent|progress|superior to|Scale|Rank\s*\d|\bBonus\b)"),
        # `ranks depending on experience and skill`（Flame Guard 的组织内部等级）本是普通名词义，
        # 却因窗口里有 `skill` 被判成 GAME。补进 COMMON 的是**组织层级**这一实义线索。
        common=(r"(ranks? (depending|based) on|"
                r"civic|social|nobil|noble|militar|clerical|within the order|of the order|"
                r"stripped of|full rank of|ranks of|swell(ing)? the ranks|rank[- ]and[- ]file|"
                r"hierarch|station|office|through the ranks|within [^.]{0,30}\branks\b|"
                r"beyond its ranks|rose to ranks|\branks (after|and)\b|rank as an? )"),
        # 第三义项：这些**不归「阶位」那条裁决管**，块内出现即整块不判（不是放宽，是分类）
        exempt=r"(rank of exhaustion|ranks? of exhaustion|close ranks|join(ing)? their ranks)",
        window=90,
    ),
}


def run_sequence(cfg, rows, st, review):
    leaf_re = re.compile(cfg["leaf_gate"], 0 if cfg["case_sensitive"] else re.IGNORECASE)
    en_tok = re.compile(cfg["en_tokens"], 0 if cfg["case_sensitive"] else re.IGNORECASE)
    cn_tok = re.compile(cfg["cn_tokens"])
    for pack, path, ev, cv in rows:
        if not leaf_re.search(ev):
            continue
        st["leaf"] += 1
        eb, cb = blocks(ev), blocks(cv)
        if len(eb) != len(cb):
            st["shape"] += 1
            review.append((pack, path, -1, f"标签块数 {len(eb)}!={len(cb)}", "", ""))
            continue
        st["leaf_aligned"] += 1
        for i, (e, c) in enumerate(zip(eb, cb)):
            ee = [cfg["en_cls"](m.group()) for m in en_tok.finditer(e)]
            cc = [cfg["cn_cls"][m.group()] for m in cn_tok.finditer(c)]
            if not ee and not cc:
                continue
            st["block"] += 1
            if ee == cc:
                st["ok"] += 1
            else:
                st["bad"] += 1
                review.append((pack, path, i, "块内不对齐", "".join(ee), "".join(cc)))


def run_count_ge(cfg, rows, st, review):
    """中文各类计数**不得少于**英文（可多不可少），外加指定类的反向存在闸。

    比逐位对齐宽一格、比整叶细一个量级。宽的那一格换来的是不再把「代词还原」
    （英文 they/them → 中文点名「碎片诸神」）当成缺陷 —— 实测 13 块残差全是这一类，
    且**无一例外是中文多、英文少**。能抓的：整块漏译一处、把女神并进神、把某一类整支改名。
    """
    leaf_re = re.compile(cfg["leaf_gate"], 0 if cfg["case_sensitive"] else re.IGNORECASE)
    en_tok = re.compile(cfg["en_tokens"], 0 if cfg["case_sensitive"] else re.IGNORECASE)
    cn_tok = re.compile(cfg["cn_tokens"])
    for pack, path, ev, cv in rows:
        if not leaf_re.search(ev):
            continue
        st["leaf"] += 1
        eb, cb = blocks(ev), blocks(cv)
        if len(eb) != len(cb):
            st["shape"] += 1
            review.append((pack, path, -1, f"标签块数 {len(eb)}!={len(cb)}", "", ""))
            continue
        st["leaf_aligned"] += 1
        for i, (e, c) in enumerate(zip(eb, cb)):
            ec = collections.Counter(cfg["en_cls"](m.group()) for m in en_tok.finditer(e))
            cc = collections.Counter(cfg["cn_cls"][m.group()] for m in cn_tok.finditer(c))
            if not ec and not cc:
                continue
            st["block"] += 1
            ok = True
            for k, n in ec.items():
                if cc.get(k, 0) < n:
                    ok = False
                    review.append((pack, path, i, f"中文「{k}」类只有 {cc.get(k, 0)} 处、英文有 {n} 处",
                                   "".join(sorted(ec.elements())), "".join(sorted(cc.elements()))))
            for k in cfg.get("backward", []):
                if cc.get(k, 0) and not ec.get(k, 0):
                    ok = False
                    review.append((pack, path, i, f"中文出现「{k}」类而英文块内没有",
                                   "".join(sorted(ec.elements())), "".join(sorted(cc.elements()))))
            st["ok" if ok else "bad"] += 1


def run_sense(cfg, rows, st, review):
    leaf_re = re.compile(cfg["leaf_gate"], re.IGNORECASE)
    occ = re.compile(cfg["occ"], re.IGNORECASE)
    game = re.compile(cfg["game"], re.IGNORECASE)
    common = re.compile(cfg["common"], re.IGNORECASE)
    exempt = re.compile(cfg["exempt"], re.IGNORECASE)
    strong = re.compile(cfg["strong_game"], re.IGNORECASE)
    win = cfg["window"]
    need = cfg["cn_required"]
    for pack, path, ev, cv in rows:
        if not leaf_re.search(ev):
            continue
        st["leaf"] += 1
        eb, cb = blocks(ev), blocks(cv)
        if len(eb) != len(cb):
            st["shape"] += 1
            review.append((pack, path, -1, f"标签块数 {len(eb)}!={len(cb)}", "", ""))
            continue
        st["leaf_aligned"] += 1
        for i, (e, c) in enumerate(zip(eb, cb)):
            ms = list(occ.finditer(e))
            if not ms:
                continue
            st["block"] += 1
            seen = set()
            for m in ms:
                w = e[max(0, m.start() - win): m.end() + win]
                if exempt.search(w):
                    seen.add("EXEMPT")
                elif strong.search(w):
                    seen.add("GAME")
                elif common.search(w):
                    seen.add("COMMON")
                elif game.search(w):
                    seen.add("GAME")
                else:
                    seen.add("UNKNOWN")
            if "EXEMPT" in seen:
                st["b_exempt"] += 1
                continue
            if "UNKNOWN" in seen:
                st["b_unknown"] += 1
                continue
            if seen == {"GAME"}:
                st["b_game"] += 1
                if not c.strip():
                    st["b_cn_empty"] += 1
                    continue
                if need not in c:
                    st["bad_pos"] += 1
                    review.append((pack, path, i, "块内全机制义、中文无「阶位」",
                                   e.strip()[:70], c.strip()[:50]))
            elif seen == {"COMMON"}:
                st["b_common"] += 1
                if need in c:
                    st["bad_neg"] += 1
                    review.append((pack, path, i, "块内全普通名词义、中文却有「阶位」",
                                   e.strip()[:70], c.strip()[:50]))
            else:
                st["b_mix"] += 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--case", action="append")
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()

    rows = []
    for r in a.repo:
        rows += list(load_pairs(r))
    print(f"读入 {len(rows)} 对中英叶\n")

    for cid in (a.case or list(CASES)):
        cfg = CASES[cid]
        st = collections.Counter()
        review = []
        {"sequence": run_sequence, "count_ge": run_count_ge, "sense": run_sense}[cfg["mode"]](
            cfg, rows, st, review)
        print(f"=== {cid} ({cfg['mode']}) ===")
        print(f"  闸下 {st['leaf']} 叶 · 标签结构对齐 {st['leaf_aligned']} 叶 · 结构不同 {st['shape']} 叶")
        if cfg["mode"] in ("sequence", "count_ge"):
            print(f"  有词块 {st['block']} 块 · 对齐 {st['ok']} · 不对齐 {st['bad']}")
        else:
            print(f"  含 rank 的块 {st['block']} 块 · GAME {st['b_game']} / COMMON {st['b_common']}"
                  f" / MIX {st['b_mix']} / UNKNOWN {st['b_unknown']} / 第三义项 {st['b_exempt']}"
                  f" · 中文块为空 {st['b_cn_empty']}")
            print(f"  正向闸违规 {st['bad_pos']} · 反向闸违规 {st['bad_neg']}")
        for row in review[:a.show]:
            pack, path, i, why, x, y = row
            print(f"    [{pack[:26]}] {path[-58:]} 块{i} {why}")
            print(f"       EN={x}  CN={y}")
        if len(review) > a.show:
            print(f"    …另 {len(review) - a.show} 条")
        print()


if __name__ == "__main__":
    main()
