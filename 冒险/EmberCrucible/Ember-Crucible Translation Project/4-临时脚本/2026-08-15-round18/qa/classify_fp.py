# -*- coding: utf-8 -*-
"""Y1-B：把 scan_dropped_terms 剩下的假阳性按**成因**分类，给判据那一路用。

判别特征都写成可编码的形式，每类都在真库上数出命中数（反空转：打印扫了多少块）。

A 英文侧词形没被 `stem()` 数到 —— 英文其实还在
  A1 所有格：`Manor's` → `stem()` 砍掉词尾 s 变成 `manor'`，与 `manor` 不同桶
  A2 派生／屈折：`successful` `successfully` `sensed` `trained` `trains`（`stem()` 只处理 -s/-es/-ies）
  A3 复合／连写：`alleyway` `giantfolk` `Oakengarde` `ooze-ologist`；反向的 `coat room`（英文拆开写）
  A4 上游拼写事故：`Kkithil` `Edival` `Kynryrth` `Acturel` `Social Cvent`
B 中文一词多源 —— 这个汉字词是**另一个英文词**的正当译名
  `role→角色` `clerics→牧师` `console→控制台` `task→任务` `lockpicks→开锁工具`
  `section/part/vicinity→区域` `clandestine/covert→秘密` `affairs→社交场合` `group→队伍`
C 中英语序 / 块边界错位、代词还原、加译
  C1 `Any character who makes a successful [[/check …]] …` 英文主语在 enricher 之前、
     中文在其之后 —— 切块后落进相邻块，整叶合计其实相等
  C2 代词还原：`it→低语者` `the city→阿克图瑞尔`
  C3 加译：`the room→书房`（所在小节标题就是 Study）`the key→挑战钥匙`
D enricher / 裸 @UUID 展开后词形对不上
  D1 `[[/eventState localColorArtists]]` 展开成 `eventState`，中文写「事件」
  D2 `@CriticalSuccess[15]` 被 SEM_MARKUP 吃成 `15`，`Success` 凭空消失
  D3 裸 `@UUID[Actor.x]` 的 idmap 名带所有格（`Gohema's Head`）→ 又落回 A1
E 上游整体改名、中文已按主控裁定跟进（`Lower Arcturel`→`the Dives`）
"""
from __future__ import annotations
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "3-常用脚本", "qa")))
from scan_dropped_terms import stem, WORD  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))

# B 类：中文词 → 本轮逐块人核时确认的「另一个英文来源」
B_SOURCES = {
    "角色": ["role", "roles", "party", "they"],
    "队伍": ["group", "they", "them"],
    "区域": ["section", "part", "parts", "vicinity", "cone", "area"],
    "秘密": ["clandestine", "covert", "secretive"],
    "任务": ["task", "tasks", "mission"],
    "牧师": ["cleric", "clerics"],
    "控制": ["console"],
    "工具": ["lockpicks"],
    "社交": ["affairs", "affair"],
    "图书馆": ["catalogues", "catalogue"],
    "书房": ["room"], "兵营": ["room"], "衣帽间": ["coat"],
    "挑战": ["key", "keys"],
    "低语者": ["it", "its"],
}


def edit1(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


def classify(term_en, cn_term, blk_en_txt, blk_cn_txt, neigh_txt, idnames):
    st = stem(term_en)
    low = blk_en_txt.lower()
    toks = [w.lower() for w in WORD.findall(blk_en_txt)]
    # ---- A1 所有格
    if re.search(r"\b" + re.escape(st) + r"['’]s\b", low):
        return "A1", f"块内有所有格 {st}'s，stem() 砍成 {st}'"
    # ---- D3 裸 @UUID 的 idmap 名里带这个词（含所有格）
    for nm in idnames:
        if st in nm.lower():
            return "D3", f"裸 @UUID 目标名 {nm!r} 含该词（所有格/复合形式，stem 数不到）"
    # ---- D1/D2 enricher 名里含这个词
    for m in re.finditer(r"\[\[/([A-Za-z]+)|@([A-Za-z]+)\[", blk_en_txt):
        w = (m.group(1) or m.group(2) or "").lower()
        if st in w and w != st:
            return ("D1" if m.group(1) else "D2"), f"enricher 名 {w!r} 含该词，展开后不是独立单词"
    # ---- A2 派生／屈折
    for t in toks:
        if t != st and (t.startswith(st) or st.startswith(t[:max(4, len(t) - 2)])) and \
                len(t) >= len(st) - 1 and t[:4] == st[:4]:
            return "A2", f"块内有派生/屈折形 {t!r}（stem() 只处理 -s/-es/-ies）"
    # ---- A3 复合／连写
    for t in toks:
        if t != st and st in t:
            return "A3", f"块内有复合词 {t!r} 含该词"
    # ---- A4 上游拼写事故
    for t in toks:
        if t != st and edit1(t, st):
            return "A4", f"块内有拼写事故 {t!r}（与 {st} 差一个字母）"
    # ---- B 一词多源
    for src in B_SOURCES.get(cn_term, []):
        if re.search(r"\b" + src + r"\b", low):
            return "B", f"该汉字词译的是同块的 {src!r}，不是 {term_en!r}"
    # ---- C1 相邻块里有这个词
    if any(stem(w) == st for w in WORD.findall(neigh_txt)):
        return "C1", "相邻块里有这个词：中英语序不同，切块后落到隔壁块"
    return "C2/C3", "代词还原 / 加译（人核）"


def main():
    rep = json.load(open(os.path.join(HERE, "split_ember.json"), encoding="utf-8"))
    cru = json.load(open(os.path.join(HERE, "split_crucible.json"), encoding="utf-8"))
    # 相邻块要真的读出来 —— C1「词落到隔壁块」是本轮最大的一类，判据不能是空跑
    from scan_dropped_terms import load_json, leaves
    TAG = re.compile(r"<[^>]+>")
    blkcache = {}

    def neighbours(repo, pack, path, i):
        key = (repo, pack)
        if key not in blkcache:
            d = {}
            leaves(load_json(os.path.join(repo, "compendium", "en", pack)).get("entries", {}), [], d)
            blkcache[key] = d
        s = blkcache[key].get(path)
        if s is None:
            return ""
        bs = TAG.split(s)
        return " ".join(bs[max(0, i - 2):i] + bs[i + 1:i + 3])
    bind = json.load(open(os.path.join(HERE, "bind3.json"), encoding="utf-8"))
    idmap = {}
    for k, v in bind["ids"].items():
        for o in (v if isinstance(v, list) else [v]):
            if isinstance(o, dict) and o.get("name"):
                idmap.setdefault(k, o["name"])
                break

    REAL = {
        ("Ember Early Access.journals.The Book Of Tales.pages.The Signborn's Secret.text", "Jahud"),
        ("Ember Early Access.journals.Disgraced House.pages.To Copy a Key.text", "party"),
    }
    cause = collections.Counter()
    detail = []
    n_blk = 0
    rows = rep["rows"] + cru["rows"]
    for r in rows:
        if r["en"] == "Arcturel":
            cause["E"] += len(r["sus_blocks"])
            for b in r["sus_blocks"]:
                n_blk += 1
                detail.append({**{k: r[k] for k in ("pack", "path", "en", "cn_term")},
                               "blk": b["blk"], "cause": "E",
                               "why": "the Dives 改名已裁并落库；split_dives 逐块核 102 叶 0 待人看"})
            continue
        for b in r["sus_blocks"]:
            n_blk += 1
            if (r["path"], r["en"]) in REAL:
                cause["REAL?"] += 1
                detail.append({**{k: r[k] for k in ("pack", "path", "en", "cn_term")},
                               "blk": b["blk"], "cause": "REAL-叶", "why": "见 batches/"})
                continue
            ids = [idmap[m] for m in re.findall(r"@(?:UUID|Embed)\[[^\]]*?\.([A-Za-z0-9]{6,})\](?!\s*\{)",
                                                b["en_txt"]) if m in idmap]
            repo = "2-Crucible汉化插件" if r["pack"].startswith("crucible.") else "1-Ember汉化插件"
            neigh = neighbours(repo, r["pack"], r["path"], b["blk"])
            c, why = classify(r["en"], r["cn_term"], b["en_txt"], b["cn_txt"], neigh, ids)
            cause[c] += 1
            detail.append({**{k: r[k] for k in ("pack", "path", "en", "cn_term")},
                           "blk": b["blk"], "cause": c, "why": why,
                           "en_txt": b["en_txt"][:200], "cn_txt": b["cn_txt"][:200]})
    print(f"扫了 {len(rows)} 个「词×叶」行 / {n_blk} 个嫌疑块")
    for k, v in sorted(cause.items(), key=lambda x: -x[1]):
        print(f"  {k:8s} {v}")
    json.dump({"n_rows": len(rows), "n_sus_blocks": n_blk, "cause": dict(cause),
               "detail": detail}, open(os.path.join(HERE, "fp_causes.json"), "w",
                                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  -> fp_causes.json")
    for d in detail:
        if d["cause"] == "C2/C3":
            print(f"  [人核] {d['path'][-46:]} blk{d['blk']} {d['en']}→{d['cn_term']}")


if __name__ == "__main__":
    main()
