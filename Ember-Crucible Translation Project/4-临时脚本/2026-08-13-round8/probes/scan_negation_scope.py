# -*- coding: utf-8 -*-
"""否定作用域判据 —— 英文里「否定 + 某个机制名词」的组合，中文侧那个机制名词旁边没有否定。

和 scan_negation_drift.py 的分工
--------------------------------
`scan_negation_drift.py` 比的是**一个块里否定的总量**（预算闸）。它的盲区是：
一个长块里中文有别的「不」，总量对得上，但**否定挂错了地方**（原文说「不造成伤害」，
译文说「造成伤害」，同时别处多了个「不」）。总量判据看不见这种。

本判据比的是**否定的作用对象**：
  英文出现 NEG（not/no/cannot/without/never/…）后 WIN 字符内的机制名词 M
  -> 中文块里必须存在 M 的中文对译，且其**前后 CTX 字符内有否定字**。
  中文里根本没有 M 的对译时跳过（可能整句改写，判不了）。

同样做块级对齐（英中 HTML 块数 99.8% 相同）。

用法：
  python scan_negation_scope.py --repo <repoDir> [--repo <另一个>] --out <json>
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys
import importlib.util

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "nd", os.path.join(_HERE, "scan_negation_drift.py"))
nd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd)

NEG = re.compile(
    r"\b(?:cannot|can\s*not|can't|may\s+not|must\s+not|will\s+not|won't|"
    r"do(?:es)?\s+not|don't|doesn't|did\s+not|didn't|is\s+not|are\s+not|isn't|"
    r"aren't|was\s+not|were\s+not|has\s+not|have\s+not|hasn't|haven't|"
    r"never|no\s+longer|without|unable\s+to|neither|nor|not|no)\b", re.I)

CN_NEG_CHAR = re.compile(r"[不无没未非勿禁免缺]")

# 机制名词 -> 中文对译（任一命中即算「中文里有这个机制词」）
MECH_MAP = [
    (r"damage", ["伤害"]),
    (r"movement|move|moving|moves|moved", ["移动"]),
    (r"stride", ["步幅"]),
    (r"attacks?", ["攻击"]),
    (r"checks?", ["检定"]),
    (r"saves?|saving\s+throws?", ["豁免"]),
    (r"actions?", ["动作", "行动"]),
    (r"turns?", ["回合"]),
    (r"rounds?", ["轮"]),
    (r"spells?", ["法术"]),
    (r"hit\s+points?|health", ["生命"]),
    (r"wounds?", ["创伤"]),
    (r"bonus(?:es)?", ["加值"]),
    (r"penalt(?:y|ies)", ["减值", "惩罚"]),
    (r"resistances?", ["抗性"]),
    (r"advantage", ["优势"]),
    (r"disadvantage", ["劣势"]),
    (r"targets?", ["目标"]),
    (r"weapons?", ["武器"]),
    (r"armou?r", ["护甲"]),
    (r"shields?", ["盾"]),
    (r"rest(?:ing|s)?", ["休息"]),
    (r"initiative", ["先攻"]),
    (r"reactions?", ["反应"]),
    (r"allies|ally", ["盟友"]),
    (r"enem(?:y|ies)", ["敌人"]),
    (r"distance", ["距离"]),
    (r"ranges?", ["射程", "范围"]),
    (r"conditions?", ["状态"]),
    (r"effects?", ["效果"]),
    (r"defenses?|defences?", ["防御"]),
    (r"boon", ["恩惠骰"]),
    (r"bane", ["祸骰"]),
    (r"focus", ["专注"]),
    (r"morale", ["士气"]),
    (r"tiers?", ["阶"]),
]
MECH_RX = [(re.compile(r"\b(?:" + p + r")\b", re.I), cn) for p, cn in MECH_MAP]

WIN = 34    # 英文否定之后多少字符内算「否定作用于」该机制词
CTX = 14    # 中文机制词前后多少字符内要有否定字


def cn_negated(c: str, terms: list[str]) -> tuple[bool, bool]:
    """返回 (中文里有这个机制词, 至少一处该机制词附近有否定)。"""
    found = False
    for t in terms:
        for m in re.finditer(re.escape(t), c):
            found = True
            s = max(0, m.start() - CTX)
            e = min(len(c), m.end() + CTX)
            if CN_NEG_CHAR.search(c[s:e]):
                return True, True
    return found, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--pack", default="all")
    ap.add_argument("--out")
    ap.add_argument("--show", type=int, default=60)
    a = ap.parse_args()

    st = collections.Counter()
    hits = []
    for repo in a.repo:
        ed = os.path.join(repo, "compendium", "en")
        cd = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(ed):
            print(f"!! 没有 {ed}")
            continue
        packs = (sorted(f for f in os.listdir(ed)
                        if f.endswith(".json") and not f.startswith("_"))
                 if a.pack == "all" else [x.strip() for x in a.pack.split(",")])
        for pack in packs:
            ep, cp = os.path.join(ed, pack), os.path.join(cd, pack)
            if not (os.path.isfile(ep) and os.path.isfile(cp)):
                continue
            rows = []
            nd.walk(nd.load(ep).get("entries", {}), nd.load(cp).get("entries", {}),
                    ["entries"], rows)
            for p, en, cn in rows:
                if not cn:
                    continue
                us, mode = nd.units(en, cn)
                for idx, e, c in us:
                    if not e:
                        continue
                    st["单元"] += 1
                    for m in NEG.finditer(e):
                        seg = e[m.end(): m.end() + WIN]
                        for rx, terms in MECH_RX:
                            mm = rx.search(seg)
                            if not mm:
                                continue
                            st["否定+机制词 对"] += 1
                            found, negated = cn_negated(c, terms)
                            if not found:
                                st["中文无该机制词（跳过）"] += 1
                                continue
                            if negated:
                                st["中文该机制词旁有否定"] += 1
                                continue
                            st["**中文该机制词旁无否定**"] += 1
                            s = max(0, m.start() - 90)
                            t = min(len(e), m.end() + WIN + 90)
                            hits.append({
                                "repo": os.path.basename(repo), "pack": pack,
                                "path": p,
                                "batch_path": p[len("entries."):]
                                if p.startswith("entries.") else p,
                                "unit": f"{mode}#{idx}",
                                "en_neg": m.group(0), "en_mech": mm.group(0),
                                "cn_terms": terms,
                                "en_frag": e[s:t],
                                "en": nd.clip(e, 900), "cn": nd.clip(c, 900),
                            })
                            break

    # 同一 (pack, path, unit, mech) 只留一条
    seen, uniq = set(), []
    for h in hits:
        k = (h["pack"], h["path"], h["unit"], h["en_mech"].lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(h)

    print("扫描规模：")
    for k, v in st.most_common():
        print(f"  {k:24s} {v}")
    print(f"\n命中 {len(uniq)} 条（去重前 {len(hits)}）")
    for h in uniq[: a.show]:
        print("=" * 92)
        print(f"{h['pack'][:26]} | {h['path'][-60:]} | {h['unit']} | "
              f"[{h['en_neg']} … {h['en_mech']}] -> {'/'.join(h['cn_terms'])}")
        print("  EN…", h["en_frag"])
        print("  CN :", h["cn"][:420])

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"stats": dict(st), "hits": uniq}, f,
                      ensure_ascii=False, indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
