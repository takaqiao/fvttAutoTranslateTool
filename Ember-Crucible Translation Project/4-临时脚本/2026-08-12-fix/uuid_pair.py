# -*- coding: utf-8 -*-
"""@UUID[target]{label} 标签/目标错位检测器。

判据（与审计报告 1.4 的建议一致，但独立实现）：
  以英文侧同叶的 目标↔标签 配对为权威事实；
  用全库「目标 id -> 中文标签多数写法」表重新配对中文侧的每个链接；
  某叶内某个 @UUID[t]{L} 若 L != canon_cn(t)，且 L == canon_cn(t') 而 t' 也出现在同叶，
  就判为「标签挂错了目标」（轮转/对调）。

输出 JSON：每叶给出英文链接序列、中文链接序列、每个位置的 canon 标签与判定。
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
# @UUID[...] 后面可选 {label}
UUID_RX = re.compile(r"@UUID\[([^\]]*)\](?:\{([^}]*)\})?")


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append({"path": p,
                    "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                    "en": en,
                    "cn": cn if isinstance(cn, str) else None})


def load_pairs(repo, pack):
    en = json.load(open(os.path.join(repo, "compendium", "en", pack), encoding="utf-8"))
    cp = os.path.join(repo, "compendium", "cn", pack)
    cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
    rows = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
    for r in rows:
        r["pack"] = pack
    return rows


def links(s):
    """[(target, label_or_None, span)]"""
    return [(m.group(1), m.group(2), m.span()) for m in UUID_RX.finditer(s or "")]


def tail(target):
    """最后一段 id，用来把 Compendium.x / JournalEntry.x 之类的不同写法归一。"""
    parts = target.split()
    core = parts[0] if parts else target
    return core.split(".")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--packs", required=True, help="逗号分隔")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-support", type=int, default=2,
                    help="canon 标签的最少出现次数")
    a = ap.parse_args()

    packs = [p.strip() for p in a.packs.split(",") if p.strip()]
    rows = []
    for p in packs:
        rows.extend(load_pairs(a.repo, p))

    # ---- 全库统计：目标 -> 中文标签计数 / 英文标签计数 -------------------
    cn_lab = defaultdict(Counter)
    en_lab = defaultdict(Counter)
    for r in rows:
        if not r["cn"]:
            continue
        for t, l, _ in links(r["en"]):
            if l:
                en_lab[tail(t)][l] += 1
        for t, l, _ in links(r["cn"]):
            if l:
                cn_lab[tail(t)][l] += 1

    canon = {}
    for t, c in cn_lab.items():
        lab, n = c.most_common(1)[0]
        canon[t] = {"label": lab, "n": n, "total": sum(c.values()),
                    "alts": dict(c.most_common())}

    # ---- 逐叶判定 -------------------------------------------------------
    findings = []
    for r in rows:
        if not r["cn"]:
            continue
        el, cl = links(r["en"]), links(r["cn"])
        if not cl:
            continue
        en_by_t = defaultdict(list)
        for t, l, _ in el:
            en_by_t[tail(t)].append(l)
        leaf_targets = {tail(t) for t, l, _ in cl}
        hits = []
        for i, (t, l, span) in enumerate(cl):
            if not l:
                continue
            tt = tail(t)
            cv = canon.get(tt)
            if not cv:
                continue
            if l == cv["label"]:
                continue
            # 这个标签是不是同叶另一个目标的多数写法？
            owners = [o for o in leaf_targets
                      if o != tt and canon.get(o, {}).get("label") == l]
            if not owners:
                continue
            hits.append({
                "idx": i, "target": t, "target_id": tt,
                "cn_label": l,
                "canon_label": cv["label"], "canon_n": cv["n"], "canon_total": cv["total"],
                "canon_alts": cv["alts"],
                "en_labels": en_by_t.get(tt, []),
                "label_belongs_to": owners,
                "label_canon_support": {o: canon[o]["n"] for o in owners},
            })
        if hits:
            findings.append({
                "pack": r["pack"], "path": r["path"], "batch_path": r["batch_path"],
                "en": r["en"], "cn": r["cn"],
                "en_links": [{"t": tail(t), "l": l} for t, l, _ in el],
                "cn_links": [{"t": tail(t), "l": l} for t, l, _ in cl],
                "hits": hits,
            })

    payload = {
        "packs": packs,
        "leaves_scanned": sum(1 for r in rows if r["cn"]),
        "leaves_flagged": len(findings),
        "links_flagged": sum(len(f["hits"]) for f in findings),
        "findings": findings,
    }
    open(a.out, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"leaves_scanned={payload['leaves_scanned']} "
          f"leaves_flagged={payload['leaves_flagged']} "
          f"links_flagged={payload['links_flagged']} -> {a.out}")


if __name__ == "__main__":
    main()
