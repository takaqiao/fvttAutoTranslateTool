# -*- coding: utf-8 -*-
"""把 scan_negation_drift.py 的报告按条打印，并高亮相关句子，便于人工核。"""
import argparse, json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SENT_EN = re.compile(r"[^.!?;]*[.!?;]")

PAT = {
    "unless": r"unless",
    "cannot": r"can\s*not|cannot|can't|may\s+not|must\s+not|unable",
    "no_longer": r"no\s+longer",
    "instead_of": r"instead\s+of|rather\s+than|in\s+place\s+of",
    "except": r"except|other\s+than|aside\s+from|apart\s+from|besides",
    "never": r"never",
    "without": r"without",
    "plain_not": r"\bnot\b|\bno\b|neither|\bnor\b|fails?\s+to|n't\b",
}
CNPAT = {
    "cn_unless": r"除非|若非",
    "cn_cannot": r"无法|不能|不可|不得",
    "cn_no_longer": r"不再|不复",
}


def sents(text, rx):
    out = []
    for m in re.finditer(rx, text, re.I):
        s = max(0, m.start() - 130)
        e = min(len(text), m.end() + 160)
        out.append(text[s:e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--dir", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pack")
    ap.add_argument("--grep-path")
    a = ap.parse_args()
    d = json.load(open(a.report, encoding="utf-8"))
    rows = d[a.dir]
    if a.pack:
        rows = [r for r in rows if a.pack in r["pack"]]
    if a.grep_path:
        rows = [r for r in rows if re.search(a.grep_path, r["path"])]
    for i, r in enumerate(rows[a.start: a.start + a.n], a.start):
        print("=" * 100)
        print(f"#{i} {r['repo']} / {r['pack']}")
        print(f"path: {r['path']}")
        print(f"unit={r.get('unit')} gaps: {json.dumps(r['gaps'], ensure_ascii=False)}"
              f"   en_any={r.get('en_any')} cn_any={r.get('cn_any')}")
        table = PAT if a.dir == "forward" else CNPAT
        for k in r["gaps"]:
            rx = table.get(k)
            if not rx:
                continue
            src = r["en"] if a.dir == "forward" else r["cn"]
            for s in sents(src, rx)[:6]:
                print(f"  <{k}> …{s}…")
        print("--- CN ---" if a.dir == "forward" else "--- EN ---")
        other = r["cn"] if a.dir == "forward" else r["en"]
        print(other if len(other) < 1400 else other[:1400] + " …")


if __name__ == "__main__":
    main()
