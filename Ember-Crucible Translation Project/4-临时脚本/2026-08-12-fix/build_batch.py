# -*- coding: utf-8 -*-
"""Rewrite untranslated {labels} in place and emit an apply_translations batch.

Only the text between the braces of a markup occurrence is replaced, at the exact
offset found by the same regex the plan used.  The bracketed target and every
other byte of the leaf are untouched.
"""
import argparse, json, re, sys, html
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CJK = re.compile(r'[一-鿿]')
MARK = re.compile(r'(@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])(\{([^{}]*)\})?')


def strip_tail(cnname, en):
    for cand in {en, html.unescape(en or "")}:
        if not cand:
            continue
        t = " " + cand
        if cnname.endswith(t) and CJK.search(cnname[: -len(t)]):
            return cnname[: -len(t)].strip()
    m = re.match(r"^(.*?[一-鿿])\s+([A-Za-z0-9'’&;#.,:!?()\-À-ɏ ]{2,})$", cnname)
    if m and not CJK.search(m.group(2)):
        return m.group(1).strip()
    return cnname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--decisions", required=True, help="dec_*.json from decide.py")
    ap.add_argument("--overrides", required=True, help="{EN label: CN label}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit-paths", help="regex; only leaves whose batch_path matches")
    a = ap.parse_args()

    ov = json.load(open(a.overrides, encoding="utf-8"))
    dec = {(r["path"], r["idx"]): r for r in json.load(open(a.decisions, encoding="utf-8"))}

    chosen, unresolved = {}, []
    for (p, i), r in dec.items():
        en = r["en"]
        src = None
        if en is not None and en in ov:
            val, src = ov[en], "override"
        elif ("@@cn:" + (r["cn"] or "")) in ov:
            val, src = ov["@@cn:" + r["cn"]], "override-cn"
        elif "T" in r["ev"]:
            val, src = r["ev"]["T"][0], "T"
        elif "L" in r["ev"]:
            val, src = r["ev"]["L"][0], "L"
        elif "N" in r["ev"]:
            val, src = r["ev"]["N"][0], "N"
        else:
            unresolved.append(r)
            continue
        chosen[(p, i)] = (val, src, en, r["cn"])

    out, rep = {}, []
    touched = defaultdict(int)
    for it in json.load(open(a.plan, encoding="utf-8")):
        p = it["batch_path"]
        if a.limit_paths and not re.search(a.limit_paths, p):
            continue
        s = it["cn_full"]
        edits = []
        for m in MARK.finditer(s):
            pass
        marks = list(MARK.finditer(s))
        for l in it["labels"]:
            key = (p, l["idx"])
            if key not in chosen:
                continue
            val, src, en, old = chosen[key]
            m = marks[l["idx"]]
            # the {label} span is everything after group(1) inside the match
            b0 = m.start(3)
            b1 = m.end(3)
            edits.append((b0, b1, val))
            rep.append({"path": p, "idx": l["idx"], "target": m.group(1),
                        "en": en, "before": old, "after": val, "basis": src})
        if not edits:
            continue
        for b0, b1, val in sorted(edits, reverse=True):
            s = s[:b0] + val + s[b1:]
        out[p] = s
        touched[p] = len(edits)

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"edits": rep, "unresolved": unresolved},
              open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"leaves in batch: {len(out)}   labels rewritten: {len(rep)}   unresolved: {len(unresolved)}")
    if unresolved:
        seen = set()
        for r in unresolved[:60]:
            k = r["en"]
            if k in seen:
                continue
            seen.add(k)
            print("  UNRESOLVED", repr(r["en"]), "|", repr(r["cn"]), "|", r["path"][:70])


if __name__ == "__main__":
    main()
