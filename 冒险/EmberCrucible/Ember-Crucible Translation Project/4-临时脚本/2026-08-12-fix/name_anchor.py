# -*- coding: utf-8 -*-
"""Proper-noun anchor check (single-pass, alternation regex).

Two directions, both aimed at "中文与英文不符" that length/markup gates miss:

  MISSING : a proper noun appears in the English leaf, its established Chinese
            rendering appears nowhere in the Chinese leaf  -> content dropped or
            the paragraph was written against an older English.
  INVENTED: a Chinese proper-noun rendering appears in the Chinese leaf while the
            corresponding English name is absent from the English leaf -> content
            invented (or carried over from a different scene).

Dictionary is harvested from every `*.name` leaf pair in the pack: the English
side is the proper noun, the Chinese side is `中文 English` (bilingual house
style) or plain 中文; we keep the leading hanzi run.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
HAN = r"[一-鿿··]"
TAG = re.compile(r"<[^>]+>")
MARKUP_TARGET = re.compile(r"@[A-Za-z]+\[[^\]]*\]")
INLINE = re.compile(r"\[\[[^\]]*\]\]")
REFERENCE = re.compile(r"&(?:amp;)?[Rr]eference\[[^\]]*\]")

STOP = set("""The A An And Or But If In On At To For Of With From By As It They You We
This That These Those There Here When While After Before During Once Each Every All
Some Any No Not So Then Than Their His Her Its Your Our My Who Whom Whose What Which
He She Him Them Us Me Is Are Was Were Be Been Being Have Has Had Do Does Did Will
Would Can Could Should May Might Must Gamemaster Read Note New Level Challenge
Chapter Event Events Quest Main Side Area Map Scene Page Table Item Actor Overview
Summary Aftermath Conclusion Introduction Flowchart Details Rewards Encounter
""".split())


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def plain(s):
    s = MARKUP_TARGET.sub(" ", s)
    s = INLINE.sub(" ", s)
    s = REFERENCE.sub(" ", s)
    s = TAG.sub(" ", s)
    return re.sub(r"\s+", " ",
                  s.replace("&amp;", "&").replace("{", " ").replace("}", " ")).strip()


def build_dict(rows):
    cand = {}
    for p, e, c in rows:
        if not p.endswith(".name") or not c:
            continue
        e = e.strip()
        if len(e) < 5 or len(e) > 40:
            continue
        if not re.match(r"^[A-Z][A-Za-z'’\- ]+$", e):
            continue
        toks = re.findall(r"[A-Za-z']+", e)
        if all(t in STOP for t in toks):
            continue
        m = re.match(r"^(" + HAN + r"+)", c.strip())
        if not m:
            continue
        han = m.group(1)
        if len(han) < 3:
            continue
        cand.setdefault(e, Counter())[han] += 1
    return {k: v.most_common(1)[0][0] for k, v in cand.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--journals")
    ap.add_argument("--min-en", type=int, default=80)
    ap.add_argument("--out")
    a = ap.parse_args()

    en = json.load(open(os.path.join(a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(a.repo, "compendium", "cn", a.pack), encoding="utf-8"))
    rows = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
    d = build_dict(rows)
    keys = sorted(d, key=len, reverse=True)
    han2en = defaultdict(list)
    for k in keys:
        han2en[d[k]].append(k)
    hans = sorted(han2en, key=len, reverse=True)
    print(f"name dictionary: {len(d)} EN keys / {len(hans)} CN forms")

    rx_en = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")
    rx_cn = re.compile("|".join(re.escape(h) for h in hans))

    names = None
    if a.journals:
        names = set(l.strip() for l in open(a.journals, encoding="utf-8") if l.strip())

    findings = []
    for p, e, c in rows:
        m = re.match(r"entries\.([^.]+)\.journals\.(.+?)\.pages\.", p)
        if not m:
            continue
        if names is not None and m.group(2) not in names:
            continue
        pe, pc = plain(e), plain(c)
        if len(pe) < a.min_en or not pc:
            continue
        en_keys = set(rx_en.findall(pe))
        cn_hans = set(rx_cn.findall(pc))
        # a hanzi form is "covered" if any of its EN keys is in the English
        missing = sorted(k for k in en_keys if d[k] not in pc and k not in pc)
        invented = sorted(h for h in cn_hans
                          if not any(re.search(r"\b" + re.escape(k) + r"\b", pe)
                                     for k in han2en[h]))
        if missing or invented:
            findings.append({"path": p, "journal": m.group(2),
                             "missing": [f"{k}->{d[k]}" for k in missing],
                             "invented": [f"{h}<-{'/'.join(han2en[h])}" for h in invented]})

    findings.sort(key=lambda f: -(len(f["missing"]) + len(f["invented"])))
    print(f"leaves flagged: {len(findings)}")
    for f in findings:
        print(f"  {f['path']}")
        if f["missing"]:
            print(f"      MISSING  {', '.join(f['missing'])}")
        if f["invented"]:
            print(f"      INVENTED {', '.join(f['invented'])}")
    if a.out:
        json.dump(findings, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
