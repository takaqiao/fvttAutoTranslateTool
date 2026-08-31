# -*- coding: utf-8 -*-
"""English-gated term counting for the Alien RPG project.

Ported from Ember-Crucible `5-临时脚本/2026-08-12-fix/term_gate.py`, generalised to
the three parallel corpora this project actually has.  EC's hard rule carries over
verbatim:

    NEVER adopt a term from a bare Chinese frequency count.  A bare count both
    over- and under-reports.  Always split the count by what the *paired English*
    at the same position actually says.

Corpora (pick with --mode; repeat --src to union several):

  lang        two Foundry lang JSONs sharing a key space, e.g.
                systems/alienrpg/lang/en.json  +  .../cn.json
              --src accepts either the directory (en.json/cn.json assumed) or an
              explicit "en.json::cn.json" pair.

  subs        CnSCG bilingual .ssa files.  Each Dialogue event packs ZH and EN into
              ONE Text field separated by the literal \\N.  Override tags {...} are
              stripped; the CJK-bearing segment is the CN side regardless of order,
              so both the `chs&eng` and the reversed `eng&chs` files parse.
              --src accepts a directory of .ssa files or a single .ssa.
              NOTE: all four films are one CnSCG lineage.  Cross-film agreement is
              shared provenance, not independent corroboration.  Weight the whole
              corpus as ONE vote (--vote-weight prints the reminder).

  compendium  Babele module layout <repo>/compendium/{en,cn}/*.json, walked leaf by
              leaf on identical paths.  This is EC's original mode, kept intact so
              the same command works once the Alien packs are extracted.

Buckets, per Chinese candidate:
  gated_hit   EN matches --en AND CN contains the candidate    <- the only real signal
  cn_only     CN contains the candidate but EN does NOT match  <- a different English
              word borrowed the same Chinese; NOT residue, do not "fix" blindly
  en_only     EN matches but CN uses none of the candidates    <- omission, or a
              competing rendering you have not listed yet
  no_cn       EN matches and there is no Chinese at that position at all

Examples
  python term_gate.py --mode lang --src "C:/.../systems/alienrpg/lang" \
      --en "\\bEngaged\\b" --cn 接战,近战
  python term_gate.py --mode subs --src "C:/Users/Taka/Desktop/fvtt/AlienRPG" \
      --en "\\bqueen\\b" --cn 女王,王后 --ignore-case
  python term_gate.py --mode compendium --src "<repoDir>" --en "\\bStress\\b" --cn 压力
  python term_gate.py --mode subs --src "..." --en "\\bfacehugger\\b" --cn 抱脸虫 --ignore-case
      -> gated_hit=0 en_only=0 no_cn=0 : ZERO attestation, the corpus cannot vote.

Exit code is 0 always; this is a reporting tool, not a gate in the CI sense.
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
CJK = re.compile(r"[\u4e00-\u9fff]")
SSA_OVERRIDE = re.compile(r"\{[^}]*\}")


# ---------------------------------------------------------------- row builders
def _walk(en, cn, path, out):
    """Emit (path, en_leaf, cn_leaf) for every string leaf on a shared key space."""
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            _walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            _walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def rows_lang(src):
    """src is a lang dir (en.json + cn.json) or an explicit 'en.json::cn.json'."""
    if "::" in src:
        en_p, cn_p = src.split("::", 1)
    else:
        en_p = os.path.join(src, "en.json")
        cn_p = os.path.join(src, "cn.json")
    en = json.load(open(en_p, encoding="utf-8-sig"))
    cn = json.load(open(cn_p, encoding="utf-8-sig")) if os.path.isfile(cn_p) else {}
    out = []
    _walk(en, cn, [], out)
    tag = os.path.basename(os.path.dirname(en_p)) or os.path.basename(en_p)
    return [(tag, os.path.basename(en_p), p, e, c) for p, e, c in out]


def rows_compendium(repo):
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    rows = []
    if not os.path.isdir(en_dir):
        return rows
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8-sig"))
        cp = os.path.join(cn_dir, fn)
        cn = json.load(open(cp, encoding="utf-8-sig")) if os.path.isfile(cp) else {}
        sub = []
        _walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
        for p, e, c in sub:
            rows.append((os.path.basename(repo), fn, p, e, c))
    return rows


def _ssa_files(src):
    if os.path.isfile(src):
        return [src]
    return [
        os.path.join(src, f)
        for f in sorted(os.listdir(src))
        if f.lower().endswith(".ssa")
    ]


def _film_tag(path):
    b = os.path.basename(path).lower()
    if b.startswith("alien.1979"):
        return "ALIEN1979"
    if b.startswith("aliens.2"):
        return "ALIENS1986"
    if b.startswith("alien.3"):
        return "ALIEN3"
    if b.startswith("alien.resurrection"):
        return "RESURRECTION"
    return os.path.basename(path)


def rows_subs(src):
    """Parse CnSCG bilingual SSA.  One Dialogue event = one aligned ZH/EN pair.

    Dedupe is by (film, start-time, en, cn) so the redundant reversed-order file in
    the same folder does not double-count the same line.
    """
    rows, seen = [], set()
    for path in _ssa_files(src):
        film = _film_tag(path)
        for raw in open(path, encoding="utf-8-sig", errors="replace"):
            if not raw.startswith("Dialogue:"):
                continue
            parts = raw.split(",", 9)
            if len(parts) < 10:
                continue
            start = parts[1].strip()
            text = SSA_OVERRIDE.sub("", parts[9]).replace("\\n", "\\N")
            segs = [s.strip() for s in text.split("\\N") if s.strip()]
            zh = " ".join(s for s in segs if CJK.search(s))
            en = " ".join(s for s in segs if not CJK.search(s))
            if not en and not zh:
                continue
            key = (film, start, en, zh)
            if key in seen:
                continue
            seen.add(key)
            rows.append((film, os.path.basename(path), start, en, zh))
    return rows


MODES = {"lang": rows_lang, "compendium": rows_compendium, "subs": rows_subs}


# ------------------------------------------------------------------- reporting
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(MODES), default="lang")
    ap.add_argument("--src", action="append", required=True,
                    help="repeat to union corpora; meaning depends on --mode")
    ap.add_argument("--en", required=True, help="regex applied to the English side")
    ap.add_argument("--cn", required=True,
                    help="comma-separated Chinese candidate renderings")
    ap.add_argument("--ignore-case", action="store_true")
    ap.add_argument("--show", type=int, default=3, help="sample rows per bucket")
    ap.add_argument("--trunc", type=int, default=260)
    ap.add_argument("--by-source", action="store_true",
                    help="break gated_hit down by corpus tag (film / lang file)")
    ap.add_argument("--json", metavar="OUT", help="also write the counts as JSON")
    a = ap.parse_args()

    rx = re.compile(a.en, re.I if a.ignore_case else 0)
    terms = [t.strip() for t in a.cn.split(",") if t.strip()]
    reader = MODES[a.mode]

    rows = []
    for s in a.src:
        rows.extend(reader(s))

    def clip(s):
        return s[: a.trunc] + ("…" if len(s) > a.trunc else "")

    en_match = [r for r in rows if rx.search(r[3])]
    print(f"mode={a.mode}  scanned rows: {len(rows)}   en-regex: {a.en!r}")
    print(f"rows whose ENGLISH matches: {len(en_match)}")
    if a.mode == "subs":
        print("!! subs corpus = ONE vote: all four films are a single CnSCG lineage.")
    print("-" * 78)

    report = {"mode": a.mode, "en": a.en, "src": a.src,
              "rows": len(rows), "en_match": len(en_match), "terms": {}}

    for t in terms:
        gated, cn_only = [], []
        for r in rows:
            if t in (r[4] or ""):
                (gated if rx.search(r[3]) else cn_only).append(r)
        bysrc = {}
        for r in gated:
            bysrc[r[0]] = bysrc.get(r[0], 0) + 1
        report["terms"][t] = {"gated_hit": len(gated),
                              "cn_only": len(cn_only),
                              "by_source": bysrc}
        line = f"CN {t!r}:  gated_hit={len(gated)}   cn_only(different English)={len(cn_only)}"
        if a.by_source and bysrc:
            line += "   " + " ".join(f"{k}={v}" for k, v in sorted(bysrc.items()))
        print(line)
        for r in gated[: a.show]:
            print(f"    [gated] {r[0]}/{r[2]}")
            print(f"        EN: {clip(r[3])}")
            print(f"        CN: {clip(r[4])}")
        for r in cn_only[: a.show]:
            print(f"    [cn_only] {r[0]}/{r[2]}")
            print(f"        EN: {clip(r[3])}")
            print(f"        CN: {clip(r[4])}")
    print("-" * 78)

    en_only = [r for r in en_match if r[4] and not any(t in r[4] for t in terms)]
    no_cn = [r for r in en_match if not r[4]]
    report["en_only"] = len(en_only)
    report["no_cn"] = len(no_cn)
    print(f"EN matches but CN uses none of {terms}: {len(en_only)}")
    for r in en_only[: a.show * 3]:
        print(f"    [en_only] {r[0]}/{r[2]}")
        print(f"        EN: {clip(r[3])}")
        print(f"        CN: {clip(r[4])}")
    print(f"EN matches but NO Chinese at that position at all: {len(no_cn)}")
    for r in no_cn[: a.show]:
        print(f"    [no_cn] {r[0]}/{r[2]}")
        print(f"        EN: {clip(r[3])}")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(report, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
