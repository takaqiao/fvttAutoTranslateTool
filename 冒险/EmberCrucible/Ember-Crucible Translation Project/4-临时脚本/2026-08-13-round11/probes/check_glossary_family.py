# -*- coding: utf-8 -*-
"""check_glossary_family.py -- mechanical family-consistency check for glossary_ec.json.

RULE (requested by 主控, round 11 / unit F7):
    Every glossary entry that mentions a *proper-noun* English token T must render
    T with the same Chinese string that T's own head entry uses.

Why this rule exists
--------------------
glossary_ec.json is the authority behind apply_tm / fill_missing.  When one
member of a name family drifts -- "Mutagist" -> 突变学派 but "Mutagist Excisor"
-> 嬗变师切除者 -- every later batch re-injects the drifted form, so the split
becomes self-healing in the wrong direction.  Round 11 found four families rotted
this way simultaneously (Mutagist 6 renderings, Toothbreaker 5, House Cevher 4,
the X-Lineage series 5), none of which any single-term scan can see, because each
individual entry looks internally fine.  A family-consistency check catches all
four in one pass.

Why the proper-noun filter is the whole trick
---------------------------------------------
A naive "same first word => same prefix" rule fires ~700 times on this glossary
and is useless: ordinary English words *should* vary with context ("Light" 光 vs
"Light Armor" 轻型护甲; "Fire" 火元素 vs "Alchemist's Fire" 炼金火焰).  Invented
setting names must not.  So the checker separates the two automatically, with no
hand-maintained name list:

    a token is a PROPER NOUN iff, across the packs' English corpus, it almost
    never appears lowercased (lowercase_occurrences / total < --lower-ratio).

"light" and "fire" appear lowercase constantly; "Mutagist", "Cevher", "Kivahr"
essentially never do.  The corpus is the two compendium/en trees, so the filter
stays correct as the adventure grows.  With --no-corpus the filter falls back to
the STOP list alone (noisy; use only if the packs are unavailable).

Violation classes
-----------------
  A  head-mismatch      T has a head entry (key == T / T+s / "The "+T) and some
                        other entry containing T does not carry the head's
                        Chinese rendering.   <- Mutagist, Signborn, Anachraenum
  B  headless-no-consensus
                        >= --min-family entries share T, no head entry exists,
                        and their Chinese sides share no common substring, i.e.
                        no two of them agree on how to write T.   <- House Cevher
  C  english-leak       the Chinese side still contains a raw ASCII word after
                        the bilingual tail is stripped.   <- "Hulg'run血统"

Usage
-----
    python check_glossary_family.py                      # full report
    python check_glossary_family.py --token Mutagist     # one family
    python check_glossary_family.py --json out.json      # machine-readable
    python check_glossary_family.py --kind A             # A findings only
Exit code 1 when any violation survives, so it can gate a release.
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "C:\\Users\\Taka\\Desktop\\fvtt\\Ember-Crucible Translation Project"
DEFAULT_GLOSSARY = os.path.join(ROOT, "5-\u5176\u4ed6\u5185\u5bb9", "glossary",
                                "glossary_ec.json")
PACK_DIRS = [
    os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "compendium", "en"),
    os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "compendium", "en"),
]

MIN_TOKEN_LEN = 4

# Fast pre-filter: articles/prepositions and glossary scaffolding words that are
# never a setting name.  The corpus filter removes the rest of ordinary English.
STOP = set("""
the a an and or of in on to for with without from into onto by at as is are was
were be been being it its this that these those his her their your our my
new old young elder greater lesser minor major first second third
""".split())

TOKEN_RX = re.compile(r"[A-Za-z][A-Za-z'\u2019]*")
WORD_RX = re.compile(r"[A-Za-z][A-Za-z'\u2019]{2,}")
ASCII_TAIL_RX = re.compile(r"[\x20-\x7e]+$")
CJK_RX = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")
SKIP_KEYS = {"_id", "path", "_variants", "_when"}


# ----------------------------------------------------------------- corpus ---
def en_strings():
    def walk(o, out):
        if isinstance(o, dict):
            for k, v in o.items():
                if k not in SKIP_KEYS:
                    walk(v, out)
        elif isinstance(o, list):
            for v in o:
                walk(v, out)
        elif isinstance(o, str):
            out.append(o)

    out = []
    for d in PACK_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json") and not fn.startswith("_"):
                walk(json.load(open(os.path.join(d, fn), encoding="utf-8")).get("entries", {}), out)
    return out


def case_profile():
    """word(lowercased) -> (lowercase_occurrences, total_occurrences)."""
    low, tot = {}, {}
    for s in en_strings():
        for m in WORD_RX.finditer(s):
            w = m.group(0)
            k = w.lower()
            tot[k] = tot.get(k, 0) + 1
            if w[0].islower():
                low[k] = low.get(k, 0) + 1
    return low, tot


# ---------------------------------------------------------------- helpers ---
def cn_side(key, value):
    """Strip the bilingual English tail: values are stored as '中文 English'."""
    v = value.strip()
    for cand in (key, key.rstrip("s"), key + "s"):
        if cand and v.endswith(" " + cand):
            return v[: -(len(cand) + 1)].strip()
    m = ASCII_TAIL_RX.search(v)
    if m and CJK_RX.search(v[: m.start()]):
        return v[: m.start()].strip()
    return v


def longest_common_substring(strings, minlen):
    if not strings:
        return ""
    base = min(strings, key=len)
    best = ""
    for i in range(len(base)):
        for j in range(len(base), i + minlen - 1, -1):
            if len(base[i:j]) <= len(best):
                break
            sub = base[i:j]
            if all(sub in s for s in strings):
                best = sub
                break
    return best


# ------------------------------------------------------------------- core ---
def run(a):
    g = json.load(open(a.glossary, encoding="utf-8"))
    cn = {k: cn_side(k, v) for k, v in g.items()}

    if a.no_corpus:
        low, tot = {}, {}
    else:
        low, tot = case_profile()

    def is_proper(t):
        if len(t) < MIN_TOKEN_LEN or t.lower() in STOP:
            return False
        if a.no_corpus:
            return True
        k = t.lower()
        n = tot.get(k, 0)
        if n == 0:
            # never seen in the packs at all -> treat as a name only if the
            # glossary itself always capitalises it
            return True
        return (low.get(k, 0) / n) < a.lower_ratio

    def key_tokens(key):
        out = []
        for m in TOKEN_RX.finditer(key):
            t = m.group(0)
            if not (t[0].isupper() or key.strip() == t):
                continue
            if is_proper(t):
                out.append(t)
        return out

    fam = {}
    for k in g:
        for t in key_tokens(k):
            fam.setdefault(t, []).append(k)

    lower_keys = {}
    for k in g:
        lower_keys.setdefault(k.lower(), k)

    def head_of(t):
        for cand in (t, t + "s", t.rstrip("s"), "The " + t, "The " + t + "s"):
            hk = lower_keys.get(cand.lower())
            if hk:
                stripped = re.sub(r"^(the|a|an)\s+", "", hk, flags=re.I)
                if len(TOKEN_RX.findall(stripped)) == 1:
                    return hk
        return None

    findings = []
    for t, keys in sorted(fam.items()):
        if a.token and t.lower() != a.token.lower():
            continue
        if len(keys) < 2:
            continue
        hk = head_of(t)
        if hk:
            canon = cn[hk]
            if not canon or not CJK_RX.search(canon):
                continue
            bad = [k for k in keys if k != hk and cn[k] and canon not in cn[k]]
            if bad:
                findings.append({
                    "kind": "A-head-mismatch", "token": t, "head_key": hk,
                    "canonical_cn": canon, "family_size": len(keys),
                    "offenders": [{"key": k, "value": g[k], "cn": cn[k]} for k in sorted(bad)],
                })
        else:
            if len(keys) < a.min_family:
                continue
            vals = [cn[k] for k in keys if cn[k] and CJK_RX.search(cn[k])]
            if len(vals) < a.min_family:
                continue
            if not longest_common_substring(vals, 2):
                findings.append({
                    "kind": "B-headless-no-consensus", "token": t, "head_key": None,
                    "canonical_cn": None, "family_size": len(keys),
                    "offenders": [{"key": k, "value": g[k], "cn": cn[k]} for k in sorted(keys)],
                })

    for k, v in g.items():
        c = cn[k]
        if not (c and CJK_RX.search(c)):
            continue
        leak = WORD_RX.search(c)
        if not leak:
            continue
        if a.token and a.token.lower() not in k.lower():
            continue
        findings.append({
            "kind": "C-english-leak", "token": leak.group(0), "head_key": None,
            "canonical_cn": None, "family_size": 1,
            "offenders": [{"key": k, "value": v, "cn": c}],
        })
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", default=DEFAULT_GLOSSARY)
    ap.add_argument("--token", help="report one family only")
    ap.add_argument("--kind", choices=["A", "B", "C"], help="one violation class only")
    ap.add_argument("--json", help="also write machine-readable findings here")
    ap.add_argument("--min-family", type=int, default=3,
                    help="minimum members before a headless family is reported")
    ap.add_argument("--lower-ratio", type=float, default=0.2,
                    help="a token counts as a proper noun below this "
                         "lowercase/total ratio in the English corpus")
    ap.add_argument("--no-corpus", action="store_true",
                    help="skip the corpus-derived proper-noun filter (noisy)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    f = run(a)
    if a.kind:
        f = [d for d in f if d["kind"].startswith(a.kind)]
    f.sort(key=lambda d: (d["kind"], -d["family_size"], d["token"] or ""))
    if a.json:
        json.dump(f, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for d in (f[: a.limit] if a.limit else f):
        print("[%s] token=%r family=%d head=%r canonical=%r"
              % (d["kind"], d["token"], d["family_size"], d["head_key"], d["canonical_cn"]))
        for o in d["offenders"]:
            print("      %-44r => %r" % (o["key"], o["value"]))
    print("-" * 70)
    print("violations: %d  (A=%d  B=%d  C=%d)"
          % (len(f),
             sum(1 for d in f if d["kind"].startswith("A")),
             sum(1 for d in f if d["kind"].startswith("B")),
             sum(1 for d in f if d["kind"].startswith("C"))))
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
