# -*- coding: utf-8 -*-
"""Find Latin-script residue *inside* Chinese strings.

Neither validate_translations.py (cn.strip()==en.strip()) nor any coverage /
markup scan sees this class: the leaf *does* have Chinese, it just also has
untranslated English words sitting in the middle of it.

Method: strip everything that is legitimately non-Chinese
  - HTML tags        <...>            (attribute values are markup, not prose)
  - enrichers        @Foo[target]{label}  -> keep only {label}
  - roll/inline      [[ ... ]]
  - html entities    &nbsp; &amp; ...
  - <span class="reference">...</span> bodies (coordinate tokens)
then look at what Latin words remain in the Chinese side.

By-design exclusions (PROJECT.md sec.8):
  - bilingual proper nouns  "中文 English"  -> Capitalized word(s) right after CJK
  - DC / ∞ / ??? / system name Crucible / Ember / dnd5e ...
  - pronunciation fields (IPA-ish respelling)
  - [[/item 中文名]] handled by the [[ ]] strip

Usage:
  python scan_latin_residue.py --repo <repo> [--pack p.json] [--out x.json]
                               [--mode all|lower|strict] [--limit N]
"""
import argparse, json, os, re, sys, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# paths whose content is legitimately Latin
SKIP_PATH_RX = re.compile(
    r"(?:^|\.)(?:pronunciation|img|src|texture|type|folder|sort|uuid|id)$", re.I)

CJK = r"一-鿿㐀-䶿"

# ---- markup strippers -------------------------------------------------------
RX_REF_SPAN = re.compile(r'<span class="reference">.*?</span>', re.S)
RX_ENRICH = re.compile(r'(?:@|&amp;|&)[A-Za-z]+\[[^\]]*\](?:\{([^}]*)\})?')
# dnd5e / crucible pronoun notation in NPC stat lines: "LN, Human, he/him"
RX_PRONOUN = re.compile(r'\b(?:he|she|they|it)\s*/\s*(?:him|her|them|its)\b', re.I)
RX_INLINE = re.compile(r'\[\[.*?\]\]')
RX_TAG = re.compile(r'<[^>]*>')
RX_ENTITY = re.compile(r'&[A-Za-z#0-9]+;')


def strip_markup(s):
    s = RX_REF_SPAN.sub(' ', s)
    s = RX_INLINE.sub(' ', s)
    s = RX_ENRICH.sub(lambda m: ' ' + (m.group(1) or '') + ' ', s)
    s = RX_TAG.sub(' ', s)
    s = RX_ENTITY.sub(' ', s)
    s = RX_PRONOUN.sub(' ', s)
    return s


# words that are by-design kept in English everywhere
WHITELIST = {
    'dc', 'crucible', 'ember', 'dnd', 'dnd5e', 'foundry', 'vtt', 'gm', 'pc', 'npc',
    'hp', 'ac', 'xp', 'beta', 'alpha', 'one', 'two',
}

RX_WORD = re.compile(r'[A-Za-z][A-Za-z\'’-]{2,}')


def scan_string(cn):
    """return list of (word, context) latin words remaining after stripping"""
    t = strip_markup(cn)
    hits = []
    for m in RX_WORD.finditer(t):
        w = m.group(0)
        if w.lower().strip("'’-") in WHITELIST:
            continue
        # bilingual "中文 English": Capitalized token whose left neighbour
        # (skipping spaces) is CJK  -> by design
        left = t[:m.start()]
        # walk left over an existing run of Capitalized latin words + spaces
        j = len(left)
        while True:
            k = j
            while k > 0 and left[k - 1] == ' ':
                k -= 1
            m2 = re.search(r'[A-Za-z\'’-]+$', left[:k])
            if m2 and m2.group(0)[:1].isupper():
                j = m2.start()
                continue
            j = k
            break
        prev = left[:j].rstrip()
        bilingual = bool(w[:1].isupper() and prev and re.search('[' + CJK + '、《》]$', prev))
        hits.append({"w": w, "bilingual": bilingual,
                     "ctx": t[max(0, m.start() - 28): m.end() + 28].replace('\n', ' ')})
    return hits


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--pack", default="all")
    ap.add_argument("--mode", default="lower",
                    choices=["all", "lower", "strict"],
                    help="all=every latin word; lower=drop bilingual-looking; "
                         "strict=lower + only lowercase-initial words")
    ap.add_argument("--grep-path")
    ap.add_argument("--word")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--ctx", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = []
    for repo in a.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        packs = (sorted(f for f in os.listdir(en_dir)
                        if f.endswith(".json") and not f.startswith("_"))
                 if a.pack == "all" else [p.strip() for p in a.pack.split(",")])
        for fn in packs:
            ep = os.path.join(en_dir, fn)
            if not os.path.isfile(ep):
                continue
            en = json.load(open(ep, encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                rows.append((os.path.basename(repo), fn, p, e, c))

    if a.grep_path:
        rx = re.compile(a.grep_path)
        rows = [r for r in rows if rx.search(r[2])]

    findings = []
    wordcount = collections.Counter()
    leafcount = 0
    for repo, fn, p, e, c in rows:
        if not c:
            continue
        if SKIP_PATH_RX.search(p):
            continue
        if not re.search('[' + CJK + ']', c):
            continue  # no Chinese at all -> a different defect class
        hits = scan_string(c)
        if a.mode != "all":
            hits = [h for h in hits if not h["bilingual"]]
        if a.mode == "strict":
            hits = [h for h in hits if h["w"][:1].islower()]
        if a.word:
            hits = [h for h in hits if h["w"].lower() == a.word.lower()]
        if not hits:
            continue
        leafcount += 1
        for h in hits:
            wordcount[h["w"]] += 1
        findings.append({"repo": repo, "pack": fn, "path": p,
                         "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                         "n": len(hits),
                         "words": [h["w"] for h in hits],
                         "ctx": [h["ctx"] for h in hits] if a.ctx else None})

    findings.sort(key=lambda r: -r["n"])
    print(f"leaves with residue: {leafcount}   total residue words: {sum(wordcount.values())}")
    print("top words:")
    for w, n in wordcount.most_common(a.limit):
        print(f"  {n:5d}  {w}")
    print("-" * 70)
    for f in findings[:a.limit]:
        print(f"{f['n']:4d}  {f['pack']}::{f['path']}")
        print(f"       {sorted(set(f['words']))[:14]}")
        if a.ctx:
            for cx in f["ctx"][:4]:
                print(f"        … {cx} …")

    if a.out:
        json.dump({"leaves": leafcount, "words": wordcount.most_common(),
                   "findings": findings}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
