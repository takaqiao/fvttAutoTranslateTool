"""H2-B: EN / our-CN / their-FR triples on the same path, for term auditing.

French and Chinese were translated from the same English by independent teams,
so the FRENCH TREATMENT OF A WORD IS INDEPENDENT EVIDENCE about what kind of
word it is.  The failure mode this project keeps hitting is a proper noun
translated as if it were a common noun (Sockets->插孔, Waterborne->水运,
Ordain Streets->授命街道).  French makes that visible: when the French keeps the
English string verbatim, the French team judged it a proper noun.

Alignment: both sides key entries differently (we by `name`, FR by `_id`), and
both ship their own English baseline.  So each side is joined en<->translation
by its OWN key, then the two are joined on the English entry name.

Usage:
  python h2_fr_term_triples.py --out triples.json [--names-only]
"""
import argparse
import json
import os
import re

OUR_EN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\en"
OUR_CN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\cn"
FR_EN = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium\en"
FR_FR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium\fr"

CJK = re.compile(r"[\u4e00-\u9fff]")
TAG = re.compile(r"<[^>]+>")


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def pair_leaves(en, tr, prefix="", out=None):
    """Walk the English tree, carrying the translation tree in lockstep.

    Keys match by construction inside one project (the translation file was
    generated from that project's own English baseline).
    """
    if out is None:
        out = []
    if isinstance(en, dict):
        for k, v in en.items():
            t = tr.get(k) if isinstance(tr, dict) else None
            pair_leaves(v, t, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(en, list):
        tl = tr if isinstance(tr, list) else []
        for i, v in enumerate(en):
            pair_leaves(v, tl[i] if i < len(tl) else None,
                        f"{prefix}[{i}]", out)
    elif isinstance(en, str) and en.strip():
        out.append((prefix, en, tr if isinstance(tr, str) else None))
    return out


def entry_name(v):
    return v.get("name") if isinstance(v, dict) and isinstance(v.get("name"), str) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--names-only", action="store_true")
    args = ap.parse_args()

    triples = []
    for fn in sorted(f for f in os.listdir(OUR_EN)
                     if f.endswith(".json") and not f.startswith("_")):
        if not os.path.exists(os.path.join(FR_EN, fn)):
            continue
        oen, ocn = load(os.path.join(OUR_EN, fn)), load(os.path.join(OUR_CN, fn))
        fen, ffr = load(os.path.join(FR_EN, fn)), load(os.path.join(FR_FR, fn))

        # our side: en-key -> {leafpath: (en, cn)}
        ours = {}
        for k, v in oen.get("entries", {}).items():
            nm = entry_name(v)
            if not nm:
                continue
            got = pair_leaves(v, ocn.get("entries", {}).get(k, {}))
            ours.setdefault(nm, {}).update({p: (e, t) for p, e, t in got})

        theirs = {}
        for k, v in fen.get("entries", {}).items():
            nm = entry_name(v)
            if not nm:
                continue
            got = pair_leaves(v, ffr.get("entries", {}).get(k, {}))
            theirs.setdefault(nm, {}).update({p: (e, t) for p, e, t in got})

        # Join on English *string value*, per entry: both sides may put the same
        # English at differently-shaped paths, so match on (entry, en-text).
        for nm in set(ours) & set(theirs):
            fr_by_en = {}
            for p, (e, t) in theirs[nm].items():
                if t:
                    fr_by_en.setdefault(e, (p, t))
            for p, (e, cn) in ours[nm].items():
                if args.names_only and not p.endswith("name"):
                    continue
                hit = fr_by_en.get(e)
                if not hit:
                    continue
                triples.append({
                    "pack": fn, "entry": nm, "our_path": p, "fr_path": hit[0],
                    "en": e, "cn": cn, "fr": hit[1],
                })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=1)
    print(f"{len(triples)} aligned EN/CN/FR triples -> {args.out}")


if __name__ == "__main__":
    main()
