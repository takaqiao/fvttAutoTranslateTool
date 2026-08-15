"""H2-B: build an EN / ZH / FR triple table for crucible.

FR (Padhiver/Crucible-FR) and ZH (this project) were both translated from the
same English, by independent teams. Where the two disagree about *what kind of
word* something is — a proper name to transliterate vs a common word to
translate — one of them is probably wrong. That is exactly the failure mode this
project has hit repeatedly (Sockets->插孔, Waterborne->水运, Ordain Streets->
授命街道).

Join key is the ENGLISH STRING (both sides' key shapes differ: they key embedded
collections by _id, we key by name), so this needs no path alignment.

Usage: python h2_triple.py --out OUT.json [--max-len 60]
"""
import argparse
import json
import os
import re

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium"
FR_ROOT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium"
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]+")


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def pairs(en_node, tr_node, path, out):
    """Walk the English tree, carrying the parallel translated tree."""
    if isinstance(en_node, dict):
        for k, v in en_node.items():
            t = tr_node.get(k) if isinstance(tr_node, dict) else None
            pairs(v, t, path + [k], out)
    elif isinstance(en_node, list):
        for i, v in enumerate(en_node):
            t = tr_node[i] if isinstance(tr_node, list) and i < len(tr_node) else None
            pairs(v, t, path + [str(i)], out)
    elif isinstance(en_node, str) and en_node.strip():
        out.append((".".join(path), en_node, tr_node if isinstance(tr_node, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=70)
    args = ap.parse_args()

    table = {}   # en -> {"zh": set, "fr": set, "paths": [...], "packs": set}
    for fn in sorted(os.listdir(os.path.join(ROOT, "en"))):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        en = load(os.path.join(ROOT, "en", fn)).get("entries", {})
        zh_p = os.path.join(ROOT, "cn", fn)
        zh = load(zh_p).get("entries", {}) if os.path.exists(zh_p) else {}
        ours = []
        pairs(en, zh, [], ours)

        fr_en_p = os.path.join(FR_ROOT, "en", fn)
        fr_p = os.path.join(FR_ROOT, "fr", fn)
        theirs = []
        if os.path.exists(fr_en_p) and os.path.exists(fr_p):
            pairs(load(fr_en_p).get("entries", {}),
                  load(fr_p).get("entries", {}), [], theirs)

        for p, e, t in ours:
            if len(e) > args.max_len:
                continue
            d = table.setdefault(e, {"zh": {}, "fr": {}, "packs": set()})
            d["packs"].add(fn)
            if t:
                d["zh"][t] = d["zh"].get(t, 0) + 1
                d.setdefault("zh_path", p)
        for p, e, t in theirs:
            if len(e) > args.max_len:
                continue
            d = table.setdefault(e, {"zh": {}, "fr": {}, "packs": set()})
            if t:
                d["fr"][t] = d["fr"].get(t, 0) + 1

    out = {}
    for e, d in table.items():
        if not d["zh"] or not d["fr"]:
            continue
        out[e] = {
            "zh": sorted(d["zh"], key=lambda k: -d["zh"][k]),
            "fr": sorted(d["fr"], key=lambda k: -d["fr"][k]),
            "packs": sorted(d["packs"]),
            "path": d.get("zh_path"),
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"triples with both zh and fr: {len(out)}")

    # --- signal 1: FR left the English verbatim, we produced pure Chinese ---
    kept = []
    for e, d in out.items():
        fr0 = d["fr"][0]
        zh0 = d["zh"][0]
        if fr0.strip() == e.strip() and CJK.search(zh0) and e.lower() not in zh0.lower():
            kept.append((e, zh0, fr0, d["packs"], d["path"]))
    print(f"\n[S1] FR kept English verbatim, our zh has no English tail: {len(kept)}")
    for e, z, f_, pk, pa in sorted(kept):
        print(f"   EN={e!r}  ZH={z!r}  FR={f_!r}  {pk} {pa}")

    # --- signal 2: FR translation still contains the English word (loan/translit
    #     anchor) while our Chinese dropped it entirely ---
    anch = []
    for e, d in out.items():
        fr0, zh0 = d["fr"][0], d["zh"][0]
        if fr0.strip() == e.strip():
            continue
        words = [w for w in LATIN_WORD.findall(e) if len(w) > 3]
        if not words:
            continue
        keptw = [w for w in words if re.search(rf"\b{re.escape(w)}\b", fr0)]
        if keptw and CJK.search(zh0) and not any(
                re.search(rf"{re.escape(w)}", zh0, re.I) for w in keptw):
            anch.append((e, zh0, fr0, keptw, d["packs"], d["path"]))
    print(f"\n[S2] FR kept a specific English word, our zh dropped it: {len(anch)}")
    for e, z, f_, w, pk, pa in sorted(anch):
        print(f"   EN={e!r} kept={w}  ZH={z!r}  FR={f_!r}  {pk} {pa}")


if __name__ == "__main__":
    main()
