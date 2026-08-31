# -*- coding: utf-8 -*-
"""U1: dump EN/CN context around each @UUID finding in [lo, hi)."""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SWAP = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def get(root, path):
    """Same tolerant split apply_translations.py uses: keys may contain dots
    (`Patch 0.2.0`), so a naive split shreds them."""
    naive = path.split(".")
    v = get_at(root, naive)
    if v is not None:
        return v
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + ".")]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition(".")
        parts.append(head)
        if isinstance(node, list):
            try:
                node = node[int(head)]
            except (ValueError, IndexError):
                node = None
        elif isinstance(node, dict):
            node = node.get(head)
        else:
            node = None
    return get_at(root, parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=76)
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--only", help="comma list of finding indices")
    a = ap.parse_args()
    only = set(int(x) for x in a.only.split(",")) if a.only else None

    findings = json.load(open(SWAP, encoding="utf-8"))["findings"]
    cache = {}
    for i in range(a.lo, a.hi):
        f = findings[i]
        if only is not None and i not in only:
            continue
        repo, pack = f["repo"], f["pack"]
        for side in ("en", "cn"):
            key = (repo, pack, side)
            if key not in cache:
                cache[key] = json.load(open(os.path.join(P, repo, "compendium", side, pack), encoding="utf-8"))
        en = cache[(repo, pack, "en")]
        cn = cache[(repo, pack, "cn")]
        bp = f["batch_path"]
        try:
            ent = get(en["entries"], bp)
        except Exception as e:
            ent = f"<EN MISSING {e}>"
        try:
            cnt = get(cn["entries"], bp)
        except Exception as e:
            cnt = f"<CN MISSING {e}>"
        # links may be written relative (`@UUID[.<pageId>]`) or with a section
        # anchor (`...#fighting-kaftor`); the scanner normalises both away.
        full = f["target"]
        short = "." + full.split(".")[-1]
        tgt = "(?:" + re.escape(full) + "|" + re.escape(short) + ")(?:#[^\\]]*)?"
        print("=" * 100)
        print(f"[{i}] {pack} :: {bp}")
        print(f"    target={f['target']}  en_label={f['en_label']!r}  cn_label={f['cn_label']!r}  maj={f['majority']['label']!r} {f['majority']['support']}/{f['majority']['total']}")
        for name, txt in (("EN", ent), ("CN", cnt)):
            if not isinstance(txt, str):
                print(f"  {name}: {txt}")
                continue
            hits = list(re.finditer(r"@UUID\[" + tgt + r"\](\{[^{}]*\})?", txt))
            if not hits:
                print(f"  {name}: <no hit>  len={len(txt)}")
            for h in hits:
                s = max(0, h.start() - a.win)
                e = min(len(txt), h.end() + a.win)
                print(f"  {name} @{h.start()}: ...{txt[s:e]}...")
        print()


if __name__ == "__main__":
    main()
