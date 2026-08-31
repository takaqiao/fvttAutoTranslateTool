# -*- coding: utf-8 -*-
"""Ad-hoc probe: English-gated leaf listing with occurrence counts + context.

  python probe.py --en "\\bAcid\\b" --cn 酸液 --ctx 40 --limit 60
  python probe.py --en "\\bPresence\\b" --cn 气场 --also-en "Formidable Presence"
"""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPOS = [BASE + "/1-Ember汉化插件", BASE + "/2-Crucible汉化插件"]
SKIP = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def rows(repos=None):
    out = []
    for r in (repos or REPOS):
        d = os.path.join(r, "compendium", "en")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            en = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            cp = os.path.join(r, "compendium", "cn", fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                out.append((os.path.basename(r), fn, p, e, c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True)
    ap.add_argument("--cn", required=True, help="regex on the Chinese side")
    ap.add_argument("--also-en", help="extra regex; report whether EN also matches it")
    ap.add_argument("--unless", help="skip leaves whose EN matches this")
    ap.add_argument("--ctx", type=int, default=34)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--enctx", action="store_true")
    a = ap.parse_args()

    rx_en = re.compile(a.en)
    rx_cn = re.compile(a.cn)
    rx_also = re.compile(a.also_en) if a.also_en else None
    rx_un = re.compile(a.unless) if a.unless else None

    tot_leaf = tot_occ = 0
    shown = 0
    for repo, fn, p, e, c in rows():
        if not rx_en.search(e) or not c:
            continue
        if rx_un and rx_un.search(e):
            continue
        ms = list(rx_cn.finditer(c))
        if not ms:
            continue
        tot_leaf += 1
        tot_occ += len(ms)
        if shown >= a.limit:
            continue
        shown += 1
        flag = ""
        if rx_also:
            flag = "  ALSO-EN=%s" % bool(rx_also.search(e))
        print(f"[{tot_leaf}] {repo}/{fn}::{p}  x{len(ms)}{flag}")
        for m in ms[:6]:
            print("    CN …" + c[max(0, m.start() - a.ctx):m.end() + a.ctx].replace("\n", " ") + "…")
        if a.enctx:
            for m in list(rx_en.finditer(e))[:3]:
                print("    EN …" + e[max(0, m.start() - a.ctx * 2):m.end() + a.ctx * 2].replace("\n", " ") + "…")
    print(f"--- leaves={tot_leaf}  occurrences={tot_occ}")


if __name__ == "__main__":
    main()
