# -*- coding: utf-8 -*-
"""Dump the upstream-authored patch-note journal pages from ember's
Gamemaster's Guide as plain text. Source of truth: the module's own content,
not any third-party changelog."""
import json, io, os, sys, re, html

def strip(h):
    h = re.sub(r"(?is)<(script|style).*?</\1>", "", h)
    h = re.sub(r"(?i)<li[^>]*>", "\n  - ", h)
    h = re.sub(r"(?i)<(h[1-6])[^>]*>", "\n\n### ", h)
    h = re.sub(r"(?i)</(h[1-6])>", "\n", h)
    h = re.sub(r"(?i)<(p|div|tr|br\s*/?)[^>]*>", "\n", h)
    h = re.sub(r"(?i)<td[^>]*>", " | ", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = html.unescape(h)
    h = re.sub(r"\n{3,}", "\n\n", h)
    return h.strip()

def dump(pack_path, want_prefix="Patch "):
    d = json.load(io.open(pack_path, encoding="utf-8"))
    adv = list(d["entries"].values())[0]
    gm = adv["journals"]["Gamemaster's Guide"]["pages"]
    out = {}
    for k, v in gm.items():
        if k.startswith(want_prefix):
            body = []
            for fld in ("text", "text.content", "content"):
                cur = v
                ok = True
                for part in fld.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        ok = False; break
                if ok and isinstance(cur, str) and fld not in ("name", "title"):
                    body.append(cur)
            out[k] = strip("\n".join(body)) if body else "(no html body field found; keys=%s)" % list(v.keys())
    return out

if __name__ == "__main__":
    pack, only = sys.argv[1], sys.argv[2:]
    pages = dump(pack)
    keys = [k for k in pages if not only or k in only]
    def vkey(k):
        return [int(x) for x in k.split()[1].split(".")]
    for k in sorted(keys, key=vkey):
        print("\n" + "="*70 + f"\n# {k}\n" + "="*70)
        print(pages[k])
