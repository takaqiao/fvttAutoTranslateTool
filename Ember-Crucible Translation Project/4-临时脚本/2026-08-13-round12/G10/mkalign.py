# -*- coding: utf-8 -*-
"""Emit block-aligned EN/CN pairs for selected pages."""
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\G10"
rows = json.load(open(os.path.join(D, "j02.ember.json"), encoding="utf-8"))["rows"]
PRE = "entries.Ember Early Access.journals.Arctus Plateau Gazetteer."
by = collections.OrderedDict()
for x in rows:
    p = x["path"][len(PRE):]
    m = re.match(r"pages\.(.*?)\.(name|overview|exposition|text)$", p)
    if m:
        by.setdefault(m.group(1), {})[m.group(2)] = x

OPEN = re.compile(r"<(p|li|h[1-6]|dt|dd|td|th|blockquote)\b[^>]*>", re.I)


def blocks(s):
    s = re.sub(r"[ \t]*\n[ \t]*", "", s)
    s = OPEN.sub(lambda m: "\x00" + m.group(1) + "\x01", s)
    s = re.sub(r"</(p|li|h[1-6]|dt|dd|td|th|blockquote)>", "\x00", s, flags=re.I)
    parts = s.split("\x00")
    out = []
    for seg in parts:
        if "\x01" in seg:
            tag, body = seg.split("\x01", 1)
        else:
            tag, body = "-", seg
        body = re.sub(r"<span[^>]*>", "«", body)
        body = re.sub(r"</span>", "»", body)
        body = re.sub(r"<[^>]*>", "", body).strip()
        if body:
            out.append((tag, body))
    return out


for pg in sys.argv[1:]:
    print("\n" + "#" * 60 + f"\n# {pg}\n" + "#" * 60)
    for f in ["name", "overview", "exposition"]:
        r = by[pg].get(f)
        if r:
            print(f"[{f}] EN: {re.sub(r'<[^>]*>','',r['en'])}")
            print(f"[{f}] CN: {re.sub(r'<[^>]*>','',r['cn'] or '')}")
    r = by[pg]["text"]
    be, bc = blocks(r["en"]), blocks(r["cn"] or "")
    print(f"--- text blocks EN={len(be)} CN={len(bc)} ---")
    for i in range(max(len(be), len(bc))):
        e = be[i] if i < len(be) else ("?", "")
        c = bc[i] if i < len(bc) else ("?", "")
        print(f"{i:>3} {e[0]:>2} E| {e[1]}")
        print(f"    {c[0]:>2} C| {c[1]}")
