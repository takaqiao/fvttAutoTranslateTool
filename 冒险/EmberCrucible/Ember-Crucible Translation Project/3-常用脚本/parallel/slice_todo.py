#!/usr/bin/env python3
"""Group the campaign pack's todo list by journal (and page) so a batch can be
cut along a self-contained unit."""
import json, sys, collections, os

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TODO = os.path.join(P, "5-其他内容", "reports", "ember", "todo", "ember.crucible-adventure.todo.json")

d = json.load(open(TODO, encoding="utf-8"))
items = d["items"]
print("total todo:", len(items), "chars:", d["_meta"]["chars"])

# show a few sample paths at various depths
for it in items[:3]:
    print("SAMPLE", it["path"][:200])

def journal_of(p):
    parts = p.split(".")
    # find 'journals' segment
    if "journals" in parts:
        i = parts.index("journals")
        return parts[i+1] if i + 1 < len(parts) else "?"
    return "(" + parts[0] + ")"

def page_of(p):
    parts = p.split(".")
    if "pages" in parts:
        i = parts.index("pages")
        return parts[i+1] if i + 1 < len(parts) else "?"
    return "(no-page)"

by = collections.Counter()
ch = collections.Counter()
for it in items:
    k = journal_of(it["path"])
    by[k] += 1
    ch[k] += it["chars"]

arg = sys.argv[1] if len(sys.argv) > 1 else None
if not arg:
    print(f"\n{'journal':<48}{'items':>7}{'chars':>10}")
    for k, c in ch.most_common(90):
        print(f"{k[:47]:<48}{by[k]:>7}{c:>10}")
else:
    sel = [it for it in items if arg.lower() in journal_of(it["path"]).lower()]
    pby = collections.Counter(); pch = collections.Counter()
    for it in sel:
        pby[page_of(it["path"])] += 1
        pch[page_of(it["path"])] += it["chars"]
    print(f"\n{arg}: {len(sel)} items / {sum(i['chars'] for i in sel)} chars")
    print(f"{'page':<58}{'items':>7}{'chars':>10}")
    for k, c in pch.most_common():
        print(f"{k[:57]:<58}{pby[k]:>7}{c:>10}")
