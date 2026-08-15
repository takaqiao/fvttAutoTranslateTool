# -*- coding: utf-8 -*-
"""Mechanical prescreen for J02 Arctus Plateau Gazetteer."""
import json, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\G10\j02.ember.json"
rows = json.load(open(SRC, encoding="utf-8"))["rows"]

TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>")

def tags(s):
    return [(m.group(1), m.group(2).lower()) for m in TAG.finditer(s)]

def ids(s):
    return re.findall(r'id="([^"]*)"', s)

def uuids(s):
    return re.findall(r"@UUID\[([^\]]*)\]", s)

def dtdd(s):
    return re.findall(r"<(dt|dd)\b[^>]*>(.*?)</\1>", s, re.S)

problems = collections.defaultdict(list)

for x in rows:
    p = x["path"].replace("entries.Ember Early Access.journals.Arctus Plateau Gazetteer.", "")
    en, cn = x["en"], x["cn"] or ""
    # 1 tag multiset
    te, tc = collections.Counter(tags(en)), collections.Counter(tags(cn))
    if te != tc:
        diff = {k: (te.get(k, 0), tc.get(k, 0)) for k in set(te) | set(tc) if te.get(k, 0) != tc.get(k, 0)}
        problems["TAGDIFF"].append((p, diff))
    # 2 id preservation
    ie, ic = collections.Counter(ids(en)), collections.Counter(ids(cn))
    if ie != ic:
        problems["IDDIFF"].append((p, sorted(set(ie) ^ set(ic))))
    # 3 uuid preservation
    ue, uc = collections.Counter(uuids(en)), collections.Counter(uuids(cn))
    if ue != uc:
        problems["UUIDDIFF"].append((p, [k for k in set(ue) | set(uc) if ue.get(k, 0) != uc.get(k, 0)]))
    # 4 dt/dd count
    de, dc = dtdd(en), dtdd(cn)
    if len(de) != len(dc):
        problems["DTDDCOUNT"].append((p, len(de), len(dc)))
    # 5 block count
    be = len(re.findall(r"<(p|li|h[1-6]|dt|dd|td|th|tr|blockquote)\b", en))
    bc = len(re.findall(r"<(p|li|h[1-6]|dt|dd|td|th|tr|blockquote)\b", cn))
    if be != bc:
        problems["BLOCKCOUNT"].append((p, be, bc))
    # 6 strong position: text before/after each <strong>
    se = re.findall(r"<strong>(.*?)</strong>", en, re.S)
    sc = re.findall(r"<strong>(.*?)</strong>", cn, re.S)
    if len(se) != len(sc):
        problems["STRONGCOUNT"].append((p, len(se), len(sc)))
    # 7 numbers
    ne = collections.Counter(re.findall(r"\d+", re.sub(r"<[^>]*>", "", en)))
    nc = collections.Counter(re.findall(r"\d+", re.sub(r"<[^>]*>", "", cn)))
    if ne != nc:
        d = {k: (ne.get(k, 0), nc.get(k, 0)) for k in set(ne) | set(nc) if ne.get(k, 0) != nc.get(k, 0)}
        problems["NUMDIFF"].append((p, d))
    # 8 gamemaster block count
    ge = en.count("gamemaster")
    gc = cn.count("gamemaster")
    if ge != gc:
        problems["GMBLOCK"].append((p, ge, gc))

for k in problems:
    print("=" * 20, k, len(problems[k]))
    for item in problems[k]:
        print("  ", item)
