# -*- coding: utf-8 -*-
import json, os, re, sys
REPO = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"
EN = os.path.join(REPO, "compendium", "en")
CN = os.path.join(REPO, "compendium", "cn")
PACKS = ["ember.adventure.json","ember.crucible-adventure.json","ember.crucible-adversary.json"]

def load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def walk(node, prefix=""):
    """yield (path, value) for every string leaf under node"""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, prefix + "/" + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, prefix + "/" + str(i))
    elif isinstance(node, str):
        yield (prefix, node)

def leaves(path):
    d = load(path)
    out = {}
    for p, v in walk(d.get("entries", {}), "/entries"):
        out[p] = v
    return out

def getpath(d, path):
    cur = d
    for seg in path.strip("/").split("/"):
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            if seg not in cur: return None
            cur = cur[seg]
    return cur

def setpath(d, path, val):
    segs = path.strip("/").split("/")
    cur = d
    for i, seg in enumerate(segs[:-1]):
        nxt = segs[i+1]
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            if seg not in cur:
                cur[seg] = [] if nxt.isdigit() else {}
            cur = cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        idx = int(last)
        while len(cur) <= idx: cur.append("")
        cur[idx] = val
    else:
        cur[last] = val
