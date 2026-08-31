# -*- coding: utf-8 -*-
"""Fingerprint which upstream ember version each ember_cn release tag was built
against, using two markers that only upstream controls:
  1. the Adventure document's own name ("Ember Beta Two" -> "Ember Early Access")
  2. the highest "Patch X.Y.Z" page present in the Gamemaster's Guide journal
Both live inside the translation file because Babele keys by upstream name.
"""
import json, subprocess, sys, os, re

REPO = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"

def git(*a):
    return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True)

def tags():
    out = git("tag").stdout.decode().split()
    def key(t):
        return [int(x) for x in t.lstrip("v").split(".")]
    return sorted(out, key=key)

CANDIDATES = ["compendium/cn/ember.crucible-adventure.json",
              "compendium/en/ember.crucible-adventure.json"]

def probe(tag):
    for path in CANDIDATES:
        r = git("show", f"{tag}:{path}")
        if r.returncode != 0:
            continue
        try:
            d = json.loads(r.stdout.decode("utf-8"))
        except Exception as e:
            return (path, "PARSE-ERROR: %s" % e, None, None)
        e = d.get("entries", {})
        if not e:
            return (path, "no entries", None, None)
        advname = list(e.keys())[0]
        adv = list(e.values())[0]
        j = adv.get("journals", {}) or {}
        gm = j.get("Gamemaster's Guide")
        patches = []
        if gm:
            for k in (gm.get("pages", {}) or {}):
                m = re.match(r"^Patch (\d+)\.(\d+)\.(\d+)$", k)
                if m:
                    patches.append(tuple(int(x) for x in m.groups()))
        top = ".".join(str(x) for x in max(patches)) if patches else None
        return (path, advname, top, len(patches))
    return (None, "no adventure file at this tag", None, None)

if __name__ == "__main__":
    # ---- pre-flight self-assertions against independently known truths ----
    p, name, top, n = probe("v1.0.0")
    assert name == "Ember Beta Two", f"v1.0.0 adventure name = {name!r}"
    assert top == "0.4.6", f"v1.0.0 top patch = {top!r}"
    p2, name2, top2, n2 = probe("v1.1.24")
    assert name2 == "Ember Early Access", f"v1.1.24 adventure name = {name2!r}"
    assert top2 == "0.6.0", f"v1.1.24 top patch = {top2!r}"
    print("SELF-CHECK OK: v1.0.0 -> Ember Beta Two / Patch 0.4.6 ; "
          "v1.1.24 -> Ember Early Access / Patch 0.6.0\n")
    ts = tags()
    print(f"{'tag':<10} {'date':<12} {'adventure entry name':<22} {'top patch page':<14} {'#patch pages':<12} source")
    for t in ts:
        date = git("log", "-1", "--format=%ad", "--date=short", t).stdout.decode().strip()
        p, name, top, n = probe(t)
        print(f"{t:<10} {date:<12} {str(name):<22} {str(top):<14} {str(n):<12} {p}")
