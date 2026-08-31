# -*- coding: utf-8 -*-
"""Probe: inventory every enricher occurrence in both CN and EN sides of both repos.

Read-only. Emits a JSON inventory to stdout / --out.
Enricher forms handled:
  @Name[args]{label}      (documentLink-style, incl. @UUID @Check @Condition @Embed @ref ...)
  [[/cmd args]]{label}    (roll-command-style, incl. /skillCheck /hazard /attunement /eventState /r ...)
  [[cmd args]]            (bare bracket roll)
"""
import json, os, re, sys, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}

# @Name[ ... ] with balanced-ish bracket handling (no nested [] inside args in practice,
# but UUIDs can contain nothing weird). We scan manually to be safe.
AT_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_]*)\[")
BB_RE = re.compile(r"\[\[")


def _scan_at(text):
    out = []
    for m in AT_RE.finditer(text):
        name = m.group(1)
        i = m.end()  # just after '['
        depth = 1
        while i < len(text) and depth:
            c = text[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            i += 1
        if depth:
            out.append((name, text[m.end():m.end() + 120], None, "UNTERMINATED"))
            continue
        args = text[m.end():i - 1]
        label = None
        if i < len(text) and text[i] == "{":
            j = i + 1
            d2 = 1
            while j < len(text) and d2:
                if text[j] == "{":
                    d2 += 1
                elif text[j] == "}":
                    d2 -= 1
                j += 1
            label = text[i + 1:j - 1]
        out.append((name, args, label, None))
    return out


def _scan_bb(text):
    out = []
    for m in BB_RE.finditer(text):
        i = m.end()
        depth = 2
        # find matching ]]
        j = i
        while j < len(text) - 1:
            if text[j] == "[":
                depth += 1
                j += 1
            elif text[j] == "]":
                depth -= 1
                j += 1
                if depth == 0:
                    break
            else:
                j += 1
        if depth != 0:
            out.append((text[i:i + 120], None, "UNTERMINATED"))
            continue
        inner = text[i:j - 1] if text[j - 1] == "]" else text[i:j]
        # strip one trailing ] if we overcounted
        inner = inner.rstrip("]") if inner.endswith("]]") else inner
        label = None
        if j < len(text) and text[j] == "{":
            k = j + 1
            d2 = 1
            while k < len(text) and d2:
                if text[k] == "{":
                    d2 += 1
                elif text[k] == "}":
                    d2 -= 1
                k += 1
            label = text[j + 1:k - 1]
        out.append((inner, label, None))
    return out


def walk_json(obj, path, sink):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(v, path + [str(k)], sink)
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            walk_json(v, path + [str(idx)], sink)
    elif isinstance(obj, str):
        if "@" in obj or "[[" in obj:
            sink.append(("/".join(path), obj))


def collect(repo_key, side):
    base = os.path.join(REPOS[repo_key], "compendium", side)
    recs = []
    if os.path.isdir(base):
        for fn in sorted(os.listdir(base)):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(base, fn)
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception as e:
                print("PARSE FAIL", p, e, file=sys.stderr)
                continue
            sink = []
            walk_json(d, [], sink)
            for jpath, s in sink:
                recs.append((fn, jpath, s))
    # lang files
    langf = "cn.json" if side == "cn" else "en.json"
    p = os.path.join(REPOS[repo_key], "lang", langf)
    if os.path.isfile(p):
        d = json.load(open(p, encoding="utf-8"))
        sink = []
        walk_json(d, [], sink)
        for jpath, s in sink:
            recs.append(("lang/" + langf, jpath, s))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = {}
    for repo in REPOS:
        for side in ("cn", "en"):
            recs = collect(repo, side)
            items = []
            for fn, jpath, s in recs:
                for name, args, label, err in _scan_at(s):
                    items.append({"kind": "at", "name": name, "args": args,
                                  "label": label, "err": err,
                                  "file": fn, "jpath": jpath})
                for inner, label, err in _scan_bb(s):
                    items.append({"kind": "bb", "name": None, "args": inner,
                                  "label": label, "err": err,
                                  "file": fn, "jpath": jpath})
            result["%s/%s" % (repo, side)] = items
            print("%s/%s: %d strings scanned, %d enricher hits" % (repo, side, len(recs), len(items)), file=sys.stderr)
    json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
