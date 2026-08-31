# -*- coding: utf-8 -*-
"""U3: print EN/CN context around the exact @UUID occurrence of each finding.

Reuses scan_uuid_swap's link tokenizer so the finding's `i` (link ordinal inside
the leaf) locates the same occurrence.
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SC = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"

LINK = re.compile(r'@([A-Za-z]+)\[([^\]\n]*)\]\{([^}\n]*)\}')
CMD = re.compile(r'\[\[([^\]\n]*)\]\]\{([^}\n]*)\}')
WS = re.compile(r'\s+')


def links_in(s):
    out = []
    for m in LINK.finditer(s):
        out.append({'at': m.start(), 'end': m.end(), 'label': m.group(3),
                    'body': m.group(2), 'whole': m.group(0)})
    for m in CMD.finditer(s):
        out.append({'at': m.start(), 'end': m.end(), 'label': m.group(2),
                    'body': m.group(1), 'whole': m.group(0)})
    out.sort(key=lambda d: d['at'])
    return out


def leaf_strings(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaf_strings(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaf_strings(v, path + [str(i)], out)
    elif isinstance(node, str):
        out['.'.join(path)] = node


packs = {}


def get(repo, pack, side):
    """dotted path -> string, built exactly like scan_uuid_swap's leaf_strings so
    keys that contain dots (`Patch 0.2.0`) resolve identically."""
    k = (repo, pack, side)
    if k not in packs:
        doc = json.load(open(os.path.join(P, repo, 'compendium', side, pack),
                             encoding='utf-8'))
        flat = {}
        leaf_strings(doc, [], flat)
        packs[k] = flat
    return packs[k]


def leaf(flat, path):
    return flat[path]


d = json.load(open(SC + "/uuid_swap.json", encoding="utf-8"))
lo, hi = int(sys.argv[1]), int(sys.argv[2])
W = int(sys.argv[3]) if len(sys.argv) > 3 else 180

for i in range(lo, hi):
    x = d["findings"][i]
    cn = leaf(get(x['repo'], x['pack'], 'cn'), x['path'])
    try:
        en = leaf(get(x['repo'], x['pack'], 'en'), x['path'])
    except Exception:
        en = None
    cls = links_in(cn)
    L = cls[x['i']] if x['i'] < len(cls) else None
    print(f"\n===== {i} | {x['repo']} | {x['pack']} | {x['path']}")
    print(f"  target={x['target']}  en={x['en_label']!r} cn={x['cn_label']!r} "
          f"maj={x['majority']['label']!r} {x['majority']['support']}/{x['majority']['total']} basis={x.get('basis')}")
    if L:
        a, b = max(0, L['at'] - W), min(len(cn), L['end'] + W)
        print(f"  CN …{cn[a:b]}…")
        print(f"  HIT  {L['whole']}")
    if en:
        # find the EN occurrence with the same target tail
        tail = x['key']
        hits = [m for m in links_in(en) if tail in m['body']]
        for h in hits:
            a, b = max(0, h['at'] - W), min(len(en), h['end'] + W)
            print(f"  EN …{en[a:b]}…")
