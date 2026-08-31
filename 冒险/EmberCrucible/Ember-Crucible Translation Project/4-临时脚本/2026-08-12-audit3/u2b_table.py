# -*- coding: utf-8 -*-
"""Emit the U2 per-finding verdict table (findings 76..150) as markdown.

Columns: 序号 | 目标 | 英文标签 | 原中文 | 判定 | 依据.
"原中文" is the label in the working tree (the audit-3 batches are already applied);
"判定" is computed from whether the U2 batch changes that label.
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SCR = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
SW = json.load(open(SCR + "/uuid_swap.json", encoding="utf-8"))["findings"]
IDS = json.load(open(ROOT + r"/4-临时脚本/2026-08-12-fix/reports/ember_ids.json", encoding="utf-8"))
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?')
BAT = {}
for fn in ["ember.adventure.json", "ember.crucible-adventure.json"]:
    BAT[fn] = json.load(open(SCR + f"/batches/U2__ember__{fn}", encoding="utf-8"))
_c = {}


def pack(repo, fn, side):
    k = (repo, fn, side)
    if k not in _c:
        _c[k] = json.load(open(os.path.join(ROOT, repo, "compendium", side, fn), encoding="utf-8"))
    return _c[k]


def resolve(base, p):
    segs = p.split("."); cur = base; i = 0
    while i < len(segs):
        if isinstance(cur, list):
            cur = cur[int(segs[i])]; i += 1; continue
        for j in range(len(segs), i, -1):
            k = ".".join(segs[i:j])
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]; i = j; break
        else:
            return None
    return cur


def labels(s, key):
    return [m.group(3) for m in MARK.finditer(s or "")
            if (m.group(2) or "").split()[0].split("#")[0].split(".")[-1] == key]


REASON = json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/u2b_reasons.json", encoding="utf-8"))

rows = []
for i in range(76, 151):
    x = SW[i]
    bp = x["path"][len("entries."):]
    cur = resolve(pack(x["repo"], x["pack"], "cn"), x["path"])
    en = resolve(pack(x["repo"], x["pack"], "en"), x["path"])
    new = BAT.get(x["pack"], {}).get(bp)
    el = labels(en, x["key"])
    cl = labels(cur, x["key"])
    nl = labels(new, x["key"]) if new else cl
    # position of this finding inside the leaf
    pos = None
    for k, v in enumerate(cl):
        if v == x["cn_label"]:
            pos = k; break
    if pos is None:
        pos = 0
    old_lab = cl[pos] if pos < len(cl) else x["cn_label"]
    new_lab = nl[pos] if pos < len(nl) else old_lab
    enl = el[pos] if pos < len(el) else None
    verdict = "**改**" if new_lab != old_lab else "不改"
    info = IDS.get(x["key"])
    dn = info["name"] if info else "(外部包)"
    rows.append((i, dn, enl, old_lab, new_lab, verdict, REASON.get(str(i), "")))

print("| 序号 | 目标（EN 文档名） | 英文标签 | 原中文 | 判定 | 依据 |")
print("|---|---|---|---|---|---|")
for i, dn, enl, old, new, v, r in rows:
    lab = f"`{enl}`" if enl else "*(无标签)*"
    val = f"{old} → **{new}**" if v.startswith("**") else old
    print(f"| {i} | {dn} | {lab} | {val} | {v} | {r} |")
