"""把 203 条 OPEN verdict 归并成「一个文件只归一个 agent」的工作包。

归并只做机械聚类，不做语义裁决 —— 聚类的作用是**防止两个 agent 改同一个文件**，
以及把明显同根因的条目（如「DialogV2 只翻标题」那一族）摆到一起给同一个 agent，
真正的去重留给拿到工作包的 agent 自己判断。
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "verdicts.json")

# 归属：一个文件只能属于一个 owner，agent 之间因此不会撞车。
OWNERS = [
    ("hc-mjs", ["ember-hardcoded-cn.mjs"]),
    ("cru-register", ["babele-register.js"]),
    ("ember-register", ["register.js"]),
    ("pipeline", ["mappings.mjs", "babele-mappings.js", "runtime-converters.js", "generate_runtime.mjs"]),
    ("lang", ["lang/cn.json", "cn.json"]),
    ("manifest", ["module.json", "ember-cn.css", "crucible-cn.css", ".css", "release-body-template"]),
    ("tooling", ["3-常用脚本", "build_glossary.py", "fix_word_leaks.py", "scan_", "fill_", "apply_", "propagate_", "prune_", "sync_", "normalize_", "repair_"]),
    ("docs", ["PROJECT.md", "LOCAL-PATCHES.md", "README.md", "PARALLEL-RUNBOOK", "STAGE-LOG"]),
    ("compendium", ["compendium/", "compendium\\", ".json"]),
]


def owner_of(files, blob):
    """一条 finding 可能列了多个文件；取**第一个能定归属的** owner。"""
    for f in files:
        norm = f.replace("\\", "/")
        for name, needles in OWNERS:
            if any(n in norm for n in needles):
                return name
    for name, needles in OWNERS:
        if any(n in blob for n in needles):
            return name
    return "other"


STOP = set("的了是在与和或不也都很就把被从对为以及等这那有仍已未还只再又更同一个两三多少条处个人".split())
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}|[一-鿿]{2,4}")


def sig(v):
    text = f"{v.get('short','')} {v.get('fix_sketch','')}"
    toks = {t.lower() for t in TOKEN.findall(text) if t not in STOP}
    return toks


def cluster(items, threshold=0.34):
    """朴素的连通分量聚类：token Jaccard 超阈值即同族。"""
    sigs = [sig(v) for v in items]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = sigs[i], sigs[j]
            if not a or not b:
                continue
            inter = len(a & b)
            if inter and (inter / min(len(a), len(b))) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = defaultdict(list)
    for i, v in enumerate(items):
        groups[find(i)].append(v)
    return list(groups.values())


SEV_ORDER = {"阻断": 0, "严重": 1, "一般": 2, "观感": 3}


def main():
    verdicts = json.load(open(SRC, encoding="utf-8"))
    op = [v for v in verdicts if v["status"] in ("OPEN", "UNCERTAIN")]

    by_owner = defaultdict(list)
    for v in op:
        blob = f"{v.get('short','')} {v.get('evidence','')[:400]}"
        by_owner[owner_of(v.get("target_files") or [], blob)].append(v)

    packages = []
    for owner in sorted(by_owner, key=lambda k: -len(by_owner[k])):
        items = sorted(by_owner[owner], key=lambda v: (SEV_ORDER.get(v.get("severity"), 9), v["idx"]))
        for group in sorted(cluster(items), key=lambda g: (min(SEV_ORDER.get(x.get("severity"), 9) for x in g), -len(g))):
            group.sort(key=lambda v: (SEV_ORDER.get(v.get("severity"), 9), v["idx"]))
            files = sorted({f for v in group for f in (v.get("target_files") or [])})
            packages.append({
                "owner": owner,
                "severity": min((v.get("severity") for v in group), key=lambda s: SEV_ORDER.get(s, 9)),
                "n": len(group),
                "idx": [v["idx"] for v in group],
                "files": files,
                "items": group,
            })

    json.dump(packages, open(os.path.join(HERE, "workpackages.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = ["# 第十四轮工作包（机械聚类，203 条 OPEN）\n"]
    tot = defaultdict(int)
    for owner in sorted(by_owner, key=lambda k: -len(by_owner[k])):
        pkgs = [p for p in packages if p["owner"] == owner]
        lines.append(f"\n## {owner} —— {len(by_owner[owner])} 条 / {len(pkgs)} 族\n")
        for p in pkgs:
            tot[p["severity"]] += p["n"]
            lines.append(f"- **[{p['severity']}] {p['n']} 条** idx {p['idx']}")
            for v in p["items"][:4]:
                lines.append(f"  - `{v['idx']}` {v['short']}")
            if p["n"] > 4:
                lines.append(f"  - …另 {p['n'] - 4} 条")
    lines.append(f"\n\n合计：{dict(tot)}")
    open(os.path.join(HERE, "WORKPACKAGES.md"), "w", encoding="utf-8").write("\n".join(lines))

    print(f"{len(op)} OPEN → {len(packages)} 个族")
    for owner in sorted(by_owner, key=lambda k: -len(by_owner[k])):
        pkgs = [p for p in packages if p["owner"] == owner]
        sev = defaultdict(int)
        for p in pkgs:
            sev[p["severity"]] += p["n"]
        print(f"  {owner:14s} {len(by_owner[owner]):4d} 条 / {len(pkgs):3d} 族   {dict(sev)}")


if __name__ == "__main__":
    main()
