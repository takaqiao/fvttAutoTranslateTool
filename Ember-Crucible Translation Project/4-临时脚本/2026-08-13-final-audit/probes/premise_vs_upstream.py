#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
premise_vs_upstream.py  —  只读探针，不写库。

把「形状强转的前提与上游 schema 相反」抽象成一条可机械化的判据，全库扩查。

判据（三条子判据，全部只读）：

  P1  路径存在性：我们代码里写死的每一条 `system.*` 文档路径，在上游
      (crucible 0.10.1 `crucible-compiled.mjs` 的 defineSchema / ember 0.6.0
      `ember.mjs`) 的 schema 里是否真的存在？不存在 = 前提落空（死路径）。

  P2  形状一致性：路径存在时，上游声明的 DataField 类型是什么？
      我们代码对它做的 `typeof x === 'string'` / `Array.isArray(x)` 判据，
      其「为真」的分支是否恰好落在**上游声明的正确形状**上？
      是 = 判据反转（把好数据当坏数据修）。

  P3  兄弟键枚举：我们的补丁遍历某个上游配置对象时，枚举的键集合
      与上游实际写入的键集合是否一致？漏掉的兄弟键 = 静默盲区。

假阳性模式（必须自己交代清楚）：
  * schema 提取用正则扫 `defineSchema()` 里的 `name: new fields.XField(`，
    只认**缩进两级以内**的直接字段；`Object.assign(schema, {...})`、
    `schema.x = ...`、`...Xxx.defineSchema()` 展开这三种写法需要单独兜，
    脚本对后两种做了处理，对第一种可能漏。所以 P1 报的「缺失」必须回到
    crucible-compiled.mjs 逐条肉眼复核（本轮 4 条候选全部人工复核过）。
  * 派生属性（class field，如 CrucibleBaseActor 的 `actions = this.actions`）
    不在 schema 里但运行时存在 —— 脚本会额外扫 class field，避免把
    「不是 schema 字段但运行时有值」误报成「完全不存在」。
  * P3 只能扫 `crucible.CONFIG.<key>` 形式的字面写入，动态 key 扫不到。

用法：
    python premise_vs_upstream.py
"""

import json
import os
import re
import sys

PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CRUCIBLE = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"
CRUCIBLE_JSON = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\system.json"
EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"

OURS = [
    os.path.join(PROJ, "1-Ember汉化插件", "register.js"),
    os.path.join(PROJ, "1-Ember汉化插件", "babele-mappings.js"),
    os.path.join(PROJ, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(PROJ, "2-Crucible汉化插件", "babele-mappings.js"),
    os.path.join(PROJ, "2-Crucible汉化插件", "babele-register.js"),
    os.path.join(PROJ, "3-常用脚本", "extract", "mappings.mjs"),
    os.path.join(PROJ, "3-常用脚本", "release", "runtime-converters.js"),
]


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------- upstream

FIELD_RE = re.compile(r"^\s{2,8}(\w+):\s*new (?:fields\.|foundry\.data\.fields\.)?(\w+)\(")
SCHEMA_ASSIGN_RE = re.compile(r"^\s{2,8}schema\.(\w+)\s*=\s*new (?:fields\.)?(\w+)\(")
SPREAD_RE = re.compile(r"^\s*\.\.\.(\w+)\.defineSchema\(\)")
CLASSFIELD_RE = re.compile(r"^\s{2}(\w+)\s*=\s*this\.\1;")


def index_classes(src):
    """class name -> (start_line_idx, end_line_idx) over a rollup-compiled bundle."""
    lines = src.split("\n")
    starts = []
    for i, l in enumerate(lines):
        m = re.match(r"^\}?class (\w+)(?: extends ([\w.$()]+))?", l)
        if m:
            starts.append((i, m.group(1), m.group(2)))
    out = {}
    for n, (i, name, parent) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        out[name] = (i, end, parent)
    return lines, out


def schema_of(lines, span, classes, seen=None):
    """Shallow {field: FieldType} for one class, following `...X.defineSchema()`
    spreads and `schema.x = new fields.Y(` assignments, plus the parent chain."""
    seen = seen or set()
    i, end, parent = span
    fields = {}
    if parent and parent in classes and parent not in seen:
        seen.add(parent)
        fields.update(schema_of(lines, classes[parent], classes, seen))
    in_define = False
    for n in range(i, end):
        l = lines[n]
        if "defineSchema()" in l and "static" in l:
            in_define = True
            continue
        if in_define:
            m = SPREAD_RE.match(l)
            if m and m.group(1) in classes and m.group(1) not in seen:
                seen.add(m.group(1))
                fields.update(schema_of(lines, classes[m.group(1)], classes, seen))
                continue
            m = FIELD_RE.match(l)
            if m:
                fields.setdefault(m.group(1), m.group(2))
                continue
            m = SCHEMA_ASSIGN_RE.match(l)
            if m:
                fields[m.group(1)] = m.group(2)
                continue
            if re.match(r"^\s{2}\}", l) and n > i + 3:
                in_define = False
        # derived class fields (`actions = this.actions;`) — runtime-only
        m = CLASSFIELD_RE.match(l)
        if m:
            fields.setdefault(m.group(1), "<DERIVED class field, not persisted>")
    return fields


def main():
    csrc = read(CRUCIBLE)
    clines, cclasses = index_classes(csrc)
    sysjson = json.load(open(CRUCIBLE_JSON, encoding="utf-8"))

    report = {"P1_paths": [], "P2_predicates": [], "P3_sibling_keys": []}

    # ---- ground truth tables we care about ------------------------------
    targets = {
        "Item(physical)": "CruciblePhysicalItem",
        "Item(talent)": "CrucibleTalentItem",
        "Item(spell)": "CrucibleSpellItem",
        "Item(ancestry)": "CrucibleAncestryItem",
        "Item(archetype)": "CrucibleArchetypeItem",
        "Item(background)": "CrucibleBackgroundItem",
        "Item(taxonomy)": "CrucibleTaxonomyItem",
        "Actor(base)": "CrucibleBaseActor",
        "Actor(hero)": "CrucibleHeroActor",
        "Actor(adversary)": "CrucibleAdversaryActor",
        "Actor(group)": "CrucibleGroupActor",
        "ActiveEffect(affix)": "CrucibleAffixActiveEffect",
        "Action": "CrucibleAction",
    }
    schemas = {}
    for label, cls in targets.items():
        if cls not in cclasses:
            print(f"  [!] class not found: {cls}", file=sys.stderr)
            continue
        schemas[label] = schema_of(clines, cclasses[cls], cclasses)

    # ---- P1: every `system.<top>` path our code touches -------------------
    paths = set()
    for f in OURS:
        if not os.path.exists(f):
            continue
        for m in re.finditer(r"['\"]system\.([A-Za-z0-9_.]+)['\"]", read(f)):
            paths.add(m.group(1))
    for top in sorted({p.split(".")[0] for p in paths}):
        row = {"path": "system." + top, "present_in": [], "absent_in": []}
        for label, fields in schemas.items():
            if top in fields:
                row["present_in"].append(f"{label}:{fields[top]}")
            else:
                row["absent_in"].append(label)
        report["P1_paths"].append(row)

    # ---- P2: shape predicates in our JS ----------------------------------
    pred_re = re.compile(r"typeof\s+([\w.?\[\]']+)\s*([!=]==)\s*'(\w+)'|Array\.isArray\(([^)]*)\)")
    for f in OURS:
        if not os.path.exists(f):
            continue
        for n, l in enumerate(read(f).split("\n"), 1):
            for m in pred_re.finditer(l):
                report["P2_predicates"].append(
                    {"file": os.path.relpath(f, PROJ), "line": n, "src": l.strip()[:150]}
                )

    # ---- P3: sibling-key enumeration on crucible.CONFIG -------------------
    upstream_writes = set()
    esrc = read(EMBER)
    for m in re.finditer(r"crucible\.CONFIG\.(\w+)", esrc):
        upstream_writes.add(m.group(1))
    ours_enumerated = set()
    hc = read(os.path.join(PROJ, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"))
    for m in re.finditer(r'\[\["(\w+)", *[A-Z_]+\], *\["(\w+)", *[A-Z_]+\]\]', hc):
        ours_enumerated.update([m.group(1), m.group(2)])
    report["P3_sibling_keys"] = {
        "ember_touches_crucible_CONFIG": sorted(upstream_writes),
        "our_patch_enumerates": sorted(ours_enumerated),
        "gap": sorted(upstream_writes - ours_enumerated),
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "premise_vs_upstream.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # human summary
    print("== P1 paths absent from every schema we know ==")
    for r in report["P1_paths"]:
        if not r["present_in"]:
            print("  MISSING EVERYWHERE:", r["path"])
        else:
            print(f'  {r["path"]:28s} -> {", ".join(r["present_in"])[:160]}')
    print(f'\n== P2: {len(report["P2_predicates"])} shape predicates collected (manual review) ==')
    print("\n== P3 sibling-key gap ==")
    print("  ember writes :", report["P3_sibling_keys"]["ember_touches_crucible_CONFIG"])
    print("  we enumerate :", report["P3_sibling_keys"]["our_patch_enumerates"])
    print("  GAP          :", report["P3_sibling_keys"]["gap"])
    print("\nwrote", out)


if __name__ == "__main__":
    main()
