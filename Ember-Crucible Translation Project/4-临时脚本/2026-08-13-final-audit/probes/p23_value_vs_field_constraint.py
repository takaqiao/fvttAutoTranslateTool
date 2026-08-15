#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p23_value_vs_field_constraint.py  —  同一类别的第二条判据 (只读)

类别抽象 (同 p22)：**写出去的值的类型/约束 ≠ 目标 schema field 的类型/约束
→ Foundry 静默 _cast / 静默回落到 initial / 静默丢键，不抛错不提示。**

p22 查的是「插件 JS 代码」这一侧的写入。
p23 查的是「Babele 译文」这一侧的写入 —— 译文 JSON 的每一个叶子最终都会被
Babele 塞进 document 源数据，同样要过 DataField.clean。

三条子判据：
  C1 blank-fallback : 目标 field 是 blank:false 的 String/HTMLField，
                      而译文给了空串 / 只有空白 / 只有空标签
                      -> clean 抛 validation，落回 `initial`
                      (ember 的 initial 就是英文 "To do in Phase 1/2")
  C2 shape-mismatch : 译文叶子是 object/array/number，而目标 field 是标量；
                      或反过来
  C3 dead-target    : mapping 指向的 system 路径在**任何** subtype 的 schema 里
                      都不存在 -> SchemaField.#cleanKeys 静默剪掉

假阳性模式：
  * C1 只能证明「如果这一条真的进了 clean」；Babele 在 translate 时对
    undefined/空值有自己的短路（converter 里的 isStr 闸），所以要回读 converter
    才能定这一条到底会不会走到 clean。脚本把 converter 名一并打出来。
  * C3 依赖 p22 的正则 schema 解析，effect-affix.mjs 那种
    `schema.x = new fields.Y()` 赋值式声明会被漏掉 -> 已知漏报。
  * 译文 JSON 的键名空间是 babele mapping 的 key（不是 document 路径），
    脚本按 mappings.mjs 的 key->path 表换算。
"""
import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
PACKS = [
    (os.path.join(ROOT, r"1-Ember汉化插件\compendium\cn"), "ember"),
    (os.path.join(ROOT, r"2-Crucible汉化插件\compendium\cn"), "crucible"),
]

# key -> (document path, field-kind) taken verbatim from 3-常用脚本/extract/mappings.mjs
# field-kind verified by grep against crucible 0.10.1 / ember 0.6.0 defineSchema().
KEY_PATH = {
    # Item / ActiveEffect
    "description": ("system.description", "poly:HTMLField|SchemaField{public,private}"),
    "adjective": ("system.adjective", "StringField (ONLY on ActiveEffect.affix; NOT on any Item)"),
    "actions": ("system.actions", "ArrayField(CrucibleActionField)"),
    # Ember pages
    "overview": ("system.overview", "HTMLField blank:false initial='To do in Phase 1'"),
    "exposition": ("system.exposition", "HTMLField blank:false initial='To do in Phase 2'"),
    "summary": ("system.summary", "HTMLField blank:false initial='To do in Phase 2'"),
    "contentOverview": ("system.content.overview", "HTMLField"),
    "contentGamemaster": ("system.content.gamemaster", "HTMLField"),
    "pronunciation": ("system.pronunciation", "StringField"),
    "subtitle": ("system.subtitle", "StringField"),
    "bannerCaption": ("system.banner.caption", "StringField"),
    "height": ("system.height", "StringField"),
    "lifespan": ("system.lifespan", "StringField"),
    "origin": ("system.origin", "StringField"),
    "outcomes": ("system.outcomes", "ArrayField(SchemaField{id,label,summary blank:false,retry})"),
    # base
    "name": ("name", "StringField"),
    "text": ("text.content", "HTMLField"),
    "caption": ("image.caption", "StringField"),
    "tokenName": ("prototypeToken.name", "StringField"),
    "biography": ("system.details.biography", "SchemaField{appearance,age,height,pronouns,weight,public,private}"),
    "levels": ("levels", "Scene.levels[].name"),
    "navName": ("navName", "StringField"),
}

BLANK_FALSE = {"overview", "exposition", "summary"}
EMPTYISH = re.compile(r"^(?:\s|&nbsp;|<p>|</p>|<br\s*/?>|<div>|</div>)*$", re.I)


def walk(obj, path, out):
    """Yield (json-path, key, value) for every leaf and every dict node."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((path + [k], k, v))
            walk(v, path + [k], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, path + [f"[{i}]"], out)


def main():
    c1, c2, c3 = [], [], []
    stats = {"files": 0, "entries": 0, "nodes": 0}
    keyspace = {}
    for pdir, tag in PACKS:
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".json"):
                continue
            stats["files"] += 1
            data = json.load(open(os.path.join(pdir, fn), encoding="utf-8"))
            entries = data.get("entries", {})
            stats["entries"] += len(entries)
            nodes = []
            walk(entries, [], nodes)
            stats["nodes"] += len(nodes)
            for jpath, key, val in nodes:
                keyspace[key] = keyspace.get(key, 0) + 1
                if key not in KEY_PATH:
                    continue
                where = f"{tag}/{fn}:{'/'.join(jpath)}"
                # C1
                if key in BLANK_FALSE and isinstance(val, str) and EMPTYISH.match(val):
                    c1.append({"where": where, "key": key, "value": val,
                               "field": KEY_PATH[key][1]})
                # C2
                expect_scalar = key not in ("actions", "outcomes", "biography",
                                            "levels", "description")
                if expect_scalar and not isinstance(val, str):
                    c2.append({"where": where, "key": key,
                               "type": type(val).__name__,
                               "field": KEY_PATH[key][1]})
                if key == "description" and not isinstance(val, (str, dict)):
                    c2.append({"where": where, "key": key,
                               "type": type(val).__name__, "field": KEY_PATH[key][1]})
                # C3
                if key == "adjective":
                    # dead unless the enclosing document is an ActiveEffect
                    c3.append({"where": where, "key": key, "value": str(val)[:60]})

    print("== 规模 ==", stats)
    print("\n== C1 blank -> 回落到英文 initial ==", len(c1))
    for r in c1[:40]:
        print("  ", r)
    print("\n== C2 类型不符 ==", len(c2))
    for r in c2[:40]:
        print("  ", r)
    print("\n== C3 adjective 出现位置（Item 上是死写入） ==", len(c3))
    for r in c3[:40]:
        print("  ", r)
    print("\n== 译文 JSON 出现过的全部键名（用来找 mapping 覆盖不到的键） ==")
    for k, n in sorted(keyspace.items(), key=lambda x: -x[1])[:60]:
        mark = "" if k in KEY_PATH else "   <-- 不在 KEY_PATH 里"
        print(f"   {k:<28} {n:>7}{mark}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
