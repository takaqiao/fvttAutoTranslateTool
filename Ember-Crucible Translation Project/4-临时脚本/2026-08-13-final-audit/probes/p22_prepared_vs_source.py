#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p22_prepared_vs_source.py  —  举一反三探针 (只读)

被抽象的缺陷类别（已确认实例：register.js:60-131 ready 迁移把 HTMLField 描述写成对象）：

    「模块代码从一个 **已 prepare / 已初始化** 的对象（Document / DataModel /
      CONFIG 实例 / DOM）上读一个值，按 JS 运行时类型做判断，再把改写后的值
      **写回持久化层**。因为 prepared 值的类型 ≠ 源 schema 的类型，落库时会被
      DataField._cast 静默强转，或者反过来写出去的东西被 toObject() 吞掉。」

判据可机械化为三步：
  A. 枚举两个插件仓库里所有 **持久化写入 sink**
     (`.update(`, `.updateEmbeddedDocuments(`, `.setFlag(`, `updateSource(`,
      `modifyBatch(`, 以及会被 Foundry 采纳的 preUpdate* 钩子里对 changes 的原地改写)
  B. 对每个 sink，抽出它写的 **路径字面量**
  C. 用 crucible 0.10.1 / ember 0.6.0 的真实 defineSchema() 解析该路径的
     **field 类**，与代码写出的 JS 值形状比对

假阳性模式（必须人工复核，脚本自己说清楚）：
  * 正则抽路径只认字面量；`foundry.utils.setProperty(patch, X, …)` 里 X 是变量时抓不到。
  * 一个路径在不同 document subtype 下 field 类不同（system.description 就是），
    脚本对每个 subtype 分别列出，"混合" 结果需要人判。
  * sink 的值是否来自 prepared 对象，脚本只能给出 "同一函数内是否出现
    getProperty(<live doc>, …)" 这一弱信号，必须回源码确认。

用法:  python p22_prepared_vs_source.py
输出:  stdout（表格）+ p22_out.json
"""
import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CRUCIBLE = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
OUT = os.path.join(ROOT, r"4-临时脚本\2026-08-13-final-audit\probes\p22_out.json")

PLUGIN_JS = [
    os.path.join(ROOT, r"1-Ember汉化插件\register.js"),
    os.path.join(ROOT, r"1-Ember汉化插件\babele-mappings.js"),
    os.path.join(ROOT, r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, r"2-Crucible汉化插件\babele-register.js"),
    os.path.join(ROOT, r"2-Crucible汉化插件\babele-mappings.js"),
    os.path.join(ROOT, r"3-常用脚本\release\runtime-converters.js"),
    os.path.join(ROOT, r"3-常用脚本\extract\mappings.mjs"),
]

# ---------------------------------------------------------------- A. sinks
SINK_RE = re.compile(
    r"(?P<call>\.update\s*\(|\.updateEmbeddedDocuments\s*\(|\.updateSource\s*\(|"
    r"\.setFlag\s*\(|modifyBatch\s*\(|\.createEmbeddedDocuments\s*\(|"
    r"\.create\s*\(|setProperty\s*\()")
# in-place mutation of a hook payload counts as a sink too: Foundry re-cleans
# `changes` after preUpdate* (client/data/client-backend.mjs:239-246).
HOOK_RE = re.compile(r"Hooks\.on\(\s*['\"](pre(?:Update|Create)\w+)['\"]")
PATH_RE = re.compile(r"['\"]((?:system|prototypeToken|flags|text|image|effects|items)[\w.\[\]]*)['\"]")


def scan_sinks():
    rows = []
    for path in PLUGIN_JS:
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            m = SINK_RE.search(line)
            if not m:
                continue
            # look at the sink line plus the next 4 lines for a path literal
            window = "\n".join(lines[i - 1:i + 4])
            paths = sorted(set(PATH_RE.findall(window)))
            rows.append({
                "file": os.path.relpath(path, ROOT),
                "line": i,
                "call": m.group("call").strip(),
                "text": line.strip()[:160],
                "paths": paths,
            })
        for m in HOOK_RE.finditer(src):
            ln = src[:m.start()].count("\n") + 1
            rows.append({
                "file": os.path.relpath(path, ROOT),
                "line": ln,
                "call": "HOOK:" + m.group(1),
                "text": src[m.start():m.start() + 120].splitlines()[0],
                "paths": [],
            })
    return rows


# ------------------------------------------------- B/C. schema field table
# Parse `key: new fields.XField(` (and `new crucibleFields.YField(`) out of
# every defineSchema()/schema.X = block in the two upstream packages. Nesting is
# tracked by brace depth so `system.details.biography.public` resolves.
FIELD_RE = re.compile(
    r"(?P<key>[A-Za-z_$][\w$]*)\s*:\s*new\s+(?:fields|crucibleFields|fields\$\d+)\.(?P<cls>\w+)\s*\(")


def parse_model_file(path):
    """Return {dotted.key: FieldClass} for one model file, best-effort."""
    src = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    stack = []          # list of (key, depth_at_open)
    depth = 0
    i = 0
    pending = None
    while i < len(src):
        ch = src[i]
        if ch in "({[":
            depth += 1
            if pending:
                stack.append((pending, depth))
                pending = None
        elif ch in ")}]":
            while stack and stack[-1][1] > depth:
                stack.pop()
            depth -= 1
            while stack and stack[-1][1] > depth:
                stack.pop()
        else:
            m = FIELD_RE.match(src, i)
            if m:
                dotted = ".".join([k for k, _ in stack] + [m.group("key")])
                out.setdefault(dotted, []).append(m.group("cls"))
                # only SchemaField-ish things nest
                if m.group("cls") in ("SchemaField", "ArrayField", "SetField",
                                      "EmbeddedDataField", "CrucibleActionField",
                                      "TypedObjectField"):
                    pending = m.group("key")
                i = m.end() - 1
                depth += 1
                if pending:
                    stack.append((pending, depth))
                    pending = None
        i += 1
    return out


def build_schema_table():
    table = {}
    mdir = os.path.join(CRUCIBLE, "module", "models")
    for fn in sorted(os.listdir(mdir)):
        if fn.endswith(".mjs"):
            table["crucible/" + fn] = parse_model_file(os.path.join(mdir, fn))
    table["ember/ember.mjs"] = parse_model_file(os.path.join(EMBER, "scripts", "ember.mjs"))
    return table


STRINGY = {"StringField", "HTMLField", "FilePathField", "JavaScriptField",
           "ItemIdentifierField", "DocumentUUIDField", "DocumentIdField",
           "ColorField", "JSONField", "AnyField", "ObjectField"}


def main():
    sinks = scan_sinks()
    table = build_schema_table()

    # Which mapped/written leaf keys are NOT string-family fields?
    interesting = ["description", "actions", "adjective", "overview", "exposition",
                   "summary", "outcomes", "pronunciation", "subtitle", "height",
                   "lifespan", "origin", "caption", "biography", "levels", "navName",
                   "name"]
    hits = {}
    for fn, fields in table.items():
        for dotted, clss in fields.items():
            leaf = dotted.split(".")[-1]
            if leaf in interesting:
                hits.setdefault(leaf, []).append((fn, dotted, clss))

    print("=" * 78)
    print("A. 插件仓库里的持久化写入 sink（含 preUpdate 钩子原地改写）")
    print("=" * 78)
    for r in sinks:
        print(f"{r['file']}:{r['line']:<4} [{r['call']}] paths={r['paths']}")
        print(f"      {r['text']}")

    print()
    print("=" * 78)
    print("B. 上游 schema 里这些 key 的真实 field 类（按文件/路径）")
    print("=" * 78)
    for leaf in interesting:
        rows = hits.get(leaf, [])
        classes = sorted({c for _, _, cs in rows for c in cs})
        flag = "  <-- 多型/非字符串" if (len(classes) > 1 or
                                        any(c not in STRINGY for c in classes)) else ""
        print(f"\n### {leaf}: {classes}{flag}")
        for fn, dotted, cs in rows:
            print(f"    {fn:<34} {dotted:<52} {cs}")

    print()
    print("=" * 78)
    print("C. 逐 sink 人工裁决（重跑时对照；上游引用行号来自 crucible 0.10.1 / v14 core）")
    print("=" * 78)
    for line, verdict in sorted(VERDICTS.items()):
        print(f"\nregister.js:{line}\n    {verdict}")

    json.dump({"sinks": sinks, "schema_hits": hits, "verdicts": VERDICTS},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")


VERDICTS = {
    76: "已报（母任务给定实例）ready 迁移把 HTMLField 描述写成对象 -> '[object Object]'。",
    89: "无害死码。crucible 四个 actor model（actor-base/hero/adversary/group）"
        "**没有任何 system.description 字段**，getProperty 恒 undefined，永不进入分支。",
    112: "已报（同 76，embedded 侧）。",
    251: "干净。逐行照抄 crucible 自己的 syncOwnedItems"
         "（crucible-compiled.mjs:48186-48216），含 '_stats.systemVersion' 与 noHook:true。",
    273: "**缺陷 F1**。同一个强转判据接在 preUpdateItem 上（434-436）。"
         "core 在 client-backend.mjs:239 调钩子、246-256 用 updateSource({clean:true}) 重清洗，"
         "所以钩子里对 changes 的原地改写是**被采纳**的 -> HTMLField._cast=String(obj)"
         "（fields.mjs:1705 / 3949）-> '[object Object]'。无 world flag，永久生效。"
         "接在 preCreateItem 上（438-440）则相反：doc 在 client-backend.mjs:92 已由"
         "deepClone(createData) 建好，122 行 operation.data=documents，改 createData 被丢弃 -> 空转。",
    279: "轻。同函数里对 system.actions 的 sanitize，在 preUpdate 侧拿到的是**源数据**"
         "（plain object），不构成 prepared/source 错配；最坏是给 causticPhial 补一个空 effect。",
    365: "**缺陷 F2**。读的是 prepared 的 CrucibleAction DataModel 实例"
         "（EmbeddedDataField.initialize，fields.mjs:2809-2813），mergeObject 只改到实例属性；"
         "写回时 DataModelSchemaField._cast 走 value.toObject()（fields.mjs:2788）"
         "= deepClone(_source)（data.mjs:821），补丁被吞 -> diff 恒空 -> 这个迁移永远修不了东西。",
    388: "同 365（embedded 侧）。",
}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
