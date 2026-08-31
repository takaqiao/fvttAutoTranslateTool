# -*- coding: utf-8 -*-
"""
举一反三探针 #2：**被翻译的字段路径，在上游 schema 里到底是什么 field 类**

判据（对已确认实例的机械化抽象）
--------------------------------
已确认实例 = `system.description` 在 crucible 的 Item 子类型之间是**多态**的
（HTMLField vs SchemaField），而我方有一段代码只按单一形状处理它。

抽象成可扫的判据：
  对**每一个我方会写入的字段路径**（babele mapping 的 path、以及插件代码 setProperty 的路径），
  在上游 schema 源码里找出该叶名的**全部** `new fields.XxxField(...)` 声明，
  然后看：
    (A) 同一个叶名是否出现 **两种以上不同的 field 类** → 多态位点，
        必须确认我方的 converter/代码对每一种形状都成立；
    (B) 该 field 类是否**不是纯字符串类**（StringField/HTMLField/FilePathField 之外的
        NumberField / SetField / ArrayField / SchemaField / BooleanField / DocumentUUIDField…）
        → 往里写译文字符串会被 _cast 静默改型或校验失败；
    (C) 该 StringField 是否带 `choices` → 写中文 = 落回 initial 或抛错（terrain 那一类）。

假阳性模式
----------
* 同名叶子分属**互不相干的模型**（例如 `name` 在 TokenDocument 和 CrucibleAction 里都有），
  同名 ≠ 同一路径。所以脚本同时打印**最近的上层 class 名**，需人工判断是否是我们真会写的那个。
* 正则只认 `leaf: new fields.XxxField` / `leaf: new fields$N.XxxField` 这种字面声明；
  用变量赋值（`const f = new ...; return {leaf: f}`）或 `extendFields` 的会漏。漏 = 假阴性，不是假阳性。
* 只扫上游打包后的单文件 bundle，注释里的示例也可能被匹配到。
"""
import re, os, sys, json, collections

sys.stdout.reconfigure(encoding="utf-8")

FOUNDRY = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
SOURCES = [
    ("crucible", os.path.join(FOUNDRY, "systems", "crucible", "crucible-compiled.mjs")),
    ("ember", os.path.join(FOUNDRY, "modules", "ember", "scripts", "ember.mjs")),
    ("ember-crucible-async", os.path.join(FOUNDRY, "modules", "ember", "scripts", "crucible-async.mjs")),
]

# 我方 mapping / 代码写入的所有叶名（来自 3-常用脚本/extract/mappings.mjs 与两个插件的 js）
LEAVES = [
    # crucible item/actor
    "description", "adjective", "actions", "effects",
    "biography", "ancestry", "background", "archetype", "taxonomy",
    "public", "private", "appearance",
    # ember journal pages
    "overview", "exposition", "summary", "outcomes",
    "gamemaster", "pronunciation", "subtitle", "caption",
    "height", "lifespan", "origin", "label",
    # generic
    "name", "navName", "content", "text", "banner", "levels", "tokens",
    # 已知反例（对照组，terrain 是已归档的「不可译」）
    "terrain",
]

CLASS_RE = re.compile(r"^\s*(?:class|export class)\s+(\w+)")
DECL_RE_TMPL = r"(?<![\w$.])%s\s*:\s*new\s+(fields\$?\d*|foundry\.data\.fields)\.(\w+)\s*\(([^\n]{0,220})"

def scan():
    results = collections.defaultdict(list)     # leaf -> [(pkg, line, cls, fieldcls, args, owner)]
    for pkg, path in SOURCES:
        if not os.path.exists(path):
            print(f"  [missing] {path}")
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # 预先建 行号->最近 class 名 索引
        owners = [None] * (len(lines) + 1)
        cur = None
        for i, ln in enumerate(lines, 1):
            m = CLASS_RE.match(ln)
            if m:
                cur = m.group(1)
            owners[i] = cur
        for leaf in LEAVES:
            rx = re.compile(DECL_RE_TMPL % re.escape(leaf))
            for i, ln in enumerate(lines, 1):
                m = rx.search(ln)
                if m:
                    results[leaf].append((pkg, i, m.group(2), m.group(3).strip()[:160], owners[i]))
    return results

if __name__ == "__main__":
    res = scan()
    STRINGY = {"StringField", "HTMLField", "FilePathField", "JavaScriptField"}
    print("=" * 100)
    for leaf in LEAVES:
        rows = res.get(leaf, [])
        classes = sorted({r[2] for r in rows})
        flag = []
        if len(classes) > 1:
            flag.append("POLYMORPHIC")
        if any(c not in STRINGY for c in classes):
            flag.append("NON-STRING")
        if any("choices" in r[3] for r in rows if r[2] == "StringField"):
            flag.append("CHOICES")
        print(f"\n### {leaf}  ({len(rows)} 处声明)  类={classes}  {' '.join(flag)}")
        for pkg, line, fcls, args, owner in rows:
            mark = "  " if fcls in STRINGY else "!!"
            ch = " <CHOICES>" if "choices" in args else ""
            print(f"  {mark} {pkg}:{line}  {owner}  ->  {fcls}({args}){ch}")
