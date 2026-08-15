# -*- coding: utf-8 -*-
"""
upstream_receiver_contract.py  —  只读探针

判据（从「preCreateItem 空转」这一实例抽象出来的一类）：
    插件对上游（Foundry core / crucible / ember / babele）的某个对象做了「写」或「调用」，
    但上游对这个成员的契约与作者假设的不一致，于是这行代码**恒为空转**。

两个可机械化的子判据：

  A) CALL-ON-MISSING-MEMBER
     形如 `recv?.member?.(...)` / `recv.member(...)` 的调用。把 recv 手工绑定到一个上游类
     （RECEIVER_CLASS），沿 `extends` 链在上游源码里找 member 的**定义**。
     链上一处都找不到 = 这个调用永远短路 / 永远抛（若无 ?. 则抛）。
     假阳性来源：mixin（`XMixin(Base)`）没被链解析器展开；动态 defineProperty 定义的成员。
     所以脚本额外做一次「全语料 member 定义」兜底搜索并把命中文件列出来，由人判断
     那些定义是否挂在本 receiver 的链上。

  B) WRITE-NEVER-READ
     插件写入上游**数据结构**的某个字段（`v.name = …`、`entry.label = …`）。
     在全部上游语料里搜索该字段在同一数据结构上的**读取**。
     零读取 = 写了没人看，这个补丁恒为空转。
     假阳性来源：字段可能被 handlebars 模板 / CSS 属性选择器 / 序列化整体读取，
     所以脚本把 .hbs/.html/.json/.css 也纳入搜索范围。

用法：
    python upstream_receiver_contract.py            # 全跑
输出：
    upstream_receiver_contract.json
只读，不写任何仓库文件。
"""

import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FOUNDRY = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"

PLUGIN_FILES = [
    os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
    os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, "1-Ember汉化插件", "babele-mappings.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-mappings.js"),
]

# 上游语料：(标签, 根目录, 子目录白名单或 None, 后缀白名单)
CORPORA = [
    ("foundry-core", FOUNDRY, ["client", "common"], (".mjs", ".js")),
    ("foundry-tpl", FOUNDRY, ["templates"], (".hbs", ".html")),
    ("crucible", os.path.join(DATA, "systems", "crucible"),
     ["module", "templates"], (".mjs", ".js", ".hbs", ".html")),
    ("crucible-compiled", os.path.join(DATA, "systems", "crucible"), None, (".mjs",)),
    ("ember", os.path.join(DATA, "modules", "ember"),
     ["scripts", "templates", "styles"], (".mjs", ".js", ".hbs", ".html", ".css")),
    ("babele", os.path.join(DATA, "modules", "babele"),
     ["script", "templates"], (".js", ".hbs", ".html")),
]

# ------------------------------------------------------------------ #
# A) receiver -> 上游类。手工绑定，写清依据，避免脚本瞎猜。
#    每项: (插件里的表达式, 上游类名, 定义该类的文件, 备注)
# ------------------------------------------------------------------ #
RECEIVER_CLASS = {
    # register.js:62 `const world = game.world;` —— 局部别名，绑到同一个上游类
    "world": ("World",
              os.path.join(FOUNDRY, "client", "packages", "world.mjs"),
              "register.js:62 const world = game.world; "
              "client/game.mjs:619 this.world = new foundry.packages.World(data.world)"),
    "game.world": ("World",
                   os.path.join(FOUNDRY, "client", "packages", "world.mjs"),
                   "client/game.mjs:619 this.world = new foundry.packages.World(data.world)"),
}

# 手工登记的 extends 链解析起点目录
CLASS_SEARCH_DIRS = [
    os.path.join(FOUNDRY, "client"),
    os.path.join(FOUNDRY, "common"),
]

# ------------------------------------------------------------------ #
# B) 插件写入的上游数据字段: (标签, 结构定位正则, 被写字段)
# ------------------------------------------------------------------ #
WRITE_TARGETS = [
    ("calendar.months.values[].name", r"months\s*\.\s*values", "name"),
    ("calendar.months.values[].abbreviation", r"months\s*\.\s*values", "abbreviation"),
    ("calendar.days.values[].name", r"days\s*\.\s*values", "name"),
    ("calendar.days.values[].abbreviation", r"days\s*\.\s*values", "abbreviation"),
    ("crucible.CONFIG.languages[].label", r"CONFIG\s*\.\s*languages", "label"),
    ("crucible.CONFIG.knowledge[].label", r"CONFIG\s*\.\s*knowledge", "label"),
]

# ------------------------------------------------------------------ #

def iter_files(root, subdirs, exts):
    if not os.path.isdir(root):
        return
    if subdirs is None:
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n)
            if os.path.isfile(p) and n.endswith(exts):
                yield p
        return
    for sd in subdirs:
        base = os.path.join(root, sd)
        for dirpath, _dn, fns in os.walk(base):
            for fn in sorted(fns):
                if fn.endswith(exts):
                    yield os.path.join(dirpath, fn)


def read(p):
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


CORPUS_CACHE = None


def corpus():
    global CORPUS_CACHE
    if CORPUS_CACHE is None:
        CORPUS_CACHE = []
        for label, root, subdirs, exts in CORPORA:
            for p in iter_files(root, subdirs, exts):
                CORPUS_CACHE.append((label, p, read(p)))
    return CORPUS_CACHE


# ---------------- A) 调用点抽取 ---------------- #

OPT_CALL = re.compile(r"([A-Za-z_$][\w$]*(?:\s*\??\.\s*[A-Za-z_$][\w$]*)*)\s*\?\.\s*\(")


def extract_optional_calls():
    """抓 `a.b?.(` 形式：receiver = a，member = b。"""
    out = []
    for path in PLUGIN_FILES:
        src = read(path)
        for i, line in enumerate(src.splitlines(), 1):
            for m in OPT_CALL.finditer(line):
                chain = re.sub(r"\s+", "", m.group(1))
                parts = re.split(r"\??\.", chain)
                if len(parts) < 2:
                    continue
                member = parts[-1]
                recv = ".".join(parts[:-1])
                out.append({
                    "file": os.path.relpath(path, ROOT),
                    "line": i,
                    "receiver_expr": recv,
                    "member": member,
                    "text": line.strip(),
                })
    return out


CLASS_DEF = re.compile(r"class\s+([A-Za-z_$][\w$]*)\s+extends\s+([A-Za-z_$][\w$]*(?:\s*\([^)]*\))?)")


def resolve_chain(class_name, max_depth=8):
    """在 core 源码里沿 `class X extends Y` 走链；Mixin(Base) 形式记下 base 继续走。"""
    chain = [class_name]
    files = []
    cur = class_name
    for _ in range(max_depth):
        found = None
        for d in CLASS_SEARCH_DIRS:
            for dirpath, _dn, fns in os.walk(d):
                for fn in fns:
                    if not fn.endswith(".mjs"):
                        continue
                    p = os.path.join(dirpath, fn)
                    src = read(p)
                    for m in CLASS_DEF.finditer(src):
                        if m.group(1) == cur:
                            base = m.group(2)
                            mm = re.match(r"[A-Za-z_$][\w$]*\s*\(\s*([A-Za-z_$][\w$]*)\s*\)", base)
                            found = (mm.group(1) if mm else base, p, base)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        chain.append(found[0])
        files.append({"class": cur, "extends_raw": found[2],
                      "file": os.path.relpath(found[1], FOUNDRY)})
        cur = found[0]
    return chain, files


MEMBER_DEF_PATTERNS = [
    r"^\s*(?:static\s+)?(?:async\s+)?{m}\s*\(",       # 方法
    r"^\s*(?:static\s+)?get\s+{m}\s*\(",              # getter
    r"^\s*(?:static\s+)?{m}\s*=",                     # 字段
    r"defineProperty\([^,]+,\s*[\"']{m}[\"']",        # 动态定义
]


def member_defined_anywhere(member):
    hits = []
    pats = [re.compile(p.format(m=re.escape(member)), re.M) for p in MEMBER_DEF_PATTERNS]
    for label, p, src in corpus():
        if member not in src:
            continue
        for pat in pats:
            mm = pat.search(src)
            if mm:
                ln = src[:mm.start()].count("\n") + 1
                hits.append({"corpus": label,
                             "file": os.path.basename(p),
                             "path": p,
                             "line": ln,
                             "text": src.splitlines()[ln - 1].strip()[:160]})
                break
    return hits


# ---------------- B) 写入字段的读取面 ---------------- #

def field_read_sites(struct_re, field):
    """在上游语料里找 <struct>...<field> 的读取。粗粒度：同一行或 struct 命中行的 ±3 行内出现 .field"""
    sre = re.compile(struct_re)
    fre = re.compile(r"\.\s*" + re.escape(field) + r"\b")
    hits = []
    for label, p, src in corpus():
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if not sre.search(line):
                continue
            window = lines[max(0, i - 3): i + 4]
            for j, w in enumerate(window):
                if fre.search(w):
                    # 排除「写」： `.field =` 且不是 `==`
                    if re.search(r"\.\s*" + re.escape(field) + r"\s*=(?!=)", w):
                        continue
                    hits.append({"corpus": label,
                                 "file": os.path.relpath(p, os.path.dirname(os.path.dirname(p))),
                                 "path": p,
                                 "line": max(0, i - 3) + j + 1,
                                 "struct_line": i + 1,
                                 "text": w.strip()[:180]})
                    break
    return hits


def main():
    result = {"probe": "upstream_receiver_contract",
              "criterion_A": "optional-call on a member the receiver's class chain never defines",
              "criterion_B": "plugin writes an upstream data field that no upstream consumer reads",
              "scanned": {}, "A": [], "B": []}

    result["scanned"]["plugin_files"] = [
        {"file": os.path.relpath(p, ROOT), "lines": len(read(p).splitlines())}
        for p in PLUGIN_FILES]
    result["scanned"]["corpus_files"] = len(corpus())
    result["scanned"]["corpus_bytes"] = sum(len(s) for _l, _p, s in corpus())

    # ---- A ----
    for site in extract_optional_calls():
        rec = RECEIVER_CLASS.get(site["receiver_expr"])
        entry = dict(site)
        if rec:
            cls, cls_file, why = rec
            chain, chain_files = resolve_chain(cls)
            entry["receiver_class"] = cls
            entry["binding_evidence"] = why
            entry["extends_chain"] = chain
            entry["chain_files"] = chain_files
            # 在链上任一类的定义文件里找 member
            on_chain = []
            for cf in chain_files + [{"class": cls, "file": os.path.relpath(cls_file, FOUNDRY)}]:
                p = os.path.join(FOUNDRY, cf["file"])
                src = read(p)
                for pat in MEMBER_DEF_PATTERNS:
                    if re.search(pat.format(m=re.escape(site["member"])), src, re.M):
                        on_chain.append(cf["file"])
                        break
            entry["defined_on_chain"] = on_chain
            entry["defined_elsewhere"] = member_defined_anywhere(site["member"])[:6]
            entry["verdict"] = "NOOP-CANDIDATE" if not on_chain else "ok"
        else:
            entry["receiver_class"] = None
            entry["verdict"] = "unbound-receiver (DOM/local; skipped)"
        result["A"].append(entry)

    # ---- B ----
    for label, struct_re, field in WRITE_TARGETS:
        reads = field_read_sites(struct_re, field)
        result["B"].append({
            "target": label,
            "struct_regex": struct_re,
            "field": field,
            "upstream_read_sites": reads[:25],
            "n_read_sites": len(reads),
            "verdict": "NOOP-CANDIDATE" if not reads else "read-by-upstream",
        })

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "upstream_receiver_contract.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("corpus files:", result["scanned"]["corpus_files"],
          "bytes:", result["scanned"]["corpus_bytes"])
    print("\n--- A: optional-call sites ---")
    for e in result["A"]:
        print(" %-28s :%-4s %-12s %-16s %s"
              % (os.path.basename(e["file"]), e["line"], e["receiver_expr"],
                 e["member"], e["verdict"]))
    print("\n--- B: write-never-read ---")
    for e in result["B"]:
        print(" %-42s reads=%-4s %s" % (e["target"], e["n_read_sites"], e["verdict"]))
    print("\nwrote", out)


if __name__ == "__main__":
    sys.exit(main())
