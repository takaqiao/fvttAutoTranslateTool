#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe: silent_noop_api
======================

判据（把已确认的 game.world.getFlag 那条抽象出来）：

  「对**上游对象**取一个成员，然后用一种**吞掉失败**的写法去用它」
  —— 一旦那个成员在上游其实不存在 / 类型不对，代码就变成**永久静默空操作**，
     而且不会有任何日志、异常、或可观测信号。

四种吞法（本探针识别）：
  S1 可选调用   `X?.m?.(...)`            方法不存在 → undefined，当成「没返回值」
  S2 守卫早退   `if (!x || typeof x!=='function') return;`  探测失败 → 静默 return（无日志）
  S3 空兜底     `X?.y ?? {}` / `?? []` / `.filter(Boolean)`  取不到 → 空集合 → 循环 0 次
  S4 空 catch   `try{...}catch{}` / catch 只 warn 而调用方不看返回值

输出：候选清单（文件 / 行 / 吞法 / 被取的成员 / 该成员的上游归属猜测）。
**候选不等于缺陷** —— 必须逐条去上游源码确认成员是否真的存在、类型是否真的对。
本探针只负责「把需要人工确认的点全部找齐」，不做判定。

假阳性模式（必须知道）：
  - 成员是本模块自己造的（如 `entry.__emberCnWrapped`）→ 与上游契约无关，正常。
  - 成员在上游确实存在 → S1/S2/S3 只是正常防御，正常。
  - 兜底的空集合本来就允许为空（如 `game.items ?? []` 空世界）→ 正常。
  - 正则会漏：动态成员名 `obj[key]`、跨行写法。所以另出一份「上游根对象成员全表」兜底。

只读，不写库。
"""
import os
import re
import sys
import json

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [
    os.path.join(ROOT, "1-Ember汉化插件"),
    os.path.join(ROOT, "2-Crucible汉化插件"),
]
CODE_EXT = (".js", ".mjs")
SKIP_DIRS = {".git", "compendium", "lang", "node_modules", "styles"}

# 上游根对象：这些名字下面的成员由 Foundry / crucible / ember / babele 决定，
# 本模块无权定义 —— 取错了就是契约错误。
UPSTREAM_ROOTS = {
    "game", "CONFIG", "ui", "foundry", "canvas", "Hooks", "crucible", "ember",
    "babele", "globalThis", "document", "Node", "CONST",
}

PAT_OPT_CALL = re.compile(r"([A-Za-z_$][\w$]*(?:\??\.[A-Za-z_$][\w$]*)*)\?\.\(")
PAT_TYPEOF_GUARD = re.compile(r"typeof\s+([A-Za-z_$][\w$.?\[\]'\"]*)\s*[!=]==\s*['\"]function['\"]")
PAT_EMPTY_FALLBACK = re.compile(r"([A-Za-z_$][\w$]*(?:\??\.[A-Za-z_$][\w$]*)+)\s*\?\?\s*(\{\}|\[\])")
PAT_FILTER_BOOL = re.compile(r"\.filter\(Boolean\)")
PAT_EMPTY_CATCH = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*(?:/\*.*?\*/)?\s*\}", re.S)
PAT_SILENT_RETURN = re.compile(r"if\s*\((?![^)]*console)[^)]*\)\s*return\s*;")


def root_of(expr):
    return re.split(r"\??[.\[]", expr, 1)[0]


def scan_file(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    out = []

    def add(kind, lineno, expr, text):
        out.append({
            "file": path,
            "line": lineno,
            "kind": kind,
            "expr": expr,
            "root": root_of(expr) if expr else None,
            "upstream_root": root_of(expr) in UPSTREAM_ROOTS if expr else None,
            "text": text.strip()[:200],
        })

    for i, ln in enumerate(lines, 1):
        for m in PAT_OPT_CALL.finditer(ln):
            add("S1_optional_call", i, m.group(1), ln)
        for m in PAT_TYPEOF_GUARD.finditer(ln):
            add("S2_typeof_guard", i, m.group(1), ln)
        for m in PAT_EMPTY_FALLBACK.finditer(ln):
            add("S3_empty_fallback", i, m.group(1), ln)
        if PAT_FILTER_BOOL.search(ln):
            add("S3_filter_boolean", i, "", ln)

    for m in PAT_EMPTY_CATCH.finditer(src):
        lineno = src[:m.start()].count("\n") + 1
        add("S4_empty_catch", lineno, "", lines[lineno - 1] if lineno <= len(lines) else "")

    # 「取上游成员」的全量表（兜底：正则吞法可能漏，成员表用来人工过一遍）
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("*") or ln.strip().startswith("//"):
            continue
        for m in re.finditer(
            r"\b(game|CONFIG|ui|foundry|canvas|Hooks|crucible|babele|globalThis)"
            r"((?:\??\.[A-Za-z_$][\w$]*)+)", ln
        ):
            chain = m.group(1) + m.group(2)
            out.append({
                "file": path, "line": i, "kind": "M_member_chain",
                "expr": chain, "root": m.group(1), "upstream_root": True,
                "text": ln.strip()[:200],
            })
    return out


def main():
    findings = []
    nfiles = 0
    for repo in REPOS:
        for dirpath, dirnames, filenames in os.walk(repo):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(CODE_EXT):
                    nfiles += 1
                    findings.extend(scan_file(os.path.join(dirpath, fn)))

    by_kind = {}
    for f in findings:
        by_kind.setdefault(f["kind"], []).append(f)

    print("scanned files:", nfiles)
    for k in sorted(by_kind):
        print("\n===== %s : %d =====" % (k, len(by_kind[k])))
        if k == "M_member_chain":
            uniq = {}
            for f in by_kind[k]:
                uniq.setdefault(f["expr"], []).append(
                    "%s:%d" % (os.path.basename(f["file"]), f["line"]))
            for expr in sorted(uniq):
                print("  %-58s %s" % (expr, ",".join(uniq[expr][:4])))
        else:
            for f in by_kind[k]:
                print("  %s:%d  [%s]  %s" % (
                    os.path.basename(f["file"]), f["line"], f["expr"], f["text"]))

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "silent_noop_api.candidates.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(findings, fh, ensure_ascii=False, indent=1)
    print("\n->", outp, len(findings), "candidates")


if __name__ == "__main__":
    sys.exit(main())
