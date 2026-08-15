#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scan_dead_guards.py —— 「失效的守卫」类判据（只读，不写库）

种子缺陷：`world?.getFlag?.(MODULE_ID, ...)` —— 对一个**根本没有该方法的上游对象**
做可选调用，`?.` 把「这条判据永远不成立」彻底藏起来。

抽象成机械判据：
  插件运行时代码里每一个「**跨包触点**」（指向 Foundry core / babele / crucible /
  ember 的符号：属性路径、钩子名、DOM 选择器、错误串匹配、babele mapping 选项名），
  都必须在上游语料里**真的存在**，且上游**真的会读它**。
  凡是「找不到」或「找得到但上游不读」的，都是同一类缺陷 —— 因为失败全是静默的
  （`?.` / `?? {}` / 空 catch / `if (!x) return` / 改一个上游会丢弃的对象）。

用法：
  python scan_dead_guards.py            # 打印候选表
  python scan_dead_guards.py --json out.json

假阳性模式（必须人工复核，本脚本只做粗筛）：
  1. 上游是压缩/rollup 打包代码，符号可能被改名（本项目 crucible-compiled.mjs 未压缩，
     ember.mjs 未压缩，所以按字面 grep 基本可靠；但 `HOOKS$6` 这类 rollup 后缀会让
     「hooks.action」这样的**路径**查不到，需要按叶子名单独查）。
  2. 「上游存在」不等于「上游会读」—— 例如 preCreateXxx 钩子里的 data 对象在
     client-backend.mjs 里确实被传进来了，但创建流水线用的是 documents 而不是它。
     这一层只能人工看上游源码，脚本只负责把触点列全。
  3. 字符串匹配类（stack.includes / message.includes）永远报「需人工」，因为它匹配的是
     运行期生成的文本，不是源码里的字面量。

本轮实际产出（脚本只负责把 103 个触点列全，判定全部人工回上游源码核对）：
  - register.js:438 preCreateItem 改的 data 被 client-backend.mjs:92/122 丢弃
  - ember-hardcoded-cn.mjs:417 patchCalendarNames 改的 months/days 名全库无渲染方
  - ember-hardcoded-cn.mjs:433 ui.windows 在 v14 只装 ApplicationV1
  - ember-hardcoded-cn.mjs:436 #ember-calendar 上没有 change 监听（tag 是 aside 不是 form）
  - ember-hardcoded-cn.mjs:472 render 钩子盖不住 animate() 的二次写入
  - babele-register.js:56 Sort/sort 两个 i18n 键上游无人读
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PLUGIN_FILES = [
    r"1-Ember汉化插件\register.js",
    r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs",
    r"1-Ember汉化插件\babele-mappings.js",
    r"2-Crucible汉化插件\babele-register.js",
    r"2-Crucible汉化插件\babele-mappings.js",
    r"3-常用脚本\release\runtime-converters.js",
]

# 上游语料。key 用来在报告里说明「在哪一份里查的」
UPSTREAM = {
    "foundry": [
        r"C:\Program Files\Foundry Virtual Tabletop\resources\app\client",
        r"C:\Program Files\Foundry Virtual Tabletop\resources\app\common",
    ],
    "crucible": [r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"],
    "ember": [r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts"],
    "babele": [r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\babele\script"],
}

# 跨包根对象：这些名字后面跟的属性链就是「触点」
FOREIGN_ROOTS = ("game", "CONFIG", "ui", "foundry", "canvas", "crucible", "ember", "Hooks", "document")

RE_CHAIN = re.compile(
    r"\b(" + "|".join(FOREIGN_ROOTS) + r")\b((?:\s*\??\.\s*[A-Za-z_$][\w$]*)+)"
)
RE_HOOK = re.compile(r"Hooks\.(?:on|once)\(\s*[\"']([^\"']+)[\"']")
RE_STRMATCH = re.compile(r"\.includes\(\s*[\"']([^\"']+)[\"']\s*\)")
RE_QUERY = re.compile(r"querySelector(?:All)?\(\s*[\"']([^\"']+)[\"']")
RE_OPTCALL = re.compile(r"([A-Za-z_$][\w$]*)\s*\?\.\s*\(")


def read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def iter_upstream_files():
    for tag, paths in UPSTREAM.items():
        for p in paths:
            if os.path.isfile(p):
                yield tag, p
            else:
                for dirpath, _dn, fn in os.walk(p):
                    for n in fn:
                        if n.endswith((".mjs", ".js", ".json", ".hbs", ".html")):
                            yield tag, os.path.join(dirpath, n)


_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = {}
        for tag, p in iter_upstream_files():
            _CORPUS.setdefault(tag, []).append((p, read(p)))
    return _CORPUS


def found_in(token):
    """返回该字面 token 在哪些上游语料里出现过。"""
    hits = []
    for tag, files in corpus().items():
        for p, src in files:
            if token in src:
                hits.append(tag)
                break
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = []
    for rel in PLUGIN_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print("MISSING PLUGIN FILE: " + path, file=sys.stderr)
            continue
        src = read(path)
        lines = src.split("\n")
        for i, line in enumerate(lines, 1):
            code = line.split("//")[0]
            if not code.strip() or code.strip().startswith("*"):
                continue

            for m in RE_CHAIN.finditer(code):
                root, tail = m.group(1), m.group(2)
                leaf = re.split(r"\??\.", tail)[-1].strip()
                chain = (root + tail).replace(" ", "")
                # 叶子名单独查：rollup 会把中间对象改名，叶子基本保真
                rows.append({
                    "file": rel, "line": i, "kind": "chain",
                    "token": chain, "probe": leaf, "hits": found_in(leaf),
                    "optional": "?." in chain,
                })

            for m in RE_HOOK.finditer(code):
                h = m.group(1)
                rows.append({
                    "file": rel, "line": i, "kind": "hook",
                    "token": h, "probe": h, "hits": found_in(h), "optional": False,
                })

            for m in RE_STRMATCH.finditer(code):
                s = m.group(1)
                rows.append({
                    "file": rel, "line": i, "kind": "string-match",
                    "token": s, "probe": s, "hits": found_in(s), "optional": False,
                })

            for m in RE_QUERY.finditer(code):
                sel = m.group(1)
                probe = sel.lstrip("#.").split()[0]
                rows.append({
                    "file": rel, "line": i, "kind": "selector",
                    "token": sel, "probe": probe, "hits": found_in(probe), "optional": True,
                })

            for m in RE_OPTCALL.finditer(code):
                rows.append({
                    "file": rel, "line": i, "kind": "optional-call",
                    "token": m.group(1) + "?.()", "probe": m.group(1),
                    "hits": found_in(m.group(1)), "optional": True,
                })

    # 去重
    seen, uniq = set(), []
    for r in rows:
        k = (r["file"], r["line"], r["kind"], r["token"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    miss = [r for r in uniq if not r["hits"]]
    print("触点总数 %d，其中上游语料查无此符号的 %d 条：\n" % (len(uniq), len(miss)))
    for r in sorted(miss, key=lambda x: (x["file"], x["line"])):
        print("  %-46s :%-4d [%-13s] %s%s" % (
            r["file"], r["line"], r["kind"], r["token"],
            "   <-- 静默(?.)" if r["optional"] else ""))

    print("\n---- 全部可选调用 `x?.()`（种子缺陷的语法签名）----")
    for r in uniq:
        if r["kind"] == "optional-call":
            print("  %-46s :%-4d %s  上游命中=%s" % (r["file"], r["line"], r["token"], r["hits"] or "无"))

    print("\n---- 全部钩子注册 ----")
    for r in uniq:
        if r["kind"] == "hook":
            print("  %-46s :%-4d %-22s 上游命中=%s" % (r["file"], r["line"], r["token"], r["hits"] or "无"))

    print("\n---- 全部字符串匹配判据（永远需要人工核） ----")
    for r in uniq:
        if r["kind"] == "string-match":
            print("  %-46s :%-4d %-46r 上游命中=%s" % (r["file"], r["line"], r["token"], r["hits"] or "无"))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(uniq, f, ensure_ascii=False, indent=1)
        print("\nwrote " + args.json)


if __name__ == "__main__":
    main()
