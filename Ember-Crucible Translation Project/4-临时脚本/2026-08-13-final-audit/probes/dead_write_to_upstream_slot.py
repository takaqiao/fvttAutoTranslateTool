#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判据 P2 —— 「写进上游一个没人再读的槽位」（死护栏 / 死改写）

抽象自已确认实例：preCreateItem 里改 `data`，而 v14 在钩子之后把 operation.data
整个换成 documents，改动一律作废。共同结构是：

    (1) 本模块 **写** 了一个不属于自己的对象上的某个具体槽位 S
        （上游对象的属性 / 上游注册表的键 / 上游钩子参数）
    (2) 上游在这次写之后，**从不读 S**（或读的是另一个对象 / 另一个字段）
    (3) 写失败没有任何可观测信号：`?.` / `?? {}` / 空 catch / 块体箭头函数返回 undefined
        —— 零异常、零日志，甚至常常还打一条「已改写 N 条」的成功日志

本脚本 **只读**，不改库。做两件事：

  A. 静态枚举：从两个插件仓库的全部 JS/MJS 里，抽出所有「对非本地标识符的赋值」
     和「调用上游注册 API」的位置，归一成 (对象链, 槽位) 二元组，作为候选。
  B. 反向核对：对每个候选槽位，在上游语料（Foundry v14 core / crucible / ember /
     babele）里 grep 读取该槽位的位置。**读取点为 0 的就是候选缺陷。**

假阳性模式（必须人工复核，脚本不下结论）：
  FP1 槽位名太通用（`name` / `label` / `value`），上游到处都读，grep 命中≠读的是这个对象。
  FP2 上游用动态键读（`obj[key]`、解构、`getProperty(o, path)`），grep 抓不到 → 假死。
  FP3 槽位由上游 **模板** 读（.hbs），而不是 JS —— 本脚本把 .hbs 一并纳入语料。
  FP4 写的是本模块自己造的对象（`globalThis.emberCN`），无所谓上游读不读。
  FP5 该写入是「为将来的上游版本预留」的兼容层。

用法：
  python dead_write_to_upstream_slot.py            # 全量
  python dead_write_to_upstream_slot.py --json out.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [
    os.path.join(ROOT, "1-Ember汉化插件"),
    os.path.join(ROOT, "2-Crucible汉化插件"),
]
# 上游语料：读取点就在这些树里找
UPSTREAM = [
    (r"C:\Program Files\Foundry Virtual Tabletop\resources\app\client", "foundry-core"),
    (r"C:\Program Files\Foundry Virtual Tabletop\resources\app\common", "foundry-common"),
    (r"C:\Program Files\Foundry Virtual Tabletop\resources\app\templates", "foundry-tpl"),
    (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible", "crucible"),
    (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember", "ember"),
    (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\babele", "babele"),
]
CODE_EXT = (".js", ".mjs", ".hbs", ".html", ".json")

# 本模块自己造的、上游本来就不该读的对象（FP4）
OWN_OBJECTS = {"emberCN", "MODULE_ID", "PROJECT_CONVERTERS", "DOCUMENT_MAPPINGS"}

# 赋值到「一条以点分隔的链」上：a.b.c = ...  /  a?.b.c = ...
ASSIGN = re.compile(
    r"(?<![\w$.])((?:globalThis|game|CONFIG|ui|foundry|crucible|ember|Hooks|document|window)"
    r"(?:\??\.[A-Za-z_$][\w$]*)+)\s*=\s*(?!=)"
)
# 局部变量承接的上游链：const x = <chain>;   后面 x.y = ... 也算写上游（人工核）
LOCALBIND = re.compile(
    r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"((?:globalThis|game|CONFIG|ui|foundry|crucible|ember)(?:\??\.[A-Za-z_$][\w$]*)+)"
)
# 静默吞掉的形态
SWALLOW = [
    ("opt_chain", re.compile(r"\?\.")),
    ("nullish_empty", re.compile(r"\?\?\s*(\{\}|\[\])")),
    ("empty_catch", re.compile(r"catch\s*(\([^)]*\))?\s*\{\s*/?\*?[^}]{0,60}\}")),
    ("silent_return", re.compile(r"^\s*if\s*\(.*\)\s*return\b")),
    ("filter_bool", re.compile(r"\.filter\(Boolean\)")),
]


def repo_files(repo):
    out = []
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if d not in (".git", ".github", "compendium", "release", "__pycache__")]
        for f in fns:
            if f.endswith((".js", ".mjs")):
                out.append(os.path.join(dp, f))
    return out


def collect_writes():
    cands = []
    for repo in REPOS:
        for path in repo_files(repo):
            try:
                lines = open(path, encoding="utf-8").read().splitlines()
            except Exception:
                continue
            binds = {}
            for i, ln in enumerate(lines, 1):
                for m in LOCALBIND.finditer(ln):
                    binds[m.group(1)] = m.group(2).replace("?.", ".")
                for m in ASSIGN.finditer(ln):
                    chain = m.group(1).replace("?.", ".")
                    parts = chain.split(".")
                    if parts[0] in OWN_OBJECTS or (len(parts) > 1 and parts[1] in OWN_OBJECTS):
                        continue
                    cands.append(dict(file=os.path.relpath(path, ROOT), line=i,
                                      chain=chain, slot=parts[-1], via="direct",
                                      text=ln.strip()[:160],
                                      swallow=[n for n, r in SWALLOW if r.search(ln)]))
                # 局部变量再赋值：x.y = ...
                m2 = re.match(r"\s*([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\s*=\s*(?!=)", ln)
                if m2 and m2.group(1) in binds:
                    chain = f"{binds[m2.group(1)]}[*].{m2.group(2)}"
                    cands.append(dict(file=os.path.relpath(path, ROOT), line=i,
                                      chain=chain, slot=m2.group(2), via="localbind",
                                      text=ln.strip()[:160],
                                      swallow=[n for n, r in SWALLOW if r.search(ln)]))
    return cands


def rg_count(pattern, root):
    """用 ripgrep 数上游读取点。返回 (命中数, 前 5 条样本)。"""
    if not os.path.isdir(root):
        return -1, []
    try:
        p = subprocess.run(
            ["rg", "-n", "--no-heading", "-e", pattern, root,
             "-g", "*.mjs", "-g", "*.js", "-g", "*.hbs", "-g", "*.html", "-g", "*.json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        out = [l for l in (p.stdout or "").splitlines() if l.strip()]
        return len(out), out[:5]
    except Exception as e:
        return -1, [str(e)]


def read_sites(chain, slot):
    """构造「上游读取该槽位」的 grep 模式。故意宽松 —— 宁可多命中，人工再收。"""
    tail = chain.split(".")[-2:] if "." in chain else [chain]
    pats = []
    if len(tail) >= 2 and tail[-2] != "*":
        pats.append(re.escape(f"{tail[-2]}.{tail[-1]}"))
    pats.append(rf"[.\[]\s*[\"']?{re.escape(slot)}[\"']?\s*[\]\.\),;=}}]")
    pats.append(rf"\{{\{{[^}}]*\b{re.escape(slot)}\b")           # hbs
    pats.append(rf"\b{re.escape(slot)}\s*:")                      # 解构/对象字面量
    return "|".join(f"(?:{p})" for p in pats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-upstream", action="store_true")
    a = ap.parse_args()

    cands = collect_writes()
    # 归一去重
    seen, uniq = set(), []
    for c in cands:
        k = (c["chain"], c["file"], c["line"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    if not a.no_upstream:
        cache = {}
        for c in uniq:
            key = (c["chain"], c["slot"])
            if key not in cache:
                pat = read_sites(*key)
                tot, samp = 0, []
                for root, tag in UPSTREAM:
                    n, s = rg_count(pat, root)
                    if n > 0:
                        tot += n
                        samp += [f"[{tag}] {x[:160]}" for x in s[:2]]
                cache[key] = (tot, samp[:6])
            c["upstream_read_hits"], c["upstream_samples"] = cache[key]

    uniq.sort(key=lambda c: (c.get("upstream_read_hits", 999), c["file"], c["line"]))
    out = {"total_write_sites": len(uniq),
           "zero_read_sites": [c for c in uniq if c.get("upstream_read_hits") == 0],
           "all": uniq}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"写入上游槽位的位置：{len(uniq)}")
    for c in uniq:
        print(f"  [{c.get('upstream_read_hits','?'):>5}] {c['chain']:<52} "
              f"{c['file']}:{c['line']}  swallow={','.join(c['swallow']) or '-'}")


if __name__ == "__main__":
    main()
