#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_eager_shim_writes.py  —  第十三轮「防御性 shim 过度生效」类判据

种子实例（已记录，不重复报）：
  register.js 的 degradeActorUpdatePayload 在 try 之前就把 items/effects 删掉，
  本该只在 catch 里降级。

把它抽象成可机械化的判据：

  一处「过度生效的 shim 写入」= 同时满足以下四条的语句
   W1. 它**写**一个模块自己不拥有的东西：
       - Document 持久化写：.update( / .updateEmbeddedDocuments( / .setFlag( /
         .createEmbeddedDocuments( / .updateDocuments(
       - 上游对象的属性赋值：CONFIG.* / crucible.* / game.* / globalThis.* /
         entry.label= / v.name= / node.nodeValue= / setAttribute(
       - 钩子回调形参（changes / data / updates）的原地改写
   W2. 它在 **happy path** 上 —— 不在 catch 块内，也不在 `if (error)` 之类的
       失败分支内。（种子实例正是栽在这里。）
   W3. 它的**闸门是形状/类型判断**（typeof x === 'string' / !Array.isArray(x) /
       !x.foo / x instanceof …），而不是版本号、id、显式开关。
       形状判断＝对上游 schema 的一个假设，上游一改就悄悄错。
   W4. 这条路径上**没有任何 console.***（happy path 静默）。

  W1&&W2&&W3&&!W4 → 候选。判据只负责收敛人工核实面，不下结论。

已知假阳性模式（必须人工排除，见 --verify 输出）：
  FP-a  闸门形状判断其实是「只处理我自己造出来的那种形状」（例如 babele 转换器
        里 isStr(translation) —— translation 是本项目自己的产物，不是上游 schema）
  FP-b  写的是纯展示层且卸载即恢复（DOM 文本节点）——不改数据
  FP-c  写的是模块自己命名空间下的东西（globalThis.emberCN、__emberCnWrapped 旗标）

用法（只读，不写库）：
  python scan_eager_shim_writes.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
REPOS = [ROOT / "1-Ember汉化插件", ROOT / "2-Crucible汉化插件"]
SKIP_DIRS = {"compendium", "lang", "node_modules", ".git", "release"}

# ---- W1: 写上游 -------------------------------------------------------------
PERSIST_WRITE = re.compile(
    r"\.(?:update|updateEmbeddedDocuments|createEmbeddedDocuments|updateDocuments|"
    r"setFlag|unsetFlag|deleteEmbeddedDocuments|modifyBatch)\s*\(")
UPSTREAM_ASSIGN = re.compile(
    r"(?:^|[^\w.])(?:CONFIG|crucible|game|globalThis|ui|Hooks)\b[\w.?\[\]'\"]*\s*=\s*[^=]")
MEMBER_ASSIGN = re.compile(
    r"(?:^|[^\w])(?:entry|v|cal|node|action|effect|effects|hook|changes|update|"
    r"itemData|data|group|app|first|degraded|patched|sanitized)"
    r"(?:\.[\w$]+|\[[^\]]+\])+\s*=\s*[^=]")
DOM_WRITE = re.compile(r"\.setAttribute\s*\(|\.nodeValue\s*=|\.textContent\s*=")
DELETE_STMT = re.compile(r"(?:^|[^\w])delete\s+[\w$]+(?:\.[\w$]+|\[[^\]]+\])")
MERGE_INPLACE = re.compile(r"foundry\.utils\.mergeObject\s*\(\s*[\w$.]+\s*,[^)]*\)")
SETPROP = re.compile(r"foundry\.utils\.setProperty\s*\(")

# ---- W3: 形状/类型闸门 -------------------------------------------------------
SHAPE_GATE = re.compile(
    r"typeof\s+[\w$.?\[\]'\"]+\s*[!=]==?\s*['\"](?:string|object|function|number|undefined)['\"]"
    r"|!?\s*Array\.isArray\s*\("
    r"|instanceof\s+\w+"
    r"|[\w$.]+\s*===?\s*(?:undefined|null)"
    r"|!\s*[\w$]+(?:\.[\w$]+)*\s*(?:\)|&&|\|\||\{)")
# 版本/身份闸门（若命中，说明不是形状假设，降权）
IDENTITY_GATE = re.compile(
    r"game\.system\.(?:id|version)|\.version\b|isNewerVersion|foundry\.utils\.isNewerVersion"
    r"|game\.modules\.get\(|game\.settings\.get\(|\.type\s*===|documentName")

CONSOLE = re.compile(r"console\.(?:log|warn|error|info|debug)\s*\(")


def strip_block_comments(src: str):
    """把块注释替换成等长空白，保持行号与列号不变。"""
    out = []
    i = 0
    n = len(src)
    while i < n:
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def brace_blocks(lines):
    """返回 [(start,end,header)]，用来判断某行是否落在 catch/if(err) 块内。"""
    blocks = []
    stack = []
    for idx, ln in enumerate(lines):
        for ch in ln:
            if ch == "{":
                stack.append((idx, lines[idx].strip()))
            elif ch == "}":
                if stack:
                    s, hdr = stack.pop()
                    blocks.append((s, idx, hdr))
    return blocks


CATCH_HDR = re.compile(r"\bcatch\s*\(|\bif\s*\(\s*!?\s*(?:err|error|e)\b|\bthrow\b")


def scan_file(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    src = strip_block_comments(raw)
    lines = src.split("\n")
    rawlines = raw.split("\n")
    blocks = brace_blocks(lines)
    # 一行行地找 catch 覆盖范围
    catch_ranges = [(s, e) for (s, e, hdr) in blocks if CATCH_HDR.search(hdr)]

    # 函数粗切：以顶层 function/const fn = 起始到下一个同级 function
    fn_starts = [(i, ln.strip()) for i, ln in enumerate(lines)
                 if re.match(r"\s*(?:async\s+)?function\s+[\w$]+|^\s*(?:const|let)\s+[\w$]+\s*=\s*(?:async\s*)?(?:function|\()", ln)]

    def enclosing_fn(i):
        best = None
        for s, hdr in fn_starts:
            if s <= i and (best is None or s > best[0]):
                best = (s, hdr)
        return best or (0, "<top>")

    hits = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("//"):
            continue
        kinds = []
        if PERSIST_WRITE.search(ln):
            kinds.append("persist")
        if UPSTREAM_ASSIGN.search(ln):
            kinds.append("upstream-assign")
        if MEMBER_ASSIGN.search(ln):
            kinds.append("member-assign")
        if DOM_WRITE.search(ln):
            kinds.append("dom")
        if DELETE_STMT.search(ln):
            kinds.append("delete")
        if MERGE_INPLACE.search(ln) and "inplace: false" not in ln and "inplace:false" not in ln:
            kinds.append("merge-inplace")
        if SETPROP.search(ln):
            kinds.append("setProperty")
        if not kinds:
            continue

        in_catch = any(s <= i <= e for (s, e) in catch_ranges)
        fs, fhdr = enclosing_fn(i)
        # 函数体内（fs..下一个函数起点）的闸门与日志
        nxt = min([s for s, _ in fn_starts if s > fs] + [len(lines)])
        body = "\n".join(lines[fs:nxt])
        shape = bool(SHAPE_GATE.search(body))
        ident = bool(IDENTITY_GATE.search(body))
        logged = bool(CONSOLE.search(body))

        hits.append({
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "line": i + 1,
            "fn": fhdr[:90],
            "kinds": kinds,
            "in_catch": in_catch,
            "shape_gate": shape,
            "identity_gate": ident,
            "logged_in_fn": logged,
            "code": rawlines[i].strip()[:170],
        })
    return hits


def main():
    all_hits = []
    files = 0
    for repo in REPOS:
        for p in sorted(repo.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix not in (".js", ".mjs"):
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(repo).parts[:-1]):
                continue
            files += 1
            all_hits.extend(scan_file(p))

    flagged = [h for h in all_hits
               if not h["in_catch"] and h["shape_gate"] and not h["logged_in_fn"]]
    softer = [h for h in all_hits
              if not h["in_catch"] and h["shape_gate"] and h["logged_in_fn"]]

    print(f"scanned files: {files}   write-sites: {len(all_hits)}")
    print(f"W1&&W2&&W3&&!W4  (silent eager shape-gated writes): {len(flagged)}")
    print(f"same but function has some console.* somewhere      : {len(softer)}")
    print("-" * 100)
    for h in flagged:
        print(f'{h["file"]}:{h["line"]}  [{",".join(h["kinds"])}]  fn={h["fn"]}')
        print(f'    {h["code"]}')
    print("-" * 100)
    print("SOFTER (函数内某处有 console，但写入那一行未必被记录):")
    for h in softer:
        print(f'{h["file"]}:{h["line"]}  [{",".join(h["kinds"])}]  fn={h["fn"]}')
        print(f'    {h["code"]}')

    out = Path(__file__).with_name("eager_shim_writes.json")
    out.write_text(json.dumps({"flagged": flagged, "softer": softer, "all": all_hits},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
