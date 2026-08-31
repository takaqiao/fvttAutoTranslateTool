# -*- coding: utf-8 -*-
"""
举一反三探针：**「写负载的无条件删除 / guard 与其自述语义不符」全库扩查**

种子实例（已记录，不重复报）
----------------------------
register.js:318 `importMode ? prepareSafeActorUpdatesForImport(sanitized) : sanitized`
—— 注释自称「Import-specific fallback」（失败之后才降级），代码却是 try **之前**的
无条件预处理，且降级动作是 `delete update.items; delete update.effects` 零判据整段删。

把它抽象成三个可机械化的问句（对每一个「丢弃点」都问一遍）
------------------------------------------------------------
  Q1 **判据**：这次丢弃/改写有没有条件闸？闸判的是「输入病态」还是什么都不判？
  Q2 **时序**：闸在 `try` **之前**（＝无条件预降级）还是 `catch` **之内**（＝真兜底）？
  Q3 **反馈**：丢弃发生时有没有可见输出？（无 → 静默丢数据）
  Q4 **自述**：附近的注释/函数名/文档串是否声称了一个比代码**更窄**的语义
      （fallback / 兜底 / 只在…时 / malformed / legacy / 失败）？

命中定义 = 存在丢弃点，且 (Q1 无闸 or Q2 预降级 or Q3 静默) 且 Q4 有更窄的自述。
只有 Q1/Q2/Q3 而没有 Q4 的，降级为「候选」（可能是有意为之）。

附带的第二类判据（同一类的另一侧：**guard 根本不生效**）
--------------------------------------------------------
  S2：`Hooks.on('preCreateX', (doc, data) => …mutate(data)…)`
      Foundry v14 `client/data/client-backend.mjs` #preCreateDocumentArray：
        :92  doc = new documentClass(deepClone(createData), …)   ← 文档先构造
        :101 Hooks.call(`preCreate${type}`, doc, createData, …)  ← 钩子后调用
        :120 operation.data = documents                          ← 最终用的是 doc
      所以在 preCreate 钩子里改 `data` 是**写给一个已经被消费并随后丢弃的对象**。
      要生效必须 `doc.updateSource(...)`。凡是改 data 不改 doc 的，判为死闸。

扫描范围
--------
A. 随插件发布的运行时 JS（module.json 的 esmodules + 它 import 的同仓文件 + styles）
B. 会**写回仓库**的工具（3-常用脚本/** 与两个插件 scripts/ 下，检出 open(...,'w') /
   json.dump / writeFileSync 的文件）

假阳性模式（必须知道）
----------------------
* `{inplace:false}` 的 mergeObject、`.map()` 出新数组再 return 的纯函数 —— 不改调用方
  对象，不属本类。脚本会标注 PURE，但不会自动排除（有些 return 值就是要写回去的）。
* 正则只看**同一行**的赋值/删除，跨行链式与别名间接改写会漏（假阴性）。
* Q2 的 try 判定只按**行号是否落在 try 块的括号区间内**，用缩进/括号计数估算，
  多层嵌套时可能判错，脚本会打印它认定的 try 区间供复核。
* Q4 的「自述更窄」是关键词匹配，不是语义理解 —— 命中必须人工读注释确认。

用法：python scan_drop_site_vs_claim.py [--verbose]
只读，不写库。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(ROOT, "1-Ember汉化插件"), os.path.join(ROOT, "2-Crucible汉化插件")]
TOOL_DIRS = [
    os.path.join(ROOT, "3-常用脚本"),
    os.path.join(ROOT, "1-Ember汉化插件", "scripts"),
    os.path.join(ROOT, "2-Crucible汉化插件", "scripts"),
]

# ---------------------------------------------------------------- 丢弃点判据
JS_DROP = [
    ("JS-DELETE",     re.compile(r"\bdelete\s+([\w$]+(?:\.[\w$]+|\[[^\]]+\])*)")),
    ("JS-EMPTY",      re.compile(r"([\w$]+(?:\.[\w$]+|\[[^\]]+\])+)\s*=\s*(?:\[\]|\{\}|''|\"\"|null|undefined)\s*[;,)]")),
    ("JS-RET-EMPTY",  re.compile(r"return\s*(?:\{\}|\[\]|''|\"\")\s*;")),
    ("JS-FILTER",     re.compile(r"\.filter\(")),
    ("JS-REPL-EMPTY", re.compile(r"\.replace\([^,]+,\s*(?:''|\"\")\s*\)")),
]
PY_DROP = [
    ("PY-DEL",        re.compile(r"^\s*del\s+\w+\[")),
    ("PY-POP",        re.compile(r"\.pop\(")),
    ("PY-RET-EMPTY",  re.compile(r"return\s*(?:''|\"\"|\[\]|\{\}|DELETE)\s*$")),
    ("PY-REPL-EMPTY", re.compile(r"\.replace\([^,]+,\s*(?:''|\"\")\s*\)")),
    ("PY-SUB-EMPTY",  re.compile(r"re\.sub\([^,]+,\s*(?:r?''|r?\"\")\s*,")),
    ("PY-CONTINUE",   re.compile(r"^\s*continue\s*$")),
]

# Q4：比代码更窄的自述
NARROW_CLAIM = re.compile(
    r"fallback|兜底|失败|失效|only\b|只在|仅在|仅当|malformed|broken|legacy|旧版|异常|出错|"
    r"corrupt|invalid|防御|repair|修复|病态|如果.*才|when .*fails|in case",
    re.I)
# Q3：可见反馈
FEEDBACK = re.compile(r"console\.(warn|error|info|log)|ui\.notifications|print\(|warn\(|log\(")

FUNC_JS = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)|^\s*(?:const|let)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:function|\()")
FUNC_PY = re.compile(r"^\s*def\s+(\w+)")


def try_ranges(lines, py=False):
    """粗略求出所有 try 块的行号区间（1-based, 闭区间）。缩进法，够本库用。"""
    out = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if (py and s.startswith("try:")) or ((not py) and re.match(r"^try\s*\{", s)):
            indent = len(ln) - len(ln.lstrip())
            j = i
            for k in range(i, len(lines)):
                l2 = lines[k]
                if not l2.strip():
                    continue
                ind2 = len(l2) - len(l2.lstrip())
                if ind2 <= indent and k + 1 > i:
                    body = l2.strip()
                    if (py and body.startswith(("except", "finally"))) or \
                       ((not py) and body.startswith(("}", "catch", "finally"))):
                        j = k + 1
                        continue
                    break
                j = k + 1
            out.append((i, j))
    return out


def func_spans(lines, py=False):
    """函数名 -> (起, 止)。止取下一个同级函数起点前一行。"""
    rx = FUNC_PY if py else FUNC_JS
    marks = []
    for i, ln in enumerate(lines, 1):
        m = rx.match(ln)
        if m:
            marks.append((i, next(g for g in m.groups() if g)))
    spans = []
    for idx, (ln_no, name) in enumerate(marks):
        end = marks[idx + 1][0] - 1 if idx + 1 < len(marks) else len(lines)
        spans.append((ln_no, end, name))
    return spans


def enclosing(spans, n):
    for a, b, name in spans:
        if a <= n <= b:
            return a, b, name
    return None, None, None


def nearest_guard(lines, n, py=False):
    """向上找最近的 if 条件（同一函数内、8 行以内）。"""
    for k in range(n - 1, max(0, n - 9), -1):
        s = lines[k - 1].strip()
        if s.startswith("if ") or s.startswith("if("):
            return s[:110]
    return ""


def scan_file(path, rel, py):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    spans = func_spans(lines, py)
    tries = try_ranges(lines, py)
    pats = PY_DROP if py else JS_DROP
    hits = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith(("#", "//", "*", "/*")):
            continue
        for tag, rx in pats:
            if not rx.search(ln):
                continue
            a, b, fname = enclosing(spans, i)
            body = "\n".join(lines[(a or 1) - 1:(b or i)])
            # 自述：函数体上方 12 行注释 + 函数体内注释
            head = "\n".join(lines[max(0, (a or i) - 13):(a or i)])
            claim = bool(NARROW_CLAIM.search(head)) or bool(NARROW_CLAIM.search(body))
            in_try = any(x <= i <= y for x, y in tries)
            fb = bool(FEEDBACK.search(body))
            pure = "inplace: false" in ln or "inplace:false" in ln
            hits.append(dict(file=rel, line=i, tag=tag, fn=fname, code=s[:140],
                             guard=nearest_guard(lines, i, py), in_try=in_try,
                             feedback=fb, narrow_claim=claim, pure=pure))
            break
    return hits


def shipped_js():
    out = []
    for repo in REPOS:
        mj = os.path.join(repo, "module.json")
        if not os.path.exists(mj):
            continue
        m = json.load(open(mj, encoding="utf-8"))
        rels = list(m.get("esmodules") or []) + list(m.get("scripts") or [])
        rels += ["babele-mappings.js"]           # 被 esmodule import
        for r in rels:
            p = os.path.join(repo, r.replace("/", os.sep))
            if os.path.exists(p):
                out.append((p, f"{os.path.basename(repo)}/{r}"))
    return out


WRITE_MARK = re.compile(r"open\([^)]*['\"][wa]\+?['\"]|json\.dump\(|writeFileSync|\.write\(")


def writing_tools():
    out = []
    for d in TOOL_DIRS:
        for base, _dirs, files in os.walk(d):
            if "__pycache__" in base:
                continue
            for f in files:
                if not f.endswith((".py", ".mjs", ".js")):
                    continue
                p = os.path.join(base, f)
                try:
                    txt = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                if WRITE_MARK.search(txt):
                    out.append((p, os.path.relpath(p, ROOT)))
    return out


# ------------------------------------------------- S2：preCreate 钩子死闸判据
PRECREATE = re.compile(r"Hooks\.(?:on|once)\(\s*['\"]preCreate(\w+)['\"]\s*,\s*(?:async\s*)?\(([^)]*)\)")


def scan_precreate(path, rel):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    out = []
    for i, ln in enumerate(lines, 1):
        m = PRECREATE.search(ln)
        if not m:
            continue
        args = [x.strip() for x in m.group(2).split(",")]
        data_arg = args[1] if len(args) > 1 else None
        body = "\n".join(lines[i - 1:i + 12])
        mutates_data = bool(data_arg and re.search(rf"\b\w+\(\s*{re.escape(data_arg)}\s*\)", body))
        uses_updatesource = "updateSource" in body
        out.append(dict(file=rel, line=i, doc_type=m.group(1), data_arg=data_arg,
                        mutates_data=mutates_data, uses_updateSource=uses_updatesource,
                        code=ln.strip()[:140]))
    return out


# ------------------------------- S3：完成标记与实际结果脱钩
# 同一类的第三种形态：guard 是一个**持久化的完成标记**，它的名字/读取点声称
# 「这件事已经做完了」，而写入点在「做失败了」的路径上照样执行。
SETFLAG = re.compile(r"setFlag(?:\?\.)?\(\s*[^,]+,\s*['\"]([\w.]+)['\"]\s*,\s*true")


def scan_completion_flags(path, rel):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    spans = func_spans(lines, py=False)
    out = []
    for i, ln in enumerate(lines, 1):
        m = SETFLAG.search(ln)
        if not m:
            continue
        a, b, fname = enclosing(spans, i)
        body_lines = lines[(a or 1) - 1:(b or i)]
        body = "\n".join(body_lines)
        # 同一函数里读同名 flag 当闸？
        gated = bool(re.search(rf"getFlag[^\n]*['\"]{re.escape(m.group(1))}['\"]", body))
        # 写入点之前有多少个只 console.warn 的 catch（＝被吞掉的失败）
        swallowed = len(re.findall(r"catch\s*\([^)]*\)\s*\{\s*\n?\s*console\.warn", body))
        # 写入点是否被任何「无失败」条件包住？（往上 3 行找 if）
        cond = ""
        for k in range(i - 1, max(0, i - 4), -1):
            s = lines[k - 1].strip()
            if s.startswith("if"):
                cond = s[:100]
                break
        out.append(dict(file=rel, line=i, fn=fname, flag=m.group(1), gated=gated,
                        swallowed_failures=swallowed, guard_before_write=cond))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep-continue", action="store_true")
    a = ap.parse_args()

    targets = [(p, r, r.endswith(".py")) for p, r in shipped_js()]
    tools = [(p, r, r.endswith(".py")) for p, r in writing_tools()]

    print(f"扫描面 A（发布到运行时的 JS）：{len(targets)} 个文件")
    for _p, r, _ in targets:
        print(f"    {r}")
    print(f"扫描面 B（会写回仓库的工具）：{len(tools)} 个文件")

    all_hits = []
    for p, r, py in targets + tools:
        all_hits += scan_file(p, r, py)

    # PY-CONTINUE 在扫描类脚本里是「跳过不相干输入」的惯用法，不是丢数据，
    # 默认从强候选里剔除（--keep-continue 可保留，用来复核这条剔除本身）。
    noise = set() if a.keep_continue else {"PY-CONTINUE"}
    strong = [h for h in all_hits
              if h["tag"] not in noise
              and h["narrow_claim"] and (not h["guard"] or not h["in_try"] or not h["feedback"])]
    silent = [h for h in all_hits if not h["feedback"] and not h["guard"]]

    print(f"\n丢弃点合计 {len(all_hits)}")
    print(f"  其中 Q4「自述更窄」且 Q1/Q2/Q3 至少一项不满足 → 强候选 {len(strong)}")
    print(f"  其中 无闸且静默 → {len(silent)}")

    print("\n" + "=" * 100)
    print("强候选（人工按 Q1..Q4 逐条判定）")
    print("=" * 100)
    for h in strong:
        print(f"[{h['tag']:13s}] {h['file']}:{h['line']}  fn={h['fn']}")
        print(f"    code : {h['code']}")
        print(f"    guard: {h['guard'] or '(无)'}")
        print(f"    in_try={h['in_try']}  feedback={h['feedback']}  narrow_claim={h['narrow_claim']}")

    print("\n" + "=" * 100)
    print("S2：preCreate 钩子改 data 而不改 doc（v14 会丢弃该改动）")
    print("=" * 100)
    n2 = 0
    for p, r, py in targets:
        if py:
            continue
        for h in scan_precreate(p, r):
            n2 += 1
            verdict = "死闸" if (h["mutates_data"] and not h["uses_updateSource"]) else "需人工看"
            print(f"[{verdict}] {h['file']}:{h['line']}  preCreate{h['doc_type']}  "
                  f"data 形参={h['data_arg']}  改data={h['mutates_data']}  用updateSource={h['uses_updateSource']}")
            print(f"    {h['code']}")
    if not n2:
        print("  （无 preCreate 钩子）")

    print("\n" + "=" * 100)
    print("S3：完成标记 vs 实际结果（flag 说「已完成」，写入点却在失败路径上照样执行）")
    print("=" * 100)
    n3 = 0
    for p, r, py in targets:
        if py:
            continue
        for h in scan_completion_flags(p, r):
            n3 += 1
            bad = h["gated"] and h["swallowed_failures"] > 0 and not h["guard_before_write"]
            print(f"[{'命中' if bad else '需人工看'}] {h['file']}:{h['line']}  fn={h['fn']}  flag={h['flag']}")
            print(f"    被当闸读取={h['gated']}  函数内被吞掉的失败(catch+console.warn)={h['swallowed_failures']}  "
                  f"写入前的条件={h['guard_before_write'] or '(无)'}")
    if not n3:
        print("  （无持久化完成标记）")

    if a.verbose:
        print("\n全部丢弃点：")
        for h in all_hits:
            print(f"  [{h['tag']:13s}] {h['file']}:{h['line']} fn={h['fn']} try={h['in_try']} "
                  f"fb={h['feedback']} claim={h['narrow_claim']} :: {h['code'][:90]}")


if __name__ == "__main__":
    main()
