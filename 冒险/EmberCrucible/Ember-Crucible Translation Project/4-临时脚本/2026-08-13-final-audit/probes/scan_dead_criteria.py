# -*- coding: utf-8 -*-
"""
scan_dead_criteria.py — 同一类缺陷的第二个落点：**判据脚本本身扫了个空**

抽象自 register.js 的 game.world.getFlag 幂等闸：
  「判据建立在错误的外部契约假设上，且失败是静默的」。
在 3-常用脚本/qa/ 这一侧，同一类问题表现为：
  扫描器里写死的**路径 / glob / 目录名 / 键名**在当前仓库结构下匹配不到任何对象，
  于是它输出 0 条并被当成「这一类问题已清零」。这比运行时静默失效更危险 ——
  它直接制造「全绿」的假象。

做法（只读）：
  1. 从每个 qa/*.py 里抠出所有**看起来像路径片段**的字符串字面量
     （含 `/` 或以 .json/.mjs/.js/.md 结尾，或形如 compendium/cn 这样的目录名）
  2. 对每个片段，在项目根下做一次存在性 / glob 命中统计
  3. 命中 0 的列出来人工判断

已知假阳性：
  * 输出路径（脚本自己要写的文件，跑之前当然不存在）
  * 拼接用的片段（'/' 分隔符、URL、正则里的斜杠）
  * 只在命令行参数默认值里出现、实际调用时都被覆盖的路径
所以本探针**只产候选**，每一条都必须回到源码里看它是怎么用的。
"""
import os, re, sys, glob, json

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
QA = os.path.join(ROOT, "3-常用脚本")

STR = re.compile(r"""(['"])((?:\\.|(?!\1)[^\\])*)\1""")
PATHLIKE = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff\-. */]*$")

def looks_path(s):
    if len(s) < 3 or len(s) > 120: return False
    if not PATHLIKE.match(s): return False
    if "/" in s: return True
    if s.endswith((".json", ".mjs", ".js", ".md", ".py", ".css")): return True
    return False

def main():
    rows = []
    for dp, _dn, fn in os.walk(QA):
        if "__pycache__" in dp: continue
        for f in sorted(fn):
            if not f.endswith((".py", ".mjs", ".js")): continue
            p = os.path.join(dp, f)
            src = open(p, encoding="utf-8", errors="replace").read()
            seen = set()
            for m in STR.finditer(src):
                s = m.group(2)
                if s in seen or not looks_path(s): continue
                seen.add(s)
                line = src[:m.start()].count("\n") + 1
                # 三种解释：项目根相对 / 仓库根相对 / glob
                hits = 0
                for base in (ROOT, os.path.join(ROOT, "1-Ember汉化插件"),
                             os.path.join(ROOT, "2-Crucible汉化插件")):
                    cand = os.path.join(base, s.replace("/", os.sep))
                    if os.path.exists(cand): hits += 1
                    hits += len(glob.glob(cand))
                    hits += len(glob.glob(os.path.join(base, "**", s.replace("/", os.sep)), recursive=True))
                rows.append((hits, os.path.relpath(p, ROOT).replace("\\", "/"), line, s))
    rows.sort()
    zero = [r for r in rows if r[0] == 0]
    print(f"# 路径样字符串 {len(rows)} 条，其中 0 命中 {len(zero)} 条\n")
    for h, f, l, s in zero:
        print(f"{f}:{l}  ->  {s}")
    print("\n# 非 0 命中（抽样前 30）")
    for h, f, l, s in rows[len(zero):len(zero)+30]:
        print(f"{h:>4}  {f}:{l}  ->  {s}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
