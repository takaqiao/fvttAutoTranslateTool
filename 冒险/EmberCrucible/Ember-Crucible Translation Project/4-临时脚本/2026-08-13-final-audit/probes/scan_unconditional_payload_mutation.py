# -*- coding: utf-8 -*-
"""
举一反三探针 #3：**防御式改写被无条件套用到全部文档载荷上**

判据（对已确认实例的第二层抽象）
--------------------------------
已确认实例的根因不只是「形状搞错了」，更是**调用点没有任何条件判据**：
`normalizeDescriptionValue` 本来只想修「旧世界里写成字符串的 description」，
但 `sanitizeItemDataShape` 把它挂在 preUpdateItem 上，对**任何系统、任何子类型、
每一次 item 更新**都跑；`migrateLegacyDescriptionShape` 遍历**全部**世界 item 与
内嵌 item。中间层（DataField.clean / mergeObject / updateSource）不报错，
于是正常数据被静默改写。

抽象成可扫的判据：
  在**随插件发布的 JS**（module.json 的 esmodules/scripts）里找出所有
  「拿到别人的文档载荷（update payload / changes / data）之后**改写或删除**它的键」
  的函数，然后对每一个回答三问：
    Q1 这个改写只在**病态输入**上生效吗？（有没有判子类型 / 判形状 / 判失败的闸）
    Q2 闸装在**改写之前**还是**失败之后**？（装在 try 之后 = 无条件预降级）
    Q3 改写失败/多余时**有没有可见反馈**？（无 → 静默）
  三问里只要出现「无闸 / 预降级 / 静默」，就是同一类命中。

本脚本只做机械定位（列出候选点 + 上下文），判定必须人工。

假阳性模式
----------
* 会把 converter 里那些 `{inplace:false}` 的纯函数式 mergeObject 也列出来 —— 那些
  不改调用方对象，不是本类。脚本会把 `inplace: false` 标出来，便于快速排除。
* 只按正则找 `delete X.y` / `X.y = ...` / `mergeObject(a,b)`（缺 inplace:false）
  / `setProperty(` / `.map(` 重建数组，覆盖不到通过别名间接改写的情形（假阴性）。
"""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [
    os.path.join(ROOT, "1-Ember汉化插件"),
    os.path.join(ROOT, "2-Crucible汉化插件"),
]

PATTERNS = [
    ("DELETE",        re.compile(r"\bdelete\s+([A-Za-z_$][\w$]*)\.(\w+)")),
    ("TYPE-REPLACE",  re.compile(r"\b([A-Za-z_$][\w$]*(?:\.\w+|\[[^\]]+\])*)\s*=\s*(\[\]|\{\}|''|\"\")\s*;")),
    ("MERGE-INPLACE", re.compile(r"foundry\.utils\.mergeObject\(")),
    ("SETPROPERTY",   re.compile(r"foundry\.utils\.setProperty\(")),
    ("ARRAY-REBUILD", re.compile(r"\b([A-Za-z_$][\w$]*\.\w+)\s*=\s*\1\.map\(")),
    ("RETURN-EMPTY",  re.compile(r"return\s+\{\}\s*;")),
]

# 载荷型形参/变量名：这些名字意味着「别人的文档数据」
PAYLOAD_NAMES = {"changes", "update", "updates", "data", "itemData", "itemUpdate",
                 "payload", "degraded", "source", "actions", "action", "effects",
                 "effect", "value", "translation"}

FUNC_RE = re.compile(r"^\s*(?:async\s+)?function\s+(\w+)|^\s*const\s+(\w+)\s*=\s*(?:async\s*)?\(")

def shipped_files(repo):
    mj = os.path.join(repo, "module.json")
    if not os.path.exists(mj):
        return []
    m = json.load(open(mj, encoding="utf-8"))
    out = []
    for rel in (m.get("esmodules") or []) + (m.get("scripts") or []):
        p = os.path.join(repo, rel.replace("/", os.sep))
        if os.path.exists(p):
            out.append((m.get("id"), rel, p))
    # 被 esmodule import 的同仓文件也算
    for extra in ("babele-mappings.js",):
        p = os.path.join(repo, extra)
        if os.path.exists(p):
            out.append((m.get("id"), extra, p))
    return out

def main():
    total = 0
    for repo in REPOS:
        for mod, rel, path in shipped_files(repo):
            lines = open(path, encoding="utf-8").read().splitlines()
            cur = None
            print(f"\n{'='*90}\n{mod} :: {rel}  ({len(lines)} 行)\n{'='*90}")
            for i, ln in enumerate(lines, 1):
                fm = FUNC_RE.match(ln)
                if fm:
                    cur = fm.group(1) or fm.group(2)
                for tag, rx in PATTERNS:
                    m = rx.search(ln)
                    if not m:
                        continue
                    if tag == "MERGE-INPLACE" and "inplace: false" in ln:
                        continue          # 函数式合并，不改调用方对象
                    if tag == "TYPE-REPLACE":
                        head = m.group(1).split(".")[0].split("[")[0]
                        if head not in PAYLOAD_NAMES:
                            continue      # 只关心作用在载荷上的
                    total += 1
                    print(f"  [{tag:14s}] {rel}:{i}  fn={cur}")
                    print(f"      {ln.strip()[:150]}")
    print(f"\n候选点合计 {total} 处（需人工按 Q1/Q2/Q3 三问判定）")

if __name__ == "__main__":
    main()
