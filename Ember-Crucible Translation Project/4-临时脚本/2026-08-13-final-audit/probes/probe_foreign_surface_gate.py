# -*- coding: utf-8 -*-
"""
probe_foreign_surface_gate.py —— 只读探针，不写库

把种子缺陷抽象成的机械判据
--------------------------
种子：`patchEnrichers` 在 **CONFIG.TextEditor.enrichers**（Foundry 拥有的注册表）上动手，
决定「包哪一条」的闸门用的是 **正则字面量的源码文本里有没有某个子串**（表面特征），
而不是「这条增强器是不是 Ember 注册的」（归属）。

抽象后的判据：
  在两个插件里找出所有「**对不属于自己的运行期表面**做写入 / 包裹 / 注册」的点位，
  取出决定「作用到哪些条目」的闸门表达式，把闸门分成三类：
    OWNERSHIP —— 判据是归属证据：entry.id 前缀 / package id / pack.metadata.packageName /
                 我自己的 subtype 命名空间（`ember.*`）/ game.modules.get(...)
    SURFACE   —— 判据是数据的表面特征：对源码文本 / className / 调用栈字符串 / 值查表 /
                 裸 documentType 名做匹配
    NONE      —— 无闸门，无条件作用于该表面上的全部条目
  凡 SURFACE / NONE 的点位，都要人工核对「它实际会碰到多少条别人的东西」。

已知假阳性模式（脚本不做语义分析，必须人工复核）：
  1. 一个表面可能事实上只有本模块自己的条目（例如某 CONFIG 子树只有我注册过），
     这时 NONE 也无害 —— 但要拿上游源码证明，不能默认。
  2. 闸门可能分散在多行 / 在被调用的辅助函数里，脚本只回看 8 行。
  3. `Hooks.on(...)` 的全局钩子闸门写在 handler 内部，脚本按 handler 首 12 行找。
"""
import io
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FILES = [
    os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
    os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, "1-Ember汉化插件", "babele-mappings.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js"),
    os.path.join(ROOT, "2-Crucible汉化插件", "babele-mappings.js"),
]

# 「不属于自己的运行期表面」的写入 / 包裹 / 注册点
SINKS = [
    ("CONFIG.*",            re.compile(r"\bCONFIG\??\.[A-Za-z_$][\w$]*")),
    ("globalThis.*",        re.compile(r"\bglobalThis\.[A-Za-z_$][\w$]*\s*=")),
    ("crucible.CONFIG/API", re.compile(r"\bcrucible\??\.(CONFIG|api|CONST)\b")),
    ("game.i18n.translations", re.compile(r"\bgame\.i18n\.translations\b")),
    ("game.time.calendar",  re.compile(r"\bgame\??\.time\??\.calendar\b")),
    ("documentClass 猴补丁", re.compile(r"documentClass|\.updateDocuments\s*=|\.prepare\s*=")),
    ("全局 Hook",            re.compile(r"\bHooks\.(on|once)\(\s*['\"](preUpdate|preCreate|preDelete|render)")),
    ("babele 全局注册",       re.compile(r"\bbabele\.(registerConverters|registerMapping)\b")),
    ("ui.windows/应用实例",   re.compile(r"\bui\.windows\b|foundry\.applications\.instances")),
]

OWNERSHIP = re.compile(
    r"entry\.id|\.id\s*\.startsWith|packageName|metadata\.package|game\.modules\.get\("
    r"|moduleId|MODULE_ID|['\"]ember\.[a-z]|_stats\.compendiumSource|\.pack\b",
    re.I)

SURFACE = re.compile(
    r"String\(\s*entry\.pattern|new Error\(\)\.stack|\.stack\b|className|constructor\?\.name"
    r"|\.test\(\s*(src|cls|id)\b|table\[|\bin EXACT\b|startsWith\(`\$\{en\}|typeof .*=== ['\"]string['\"]",
    re.I)


def guard_window(lines, idx, back=8):
    lo = max(0, idx - back)
    return "\n".join(lines[lo:idx + 1])


def classify(win):
    own = bool(OWNERSHIP.search(win))
    surf = bool(SURFACE.search(win))
    if own and not surf:
        return "OWNERSHIP"
    if surf:
        return "SURFACE" + ("(+归属信息在场但未用)" if own else "")
    if re.search(r"\bif\s*\(|\bcontinue\b|\breturn\b", win):
        return "OTHER-GUARD"
    return "NONE"


def main():
    out = []
    for path in FILES:
        with io.open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("*") or line.strip().startswith("//"):
                continue  # 注释行不算点位
            for name, rx in SINKS:
                if rx.search(line):
                    kind = classify(guard_window(lines, i))
                    out.append((os.path.basename(path), i + 1, name, kind, line.strip()[:110]))
                    break

    print("文件 | 行 | 外部表面 | 闸门类别 | 代码")
    print("-" * 130)
    for row in out:
        print("%-24s %5d  %-22s %-28s %s" % row)

    print()
    from collections import Counter
    c = Counter(r[3] for r in out)
    print("闸门类别统计:", dict(c))
    print("点位总数:", len(out))
    print()
    print("需人工复核（SURFACE / NONE）：")
    for r in out:
        if r[3].startswith("SURFACE") or r[3] == "NONE":
            print("  %s:%d  %s  ->  %s" % (r[0], r[1], r[2], r[4]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
