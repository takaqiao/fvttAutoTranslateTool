#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 从 ember-hardcoded-cn.mjs 抽出所有「英文键 -> 中文值」对，按表分组。

纯正则解析（文件是纯数据字面量，不需要 JS 引擎）。输出 JSON 给后续核对脚本用。
只读，不改任何文件。
"""
import json
import re
import sys
from pathlib import Path

MJS = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

# 表名 -> 起始声明
TABLES = ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
          "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"]

PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def strip_comments(block: str) -> str:
    # 去掉 // 行注释（本文件里注释不含引号内的 //，安全）
    out = []
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def main():
    src = MJS.read_text(encoding="utf-8")
    result = {}
    for t in TABLES:
        m = re.search(r"const\s+%s\s*=\s*\{" % t, src)
        if not m:
            print(f"!! 找不到表 {t}", file=sys.stderr)
            continue
        i = m.end() - 1
        depth = 0
        for j in range(i, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        block = strip_comments(src[i:j + 1])
        pairs = PAIR.findall(block)
        d = {}
        for en, cn in pairs:
            if en in d:
                print(f"!! {t} 重复键 {en}", file=sys.stderr)
            d[en] = cn
        result[t] = d

    # PREFIXED（数组形式）
    m = re.search(r"const\s+PREFIXED\s*=\s*\[(.*?)\n\];", src, re.S)
    pref = []
    if m:
        for em in re.finditer(r'\{\s*en:\s*"([^"]+)",\s*cn:\s*"([^"]+)",\s*table:\s*(\w+)\s*\}', m.group(1)):
            pref.append({"en": em.group(1), "cn": em.group(2), "table": em.group(3)})
    result["PREFIXED"] = pref

    # PATTERNS（正则形式，只抽出正则源与模板文本）
    m = re.search(r"const\s+PATTERNS\s*=\s*\[(.*?)\n\];", src, re.S)
    pats = []
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("{ re:"):
                continue
            rm = re.match(r"\{ re: (/.*?/),\s*cn:\s*(.*?)\s*\}", line)
            if rm:
                pats.append({"re": rm.group(1), "cn": rm.group(2)})
    result["PATTERNS"] = pats

    total = sum(len(v) for k, v in result.items() if isinstance(v, dict))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n# 字典型键数 {total}；PREFIXED {len(pref)}；PATTERNS {len(pats)}；"
          f"合计 {total + len(pref) + len(pats)}", file=sys.stderr)
    for k, v in result.items():
        print(f"#   {k}: {len(v)}", file=sys.stderr)


if __name__ == "__main__":
    main()
