#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b（第 4 轮独立复核）：ember-hardcoded-cn.mjs 的**机制闸**。

与 h1_check_source_hit.py 的区别：
  1. 区分**证据来源**：js 字面量 / hbs 模板 / lang/en.json。
     只在 lang/en.json 里出现 = 该串走 Foundry i18n，.mjs 这条是冗余（不是缺陷但要标）。
  2. 对 EXACT 键额外判「它是不是整个文本节点」——补丁做的是 trim 后**整串相等**替换，
     出现在 `xxx ${y}` 模板串中间的键永远匹配不上。
  3. 反方向：从 hbs 模板与 js 里**收割**硬编码英文可见串，看 .mjs 覆盖了没有。

只读，不改任何文件。
"""
import json
import re
import sys
from pathlib import Path

MOD = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")
KEYS = Path(sys.argv[1])

SUFFIX_OK = {".mjs", ".js", ".hbs", ".html", ".json"}


def load_corpus():
    files = {}
    for p in sorted(MOD.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUFFIX_OK:
            files[str(p.relative_to(MOD)).replace("\\", "/")] = p.read_text(
                encoding="utf-8", errors="replace")
    return files


def classify(fn):
    if fn.startswith("lang/"):
        return "lang"
    if fn.endswith((".hbs", ".html")):
        return "tpl"
    if fn.endswith((".mjs", ".js")):
        return "js"
    return "other"


def main():
    raw = KEYS.read_text(encoding="utf-8")
    data = json.loads(raw[:raw.rindex("}") + 1])
    corpus = load_corpus()

    rows = []
    for table, d in data.items():
        if table == "PREFIXED":
            for e in d:
                rows.append((table, e["en"] + ": ", e["cn"]))
            continue
        if table == "PATTERNS":
            continue
        for en, cn in d.items():
            rows.append((table, en, cn))

    print("表\t英文键\t中文\tjs\ttpl\tlang\t整串证据\t示例")
    for table, en, cn in rows:
        buckets = {"js": 0, "tpl": 0, "lang": 0, "other": 0}
        whole = 0
        sample = ""
        for fn, txt in corpus.items():
            k = classify(fn)
            # 出现次数（原样子串）
            c = txt.count(en)
            if not c:
                continue
            buckets[k] += c
            if not sample:
                i = txt.find(en)
                sample = txt[max(0, i - 60):i + len(en) + 40].replace("\n", "\\n")
                sample = f"{fn}:{sample}"
            # 「整串」证据：JS 里写成完整字面量 "X" / `X` / 'X'，或模板里 >X<
            for pat in (f'"{en}"', f"'{en}'", f"`{en}`", f">{en}<",
                        f'"{en}"', f">\n        {en}"):
                whole += txt.count(pat)
        flag = ""
        if not any(buckets.values()):
            flag = "  <<<NOT_FOUND"
        elif buckets["js"] == 0 and buckets["tpl"] == 0 and buckets["lang"] > 0:
            flag = "  <<<LANG_ONLY"
        elif whole == 0:
            flag = "  <<<NO_WHOLE_STRING"
        print(f"{table}\t{en}\t{cn}\t{buckets['js']}\t{buckets['tpl']}\t"
              f"{buckets['lang']}\t{whole}\t{sample[:160]}{flag}")


if __name__ == "__main__":
    main()
