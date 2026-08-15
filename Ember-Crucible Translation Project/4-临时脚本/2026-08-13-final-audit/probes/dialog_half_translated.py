# -*- coding: utf-8 -*-
"""
dialog_half_translated.py

最紧的一版判据：只看**插件自己已经承认要翻**的那些 DialogV2（标题命中 EXACT 或
PATTERNS），检查它们的 content / 按钮 label / 表单 label,hint 里还剩多少裸英文。
命中 = 同一个对话框「标题中文、正文英文」。

这是种子那条的同构体：闸放行（DialogV2 例外分支），但选择器
`root.querySelector(".window-title")` 原理上只能够到标题一个节点。
只读。
"""
import re
import sys

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
PLUGIN = (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
          r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

Q = "[\"'`]"


def exact_keys():
    src = open(PLUGIN, encoding="utf-8").read()
    m = re.search(r"const EXACT = \{(.*?)\n\};", src, re.S)
    return set(re.findall(r'"([^"]+)":', m.group(1)))


PATTERN_TITLES = [re.compile(r"^Award Attunement: "), re.compile(r"^Revoke Attunement: "),
                  re.compile(r"^Activate Attunement: ")]


def title_covered(t, keys):
    if t in keys:
        return "EXACT"
    for p in PATTERN_TITLES:
        if p.match(t):
            return "PATTERN"
    return None


def main():
    keys = exact_keys()
    L = open(EMBER, encoding="utf-8").read().split("\n")
    sites = [i for i, l in enumerate(L)
             if "DialogV2" in l and re.search(r"\.(confirm|prompt|input|wait)\(", l)]
    TITLE = re.compile(r"title:\s*" + Q + r"([^\"'`]*)" + Q)
    BODY = re.compile(r"(content|label|hint)\s*:\s*" + Q + r"([^\"'`]*)" + Q)
    n_hit = 0
    for i in sites:
        blk_lines = L[max(0, i - 16): i + 34]
        blk = "\n".join(blk_lines)
        tm = TITLE.search(blk)
        if not tm:
            continue
        # 模板串里的 ${...} 抠掉再判 PATTERN
        raw_title = tm.group(1)
        probe = re.sub(r"\$\{[^}]*\}", "X", raw_title)
        cov = title_covered(probe, keys)
        if not cov:
            continue
        leftovers = []
        for m in BODY.finditer(blk):
            k, v = m.group(1), m.group(2)
            v2 = v.strip()
            if not re.search(r"[A-Za-z]{3,}", v2):
                continue
            if v2.startswith(("EMBER.", "fa-", "modules/", "systems/", "<code-mirror")):
                continue
            if re.fullmatch(r"[a-z][A-Za-z0-9]*", v2):
                continue
            if "_loc(" in v2:
                continue
            leftovers.append((k, v2))
        if leftovers:
            n_hit += 1
            print(f"--- line {i+1}  title={raw_title!r}  [{cov} 已覆盖]")
            for k, v in leftovers:
                print(f"      {k:<8} | {v[:140]}")
    print(f"\n共 {len(sites)} 个 DialogV2 调用点；标题已覆盖但正文/按钮仍是裸英文的：{n_hit} 个")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
