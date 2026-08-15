# -*- coding: utf-8 -*-
"""
dialog_selector_gap.py —— 「闸/选择器失配」子类 D4：选择器只够到窗口标题

patchRenderedApplications 对**非 Ember 根元素**的 DialogV2 只做这一件事：
    root.querySelector(".window-title")  →  translateText(textContent)
所以 DialogV2 的 content / 按钮 label / 表单 hint / placeholder 里的硬编码英文，
原理上不可能被这个选择器够到 —— 和种子同构（闸放行了，选择器/匹配器够不着）。

本脚本枚举 ember.mjs 里所有 DialogV2 调用点，抽出其中的硬编码英文字面量，
按「是否走 _loc() i18n」分两档：走 i18n 的不算缺陷（lang 能翻），裸英文才算。
只读。
"""
import os
import re
import sys

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"

Q = "[\"'`]"
KEY = r"(content|label|hint|placeholder|ok|yes|no|title)"
STR = re.compile(KEY + r"\s*:\s*" + Q + r"([^\"'`]*)" + Q)


def main():
    L = open(EMBER, encoding="utf-8").read().split("\n")
    sites = [i for i, l in enumerate(L)
             if "DialogV2" in l and re.search(r"\.(confirm|prompt|input|wait)\(|new foundry\S*DialogV2", l)]
    rows, seen = [], set()
    for i in sites:
        blk = "\n".join(L[max(0, i - 14): i + 32])
        for m in STR.finditer(blk):
            key, val = m.group(1), m.group(2)
            if not re.search(r"[A-Za-z]{3,}", val):
                continue
            if val.startswith(("EMBER.", "fa-", "modules/", "systems/")):
                continue
            if re.fullmatch(r"[a-z][A-Za-z0-9]*", val):      # 像标识符
                continue
            k = (i + 1, val)
            if k in seen:
                continue
            seen.add(k)
            rows.append((i + 1, key, val))
    print(f"DialogV2 调用点 {len(sites)} 个；其中裸英文字面量 {len(rows)} 条")
    for r in rows:
        print(f"{r[0]:>7}  {r[1]:<11} | {r[2][:150]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
