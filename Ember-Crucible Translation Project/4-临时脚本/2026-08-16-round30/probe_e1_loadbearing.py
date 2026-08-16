# -*- coding: utf-8 -*-
"""第三十轮 · D 的验收：E1 新加的 6 条通知反例**承不承重**。

要回答的问题只有一个：把第二十九轮删掉的那条正则
    ^"(.+)" is already registered for this (layer|color)\\.$
加回被判文件，主闸会不会因为**这 6 条**而红？

⚠ 不碰真身：用 `_trans_make_tree` 在临时树里造一份**加了那条正则**的
  `ember-hardcoded-cn.mjs` 副本，跑**发布中的那条规则**（`_trans_rule()`）。
  被判文件本体一个字节都不动（本轮硬约束 1）。

三种口径对拍：
  base   —— 副本 = 真身（没加正则）：必须 0 违规（对照，不干净就什么都说明不了）
  re-add —— 加回那条正则、规则用**落 E1 之前**的（39 条反例）
  re-add + E1 —— 加回那条正则、规则用**落 E1 之后**的（45 条反例）
差出来的那几处，就是这 6 条自己挣的。
"""
import copy
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "3-常用脚本", "qa"))
sys.stdout.reconfigure(encoding="utf-8")

import assert_resolutions as AR       # noqa: E402

HEAD = "const NOTIFICATION_PATTERNS = [\n"
READD = (HEAD + '  { re: /^"(.+)" is already registered for this (layer|color)\\.$/,\n'
                '    cn: (m) => `「${m[1]}」已在该层上注册过了。` },\n')

rule_now = AR._trans_rule()
assert rule_now is not None
before = json.load(io.open(os.path.join(HERE, "RESOLUTIONS.assertions.before-e1.json"),
                           encoding="utf-8"))
rule_before = next(r for r in before["assertions"] if r.get("kind") == "translate_cases")


def run(mutate, rule):
    with tempfile.TemporaryDirectory() as tmp:
        repos = AR._trans_make_tree(os.path.join(tmp, "t"), mutate)
        return AR.a_translate_cases(rule, AR._TransCtx(repos))


print("=" * 90)
for note, mut, rule in [
    ("对照：副本 = 真身，规则 = 落 E1 之后（45 条反例）", None, rule_now),
    ("加回那条正则 · 规则 = 落 E1 **之前**（39 条反例）",
     AR._trans_mut(HEAD, READD), rule_before),
    ("加回那条正则 · 规则 = 落 E1 **之后**（45 条反例）",
     AR._trans_mut(HEAD, READD), rule_now),
]:
    b, d = run(mut, copy.deepcopy(rule))
    print(f"\n▸ {note}")
    print(f"    违规 {len(b)} 处   —— {d[:150]}")
    for x in b:
        print(f"      · [{x[1]}] {str(x[2])[:60]}")
        print(f"          {str(x[3])[:170]}")
print("\n" + "=" * 90)
