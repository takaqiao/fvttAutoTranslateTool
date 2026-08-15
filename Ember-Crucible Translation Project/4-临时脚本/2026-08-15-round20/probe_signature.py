# -*- coding: utf-8 -*-
"""证明：`&amp;reference[Paralyzed]` 这条补丁若不重打，重抽后闸门会拒收现有中文。

不自己复写正则 —— 直接 import `3-常用脚本/qa/apply_translations.py` 的 markup_signature，
避免「我抄的正则和真闸不是同一个」这种假绿。

自证在验（反空转）：
  先跑三个**已知答案**的对照，任一条不符预期即退出码 2，不再往下跑真数据。
"""
from __future__ import annotations
import io
import json
import os
import sys
import importlib.util

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
QA = os.path.join(ROOT, "3-常用脚本", "qa", "apply_translations.py")

spec = importlib.util.spec_from_file_location("apply_translations", QA)
mod = importlib.util.module_from_spec(spec)
sys.argv = [QA]          # 该模块 main() 只在 __main__ 下跑，import 安全
spec.loader.exec_module(mod)
sig = mod.markup_signature
print(f"已载入真闸: {QA}")
print(f"REFERENCE 正则 = {mod.REFERENCE.pattern}")

# ---- 自证：三条已知答案的对照 ----
selftest = [
    ("&amp;reference[X] 应被认成 1 个 REFERENCE 记号",
     lambda: sig("<p>a &amp;reference[X] b</p>")["&reference[X]"] == 1),
    ("裸 reference[X]（没有 &）应被认成 0 个",
     lambda: "&reference[X]" not in sig("<p>a reference[X] b</p>")
             and "reference[X]" not in sig("<p>a reference[X] b</p>")),
    ("带 & 与不带 & 的签名必然不等（否则本探针无意义）",
     lambda: sig("<p>a &amp;reference[X] b</p>") != sig("<p>a reference[X] b</p>")),
]
bad = 0
for name, fn in selftest:
    ok = False
    try:
        ok = bool(fn())
    except Exception as e:      # noqa: BLE001
        print(f"  自证异常 {name}: {e}")
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    bad += 0 if ok else 1
if bad:
    print("!! 自证不过，探针本身不可信，退出")
    sys.exit(2)
print("自证 3/3 通过 —— 探针确实在验\n")


def load(p):
    return json.load(io.open(p, encoding="utf-8-sig"))


PATH = ["Ember Early Access", "items", "Paralyzing Bolt",
        "effects", "Paralyzed", "description"]


def dig(d):
    for k in PATH:
        d = d[k]
    return d


REEX = os.path.join(os.path.dirname(__file__), "reextract")
scanned = 0
for pack in ["ember.adventure.json", "ember.crucible-adventure.json"]:
    en_cur = dig(load(os.path.join(ROOT, "1-Ember汉化插件", "compendium", "en", pack))["entries"])
    en_raw = dig(load(os.path.join(REEX, pack))["entries"])
    cn = dig(load(os.path.join(ROOT, "1-Ember汉化插件", "compendium", "cn", pack))["entries"])
    scanned += 1
    print(f"--- {pack}")
    print(f"  已打补丁的 en vs cn : {'一致(闸放行)' if sig(en_cur) == sig(cn) else '不一致(闸拒收)'}")
    d = sig(cn) - sig(en_raw)
    print(f"  重抽未补的 en vs cn : {'一致(闸放行)' if sig(en_raw) == sig(cn) else '不一致(闸拒收)'}"
          f"   中文多出的记号 = {dict(d)}")

print(f"\n本次扫了 {scanned} 个包 × 1 叶 = {scanned} 叶")
