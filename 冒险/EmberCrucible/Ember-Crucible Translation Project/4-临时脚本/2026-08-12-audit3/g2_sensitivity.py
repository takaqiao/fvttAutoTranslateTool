# -*- coding: utf-8 -*-
"""闸的灵敏度自测：把已修好的译文在**内存里**改回错值，闸必须重新报出来。

只测特异度（今天报 0）是不够的 —— 「全判 OK」也能得 0。这里把三类已知缺陷各注入
一条，验证 BROKEN 会回到 3；不写任何文件。
"""
import copy
import importlib.util
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
S = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
     r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3")
spec = importlib.util.spec_from_file_location(
    "snb", os.path.join(P, "3-常用脚本", "qa", "scan_name_binding.py"))
snb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snb)

packs = snb.load_packs([os.path.join(P, "1-Ember汉化插件"),
                        os.path.join(P, "2-Crucible汉化插件")])
ids, notes, results = snb.load_bindings(
    [os.path.join(S, f) for f in ("re_bind_ember.json", "re_bind_crucible.json",
                                  "re_bind_dnd5e.json")])

base_f, base_s = snb.scan(packs, ids, notes, results, False)
print(f"基线  BROKEN={base_s['note_BROKEN'] + base_s['result_BROKEN']}  "
      f"note_OK={base_s['note_OK']} result_OK={base_s['result_OK']}")

hurt = copy.deepcopy(packs)
adv = hurt["ember.crucible-adventure"]["cn"]["entries"]["Ember Early Access"]

# ① 表结果：漏掉双语并列的英文尾巴（本轮修的 16 条就是这一类）
adv["tables"]["Corpse Loot"]["results"]["51-51"]["name"] = "解除陷阱工具包"
# ② 表结果：整个词换掉（跨包目标 crucible.equipment）
adv["tables"]["Corpse Loot"]["results"]["13-13"]["name"] = "长柄斧 Halberd"
# ③ 场景针脚：标签与它打开的那一页的中文名不同（阶段 29 修的那一类）
#    挑第一条英中两侧都有的针脚下手
en_adv = hurt["ember.crucible-adventure"]["en"]["entries"]["Ember Early Access"]
pin = None
for sname, sv in en_adv["scenes"].items():
    for text in (sv.get("notes") or {}):
        cn_scene = adv["scenes"].get(sname) or {}
        if text in (cn_scene.get("notes") or {}):
            pin = (sname, text)
            break
    if pin:
        break
print(f"注入的针脚：{pin}")
adv["scenes"][pin[0]]["notes"][pin[1]] = "故意写错的名字"

f2, s2 = snb.scan(hurt, ids, notes, results, False)
broken = [f for f in f2 if f["verdict"] == "BROKEN"]
print(f"注入后 BROKEN={len(broken)}（note {s2['note_BROKEN']} / result {s2['result_BROKEN']}）")
for f in broken:
    print(f"  [{f['kind']}] {f['batch_path']}\n     CN {f['cn_label']!r} -> 目标 {f['target_cn']}")
