# -*- coding: utf-8 -*-
"""把 ember 两个探针的结果做去重与分组，给出可核对的最终清单（只读）。"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    dlg = json.load(io.open(os.path.join(HERE, "ember_interactable_dialogs.json"), encoding="utf-8"))
    btn = json.load(io.open(os.path.join(HERE, "i18n_button_slot_ember.json"), encoding="utf-8"))

    inter_titles = set(dlg["dialog_window_title"]["gap"])
    inter_labels = set(dlg["dialog_buttons_label"]["gap"])
    dyn = dlg["configureDialog_dynamic"]["gap"]
    for k in dyn:
        kind, t = k.split("|", 1)
        (inter_titles if kind == "title" else inter_labels).add(t)

    other_t, other_l = {}, {}
    for k, v in btn.items():
        kind, t = k.split("|", 1)
        if kind == "title":
            if t not in inter_titles:
                other_t[t] = v
        else:
            if t not in inter_labels:
                other_l[t] = v

    print("A. EmberInteractable 交互物件族")
    print(f"   窗口标题 {len(inter_titles)}：", sorted(inter_titles))
    print(f"   按钮标签 {len(inter_labels)}：", sorted(inter_labels))
    print()
    print("B. 其余 DialogV2 / ApplicationV2")
    print(f"   窗口标题 {len(other_t)}：", sorted(other_t))
    print(f"   按钮标签 {len(other_l)}：", sorted(other_l))
    print()
    print("合计不同串", len(inter_titles | inter_labels | set(other_t) | set(other_l)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
