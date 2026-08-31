# -*- coding: utf-8 -*-
"""从当前 `compendium/en` 截一份**全包**英文基准，供下一次上游升级时做 diff。

    python capture_baseline.py --repo <汉化插件目录> --dest <english-baseline/xxx>

为什么要截这一份（第十九轮 Y5）
------------------------------
三条 drift 闸（dropped_terms / number_drift / marker_followup）都是
「旧英文 vs 当前英文 vs 中文」的三方比对，**旧英文必须是历史快照**。
ember 侧的历史快照 `ember-cn-v1.0.15-shipped-en/` 只装了 3 个包。
⚠⚠ **此前这里写「另 6 个包永远无法补答」，是错的，第二十轮已推翻。**
真相是：`v1.0.15` 那会儿模块**根本只有那 3 个 crucible 侧的包**，
`ember.adventure.json` 等是 **`v1.1.0` 才加进来的**（`git ls-tree v1.0.15` / `v1.1.0` 可复现），
所以它们的正确基准不是 v1.0.15 而是 **v1.1.0** —— 而 v1.1.0 的 `compendium/en/`
**一直躺在 `1-Ember汉化插件` 自己的 git 历史里**（该目录从 v1.1.0 起被跟踪）。
已截成 `english-baseline/ember-cn-v1.1.0-shipped-en/`，三条闸拿它跑：扫 10 包 / 缺 0 / 零告警。
⇒ **教训：找历史英文之前，先去插件仓的 git 历史里看一眼，每个 tag 都是一份现成基准。**
两份基准**并存、各答各的区间**，谁也替不了谁。
本工具的用途因此收窄为「**为将来备份**」：升级前把当前全部包截下来，
下一次上游升级时这三条闸就是全覆盖的。

命名里写死上游版本号（从 `compendium/en/_source.json` 读 packageId/packageVersion），
否则下一轮不知道这份基准是拿谁截的。

住在哪 / 为什么挪过来（第二十轮）
--------------------------------
本文件原先在 `4-临时脚本/2026-08-15-round19/`，但它是「**每次上游升级前必跑**」的
常驻工具，不是那一轮跑完就作废的产物；`4-临时脚本/` 按 .gitignore 的取舍规矩收的是
「跑出来的、可重跑的」那一档。第二十轮挪进 `3-常用脚本/qa/`。

⚠ 挪动的连带影响：`scan_dropped_terms` / `scan_en_drift` / `scan_marker_followup` /
`scan_number_drift` / `scan_renamed_terms` 五个 scan_*.py 里写着旧路径的**共 9 处**
（docstring + `--strict-coverage` 缺包时打印的提示语），第二十轮已全部改成
`3-常用脚本/qa/capture_baseline.py`。⚠ 此前这里写「12 处」是错的，实测 9。
验收：`grep -rn "2026-08-15-round19/capture_baseline.py" --include=*.py 3-常用脚本/` 应为 0。

⚠ 另有 **2 处故意不改**：两份升级前快照的 `_source.json` 里的 `capturedBy` 字段。
那是**历史溯源记录** —— 记的是截那份快照时脚本确实在哪儿，改了等于篡改历史。
所以报数时要说清是 **9（脚本内）还是 11（含这 2 处历史记录）**。
本文件写出的**新**快照已记录新路径。
"""
from __future__ import annotations
import argparse
import io
import json
import os
import shutil
import sys
import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(p):
    return json.load(io.open(p, encoding="utf-8-sig"))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out[".".join(path)] = node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, "compendium", "en")
    cn_dir = os.path.join(a.repo, "compendium", "cn")
    src_meta = load(os.path.join(en_dir, "_source.json"))
    os.makedirs(a.dest, exist_ok=True)

    packs = []
    for f in sorted(os.listdir(en_dir)):
        if not f.endswith(".json") or f == "_source.json":
            continue
        shutil.copyfile(os.path.join(en_dir, f), os.path.join(a.dest, f))
        en_l, cn_l = {}, {}
        leaves(load(os.path.join(en_dir, f)).get("entries", {}), [], en_l)
        cnp = os.path.join(cn_dir, f)
        if os.path.exists(cnp):
            leaves(load(cnp).get("entries", {}), [], cn_l)
        packs.append({"file": f, "en_leaves": len(en_l), "cn_leaves": len(cn_l),
                      "cn_present": os.path.exists(cnp)})
        print(f"  {f}  en {len(en_l)}  cn {len(cn_l)}")

    meta = {
        "kind": "pre-upgrade-snapshot",
        "capturedAt": datetime.datetime.now().isoformat(timespec="seconds"),
        "capturedBy": "3-常用脚本/qa/capture_baseline.py",
        "capturedFrom": os.path.abspath(en_dir),
        "upstream": {"packageId": src_meta.get("packageId"),
                     "packageVersion": src_meta.get("packageVersion"),
                     "packageType": src_meta.get("packageType")},
        "includesLocalPatches": True,
        "note": a.note,
        "packs": packs,
        "totals": {"packs": len(packs),
                   "en_leaves": sum(p["en_leaves"] for p in packs),
                   "cn_leaves": sum(p["cn_leaves"] for p in packs)},
        "upstreamSourceMeta": src_meta,
    }
    with io.open(os.path.join(a.dest, "_source.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"包 {len(packs)}  en 叶 {meta['totals']['en_leaves']}  cn 叶 {meta['totals']['cn_leaves']}")
    print(f"-> {os.path.abspath(a.dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
