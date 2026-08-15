#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 批量跑英文闸，核对 ember-hardcoded-cn.mjs 的 LANGUAGES / MOODS 等译名
是否与 compendium 的多数写法一致。只读。

用法： python h1_lang_gate.py <repo>
"""
import subprocess
import sys
from pathlib import Path

P = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
GATE = P / "4-临时脚本" / "2026-08-12-fix" / "term_gate.py"

CASES = [
    ("Arcden", "奥克登语,阿克登语,奥克登,阿克登"),
    ("Cascal", "卡斯卡语,卡斯卡尔语,卡斯卡"),
    ("Forest Speech", "森语,森林语,林语"),
    ("Hardac", "哈达克语,哈达克"),
    ("Solical", "索利卡语,索利卡尔语,索利卡"),
    ("Mithia", "密西亚语,米西亚语,密西亚"),
    ("Luma", "卢玛语,卢玛"),
    ("Kaziric", "卡兹瑞克语,卡济里克语,卡兹里克语,卡兹瑞克"),
    ("Scripta", "书文语,斯克里普塔语,书写语"),
    ("Wyrdic", "维尔迪克语,维尔迪克"),
    ("Pathward", "歧路语,径道语,通路语,歧路"),
    ("Scor", "斯科尔语,斯科语,斯科尔"),
    ("Towyr", "托威尔语,托维尔语,托威尔,托维尔"),
    ("Windclaw", "风爪语,风爪"),
    ("Abyssal", "深渊语"),
    ("Draconic", "龙语,龙族语"),
    ("Druidic", "德鲁伊语"),
    ("Lunix", "月语,卢尼克斯语"),
    ("Caligon", "卡利贡语,卡利贡"),
    ("Eonic", "永世语,永恒语,伊奥尼克语"),
    ("Harmos", "和谐语,哈莫斯语,哈尔莫斯语"),
    ("Thieves' Cant", "盗贼黑话,盗贼切口,窃贼黑话"),
    ("Aedir", "埃迪尔,艾迪尔,埃迪尔人"),
    ("Leviathans?", "利维坦,巨兽"),
    ("Shent", "申特"),
    ("Abyssals", "深渊生物,深渊众,深渊者"),
]


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "1-Ember汉化插件"
    for en, cn in CASES:
        print(f"\n===== {en}  ->  {cn}")
        r = subprocess.run(
            [sys.executable, str(GATE), "--repo", repo, "--en", rf"\b{en}\b", "--cn", cn, "--show", "0"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(P))
        for line in (r.stdout or "").splitlines():
            if any(k in line for k in ("rows whose", "gated_hit", "EN matches")):
                print("   ", line.strip())
        if r.returncode:
            print("   !!", (r.stderr or "")[:200])


if __name__ == "__main__":
    main()
