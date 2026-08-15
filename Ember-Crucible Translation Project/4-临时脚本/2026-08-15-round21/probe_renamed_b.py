# -*- coding: utf-8 -*-
"""变异回测：证明 `scan_renamed_terms --mode cn-term` 在新基准上真的在扫
`crucible.adversary-equipment.json`，而不是空转。

做法（全部在 scratchpad 里，**不碰任何真文件**）：
  1. 把 `2-Crucible汉化插件/compendium/{en,cn}` 整份拷成一个假仓 `mutant-repo/`；
  2. 把假仓 **en 侧** `crucible.adversary-equipment.json` 里的条目
     `Pseudopod` 整个改名成 `Ambulatory Bleb`（模拟「上游改名」），
     cn 侧原封不动（模拟「中文没跟上，还留着旧译名 伪足」）；
  3. `--old` 仍指真基准 `crucible-cn-0.9.0-shipped-en/`（那里还叫 `Pseudopod`）。

预期：注入前 findings 0；注入后 **必须** 报出 phrase=Pseudopod 的一条。
2026-08-15 实跑：A 0 条 / B 1 条（cn='伪足'，命中
`entries.Pseudopod.name`）—— PASS。
两次都打印 findings 数与「扫了几个包」，任何一次数字对不上就是判据坏了。
"""
from __future__ import annotations
import io
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCRATCH = r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\289d7a82-7d7b-4b2d-ac68-1439487a5f75\scratchpad"
REPO = os.path.join(P, "2-Crucible汉化插件")
OLD = os.path.join(P, "5-其他内容", "english-baseline", "crucible-cn-0.9.0-shipped-en")
GLOSS = os.path.join(P, "5-其他内容", "glossary", "glossary_ec.json")
IDS = os.path.join(P, "5-其他内容", "reports", "crucible_ids.json")
SCAN = os.path.join(P, "3-常用脚本", "qa", "scan_renamed_terms.py")
PACK = "crucible.adversary-equipment.json"
# 受害条目要挑「全库 en 里只出现 2 次（键 + name 值）」的，否则改完
# `cur.has_phrase()` 仍然为真、候选直接被跳过 —— 第一版探针挑了 `Exoskeleton`，
# 因为还有个 `Fused Exoskeleton` 含着它，注入后照样报 0，差点被当成「判据空转」。
VICTIM = "Pseudopod"
NEWNAME = "Ambulatory Bleb"


def build_mutant(dst, mutate):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "compendium"))
    for side in ("en", "cn"):
        shutil.copytree(os.path.join(REPO, "compendium", side),
                        os.path.join(dst, "compendium", side))
    if not mutate:
        return 0
    p = os.path.join(dst, "compendium", "en", PACK)
    d = json.load(io.open(p, encoding="utf-8-sig"))
    ent = d["entries"]
    assert VICTIM in ent, f"受害条目 {VICTIM} 不在 en 包里，探针失效"
    row = ent.pop(VICTIM)
    row["name"] = NEWNAME
    ent[NEWNAME] = row
    json.dump(d, io.open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 反空转：确认变异真的落了盘
    back = io.open(p, encoding="utf-8-sig").read()
    assert NEWNAME in back and f'"{VICTIM}":' not in back, "变异没落盘"
    return 1


def run(repo, out):
    cmd = [sys.executable, SCAN, "--repo", repo, "--old", OLD,
           "--mode", "cn-term", "--glossary", GLOSS, "--ids", IDS, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    txt = r.stdout + r.stderr
    packs = [l for l in txt.splitlines() if "包覆盖" in l]
    fin = [l for l in txt.splitlines() if "[B cn-term]" in l]
    print("   " + (packs[0].strip() if packs else "!! 没有包覆盖行"))
    print("   " + (fin[0].strip() if fin else "!! 没有 findings 行"))
    res = json.load(io.open(out, encoding="utf-8"))
    rows = res if isinstance(res, list) else res.get("findings", [])
    return txt, rows


def main():
    clean = os.path.join(SCRATCH, "mutant-repo-clean")
    dirty = os.path.join(SCRATCH, "mutant-repo-dirty")
    print("== A. 未注入（对照）==")
    build_mutant(clean, False)
    _, rows_a = run(clean, os.path.join(SCRATCH, "probe_a.json"))
    print(f"   findings={len(rows_a)}  phrases={[r.get('phrase') for r in rows_a][:5]}")

    print(f"== B. 注入：en 侧把 {VICTIM} 改名成 {NEWNAME}，cn 不动 ==")
    build_mutant(dirty, True)
    _, rows_b = run(dirty, os.path.join(SCRATCH, "probe_b.json"))
    print(f"   findings={len(rows_b)}  phrases={[r.get('phrase') for r in rows_b][:5]}")
    for r in rows_b:
        if r.get("phrase") == VICTIM:
            print(f"   命中详情: cn={r.get('cn')!r} hit_count={r.get('hit_count')} "
                  f"hits={[ (h['pack'], h['path']) for h in r.get('hits', [])][:3]}")

    ok = (len(rows_a) == 0) and any(r.get("phrase") == VICTIM for r in rows_b)
    print("\n判定：" + ("PASS —— 注入前 0 / 注入后报出，判据确实在扫这个包"
                       if ok else "FAIL —— 判据空转或被别的规则吃掉了，结论不可用"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
