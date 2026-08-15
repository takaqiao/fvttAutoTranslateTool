# -*- coding: utf-8 -*-
"""灵敏度回测：造一个最小合成仓库，证明补丁前后行为确实不同。

场景一（无角色回退）：`A.tokenName` 中文侧整条不存在，全库同英文的多数派来自
`.name`（双语并列）。原版会把「护盾术 Shield」写进 tokenName。
场景二（shape 键丢角色名）：`S3.levels.X` 缺，同英文在 `notes` 下是双语并列、
在 `levels` 下是裸中文，且 notes 票数更多。原版两者同桶，取到 notes 的多数派。
"""
import json, os, sys, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LAB = os.path.join(HERE, "_s4_lab")

EN = {"label": "T", "folders": {}, "entries": {
    "A": {"name": "Shield", "tokenName": "Shield"},
    "B": {"name": "Shield"},
    "S1": {"levels": {"Lake Jinro": "Lake Jinro"}},
    "S2": {"notes": {"Lake Jinro": "Lake Jinro"}},
    "S4": {"notes": {"Lake Jinro": "Lake Jinro"}},
    "S3": {"levels": {"Lake Jinro": "Lake Jinro"}},
}}
CN = {"label": "T", "folders": {}, "entries": {
    "A": {"name": "护盾术 Shield"},                      # tokenName 缺
    "B": {"name": "护盾术 Shield"},
    "S1": {"levels": {"Lake Jinro": "金罗湖"}},
    "S2": {"notes": {"Lake Jinro": "金罗湖 Lake Jinro"}},
    "S4": {"notes": {"Lake Jinro": "金罗湖 Lake Jinro"}},
    "S3": {},                                            # levels 缺
}}


def run(script, out):
    os.makedirs(out, exist_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", script, "--repo", LAB,
                        "--out-dir", out, "--report", out + ".json"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        print(r.stdout, r.stderr)
    got = {}
    for fn in sorted(os.listdir(out)):
        got[fn] = json.load(open(os.path.join(out, fn), encoding="utf-8"))
    rep = json.load(open(out + ".json", encoding="utf-8")) if os.path.exists(out + ".json") else {}
    return got, rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if os.path.isdir(LAB):
        shutil.rmtree(LAB)
    for side, doc in (("en", EN), ("cn", CN)):
        d = os.path.join(LAB, "compendium", side)
        os.makedirs(d)
        json.dump(doc, open(os.path.join(d, "lab.json"), "w", encoding="utf-8"), ensure_ascii=False)

    a, _ = run(os.path.join(ROOT, "3-常用脚本", "tm", "fill_missing.py"),
               os.path.join(LAB, "out_orig"))
    b, rep = run(os.path.join(HERE, "_s4_patched", "3-常用脚本", "tm", "fill_missing.py"),
                 os.path.join(LAB, "out_patched"))
    print("原版批次   :", json.dumps(a, ensure_ascii=False))
    print("补丁后批次 :", json.dumps(b, ensure_ascii=False))
    print("补丁后报告 roleblind_suggestions:",
          json.dumps(rep.get("roleblind_suggestions", []), ensure_ascii=False))

    orig = a.get("tm.lab.json", {})
    new = b.get("tm.lab.json", {})
    ok = True
    if orig.get("A.tokenName") != "护盾术 Shield":
        print("  ! 场景一未复现：原版没有把 name 的双语并列灌进 tokenName"); ok = False
    if "A.tokenName" in new:
        print("  ! 场景一未修好：补丁后仍然写了 tokenName"); ok = False
    if orig.get("S3.levels.Lake Jinro") != "金罗湖 Lake Jinro":
        print("  ! 场景二未复现，原版给的是", orig.get("S3.levels.Lake Jinro")); ok = False
    if new.get("S3.levels.Lake Jinro") != "金罗湖":
        print("  ! 场景二未修好，补丁后给的是", new.get("S3.levels.Lake Jinro")); ok = False
    print("\n回测", "通过" if ok else "失败")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
