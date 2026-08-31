# -*- coding: utf-8 -*-
"""把当前 `compendium/en` 截成一份**全包**英文基准，供下一次上游升级时做 diff。

    python capture_baseline.py --project "<项目根>"            # 三个仓一次全截（推荐）
    python capture_baseline.py --repo <汉化插件目录> [--repo ...] [--dest-root <目录>]
    python capture_baseline.py --repo <汉化插件目录> --dest <english-baseline/xxx>

落盘位置默认 `7-其他内容/english-baseline/<packageId>-<packageVersion>/`
（版本号写死在目录名里，否则下一轮不知道这份基准是拿谁截的；packageId/packageVersion
从各仓 `compendium/en/_source.json` 读）。

为什么要截这一份
----------------
三条 drift 闸（dropped_terms / number_drift / marker_followup）都是
「旧英文 vs 当前英文 vs 中文」的三方比对，**旧英文必须是历史快照**。
升级前不截，升级后就永远补不回来（除非插件仓的 git 历史里正好有那个 tag ——
去找一眼总是值得的，每个 tag 的 `compendium/en/` 都是一份现成基准）。

⚠⚠ 覆盖率闸：本工具**从 EC 项目继承的头号教训**
--------------------------------------------------
EC 那边最贵的一个 bug 就出在这里：一份基准只装了 10 个包里的 3 个，
三条 drift 闸于是**在 30% 的语料上跑了好几个月**，报告每次都干干净净。
缺的那 7 个包没有任何一处会喊 —— 闸子只比对「基准里有的键」，
基准里没有的包，对闸子来说就等于「没有漂移」。

所以本工具**必须**把覆盖率印出来，并且是**三方**核对，不是只数目录里的文件：
    ① 上游 manifest（system.json / module.json）声明了几个包
    ② `compendium/en/_source.json` 记录抽出了几个包（含 skippedPacks）
    ③ `compendium/en/` 目录里实际有几个包文件
三者不一致就**非零退出**。只数 ③ 是不够的 —— EC 当年 ③ 自洽得很好，
错在没人拿 ① 去比。

Alien 侧的形状：3 个仓 × 每仓 1 个包 = 3/3。
仓与上游包的对应（`--project` 模式下写死在 REPOS 里）：
    1-系统汉化插件    alienrpg                  (system)
    2-新手包汉化插件  alien-evolved-starterset  (module)
    3-核心书汉化插件  alien-evolved-corerules   (module)
"""
from __future__ import annotations
import argparse
import datetime
import io
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_PROJECT = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"

# repo dir -> upstream package dir (for the manifest cross-check).
REPOS = [
    ("1-系统汉化插件",
     r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\alienrpg"),
    ("2-新手包汉化插件",
     r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\alien-evolved-starterset"),
    ("3-核心书汉化插件",
     r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\alien-evolved-corerules"),
]

CAPTURED_BY = "4-常用脚本/qa/capture_baseline.py"


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


def manifest_packs(package_dir):
    """Declared packs from the upstream system.json / module.json, or None."""
    if not package_dir:
        return None
    for f in ("system.json", "module.json"):
        p = os.path.join(package_dir, f)
        if os.path.exists(p):
            m = load(p)
            return [pk.get("name") for pk in (m.get("packs") or [])]
    return None


def capture_one(repo, dest, package_dir, note):
    """Snapshot one repo. Returns a dict of results (never raises on shortfall)."""
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    src_meta = load(os.path.join(en_dir, "_source.json"))
    os.makedirs(dest, exist_ok=True)

    packs = []
    for f in sorted(os.listdir(en_dir)):
        if not f.endswith(".json") or f == "_source.json":
            continue
        shutil.copyfile(os.path.join(en_dir, f), os.path.join(dest, f))
        en_l, cn_l = {}, {}
        leaves(load(os.path.join(en_dir, f)).get("entries", {}), [], en_l)
        cnp = os.path.join(cn_dir, f)
        if os.path.exists(cnp):
            leaves(load(cnp).get("entries", {}), [], cn_l)
        packs.append({"file": f, "en_leaves": len(en_l), "cn_leaves": len(cn_l),
                      "cn_present": os.path.exists(cnp)})
        print(f"    {f}  en {len(en_l)}  cn {len(cn_l)}")

    declared = manifest_packs(package_dir)
    on_disk = len(packs)
    from_source = src_meta.get("extractedPacks", on_disk)
    skipped = src_meta.get("skippedPacks", []) or []
    n_declared = len(declared) if declared is not None else src_meta.get("declaredPacks", on_disk)

    complete = (on_disk == n_declared == from_source) and not skipped

    meta = {
        "kind": "pre-upgrade-snapshot",
        "capturedAt": datetime.datetime.now().isoformat(timespec="seconds"),
        "capturedBy": CAPTURED_BY,
        "capturedFrom": os.path.abspath(en_dir),
        "upstream": {"packageId": src_meta.get("packageId"),
                     "packageVersion": src_meta.get("packageVersion"),
                     "packageType": src_meta.get("packageType"),
                     "packageDir": package_dir},
        "includesLocalPatches": True,
        "note": note,
        "coverage": {
            "packsInManifest": n_declared,
            "manifestPackNames": declared,
            "packsInSourceMeta": from_source,
            "packsOnDisk": on_disk,
            "skippedPacks": skipped,
            "complete": complete,
        },
        "packs": packs,
        "totals": {"packs": on_disk,
                   "en_leaves": sum(p["en_leaves"] for p in packs),
                   "cn_leaves": sum(p["cn_leaves"] for p in packs)},
        "upstreamSourceMeta": src_meta,
    }
    with io.open(os.path.join(dest, "_source.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    return {"repo": repo, "dest": dest, "meta": meta,
            "on_disk": on_disk, "declared": n_declared,
            "from_source": from_source, "complete": complete,
            "en_leaves": meta["totals"]["en_leaves"],
            "cn_leaves": meta["totals"]["cn_leaves"],
            "pkg": src_meta.get("packageId"), "ver": src_meta.get("packageVersion")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", nargs="?", const=DEFAULT_PROJECT,
                    help="截项目内**全部**仓（缺省 %(default)s）。给了 --repo 就以 --repo 为准。")
    ap.add_argument("--repo", action="append", default=[],
                    help="单个汉化插件目录，可重复。")
    ap.add_argument("--package-dir", action="append", default=[],
                    help="与 --repo 一一对应的上游包目录（用于 manifest 三方核对）。")
    ap.add_argument("--dest", help="单仓模式的落盘目录（只在恰好一个 --repo 时有效）。")
    ap.add_argument("--dest-root", help="多仓模式的落盘根目录"
                                        "（缺省 <project>/7-其他内容/english-baseline）。")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    if a.repo:
        pkgs = a.package_dir + [None] * (len(a.repo) - len(a.package_dir))
        jobs = list(zip([os.path.abspath(r) for r in a.repo], pkgs))
        project = a.project or DEFAULT_PROJECT
    else:
        project = a.project or DEFAULT_PROJECT
        jobs = [(os.path.join(project, r), p) for r, p in REPOS]

    dest_root = a.dest_root or os.path.join(project, "7-其他内容", "english-baseline")

    results = []
    for repo, package_dir in jobs:
        en_dir = os.path.join(repo, "compendium", "en")
        if not os.path.exists(os.path.join(en_dir, "_source.json")):
            print(f"!! {repo}: 没有 compendium/en/_source.json —— 先跑 extract_en.mjs")
            results.append({"repo": repo, "dest": None, "on_disk": 0, "declared": None,
                            "from_source": 0, "complete": False, "en_leaves": 0,
                            "cn_leaves": 0, "pkg": None, "ver": None})
            continue
        src_meta = load(os.path.join(en_dir, "_source.json"))
        if a.dest and len(jobs) == 1:
            dest = os.path.abspath(a.dest)
        else:
            dest = os.path.join(dest_root,
                                f"{src_meta.get('packageId')}-{src_meta.get('packageVersion')}")
        print(f"=== {os.path.basename(repo)}  ->  {dest}")
        results.append(capture_one(repo, dest, package_dir, a.note))
        print()

    # ---------------------------------------------------------------- coverage
    total_disk = sum(r["on_disk"] for r in results)
    total_decl = sum(r["declared"] or 0 for r in results)
    ok_repos = sum(1 for r in results if r["complete"])
    print("=== coverage")
    for r in results:
        d = r["declared"] if r["declared"] is not None else "?"
        print(f"    {os.path.basename(r['repo']):<18} packs {r['on_disk']}/{d}"
              f"   {'OK' if r['complete'] else 'INCOMPLETE <<<'}"
              f"   ({r['pkg']} v{r['ver']}, en 叶 {r['en_leaves']}, cn 叶 {r['cn_leaves']})")
    print(f"    COVERAGE: packs captured {total_disk}/{total_decl} declared"
          f"  |  repos complete {ok_repos}/{len(results)}")
    print(f"    en 叶合计 {sum(r['en_leaves'] for r in results)}"
          f"   cn 叶合计 {sum(r['cn_leaves'] for r in results)}")
    if total_disk != total_decl or ok_repos != len(results):
        print("!! 覆盖率不满 —— 这正是 EC 那个「3/10 包跑了几个月还报干净」的失效模式，"
              "不要在此状态下把这份基准接给 drift 闸。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
