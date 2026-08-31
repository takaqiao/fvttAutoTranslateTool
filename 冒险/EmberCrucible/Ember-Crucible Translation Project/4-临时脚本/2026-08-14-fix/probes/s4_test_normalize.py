# -*- coding: utf-8 -*-
"""normalize_adventure_translation.py 补丁前后的空跑对照（全部在副本上跑）。

原版：无 --dry/--write，直接覆盖写入；`cn == en` 判定整条叶子删除。
补丁：默认空跑；`cn == en` 保留；pronunciation 不动；接英文前先折叠空白。
"""
import json, os, sys, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LAB = os.path.join(HERE, "_s4_lab_norm")
PACK = "ember.crucible-adventure.json"
REPO = "1-Ember汉化插件"


def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{p}.{i}")
    elif isinstance(o, str):
        yield p, o


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if os.path.isdir(LAB):
        shutil.rmtree(LAB)
    os.makedirs(LAB)
    en = os.path.join(ROOT, REPO, "compendium", "en", PACK)
    cn = os.path.join(ROOT, REPO, "compendium", "cn", PACK)
    base = dict(walk(json.load(open(cn, encoding="utf-8-sig"))))

    orig_cn = os.path.join(LAB, "orig.json")
    patched_cn = os.path.join(LAB, "patched.json")
    shutil.copy(cn, orig_cn)
    shutil.copy(cn, patched_cn)

    o = os.path.join(ROOT, REPO, "scripts", "normalize_adventure_translation.py")
    p = os.path.join(HERE, "_s4_patched", REPO, "scripts", "normalize_adventure_translation.py")

    r = subprocess.run([sys.executable, "-X", "utf8", o, "--cn", orig_cn, "--en", en],
                       capture_output=True, text=True, encoding="utf-8")
    print("原版  :", r.stdout.strip() or r.stderr.strip()[:200])
    r = subprocess.run([sys.executable, "-X", "utf8", p, "--cn", patched_cn, "--en", en],
                       capture_output=True, text=True, encoding="utf-8")
    print("补丁(默认):", r.stdout.strip() or r.stderr.strip()[:200])
    same = json.load(open(patched_cn, encoding="utf-8-sig")) == json.load(open(cn, encoding="utf-8-sig"))
    print("  补丁默认未落盘:", same)

    r = subprocess.run([sys.executable, "-X", "utf8", p, "--cn", patched_cn, "--en", en, "--write"],
                       capture_output=True, text=True, encoding="utf-8")
    print("补丁(--write):", r.stdout.strip() or r.stderr.strip()[:200])

    for label, path in (("原版", orig_cn), ("补丁", patched_cn)):
        got = dict(walk(json.load(open(path, encoding="utf-8-sig"))))
        deleted = [k for k in base if k not in got]
        changed = [k for k in base if k in got and got[k] != base[k]]
        print(f"  {label}: 叶 {len(base)} -> {len(got)}   删除 {len(deleted)}   改写 {len(changed)}")
        if label == "原版":
            from collections import Counter
            print("     删除按末段:", Counter(k.rsplit('.', 1)[-1] for k in deleted).most_common(6))
        pron = [k for k in changed if k.endswith("pronunciation")]
        print(f"     其中 pronunciation 被改 {len(pron)} 条", pron[:2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
