# -*- coding: utf-8 -*-
"""L6-① 的复现探针：把四道判定函数逐个掏空，各跑一遍 `--selftest`，报 rc 与合计行。

⚠ 前置自证（纪律 ⚠4，本项目在这上面栽过四次）：
  · 四个锚点行必须**各自恰好命中一次**，命中数 ≠ 1 就当场退出（切错了不许照跑）；
  · 掏空前先跑一遍原样，断言 rc=0 且合计行里的两个数相等 —— 「基准是绿的」也要自证。
⚠ 不用正则（纪律 ⚠3：`\\b` / `\\s` 不经改写脚本传），一律纯字符串比对。
⚠ 每跑完一条立刻从备份还原；脚本任何一步失败也走 finally 还原。
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.normpath(os.path.join(HERE, "..", "..", "3-常用脚本", "qa",
                                       "assert_resolutions.py"))
BACKUP = os.path.join(HERE, "assert_resolutions.py.orig")

# (标签, 锚点行（去掉行尾空白后逐字相等）, 掏空后插入的第一句)
CASES = [
    ("_payload_floor_check", "def _payload_floor_check(A):", "    return [], 0"),
    ("_units_verdict", "def _units_verdict(rid, n):", "    return None"),
    ("_kind_coverage_check",
     "def _kind_coverage_check(owner=None, todo=None, todo_max=None, calls=None, kinds=None):",
     '    return [], "", 0'),
    ("_selftest_shape_check", "def _selftest_shape_check(counts, shape=None):",
     "    return [], [], 0"),
]

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def run_selftest():
    p = subprocess.run([sys.executable, TARGET, "--selftest"],
                       capture_output=True, env=ENV,
                       cwd=os.path.normpath(os.path.join(HERE, "..", "..")))
    out = p.stdout.decode("utf-8", "replace")
    tail = [x for x in out.splitlines() if "回测合计" in x]
    # ⚠ 只数**判定标记**那一列（行首的 FAIL），别数正文里提到 FAIL 的用例说明 ——
    #   那会把「H1 → 真身必须打 `FAIL …`」这条**通过**的用例误记成失败。
    nfail = sum(1 for x in out.splitlines() if x.lstrip().startswith("FAIL"))
    return p.returncode, (tail[-1].strip() if tail else "(没有合计行)"), nfail


def main():
    src = open(TARGET, encoding="utf-8").read()
    lines = src.split("\n")

    # ── 前置自证 ①：每个锚点恰好命中一次
    for label, anchor, _hollow in CASES:
        hits = [i for i, ln in enumerate(lines) if ln.rstrip() == anchor]
        if len(hits) != 1:
            print(f"前置自证失败：锚点 `{anchor}` 命中 {len(hits)} 次（该是 1 次）—— 中止")
            return 3
        print(f"前置自证 ok：{label} 锚点唯一命中于第 {hits[0] + 1} 行")

    # ── 前置自证 ②：原样基准必须是绿的
    rc0, line0, nf0 = run_selftest()
    print(f"\n【基准·原样】rc={rc0} · {line0} · FAIL 行 {nf0}")
    if rc0 != 0 or nf0 != 0:
        print("前置自证失败：原样基准不是全绿 —— 掏空实验的结论不作数，中止")
        return 4

    shutil.copyfile(TARGET, BACKUP)
    results = []
    try:
        for label, anchor, hollow in CASES:
            cur = open(BACKUP, encoding="utf-8").read().split("\n")
            i = [k for k, ln in enumerate(cur) if ln.rstrip() == anchor][0]
            cur.insert(i + 1, hollow)
            with open(TARGET, "w", encoding="utf-8", newline="") as fh:
                fh.write("\n".join(cur))
            rc, line, nf = run_selftest()
            results.append((label, rc, line, nf))
            print(f"【掏空 {label}】rc={rc} · {line} · FAIL 行 {nf}")
            shutil.copyfile(BACKUP, TARGET)
    finally:
        shutil.copyfile(BACKUP, TARGET)

    print("\n===== 汇总（四道全部必须 rc=1）=====")
    allok = True
    for label, rc, line, nf in results:
        ok = rc == 1
        allok = allok and ok
        print(f"  {'ok  ' if ok else 'FAIL'} 掏空 {label:24s} rc={rc} · {line} · FAIL 行 {nf}")
    print("四道全部 rc=1" if allok else "⚠ 有掏空没被接住")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
