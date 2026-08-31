# -*- coding: utf-8 -*-
"""L6-③ 的复现探针：伪造 runner / 伪造戳记，现在拦不拦得住。

⚠ 前置自证（纪律 ⚠4）：
  · 先断言「指纹清单切出来的文件数 = 已知真值 4，且逐个 basename 对得上」——
    切错了（比如清单退化成 2 个）就当场退出，不许照跑；
  · 再断言「原样状态下主闸说的是**指纹一致、不重跑自检**」——基准是绿的也要自证。

分两段：
  A 段（真跑）：改动 `translate_cases_runner.mjs` → 主闸必须说「变过」并**点名它**、
    并就地补跑自检；换成敌意假 runner → 整条链必须以 rc≠0 收场。
  B 段（决策位）：四种戳记喂进 `_require_selftest_ran()`，只看它**接不接受**。
    ⚠ 这一段把 `subprocess.run` 换成桩（自检那一趟不真跑，只为省时间）——
      被测的是「戳记合不合格」这个判断，不是自检本身；自检本身 A 段真跑过了。
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
QA = os.path.join(ROOT, "3-常用脚本", "qa")
JUDGE = os.path.join(QA, "assert_resolutions.py")
TC_RUNNER = os.path.join(QA, "translate_cases_runner.mjs")
STAMP = os.path.join(tempfile.gettempdir(), "assert_resolutions.selftest.stamp")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

spec = importlib.util.spec_from_file_location("ar_probe", JUDGE)
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def gate():
    p = subprocess.run([sys.executable, JUDGE], capture_output=True, env=ENV, cwd=ROOT)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def say(ok, note, extra=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {note}")
    if extra:
        print(f"        {extra}")
    return ok


def main():
    allok = True

    # ── 前置自证 ①：指纹清单
    files = ar._judge_side_files()
    names = sorted(os.path.basename(p) for p in files)
    want = ["RESOLUTIONS.assertions.json", "assert_resolutions.py",
            "selfcheck_panel_runner.mjs", "translate_cases_runner.mjs"]
    if names != want:
        print(f"前置自证失败：指纹清单 {names}，已知真值 {want} —— 中止")
        return 3
    print(f"前置自证 ok：指纹盖住 {len(files)} 个判据侧文件 {names}")

    # ── 前置自证 ②：原样状态下主闸不重跑自检
    rc0, out0 = gate()
    base_ok = rc0 == 0 and "本次不重跑自检" in out0
    if not base_ok:
        print(f"前置自证失败：原样基准 rc={rc0}，"
              f"输出里没有「本次不重跑自检」—— 中止\n{out0[-600:]}")
        return 4
    print("前置自证 ok：原样基准 rc=0，主闸说「指纹一致，本次不重跑自检」")

    src = open(TC_RUNNER, encoding="utf-8").read()
    stamp_backup = open(STAMP, encoding="utf-8").read() if os.path.exists(STAMP) else None

    print("\n── A 段（真跑）：伪造 runner ──")
    try:
        # A1 只加一行注释（语义无害）→ 指纹必须变、必须点名它、必须补跑自检
        with open(TC_RUNNER, "w", encoding="utf-8", newline="") as fh:
            fh.write(src + "\n// [round32 probe] 语义无害的一行注释\n")
        rc, out = gate()
        allok &= say("translate_cases_runner.mjs" in out and "补跑一遍自检" in out and rc == 0,
                     "A1 改动 runner（加一行注释）→ 主闸必须说「变过」、点名 runner、"
                     "就地补跑自检，且自检过 ⇒ rc=0",
                     next((x for x in out.splitlines() if "变的是" in x), out[-400:]))

        # A2 敌意假 runner：不 import 任何被判文件，直接宣称「违规 0 处」
        with open(TC_RUNNER, "w", encoding="utf-8", newline="") as fh:
            fh.write('import fs from "node:fs";\n'
                     'const s = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));\n'
                     'fs.writeFileSync(s.out, JSON.stringify('
                     '{ violations: [], counts: {} }), "utf8");\n')
        rc, out = gate()
        allok &= say(rc != 0, f"A2 换成 3 行的敌意假 runner（完全不调 translateText，"
                              f"直接宣称违规 0 处）→ 整条链必须以 rc≠0 收场（实得 rc={rc}）",
                     next((x for x in out.splitlines() if "FAIL" in x), out[-300:]))
    finally:
        with open(TC_RUNNER, "w", encoding="utf-8", newline="") as fh:
            fh.write(src)

    # 还原后把戳记重新对齐（A 段跑过自检，戳记已经是新的；这里保证 B 段基准干净）
    rc, out = gate()
    if rc != 0:
        print(f"⚠ 还原后主闸不是绿的（rc={rc}）—— 后面的结论不作数")
        return 5

    print("\n── B 段（决策位）：伪造戳记 ──")
    real_fp, real_per = ar._judge_fingerprint()

    class _Proc:
        returncode = 0
        stdout = "══════ 判据自身回测合计：桩 ══════\n"
        stderr = ""

    class _Shim:
        @staticmethod
        def run(*_a, **_k):
            return _Proc()

    real_sub = ar.subprocess
    try:
        for label, payload, want_accept in [
            ("F1 塞一行垃圾（`绿`）", "绿", False),
            ("F2 旧格式那一行 hex（本轮之前戳记就长这样）", real_fp, False),
            ("F3 形状对、指纹对，但**某个文件的哈希是编的**",
             json.dumps({"fingerprint": real_fp,
                         "files": dict(real_per, **{k: hashlib.sha256(b"x").hexdigest()
                                                    for k in list(real_per)[:1]}),
                         "summary": "编的", "when": "编的"}, ensure_ascii=False), False),
            ("F4 **照实算出来的一张真表**（= 直接调 `_stamp_selftest_green()` 的效果）",
             json.dumps({"fingerprint": real_fp, "files": real_per,
                         "summary": "判据自身回测合计：322 / 322 通过",
                         "when": "2026-08-16 00:00:00"}, ensure_ascii=False), True),
        ]:
            with open(STAMP, "w", encoding="utf-8") as fh:
                fh.write(payload)
            ar.subprocess = _Shim
            failed, notes = ar._require_selftest_ran()
            ar.subprocess = real_sub
            accepted = any("本次不重跑自检" in x for x in notes)
            ok = accepted == want_accept
            allok &= say(ok, f"{label} → {'必须被接受' if want_accept else '必须失效（补跑自检）'}"
                             f"（实得：{'接受' if accepted else '补跑'}）",
                         "" if ok else f"notes={notes[:2]}")
    finally:
        ar.subprocess = real_sub
        if stamp_backup is not None:
            with open(STAMP, "w", encoding="utf-8") as fh:
                fh.write(stamp_backup)
        elif os.path.exists(STAMP):
            os.remove(STAMP)

    print("\n⚠ 照实记边界：F4 说明「**完全正确地伪造**一张表仍然会被接受」——"
          "戳记住在 %TEMP%，本来就不可能防伪造。本轮把伪造从「改 1 个字节、零痕迹」"
          "抬到「要伪造一张与 4 个文件逐一对得上的表」，并把上次自检的合计行印在主闸输出上，"
          "让痕迹落在人看得见的地方。它防的是「忘了跑」，不是「故意骗」。")
    print("\n===== 汇总 =====")
    print("全部通过" if allok else "⚠ 有条目没被接住")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
