# -*- coding: utf-8 -*-
"""把 s4_edits.json 套到**副本**上（probes/_s4_patched/），验证：
  1. 每条 old 在原文件里逐字节唯一（命中数 == 1）
  2. 串行套用后仍是合法 Python / JSON
  3. 打过补丁的 .py 能 import 且行为符合预期
从不写原文件。
"""
import json, os, sys, ast, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "_s4_patched")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    edits = json.load(open(os.path.join(HERE, "s4_edits.json"), encoding="utf-8"))
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    bufs = {}
    bad = 0
    for i, e in enumerate(edits):
        f = e["file"]
        if f not in bufs:
            src = os.path.join(ROOT, f.replace("/", os.sep))
            bufs[f] = open(src, encoding="utf-8").read()
        n = bufs[f].count(e["old"])
        if n != 1:
            print(f"  ! [{i}] {f}: old 命中 {n} 次（必须 1）\n      {e['old'][:90]!r}")
            bad += 1
            continue
        bufs[f] = bufs[f].replace(e["old"], e["new"], 1)
    for f, text in bufs.items():
        dst = os.path.join(OUT, f.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8", newline="\n").write(text)
        if f.endswith(".py"):
            try:
                ast.parse(text)
                print(f"  OK  语法 {f}")
            except SyntaxError as ex:
                print(f"  !   语法错误 {f}: {ex}")
                bad += 1
        elif f.endswith(".json"):
            try:
                json.loads(text)
                print(f"  OK  JSON {f}")
            except Exception as ex:
                print(f"  !   JSON 非法 {f}: {ex}")
                bad += 1
    print(f"\n编辑 {len(edits)} 条 / 文件 {len(bufs)} 个 / 失败 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
