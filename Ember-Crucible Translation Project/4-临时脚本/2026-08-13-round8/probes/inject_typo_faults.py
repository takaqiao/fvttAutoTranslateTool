# -*- coding: utf-8 -*-
"""灵敏度回测：把 6 类已知排版错误注入一份**临时副本**，确认 scan_cn_typography 能报出来。

绝不碰真实 `compendium/`：源仓库只读复制到 --dest（默认在 scratchpad 下）。

  python inject_typo_faults.py --src "<项目根>\\2-Crucible汉化插件" --dest "<临时目录>"
  python scan_cn_typography.py --repo "<临时目录>"
"""
import argparse, json, os, re, shutil, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    assert "compendium" not in os.path.abspath(a.dest).replace("\\", "/").split("/")[-1]
    if os.path.exists(a.dest):
        shutil.rmtree(a.dest)
    shutil.copytree(os.path.join(a.src, "compendium"), os.path.join(a.dest, "compendium"))
    print("copied ->", a.dest)

    cn_dir = os.path.join(a.dest, "compendium", "cn")
    log = []

    def patch_any(kind, find_rx, repl, limit=1):
        """在任意一个 cn 包里找到第一处可注入的地方就注入。"""
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            if patch(fn, kind, find_rx, repl, limit):
                return True
        print(f"  {kind:4s} !! 全库找不到可注入的样本")
        return False

    def patch(fn, kind, find_rx, repl, limit=1):
        p = os.path.join(cn_dir, fn)
        d = json.load(open(p, encoding="utf-8-sig"))
        done = [0]

        def rec(o):
            if done[0] >= limit:
                return o
            if isinstance(o, dict):
                return {k: rec(v) for k, v in o.items()}
            if isinstance(o, list):
                return [rec(v) for v in o]
            if isinstance(o, str) and done[0] < limit:
                new, n = re.subn(find_rx, repl, o, count=limit - done[0])
                if n:
                    done[0] += n
                    log.append((kind, fn, o[:80], new[:80]))
                return new
            return o
        d = rec(d)
        if done[0]:
            json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  {kind:4s} {fn:34s} 注入 {done[0]}")
        return done[0]

    # T1 标记功能区混入全角：标签名与属性之间插一个全角空格 + 全角方括号 enricher
    patch_any("T1a", r"<span class=", "<span　class=", 1)
    patch_any("T1b", r"@UUID\[", "@UUID［", 1)
    # T2 中文标签后的半角冒号
    patch_any("T2", r"([一-鿿]{2,6})：", r"\1:", 1)
    # T3 中文正文里的半角双引号
    patch_any("T3", r"“([^”<>]{2,20})”", r'"\1"', 1)
    # T4 直角引号
    patch_any("T4", r"“([^”<>]{2,20})”", r"「\1」", 1)
    # T5 数字区间用破折号
    patch_any("T5", r"(?<![\d.])(\d)\s*[-–]\s*(\d)(?![\d.])", r"\1—\2", 1)
    # T6 中文侧丢掉一个收括号
    patch_any("T6", r"（([^）<>]{2,20})）", r"（\1", 1)

    print("\n注入明细：")
    for k, fn, o, n in log:
        print(f"  [{k}] {fn}\n      前 {o}\n      后 {n}")


if __name__ == "__main__":
    main()
