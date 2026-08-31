"""
清理 translated/ 中「汉字——汉字」滥用的中文破折号。

规则（保守）：仅当 `——` 前后都是汉字时删除。
保留：
  - 句首或句尾 `——`（前/后为标点、换行、空格）
  - `字——，` 或 `，——字` 这种含标点的合法破折号引出语
  - 半角范围 `10-20`（用 `-` 而非 `——`，本不冲突）

用法：
  python clean_dashes.py            # dry-run，仅打印统计
  python clean_dashes.py --apply    # 写回源文件
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
TRANSLATED = ROOT / "translated"

# CJK Unified Ideographs + Extension A
CJK_CHAR = r"[一-鿿㐀-䶿]"
PATTERN = re.compile(rf"(?<={CJK_CHAR})——(?={CJK_CHAR})")


def clean_text(text: str) -> tuple[str, int]:
    """Return (new_text, removed_count)."""
    matches = PATTERN.findall(text)
    if not matches:
        return text, 0
    return PATTERN.sub("", text), len(matches)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写回；默认 dry-run")
    ap.add_argument("--show-samples", type=int, default=3, help="每页显示前 N 行样例")
    args = ap.parse_args()

    files = sorted(TRANSLATED.glob("page_*.md"))
    total_removed = 0
    pages_changed = 0
    per_page = []

    for pf in files:
        text = pf.read_text(encoding="utf-8")
        new_text, removed = clean_text(text)
        if removed > 0:
            pages_changed += 1
            total_removed += removed
            per_page.append((pf.name, removed))
            if args.apply:
                pf.write_text(new_text, encoding="utf-8")

    per_page.sort(key=lambda x: -x[1])

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"=== {mode} ===")
    print(f"总文件:       {len(files)}")
    print(f"受影响页:     {pages_changed}")
    print(f"删除 `——`:    {total_removed}")
    print()
    print(f"--- TOP 10 受影响页 ---")
    for name, n in per_page[:10]:
        print(f"  {name}: {n}")

    if not args.apply:
        print()
        print("（dry-run；加 --apply 实际写回）")


if __name__ == "__main__":
    main()
