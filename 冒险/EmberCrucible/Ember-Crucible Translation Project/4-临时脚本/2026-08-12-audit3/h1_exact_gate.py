#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 对 .mjs 的每个「英文键 -> 中文」跑一次英文闸，看 compendium 是否用同一个词。

输出三列：EN 命中行数 / 中文命中(gated) / 英文命中但中文用了别的词(en_only)。
en_only 大的就是要人看的。只读。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

P = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
GATE = P / "4-临时脚本" / "2026-08-12-fix" / "term_gate.py"

SKIP_TABLES = {"PATTERNS", "PREFIXED", "CALENDAR_DAY_ABBR"}


def main():
    keys_path = Path(sys.argv[1])
    repo = sys.argv[2] if len(sys.argv) > 2 else "1-Ember汉化插件"
    only = sys.argv[3] if len(sys.argv) > 3 else None
    raw = keys_path.read_text(encoding="utf-8")
    data = json.loads(raw[:raw.rindex("}") + 1])

    for table, d in data.items():
        if table in SKIP_TABLES or not isinstance(d, dict):
            continue
        if only and table != only:
            continue
        for en, cn in d.items():
            pat = r"\b" + re.escape(en).replace(r"\ ", " ") + r"\b"
            r = subprocess.run(
                [sys.executable, str(GATE), "--repo", repo, "--en", pat, "--cn", cn, "--show", "0"],
                capture_output=True, text=True, encoding="utf-8", cwd=str(P))
            out = r.stdout or ""
            rows = gated = enonly = "?"
            for line in out.splitlines():
                m = re.search(r"rows whose ENGLISH matches: (\d+)", line)
                if m:
                    rows = m.group(1)
                m = re.search(r"gated_hit=(\d+)", line)
                if m:
                    gated = m.group(1)
                m = re.search(r"EN matches but CN uses none of .*: (\d+)", line)
                if m:
                    enonly = m.group(1)
            flag = "  <<<" if (enonly not in ("0", "?") ) else ""
            print(f"{table}\t{en}\t{cn}\ten_rows={rows}\tgated={gated}\ten_only={enonly}{flag}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
