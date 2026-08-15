#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b：产出批次文件（只写 scratchpad，不碰 compendium）。

两组编辑：
  R  `<span class="reference">X</span>` 里留着英文、而 .mjs 已把该按钮翻成中文
     → 把 span 内容换成 .mjs 的中文（否则 GM 指南指的按钮在界面上找不到）
  S  「严重成功」是把 Critical Failure 的「严重失败」机械套到 Critical Success 上
     → 「大成功」（crucible lang ACTION.EFFECT_RESULT_TYPES.CriticalSuccess）
        带英文闸：该叶英文必须出现 critical success（大小写不敏感）

用法： python h1b_build_batches.py <输出目录>
"""
import json
import os
import re
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = {"1-Ember汉化插件": "ember", "2-Crucible汉化插件": "crucible"}
SKIP_KEYS = {"_id", "path", "_variants", "_when"}

REF_MAP = {
    "Begin Event": "开始事件",
    "Complete Event": "完成事件",
    "Award Attunements": "授予同调",
    "No Awarded Attunements": "无可授予的同调",
}
CRIT_RX = re.compile(r"critical\s+success", re.I)


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append((p[len("entries."):] if p.startswith("entries.") else p, en,
                    cn if isinstance(cn, str) else None))


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    report = []
    for repo, tag in REPOS.items():
        endir = os.path.join(P, repo, "compendium", "en")
        cndir = os.path.join(P, repo, "compendium", "cn")
        for fn in sorted(os.listdir(endir)):
            if not fn.endswith(".json"):
                continue
            en = json.load(open(os.path.join(endir, fn), encoding="utf-8"))
            cp = os.path.join(cndir, fn)
            if not os.path.isfile(cp):
                continue
            cn = json.load(open(cp, encoding="utf-8"))
            rows = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
            batch = {}
            for bp, e, c in rows:
                if not c:
                    continue
                new = c
                why = []
                for k, v in REF_MAP.items():
                    tok = f'<span class="reference">{k}</span>'
                    if tok in new:
                        n = new.count(tok)
                        new = new.replace(tok, f'<span class="reference">{v}</span>')
                        why.append(f"R:{k}x{n}")
                if "严重成功" in new and CRIT_RX.search(e):
                    n = new.count("严重成功")
                    new = new.replace("严重成功", "大成功")
                    why.append(f"S:严重成功x{n}")
                if new != c:
                    batch[bp] = new
                    report.append((tag, fn, bp, ";".join(why)))
            if batch:
                out = os.path.join(outdir, f"H1__{tag}__{fn[:-5]}.json")
                json.dump(batch, open(out, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                print(f"写出 {out}  ({len(batch)} 条)")
    print("\n# 明细")
    for r in report:
        print("\t".join(r)[:200])
    print(f"# 共 {len(report)} 条叶子")


if __name__ == "__main__":
    main()
