#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compendium 里英文 name 恰为 crucible 系统状态名的条目，中文必须与 lang 的状态译名逐字相同。

为什么需要这一档：玩家在 token 上看到的状态标签由 `lang/cn.json` 的
`ACTIVE_EFFECT.STATUSES.*` 渲染，而 compendium 里的 ActiveEffect / 表结果行
是另一份独立的中文。两者不同 = 同一个状态在同屏出现两个名字。

既有判据全盲：
  - scan_cross_channel 的 B 段只比 `.mjs` ↔ lang 的键，不看 compendium
  - scan_name_splits 只在「同一英文 name 有两套中文」时才响；若某状态在
    compendium 里**始终**用同一个错译（本例 Confused 全是「神志混乱」），
    它内部自洽，name_splits 一声不响
  - 标记 / 数字 / 覆盖率 / class 全都与之无关

判据取严格版：中文剥掉双语并列的英文尾巴后，必须与 lang 值**逐字相同**。
不能用子串匹配 —— 「混乱」是「神志混乱」的子串，宽松比较会漏掉本例。

用法：
  python scan_status_name.py --repo 1-Ember汉化插件 [--repo 2-Crucible汉化插件] \
         --lang-en <crucible包>/lang/en.json --lang-cn 2-Crucible汉化插件/lang/cn.json \
         [--out <json>]
"""
import argparse
import json
import os
import sys


def leaves(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, (path + "." + str(k)) if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, path + "[" + str(i) + "]", out)
    elif isinstance(obj, str):
        out[path] = obj


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = prefix + "." + k if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        else:
            out[kk] = v
    return out


def load_statuses(lang_en, lang_cn):
    fe = flatten(json.load(open(lang_en, encoding="utf-8")))
    fc = flatten(json.load(open(lang_cn, encoding="utf-8")))
    stat = {}
    for k, v in fe.items():
        if not k.startswith("ACTIVE_EFFECT.STATUSES."):
            continue
        # 跳过整句提示文案（Flanked 那条是一整句话，不是状态名）
        if not isinstance(v, str) or len(v) > 30:
            continue
        cn = fc.get(k)
        if cn:
            stat[v] = cn
    return stat


def strip_bilingual_tail(cn, en):
    """双语并列格式是 '中文 English'，剥掉尾巴只留中文。"""
    if en and cn.endswith(" " + en):
        return cn[: -(len(en) + 1)]
    return cn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True)
    ap.add_argument("--lang-en", required=True)
    ap.add_argument("--lang-cn", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    stat = load_statuses(args.lang_en, args.lang_cn)
    consistent, findings = 0, []

    for repo in args.repo:
        cdir = os.path.join(repo, "compendium", "cn")
        edir = os.path.join(repo, "compendium", "en")
        if not os.path.isdir(cdir):
            print("跳过（无 compendium/cn）:", repo)
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".json"):
                continue
            ep = os.path.join(edir, fn)
            if not os.path.exists(ep):
                continue
            cn_leaves, en_leaves = {}, {}
            leaves(json.load(open(os.path.join(cdir, fn), encoding="utf-8")), "", cn_leaves)
            leaves(json.load(open(ep, encoding="utf-8")), "", en_leaves)
            for path, en_val in en_leaves.items():
                if not path.endswith(".name"):
                    continue
                if en_val not in stat:
                    continue
                cn_val = cn_leaves.get(path)
                if not cn_val:
                    continue
                want = stat[en_val]
                got = strip_bilingual_tail(cn_val, en_val)
                if got == want:
                    consistent += 1
                else:
                    findings.append({
                        "repo": repo,
                        "pack": fn,
                        "path": path,
                        "batch_path": path[len("entries."):] if path.startswith("entries.") else path,
                        "en": en_val,
                        "cn": cn_val,
                        "lang_expects": want,
                    })

    print("英文 name 恰为 crucible 状态名的叶子")
    print("  一致       %d" % consistent)
    print("  **不一致**  %d" % len(findings))
    groups = {}
    for f in findings:
        groups.setdefault((f["en"], f["cn"], f["lang_expects"]), []).append(f)
    for (en_val, cn_val, want), rows in sorted(groups.items(), key=lambda t: -len(t[1])):
        print("\n  %dx  EN=%s   CN=%s   lang 应为=%s" % (len(rows), en_val, cn_val, want))
        for r in rows:
            print("        %-28s %s" % (r["pack"][:28], r["path"][:80]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"statuses_known": len(stat), "consistent": consistent,
                       "findings": findings}, fh, ensure_ascii=False, indent=1)
        print("\n-> " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
