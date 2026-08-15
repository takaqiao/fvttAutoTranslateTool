#!/usr/bin/env python3
"""A7 注入回测：往副本树里造一个**新的分叉组**，确认 `R-exclusions-closed` 会变红。

为什么要这一步：这条断言本轮刚从「读旧报告快照」改成「现场重跑扫描器」，
而 A7 又把它读的豁免表从 `4-临时脚本/…/EXCLUSIONS.json` 挪到了 `5-其他内容/`。
只证明它「还能跑绿」是不够的 —— 空转的断言也一直是绿的。必须证明**库里冒出新分叉时它会红**。

做法：在副本树（`--root` 指向的那棵）里挑一个「英文唯一串出现 ≥2 次、中文只有一种写法」的叶，
把其中一处中文前面加上一个显眼前缀，制造一个原表里不存在的分叉组。只改副本，不碰真库。
"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T = os.path.dirname(os.path.abspath(__file__))
REPOS = [os.path.join(T, "1-Ember汉化插件"), os.path.join(T, "2-Crucible汉化插件")]
MARK = "【A7回测注入】"


def leaves(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, (path + "." + str(k)) if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, path + "[" + str(i) + "]", out)
    elif isinstance(obj, str):
        out[path] = obj


def set_at(obj, path, val):
    """按 leaves() 的路径写回。分段规则与 leaves 对称：`.` 分层，`[i]` 进 list。"""
    cur = obj
    parts = path.split(".")
    for i, p in enumerate(parts):
        idxs = []
        while p.endswith("]"):
            p, _, tail = p.rpartition("[")
            idxs.append(int(tail[:-1]))
        idxs.reverse()
        last = (i == len(parts) - 1) and not idxs
        if last:
            cur[p] = val
            return
        cur = cur[p]
        for j, ix in enumerate(idxs):
            if i == len(parts) - 1 and j == len(idxs) - 1:
                cur[ix] = val
                return
            cur = cur[ix]


def main():
    by_en = collections.defaultdict(lambda: collections.defaultdict(list))
    for repo in REPOS:
        cdir, edir = os.path.join(repo, "compendium", "cn"), os.path.join(repo, "compendium", "en")
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".json"):
                continue
            ep = os.path.join(edir, fn)
            if not os.path.exists(ep):
                continue
            cn_l, en_l = {}, {}
            leaves(json.load(open(os.path.join(cdir, fn), encoding="utf-8-sig")), "", cn_l)
            leaves(json.load(open(ep, encoding="utf-8-sig")), "", en_l)
            for path, ev in en_l.items():
                cv = cn_l.get(path)
                if cv:
                    by_en[ev][cv].append([repo, fn, path])

    # 挑一个：英文短（判据用整串当 key，短串更好认）、中文只有一种写法、出现 ≥2 次。
    cand = None
    for en, variants in sorted(by_en.items()):
        if len(variants) != 1 or not (6 <= len(en) <= 30):
            continue
        (cn, occ), = variants.items()
        if len(occ) >= 2 and MARK not in cn:
            cand = (en, cn, occ)
            break
    if not cand:
        print("✗ 没找到可注入的候选叶")
        return 1
    en, cn, occ = cand
    repo, fn, path = occ[0]
    p = os.path.join(repo, "compendium", "cn", fn)
    doc = json.load(open(p, encoding="utf-8-sig"))
    set_at(doc, path, MARK + cn)
    json.dump(doc, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"注入完成：EN={en!r}  原中文={cn!r}（{len(occ)} 处）")
    print(f"  改的是 {os.path.relpath(p, T)} 的 {path} → {MARK + cn!r}")
    print("  期望：R-exclusions-closed 变红，报「该分叉组不在已归档豁免表里」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
