# -*- coding: utf-8 -*-
"""英文文档名 -> 中文文档名 的全库对照表（走 compendium/en 与 compendium/cn 的
同构 walk，凡是英文侧有 name 字符串、中文侧也有的，就记一条）。

用途：@UUID 链接的目标 id 可以经 LevelDB 表解析成英文文档名，再经本表拿到
中文文档名 —— 这是「某个中文标签到底指哪个文档」的**直接证据**，
不受「多数写法」被同一处错误批量污染的影响。
"""
import json, os, re, sys
from collections import defaultdict

CJK = re.compile(r"[一-鿿]")


def walk_names(en, cn, out):
    if isinstance(en, dict):
        if isinstance(en.get("name"), str) and isinstance(cn, dict) \
                and isinstance(cn.get("name"), str):
            out[en["name"]][cn["name"]] += 1
        for k, v in en.items():
            walk_names(v, cn.get(k) if isinstance(cn, dict) else None, out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk_names(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, out)


def build(repos):
    from collections import Counter
    out = defaultdict(Counter)
    for repo in repos:
        end = os.path.join(repo, "compendium", "en")
        for fn in sorted(os.listdir(end)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(end, fn), encoding="utf-8"))
            cp = os.path.join(repo, "compendium", "cn", fn)
            if not os.path.isfile(cp):
                continue
            cn = json.load(open(cp, encoding="utf-8"))
            walk_names(en.get("entries", {}), cn.get("entries", {}), out)
    # 保留一个英文名见过的**全部**中文写法，不取多数：
    # 库里有 3 个文档都叫 Ordain（城市=奥尔丹、词条=授任），取多数会丢掉一支。
    return {k: [x for x, _ in v.most_common()] for k, v in out.items() if v}


def cn_core(name):
    """从「中文 English」双语并列里取中文部分（没有中文就返回原串）。"""
    if not name:
        return None
    m = list(CJK.finditer(name))
    if not m:
        return name.strip()
    # 中文部分总在前面；截到最后一个中文字符（含中间的标点）
    end = m[-1].end()
    return name[:end].strip()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    m = build(sys.argv[1:-1])
    json.dump(m, open(sys.argv[-1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"names={len(m)} -> {sys.argv[-1]}")
