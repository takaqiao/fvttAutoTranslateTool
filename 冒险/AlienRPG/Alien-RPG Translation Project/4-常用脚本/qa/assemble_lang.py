#!/usr/bin/env python3
"""把 6-工作区/phase1/ 的分片装配成交付用的 1-系统汉化插件/lang/cn.json。

  python assemble_lang.py [--check] [--out <path>]

为什么单独写一个装配器，而不是用 qa/apply_lang.py
------------------------------------------------
`apply_lang.py:170` 有一条 `if key not in en: reject`。那对**上游已有的键**是正确的闸，
但本项目有 18 个键**是代码引用了、而 en.json 里根本不存在的**
（`alienrpgct.Passive` / `ALIENRPG.ActiveEffect.Temporary` / …），今天在任何语言下都渲染成裸键。
Foundry 是**按语言分别 merge** 的，en 只是 fallback，所以 cn.json 里写一个 en.json 没有的键
**是有效的、且是我们修那 18 处裸键的唯一办法**。apply_lang 会把它们整批拒掉。

⚠ 另一条比这更要命的：`apply_lang.py:206` 是**裸 set_path**，多个 batch 落进同一个 dict 时
**没有任何冲突检测**。第一轮四个分片实测 505 条进、487 条"applied"、实际只有 446 条不同——
**59 次静默覆盖，零日志**。本装配器因此把「同一个键出现在两个分片里」当**硬错误**而不是当合并。

不变式（任何一条不过就 NOTHING WRITTEN + exit 1）
-------------------------------------------------
 1 分片之间零重复键
 2 覆盖 en.json 的键数 + 有意不写的键数 == en.json 键总数
 3 有意不写的键必须在 _DELIBERATELY_UNWRITTEN 里带理由
 4 每个 code-only 键确实不在 en.json 里（否则它就该走正常分片）
 5 lang_keep_english.json 里的键，值必须与 en.json 逐字节相同
 6 占位符 {x} 多重集与英文 1:1
 7 HTML 标签多重集与英文 1:1
 8 首尾空白/U+00A0 与英文不同的键，**必须**在 WHITESPACE-EXCEPTIONS.json 里申报并写明拼接点
 9 反向：申报了但其实已经不再有差异的条目要清掉（防豁免表腐烂）
10 展开成嵌套之后再压平，键集必须与压平前完全一致（防「顶层点号键 + 嵌套值」互相顶掉）
11 没有任何值等于 DO-NOT-TRANSLATE.json 里的冻结字面量
"""
import argparse
import io
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(ROOT, "6-工作区", "phase1")
HUB = os.path.join(ROOT, "1-系统汉化插件")
SYSLANG = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang"

SLICES = ["lang-core.json", "lang-combat.json", "lang-ship.json", "lang-ui.json", "lang-gap.json"]
CODE_ONLY = "lang-ui.extra.json"

# 有意不写的 en.json 键 —— 每条必须带理由
DELIBERATELY_UNWRITTEN = {
    "ALIENRPG.AbbreviationKg": "计量单位符号 kg，中文照用。",
    "ALIENRPG.EVCriticalInjuries": (
        "上游是 JSON null，且它的值是 T-FROZEN 的 RollTable 名。"
        "localize() 对非字符串会落回英文 fallback，正好是我们要的——**写任何值都会变坏**。"),
    "ALIENRPG.NPC": "NPC 三字母在中文 TRPG 语境里通用，翻成「非玩家角色」反而更长更陌生。",
    "ALIENRPG.Roboto-Regular": "字体名，必须原样。",
    "ALIENRPG.SansitaSwashed": "字体名，必须原样。",
    "ALIENRPG.Wallpoet": "字体名，必须原样。",
}

PLACEHOLDER = re.compile(r"\{[^{}]*\}")
TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")


def jload(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        kk = "%s.%s" % (p, k) if p else k
        if isinstance(v, dict):
            out.update(flat(v, kk))
        else:
            out[kk] = v
    return out


def expand(flat_map):
    """把点号键展开成嵌套 dict。撞到「既是叶又是枝」的键直接报错。"""
    root = {}
    for key in sorted(flat_map):
        parts = key.split(".")
        node = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if last:
                if isinstance(node.get(part), dict):
                    raise SystemExit("SHAPE: %r is both a leaf and a branch" % key)
                node[part] = flat_map[key]
            else:
                nxt = node.get(part)
                if nxt is None:
                    nxt = node[part] = {}
                elif not isinstance(nxt, dict):
                    raise SystemExit("SHAPE: %r collides with leaf %r" % (key, ".".join(parts[:i + 1])))
                node = nxt
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑不变式，不写盘")
    ap.add_argument("--out", default=os.path.join(HUB, "lang", "cn.json"))
    args = ap.parse_args()

    en = flat(jload(os.path.join(SYSLANG, "en.json")))
    keep_path = os.path.join(HUB, "lang", "lang_keep_english.json")
    keep = jload(keep_path) if os.path.exists(keep_path) else {}
    keep_keys = set(keep.get("keep_english", keep) if isinstance(keep, dict) else [])
    if isinstance(keep.get("keep_english"), dict):
        keep_keys = set(keep["keep_english"].keys())

    ws_path = os.path.join(WORK, "WHITESPACE-EXCEPTIONS.json")
    WS_EXCEPT = set((jload(ws_path).get("exceptions") or {}).keys()) if os.path.exists(ws_path) else set()

    dnt_path = os.path.join(ROOT, "7-其他内容", "DO-NOT-TRANSLATE.json")
    frozen = set()
    if os.path.exists(dnt_path):
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("string_compared", "actual_name", "string") and isinstance(v, str):
                        frozen.add(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(jload(dnt_path))

    # --- gather -------------------------------------------------------------
    merged, owner, fail = {}, {}, []
    for name in SLICES:
        path = os.path.join(WORK, name)
        if not os.path.exists(path):
            fail.append("MISSING SLICE %s" % name)
            continue
        for k, v in jload(path).items():
            if k in owner:
                fail.append("DUP %s: %s and %s" % (k, owner[k], name))   # 不变式 1
                continue
            owner[k] = name
            merged[k] = v

    code_only = jload(os.path.join(WORK, CODE_ONLY))
    for k, v in code_only.items():
        if k in owner:
            fail.append("DUP-CODEONLY %s also in %s" % (k, owner[k]))
        if k in en:
            fail.append("CODEONLY-IN-EN %s is in en.json; it belongs in a normal slice" % k)  # 4
        owner[k] = CODE_ONLY
        merged[k] = v

    # --- invariants ---------------------------------------------------------
    covered = {k for k in merged if k in en}
    unwritten = set(en) - covered
    if unwritten != set(DELIBERATELY_UNWRITTEN):                                  # 2 + 3
        for k in sorted(unwritten - set(DELIBERATELY_UNWRITTEN)):
            fail.append("UNCOVERED %s = %r (no reason recorded)" % (k, en[k]))
        for k in sorted(set(DELIBERATELY_UNWRITTEN) - unwritten):
            fail.append("REASON-BUT-WRITTEN %s" % k)

    for k in sorted(keep_keys & set(merged)):                                     # 5
        if merged[k] != en.get(k):
            fail.append("KEEP-EN %s: %r != en %r" % (k, merged[k], en.get(k)))

    for k in sorted(covered):
        e, c = str(en[k]), str(merged[k])
        if Counter(PLACEHOLDER.findall(e)) != Counter(PLACEHOLDER.findall(c)):    # 6
            fail.append("PLACEHOLDER %s" % k)
        if Counter(m.lower() for m in TAG.findall(e)) != Counter(m.lower() for m in TAG.findall(c)):  # 7
            fail.append("TAG %s" % k)
        ws_differs = (len(e) - len(e.lstrip()), len(e) - len(e.rstrip())) != \
                     (len(c) - len(c.lstrip()), len(c) - len(c.rstrip()))
        nbsp_differs = e.count("\u00a0") != c.count("\u00a0")
        if (ws_differs or nbsp_differs) and k not in WS_EXCEPT:                   # 8 + 9
            fail.append("WHITESPACE/NBSP %s (undeclared): en=%r cn=%r" % (k, e, c))
        if not (ws_differs or nbsp_differs) and k in WS_EXCEPT:                   # 8b \u53cd\u5411
            fail.append("WS-EXCEPT-STALE %s declared but no longer differs" % k)

    for k, v in merged.items():                                                   # 11
        if isinstance(v, str) and v in frozen and k not in keep_keys:
            fail.append("FROZEN-COLLISION %s = %r" % (k, v))

    nested = expand(merged)
    if set(flat(nested)) != set(merged):                                          # 10
        fail.append("ROUNDTRIP: expand->flatten changed the key set")

    if fail:
        print("INVARIANTS FAILED (%d):" % len(fail))
        for f in fail[:60]:
            print("  " + f)
        if len(fail) > 60:
            print("  ... and %d more" % (len(fail) - 60))
        print("NOTHING WRITTEN.")
        return 1

    print("INVARIANTS: all 11 pass.")
    print("  en.json keys      : %d" % len(en))
    print("  covered           : %d" % len(covered))
    print("  deliberately not  : %d" % len(unwritten))
    print("  code-only extras  : %d" % len(code_only))
    print("  total written     : %d" % len(merged))
    if args.check:
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote %s  (%d bytes)" % (args.out, os.path.getsize(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
