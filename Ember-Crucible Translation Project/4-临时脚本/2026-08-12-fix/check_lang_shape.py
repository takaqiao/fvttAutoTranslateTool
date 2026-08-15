# -*- coding: utf-8 -*-
"""lang/cn.json 的两项机械体检（对**全表**做，不只是批次）：

  1. 占位符 {xxx} 与英文侧逐键比对 —— 集合 + **计数**（多重集）都要一致
  2. 键形态 —— 必须全是扁平点号键；任何「顶层键带点、值却是嵌套对象」都致命
     （Foundry getProperty 先试整键、再按点下探，混合形态两条路都断）
  另附：把批次套用到内存副本后再跑一遍，确认批次不引入新问题、也不重造嵌套坑。
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\-]*\}")
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def flatten(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{p}.{k}" if p else k))
    elif isinstance(o, str):
        out[p] = o
    return out


def foundry_lookup(root, key):
    """复刻 Foundry getProperty：先试整键，再按点逐级下探。"""
    if key in root:
        return root[key]
    node = root
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


def audit(label, cn_doc, en_flat):
    print(f"===== {label}")
    nested = [k for k, v in cn_doc.items() if isinstance(v, dict)]
    print(f"  顶层键 {len(cn_doc)}；混合形态（顶层键的值是对象）: {len(nested)} {nested[:5]}")
    flat = flatten(cn_doc)
    unreachable = [k for k in flat if foundry_lookup(cn_doc, k) != flat[k]]
    print(f"  扁平化后 {len(flat)} 键；Foundry 查不到（UNREACHABLE）: {len(unreachable)} {unreachable[:5]}")

    ph_bad = tag_bad = missing_en = 0
    for k, v in flat.items():
        e = en_flat.get(k)
        if e is None:
            missing_en += 1
            continue
        ce, cc = Counter(PLACEHOLDER.findall(e)), Counter(PLACEHOLDER.findall(v))
        if ce != cc:
            ph_bad += 1
            print(f"  [占位符] {k}\n      EN {sorted(ce.elements())}\n      CN {sorted(cc.elements())}")
        te = sorted(t.lower() for t in HTML_TAG.findall(e))
        tc = sorted(t.lower() for t in HTML_TAG.findall(v))
        if te != tc:
            tag_bad += 1
            print(f"  [HTML] {k}\n      EN {te}\n      CN {tc}")
    print(f"  占位符不符 {ph_bad} / HTML 标签不符 {tag_bad} / 英文侧没有此键 {missing_en}")
    return ph_bad + tag_bad + len(nested) + len(unreachable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--batch")
    a = ap.parse_args()
    cn = json.loads((Path(a.repo) / "lang" / "cn.json").read_text(encoding="utf-8-sig"))
    en = flatten(json.loads((Path(a.package) / "lang" / "en.json").read_text(encoding="utf-8-sig")))
    bad = audit("现状", cn, en)
    if a.batch:
        batch = json.loads(Path(a.batch).read_text(encoding="utf-8-sig"))
        after = dict(cn)
        for k, v in batch.items():
            after[k] = v          # 与 apply_lang.set_path 同语义：扁平键直接赋值
        bad += audit(f"套用批次后（{len(batch)} 条）", after, en)
    print("\nTOTAL problems:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
