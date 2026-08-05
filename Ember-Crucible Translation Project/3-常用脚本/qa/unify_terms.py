# -*- coding: utf-8 -*-
"""按「英文对得上」筛选，把全库同名异译统一到既定译名。

为什么不能全局替换：`闪电` 有时确实译自 lightning，`阶级` 也可能真的是社会阶层。
所以每条规则都必须同时满足：
  1. 该条**英文原文**里出现了 `en` 正则；
  2. 该条**译文**里出现了某个 `variants` 里的旧写法；
  3. 该条英文原文**不**匹配 `unless`（可选，用来排开撞名的另一个概念）。

用法：
  # 出评审清单（不写任何东西）
  python unify_terms.py --repo <repo> --rules <rules.json> --review <out.md>
  # 出批次文件，交给 apply_translations.py --force / apply_lang.py 落盘
  python unify_terms.py --repo <repo> --rules <rules.json> --batch-dir <dir> [--package <foundry包>]

rules.json 格式：
  [{"term": "Electricity", "en": "\\bElectricity\\b", "canonical": "电击",
    "variants": ["电力", "电能", "闪电"], "unless": "\\bLightning\\b", "note": "…"}]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')
TAGNAME = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z0-9_.\-]*\}')


def markup_signature(s):
    """术语统一只应该动中文词，标记必须与**改之前**逐一相同。

    注意基线是旧译文而不是英文：库里本来就有一批多包一层 <strong> 的旧译文，
    拿英文当基线会把它们全拦下来，而那和本次替换无关。
    """
    return (Counter(MARKUP.findall(s)) + Counter(INLINE_CMD.findall(s))
            + Counter(PLACEHOLDER.findall(s))
            + Counter(f'<{slash}{name.lower()}' for slash, name in TAGNAME.findall(s)))


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out


def context(s, needle, span=18):
    i = s.find(needle)
    return s[max(0, i - span):i + len(needle) + span].replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--package", help="给出则一并处理 lang/cn.json（英文取自包内 lang/en.json）")
    ap.add_argument("--review", help="输出人工评审清单")
    ap.add_argument("--batch-dir", help="输出可回写的批次文件")
    ap.add_argument("--write", action="store_true",
                    help="直接落盘。闸门：新译文的标记必须与旧译文逐一相同")
    args = ap.parse_args()

    repo = Path(args.repo)
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8-sig"))
    for r in rules:
        r["_en"] = re.compile(r["en"])
        r["_unless"] = re.compile(r["unless"]) if r.get("unless") else None
        # variants 是字面量；variants_re 是正则，用来把替换限制在「伤害类型」这类搭配里
        # （否则 `原始电能的混沌之力` 会被改成 `原始电击的…`，`针对其心灵的攻击` 里的
        #  心灵译自 mind 而不是 Psychic）
        r["_res"] = [re.compile(p) for p in r.get("variants_re", [])]
        r["_skip"] = r.get("exclude_paths", [])

    # (source, pack, path) -> (en, cn)
    corpus = []
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in en.items():
            if p in cn:
                corpus.append(("compendium", f.name, p, s, cn[p]))
    if args.package:
        en = flatten(json.loads((Path(args.package) / "lang" / "en.json").read_text(encoding="utf-8-sig")))
        cn = flatten(json.loads((repo / "lang" / "cn.json").read_text(encoding="utf-8-sig")))
        for p, s in en.items():
            if p in cn:
                corpus.append(("lang", "lang/cn.json", p, s, cn[p]))

    changed = {}          # (source, pack, path) -> new cn
    per_rule = defaultdict(list)
    for source, pack, path, en_s, cn_s in corpus:
        new = cn_s
        for r in rules:
            if not r["_en"].search(en_s):
                continue
            if r["_unless"] and r["_unless"].search(en_s):
                continue
            if any(x in path for x in r["_skip"]):
                continue
            for v in r.get("variants", []):
                if v in new and v != r["canonical"]:
                    before = new
                    new = new.replace(v, r["canonical"])
                    per_rule[r["term"]].append((pack, path, en_s, before, new, v))
            for rx in r["_res"]:
                if rx.search(new):
                    before = new
                    new = rx.sub(r["canonical"], new)
                    if new != before:
                        per_rule[r["term"]].append((pack, path, en_s, before, new, rx.pattern))
        if new != cn_s:
            if markup_signature(new) != markup_signature(cn_s):
                raise SystemExit(f"标记被改动了，中止：{pack} :: {path}")
            changed[(source, pack, path)] = new

    if args.review:
        out = ["# 术语统一评审清单\n"]
        for r in rules:
            hits = per_rule.get(r["term"], [])
            out.append(f"\n## {r['term']} → {r['canonical']}   （{len(hits)} 处）")
            if r.get("note"):
                out.append(f"> {r['note']}")
            for pack, path, en_s, before, after, v in hits:
                out.append(f"\n- `{pack}` :: `{path}`  〔{v} → {r['canonical']}〕")
                out.append(f"    EN  …{context(en_s, r['_en'].search(en_s).group(0))}…")
                # 逐字符对齐，指出这次改动落在哪儿（正则替换时字面量 v 不一定能定位）
                i = next((k for k in range(min(len(before), len(after))) if before[k] != after[k]), 0)
                out.append(f"    旧  …{before[max(0, i - 18):i + 24]}…".replace("\n", " "))
                out.append(f"    新  …{after[max(0, i - 18):i + 24]}…".replace("\n", " "))
        Path(args.review).write_text("\n".join(out), encoding="utf-8")
        print(f"评审清单 → {args.review}")

    for r in rules:
        print(f"  {r['term']:<14} → {r['canonical']:<6} {len(per_rule.get(r['term'], [])):>4} 处")
    print(f"  受影响条目 {len(changed)} 条")

    if args.write:
        by_file = defaultdict(dict)
        for (source, pack, path), new in changed.items():
            f = repo / "lang" / "cn.json" if source == "lang" else repo / "compendium" / "cn" / pack
            by_file[f][path] = new
        for f, items in by_file.items():
            doc = json.loads(f.read_text(encoding="utf-8-sig"))
            for path, new in items.items():
                node = doc
                parts = re.findall(r"[^.\[\]]+", path)
                for p in parts[:-1]:
                    node = node[int(p)] if isinstance(node, list) else node[p]
                if isinstance(node, list):
                    node[int(parts[-1])] = new
                else:
                    node[parts[-1]] = new
            f.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"    写入 {f.name}  ({len(items)} 条)")

    if args.batch_dir:
        d = Path(args.batch_dir)
        d.mkdir(parents=True, exist_ok=True)
        packs = defaultdict(dict)
        for (source, pack, path), new in changed.items():
            key = path[len("entries."):] if source == "compendium" and path.startswith("entries.") else path
            packs[(source, pack)][key] = new
        for (source, pack), items in packs.items():
            name = "lang.json" if source == "lang" else f"unify.{pack}"
            p = d / name
            p.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"    {p}  ({len(items)} 条)")


if __name__ == "__main__":
    main()
