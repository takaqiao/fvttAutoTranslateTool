# -*- coding: utf-8 -*-
"""并排导出 lang/cn.json 与 Foundry 包 lang/en.json 的键值对，便于逐键通读。"""
import argparse, json, re
from pathlib import Path


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--grep-key")
    ap.add_argument("--grep-en")
    ap.add_argument("--grep-cn")
    ap.add_argument("--slice")           # i/k
    ap.add_argument("--out")
    args = ap.parse_args()

    cn_raw = json.loads((Path(args.repo) / "lang" / "cn.json").read_text(encoding="utf-8-sig"))
    en = flatten(json.loads((Path(args.package) / "lang" / "en.json").read_text(encoding="utf-8-sig")))
    cn = flatten(cn_raw)

    # 顶层键形态自检
    nested_top = [k for k, v in cn_raw.items() if isinstance(v, dict)]
    hdr = [f"# cn keys(flat_view)={len(cn)} raw_top={len(cn_raw)} en keys={len(en)} nested_top={len(nested_top)}"]
    if nested_top:
        hdr.append("# !! 混合形态顶层键: " + ", ".join(nested_top[:20]))

    keys = sorted(set(cn) | set(en))
    if args.grep_key:
        r = re.compile(args.grep_key)
        keys = [k for k in keys if r.search(k)]
    if args.grep_en:
        r = re.compile(args.grep_en)
        keys = [k for k in keys if r.search(en.get(k, ""))]
    if args.grep_cn:
        r = re.compile(args.grep_cn)
        keys = [k for k in keys if r.search(cn.get(k, ""))]
    if args.slice:
        i, k = (int(x) for x in args.slice.split("/"))
        keys = keys[(i - 1) * len(keys) // k: i * len(keys) // k]

    lines = list(hdr)
    for key in keys:
        lines.append(f"{key}\n  EN: {en.get(key, '<<MISSING-EN>>')}\n  CN: {cn.get(key, '<<MISSING-CN>>')}")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(keys)} keys -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
