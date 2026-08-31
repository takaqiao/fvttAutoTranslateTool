# -*- coding: utf-8 -*-
"""把翻译批次回写进 <repo>/lang/cn.json（compendium 侧的对应物是 apply_translations.py）

批次格式：扁平 {"<点分 key>": "<中文>"}，key 来自 lang_gap.py 的报告。

四道闸（任一不过就整条拒绝，不写入）：
  1. key 必须存在于新版英文里     —— 防打错 key / 翻已删除的 key
  2. 占位符 {xxx} 必须与英文完全一致 —— 少一个就会在 UI 里露出 "{name}"
  3. HTML 标签必须与英文一一对应
  4. Foundry 行内标记（@UUID[...]、@Condition[...]、[[/xxx]]）的**目标**必须一致
     （@UUID[目标]{标签} 的标签本就该翻译，只比对目标）
  5. 必须含中文；除非英文本身没有拉丁字母（如 "∞"），或该 key 在 keep-english 名单里

用法：
  python apply_lang.py --repo <repo> --package <foundry包> --batch <batch.json> [--dry]
  python apply_lang.py --repo <repo> --package <foundry包> --clean-stale
"""
import argparse
import json
import re
from pathlib import Path

CJK = re.compile(r'[一-鿿㐀-䶿]')
LATIN = re.compile(r'[A-Za-z]')
# Foundry 的插值占位符一律是 {identifier}；@UUID[...]{标签} 的标签含空格/中文，不算占位符
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z0-9_.\-]*\}')
HTML_TAG = re.compile(r'</?[a-zA-Z][^>]*>')
UUID_TARGET = re.compile(r'@(\w+)\[([^\]]+)\]')
INLINE_ROLL = re.compile(r'\[\[([^\]]+)\]\]')


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out


def set_path(root, dotted, value):
    """**必须写成扁平点号键，绝不能按点建嵌套。**

    Foundry 的 getProperty 先试整键、再按点逐级下探。我们的 cn.json 是扁平的
    （顶层就是 `"EMBER.CALENDAR.WORLD"` 这样的键），此处若按点建出
    `{"EMBER": {"CALENDAR": {"WORLD": …}}}`，就会与原来的扁平键**并存**，
    而整键那条路先命中扁平旧值 —— 新译文永远不生效，且没有任何报错。

    这正是 ember 侧 372 个键集体失效的成因（见 flatten_lang.py 的说明）。
    此函数一度把那个坑重新挖了一遍：Boon 统一批的两条 lang 改动写成了嵌套副本，
    文件里同时存在 423 行的旧「恩惠」与 1846 行的新「恩惠骰」，UI 显示的是旧的。
    """
    root[dotted] = value


def del_path(root, dotted):
    """扁平键优先删；旧的嵌套形态仍兜底处理，顺带清掉空父节点。"""
    if dotted in root:
        del root[dotted]
        return True
    parts = dotted.split(".")
    stack = [root]
    node = root
    for p in parts[:-1]:
        node = node.get(p)
        if not isinstance(node, dict):
            return False
        stack.append(node)
    if parts[-1] not in node:
        return False
    del node[parts[-1]]
    # 清理空掉的父节点
    for parent, key in zip(reversed(stack[:-1]), reversed(parts[:-1])):
        if isinstance(parent.get(key), dict) and not parent[key]:
            del parent[key]
        else:
            break
    return True


def tags_of(s):
    return sorted(t.lower() for t in HTML_TAG.findall(s))


def markup_of(s):
    return (sorted(UUID_TARGET.findall(s)),
            sorted(m.strip() for m in INLINE_ROLL.findall(s)))


def check(key, en, cn, keep_english):
    """返回 (problems, warnings)"""
    problems, warnings = [], []
    if set(PLACEHOLDER.findall(en)) != set(PLACEHOLDER.findall(cn)):
        problems.append(f"占位符不匹配 EN{sorted(set(PLACEHOLDER.findall(en)))} vs CN{sorted(set(PLACEHOLDER.findall(cn)))}")
    if tags_of(en) != tags_of(cn):
        problems.append(f"HTML 标签不匹配 EN{tags_of(en)} vs CN{tags_of(cn)}")
    if markup_of(en) != markup_of(cn):
        problems.append(f"行内标记目标不匹配 EN{markup_of(en)} vs CN{markup_of(cn)}")
    if cn == en and key not in keep_english:
        problems.append("译文与英文完全相同（若确为有意保留英文，请加入 lang_keep_english.json）")
    elif not CJK.search(cn) and LATIN.search(en):
        warnings.append("译文不含中文（纯占位符/符号串则属正常）")
    return problems, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--batch", action="append", default=[])
    ap.add_argument("--lang-file", default="en.json")
    ap.add_argument("--clean-stale", action="store_true", help="删除新版英文里已不存在的 key")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    cn_path = repo / "lang" / "cn.json"
    cn_doc = json.loads(cn_path.read_text(encoding="utf-8-sig"))
    en = flatten(json.loads((Path(args.package) / "lang" / args.lang_file).read_text(encoding="utf-8-sig")))

    keep_path = repo / "lang" / "lang_keep_english.json"
    keep_english = set(json.loads(keep_path.read_text(encoding="utf-8-sig"))) if keep_path.exists() else set()

    applied = rejected = 0
    for bp in args.batch:
        batch = json.loads(Path(bp).read_text(encoding="utf-8-sig"))
        print(f"== 批次 {bp}（{len(batch)} 条）")
        for key, value in batch.items():
            if key not in en:
                print(f"  [拒] {key} —— 新版英文里没有这个 key")
                rejected += 1
                continue
            problems, warnings = check(key, en[key], value, keep_english)
            if problems:
                print(f"  [拒] {key}")
                for p in problems:
                    print(f"        {p}")
                rejected += 1
                continue
            for w in warnings:
                print(f"  [警] {key} —— {w}")
            if not args.dry:
                set_path(cn_doc, key, value)
            applied += 1

    removed = 0
    if args.clean_stale:
        for key in sorted(set(flatten(cn_doc)) - set(en)):
            print(f"  [清] {key}")
            if not args.dry:
                del_path(cn_doc, key)
            removed += 1

    if not args.dry:
        cn_path.write_text(json.dumps(cn_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n应用 {applied} / 拒绝 {rejected} / 清理 {removed}"
          + ("（dry-run，未写入）" if args.dry else f" → {cn_path}"))
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
