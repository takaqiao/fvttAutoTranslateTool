# -*- coding: utf-8 -*-
"""把译文里被写成裸中文的 Foundry enricher 还原回去。

旧译文里大量出现这种情况：英文是 `@Condition[exposed]`（可点击、带说明浮窗），
译文直接写成「暴露」两个字，链接就没了。玩家看不到状态说明，也点不动。

`@Condition[id]` / `@Action[default id]` / `@Rule[x]` 都**不带标签**，渲染出来的
就是 lang 里的译名 —— 所以还原方式是：把译文里那个裸词换回 enricher 本身。

安全策略：
  · 只在该中文词在这条译文里**恰好出现一次**时才替换（多次出现无法判断该改哪个）；
  · 替换后重新比对标记，必须与英文一致，否则回滚这一条；
  · 处理不了的逐条列出，绝不猜。

用法：
  python restore_enrichers.py --repo <repo> --package <foundry包> [--write]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')


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


def sig(s):
    return Counter(MARKUP.findall(s)) + Counter(INLINE_CMD.findall(s))


def set_at(doc, path, value):
    node = doc
    parts = re.findall(r"[^.\[\]]+", path)
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value


def build_terms(lang_en, lang_cn):
    """enricher 目标 → 中文词

    · @Condition[id] → ACTIVE_EFFECT.STATUSES.*（id 是英文状态名的小写）
    · @Action[default id] → ACTION.DEFAULT_ACTIONS.<Name>.Name
    · @Rule[action.id] → ACTION.TAG.<Name>（SYSTEM.RULES.action 就是 ACTION.TAGS）
    · @Rule[condition.id] → 同 @Condition
    """
    terms = {}
    for k, v in lang_en.items():
        cn = lang_cn.get(k)
        if not cn:
            continue
        if k.startswith("ACTIVE_EFFECT.STATUSES.") and k.count(".") == 2:
            terms[f"@Condition[{v.lower()}]"] = cn
            terms[f"@Rule[condition.{v.lower()}]"] = cn
        m = re.fullmatch(r"ACTION\.DEFAULT_ACTIONS\.(\w+)\.Name", k)
        if m:
            aid = m.group(1)[0].lower() + m.group(1)[1:]
            terms[f"@Action[default {aid}]"] = cn
        m = re.fullmatch(r"ACTION\.TAG\.(\w+)", k)
        if m and not m.group(1).endswith("Tooltip"):
            tid = m.group(1)[0].lower() + m.group(1)[1:]
            terms[f"@Rule[action.{tid}]"] = cn
    return terms


SPLIT = re.compile(r"(?<=</p>)|(?<=</h1>)|(?<=</h2>)|(?<=</h3>)|(?<=</h4>)|(?<=</li>)|(?<=</dd>)|(?<=</dt>)")


def segments(s):
    return [x for x in SPLIT.split(s) if x]


def aligned_pairs(en_s, cn_s):
    """长规则页里同一个词会出现十几次，整串匹配必然失败。
    英文与译文的块级结构若一一对应，就按段落配对，在段内做替换 —— 段内基本只出现一次。
    结构对不上就退回整串（由调用方处理）。"""
    a, b = segments(en_s), segments(cn_s)
    return list(zip(a, b)) if len(a) == len(b) > 1 else None


def ember_candidates(token):
    """Ember 自己的增强器：按 token 现算候选写法。

    这些增强器渲染出来一律是 <strong>…</strong>：
      @Advantage[2]        → <strong>+2 Boons</strong>   → 译文多半是「+2 恩惠骰」
      @Advantage[-2]       → <strong>-2 Banes</strong>   → 「-2 祸骰」
      @CriticalSuccess[13] → <strong>Critical Success</strong>
    注意 CriticalSuccess 的 DC **不出现在渲染结果里**，所以译文那侧没有任何线索，
    只能靠段落对齐从英文同一段里取回 N —— 这也是为什么必须先做段落对齐再落到这里。
    """
    out = []
    m = re.fullmatch(r"@Advantage\[(-?\d+)\]", token)
    if m:
        n = int(m.group(1))
        body = f"+{n} 恩惠骰" if n > 0 else f"{n} 祸骰"
        alt = f"+{n} 恩惠" if n > 0 else f"{n} 祸"
        out += [f"<strong>{body}</strong>", f"<strong>{alt}</strong>", body]
    m = re.fullmatch(r"@Critical(Success|Failure)\[\d+\]", token)
    if m:
        body = "重大成功" if m.group(1) == "Success" else "重大失败"
        out += [f"<strong>{body}</strong>", body]
    return out


def candidate_forms(word):
    """译文里同一个状态可能写成「被夹击」「处于倒地状态」「<strong>暴露</strong>」等。
    按从具体到宽泛的顺序试，第一个数量对得上的就用它 —— 顺带把多包的 <strong> 一起收掉。"""
    return [f"<strong>被{word}</strong>", f"<strong>处于{word}</strong>", f"<strong>{word}</strong>",
            f"被{word}", f"处于{word}", word]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--surface-forms", help="旧译文的各种裸写法 → enricher 的对照表")
    ap.add_argument("--lang-repo", help="术语表取自另一个仓库的 lang/cn.json（修 ember 时要指向 crucible-cn，"
                                        "因为 @Condition / @Action 的名字都归 crucible 管）")
    ap.add_argument("--lang-package", help="与 --lang-repo 配套的 Foundry 包目录")
    args = ap.parse_args()

    repo = Path(args.repo)
    lang_pkg = Path(args.lang_package or args.package)
    lang_repo = Path(args.lang_repo or args.repo)
    lang_en = flatten(json.loads((lang_pkg / "lang" / "en.json").read_text(encoding="utf-8-sig")))
    lang_cn = flatten(json.loads((lang_repo / "lang" / "cn.json").read_text(encoding="utf-8-sig")))
    terms = build_terms(lang_en, lang_cn)
    surface = {}
    if args.surface_forms:
        raw = json.loads(Path(args.surface_forms).read_text(encoding="utf-8-sig"))
        surface = {k: v for k, v in raw.items() if not k.startswith("_")}

    fixed = defaultdict(dict)
    stats = Counter()
    leftovers = []

    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn_doc = json.loads(cnp.read_text(encoding="utf-8-sig"))
        cn = dict(leaves(cn_doc))
        for p, s in en.items():
            t = cn.get(p)
            if not t:
                continue
            missing = {k: a - b for k, (a, b) in
                       ((k, (sig(s).get(k, 0), sig(t).get(k, 0))) for k in sig(s))
                       if a > b}
            if not missing:
                continue
            new = t
            unresolved = []

            # 先试按段落对齐：段内替换比整串替换可靠得多
            pairs = aligned_pairs(s, t)
            if pairs:
                rebuilt, changed_any = [], False
                for en_seg, cn_seg in pairs:
                    seg_missing = {k: a - b for k, (a, b) in
                                   ((k, (sig(en_seg).get(k, 0), sig(cn_seg).get(k, 0))) for k in sig(en_seg))
                                   if a > b}
                    out_seg = cn_seg
                    for token, n in seg_missing.items():
                        word = terms.get(token)
                        cands = (candidate_forms(word) if word else []) \
                            + ember_candidates(token) \
                            + surface.get(token, []) + surface.get(token + "{生物}", [])
                        form = next((c for c in cands if out_seg.count(c) == n), None)
                        if form:
                            out_seg = out_seg.replace(form, token)
                            stats[token] += n
                            changed_any = True
                    rebuilt.append(out_seg)
                if changed_any:
                    new = "".join(rebuilt)
                    missing = {k: a - b for k, (a, b) in
                               ((k, (sig(s).get(k, 0), sig(new).get(k, 0))) for k in sig(s))
                               if a > b}

            # ① 译文把 @ref[x] 写成了别的路径（@ref[name] / @Ref[...]）——路径写错等于取错字段
            for token in list(missing):
                m = re.fullmatch(r"@ref\[([\w.]+)\]", token)
                if not m:
                    continue
                wrong = [w for w in re.findall(r"@[Rr]ef\[[\w.]+\]", new) if w not in sig(s)]
                if len(wrong) == missing[token] and len(set(wrong)) == 1:
                    default = "{生物}" if m.group(1).endswith("actor.name") else ""
                    # 连带吃掉紧跟的 {标签}：原标签若已译好就留用，否则补默认标签
                    pat = re.compile(re.escape(wrong[0]) + r"(\{[^}]*\})?")
                    new = pat.sub(lambda mm: token + (mm.group(1) or default), new)
                    stats[token] += missing.pop(token)

            for token, n in missing.items():
                word = terms.get(token)
                # lang 里没有对应译名的（@Action[Compendium…] 这类指向具体物品的），
                # 只能靠 surface-forms 对照表
                cands = (candidate_forms(word) if word else []) \
                    + ember_candidates(token) \
                    + surface.get(token, []) + surface.get(token + "{生物}", [])
                if not cands:
                    unresolved.append(f"{token} ×{n}（没有对应中文词，也没有 surface-forms 条目）")
                    continue
                form = next((c for c in cands if new.count(c) == n), None)
                if not form:
                    hint = f"「{word}」出现 {new.count(word)} 次" if word else "对照表里的写法都没命中"
                    unresolved.append(f"{token} ×{n}（{hint}，无法确定改哪个）")
                    continue
                new = new.replace(form, token)
                stats[token] += n
            # 允许「改好一部分」：长规则页往往还剩个别无法自动定位的链接，
            # 但已经还原的那些不该因此被回滚。条件是只增不减、且不会超过英文的数量。
            if new != t:
                sn, st, ss = sig(new), sig(t), sig(s)
                only_added = all(sn[k] >= st.get(k, 0) for k in sn) \
                    and all(sn.get(k, 0) >= v for k, v in st.items())
                no_overshoot = all(sn[k] <= ss.get(k, 0) for k in sn if k in ss or sn[k] > 0)
                if only_added and no_overshoot:
                    fixed[f.name][p] = new
                else:
                    unresolved.append("替换后标记与英文对不上，已回滚")
            if unresolved:
                leftovers.append({"pack": f.name, "path": p, "issues": unresolved,
                                  "en": s[:200], "cn": t[:200]})

        if args.write and fixed.get(f.name):
            for p, v in fixed[f.name].items():
                set_at(cn_doc, p, v)
            cnp.write_text(json.dumps(cn_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total = sum(len(v) for v in fixed.values())
    print(f"可自动还原 {total} 条译文，共 {sum(stats.values())} 个 enricher")
    for token, n in stats.most_common(12):
        print(f"    {token:<40} ×{n}")
    print(f"需人工处理 {len(leftovers)} 条")
    out = repo.parent / "5-其他内容" / "reports" / "crucible" / "enricher_leftovers.json"
    if out.parent.exists():
        out.write_text(json.dumps(leftovers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    → {out}")
    if args.write:
        print("已写入")


if __name__ == "__main__":
    main()
