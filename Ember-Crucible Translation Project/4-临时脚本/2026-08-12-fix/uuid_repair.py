# -*- coding: utf-8 -*-
"""@UUID[目标]{标签} 错位：检测 + 生成修复批次。

判据（审计 1.4）：**英文侧同叶的 目标↔标签 配对是权威**。
  1. 全库统计 M[(target, EN_label)] -> 中文标签多数写法。
  2. 对每一叶，拿英文侧的 (t_i, E_i) 列表算出「这一叶应该出现的中文标签集合」
     expected_i = M[(t_i, E_i)]。
  3. 把中文侧的每个标签 C_j 匹配回 expected_i，得到一个双射 π；
     位置 j 的目标应当是 t_{π(j)}。
  4. π 不是恒等映射 ⇒ 标签挂错了目标。修复 = 让每个标签配上它自己的目标，
     **中文可见文字一个字都不动**（语序保持现状），只把整个 @UUID[...] 方括号
     内容按 π 重新落位。方括号内容逐字未改，只是随中文语序换了位置；
     @UUID[...] 的多重集与英文完全相同，标记闸照旧通过。

  匹配不唯一 / 英文中文链接数对不上 / 缺 M 数据 ⇒ 判 UNCERTAIN，不动。

用法:
  python uuid_repair.py --repo <repo> --packs a.json,b.json --ids reports/ember_ids.json
                        --report reports/x.json [--batch-dir batches]
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from itertools import permutations

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
UUID_RX = re.compile(r"@UUID\[([^\]]*)\](\{([^}]*)\})?")


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append({"path": p,
                    "batch_path": p[len("entries."):] if p.startswith("entries.") else p,
                    "en": en, "cn": cn if isinstance(cn, str) else None})


def load_pairs(repo, pack):
    en = json.load(open(os.path.join(repo, "compendium", "en", pack), encoding="utf-8"))
    cp = os.path.join(repo, "compendium", "cn", pack)
    cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
    rows = []
    walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], rows)
    for r in rows:
        r["pack"] = pack
    return rows


def links(s):
    """[(target_raw, label|None, start, end)] —— 只保留带标签的由调用方过滤。"""
    return [(m.group(1), m.group(3), m.start(), m.end()) for m in UUID_RX.finditer(s or "")]


def key_of(target):
    """归一化目标键：取最后一段 id（保留 #fragment，它指向页内小节，标签不同）。"""
    core = target.split()[0] if target.split() else target
    if "#" in core:
        head, frag = core.split("#", 1)
        return head.split(".")[-1] + "#" + frag
    return core.split(".")[-1]


def bare_id(k):
    return k.split("#")[0]


def assign(expected, en_targets, cn_labels, cn_targets):
    """按标签分桶做配对，返回 (每个中文位置应有的目标, 说明)。

    桶内若英文侧给出多个不同目标而中文位置也不止一个，谁配谁没有证据 —— 除非
    中文当前的目标多重集与该桶相同（那就保持现状），否则判 UNCERTAIN。
    """
    en_by = defaultdict(list)
    for i, x in enumerate(expected):
        if x is None:
            return None, "英文侧某链接没有全库多数中文写法（M 表缺）"
        en_by[x].append(i)
    cn_by = defaultdict(list)
    for j, l in enumerate(cn_labels):
        cn_by[l].append(j)
    if set(en_by) != set(cn_by) or any(len(en_by[k]) != len(cn_by[k]) for k in en_by):
        return None, "中文标签集合与英文侧应有的中文标签集合对不上"
    out = [None] * len(cn_labels)
    for lab, js in cn_by.items():
        tg = [en_targets[i] for i in en_by[lab]]
        if len(set(tg)) == 1:
            for j in js:
                out[j] = tg[0]
        else:
            cur = [cn_targets[j] for j in js]
            if Counter(cur) == Counter(tg):
                for j in js:
                    out[j] = cn_targets[j]   # 无证据，保持现状
            else:
                return None, f"标签「{lab}」在英文侧对应多个不同目标，无法判定谁配谁"
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--packs", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--report", required=True)
    ap.add_argument("--batch-dir")
    a = ap.parse_args()

    ids = json.load(open(a.ids, encoding="utf-8")) if a.ids else {}
    packs = [p.strip() for p in a.packs.split(",") if p.strip()]
    rows = []
    for p in packs:
        rows.extend(load_pairs(a.repo, p))

    # --- M[(target_key, en_label)] -> Counter(中文标签) ------------------
    M = defaultdict(Counter)
    for r in rows:
        if not r["cn"]:
            continue
        el = [(key_of(t), l) for t, l, _, _ in links(r["en"]) if l]
        cl = [(key_of(t), l) for t, l, _, _ in links(r["cn"]) if l]
        if len(el) != len(cl):
            continue
        # 只有当同叶英文链接的 (目标,标签) 两两不同、且中文目标多重集与英文相等时，
        # 这一叶才对统计有贡献；位置对齐用英文顺序（错位的叶子会给出错误的一票，
        # 但它们是少数，多数表仍然正确 —— 之后再用文档名做交叉验证）。
        if Counter(t for t, _ in el) != Counter(t for t, _ in cl):
            continue
        for (t, e), (_, c) in zip(el, cl):
            M[(t, e)][c] += 1

    canon = {k: v.most_common(1)[0][0] for k, v in M.items()}
    support = {k: (v.most_common(1)[0][1], sum(v.values())) for k, v in M.items()}

    # --- 逐叶判定 --------------------------------------------------------
    report = {"packs": packs, "leaves": 0, "ok": 0, "fixed": [], "uncertain": []}
    batches = defaultdict(dict)

    for r in rows:
        if not r["cn"]:
            continue
        report["leaves"] += 1
        el = [(t, l, s, e) for t, l, s, e in links(r["en"]) if l]
        cl = [(t, l, s, e) for t, l, s, e in links(r["cn"]) if l]
        if not cl or len(el) != len(cl):
            continue
        ek = [key_of(t) for t, _, _, _ in el]
        ck = [key_of(t) for t, _, _, _ in cl]
        if Counter(ek) != Counter(ck):
            continue
        if len(set(ek)) < 2:
            continue  # 同叶只有一个目标，不可能错位
        expected = [canon.get((k, l)) for k, (_, l, _, _) in zip(ek, el)]
        cn_labels = [l for _, l, _, _ in cl]
        if expected == cn_labels and ek == ck:
            report["ok"] += 1
            continue
        cur = tuple(t for t, _, _, _ in cl)
        new, why = assign(expected, [t for t, _, _, _ in el], cn_labels, list(cur))
        entry = {
            "pack": r["pack"], "path": r["path"], "batch_path": r["batch_path"],
            "en_links": [{"k": k, "t": t, "label": l, "expect_cn": x,
                          "support": support.get((k, l)),
                          "doc": (ids.get(bare_id(k)) or {}).get("name")}
                         for (t, l, _, _), k, x in zip(el, ek, expected)],
            "cn_links": [{"k": k, "t": t, "label": l,
                          "doc": (ids.get(bare_id(k)) or {}).get("name")}
                         for (t, l, _, _), k in zip(cl, ck)],
        }
        if new is None:
            # 只有「某个中文标签明显长在别的目标上」才值得人看；其余多半是
            # 标签留英/一处换了说法这类与错位无关的差异。
            own = defaultdict(set)   # 目标 -> 这一叶里它自己应有的中文标签
            for i in range(len(expected)):
                own[ek[i]].add(expected[i])
            susp = [{"pos": j, "label": cn_labels[j], "on": ck[j],
                     "own_should_be": sorted(x for x in own[ck[j]] if x),
                     "belongs_to": [ek[i] for i in range(len(expected))
                                    if expected[i] == cn_labels[j] and ek[i] != ck[j]]}
                    for j in range(len(cn_labels))
                    if cn_labels[j] not in own[ck[j]]
                    and any(expected[i] == cn_labels[j] and ek[i] != ck[j]
                            for i in range(len(expected)))]
            report["uncertain_total"] = report.get("uncertain_total", 0) + 1
            if susp:
                entry["why"] = why
                entry["suspect"] = susp
                report["uncertain"].append(entry)
            continue
        new = tuple(new)
        if new == cur:
            report["ok"] += 1
            continue
        # 重建中文串：可见文字一字不动，只换方括号内容
        s, out, prev = r["cn"], [], 0
        for (t, l, st, en_), nt in zip(cl, new):
            out.append(s[prev:st])
            out.append(f"@UUID[{nt}]{{{l}}}")
            prev = en_
        out.append(s[prev:])
        fixed = "".join(out)
        # 安全网：可见文字必须逐字不变；@UUID 多重集必须不变
        assert UUID_RX.sub(lambda m: (m.group(3) or ""), fixed) == \
               UUID_RX.sub(lambda m: (m.group(3) or ""), s), r["path"]
        assert Counter(x[0] for x in links(fixed)) == Counter(x[0] for x in links(s)), r["path"]
        entry["new_cn"] = fixed
        entry["moved"] = [{"pos": j, "from": cur[j], "to": new[j], "label": cl[j][1]}
                          for j in range(len(cur)) if cur[j] != new[j]]
        report["fixed"].append(entry)
        batches[r["pack"]][r["batch_path"]] = fixed

    report["n_fixed"] = len(report["fixed"])
    report["n_uncertain"] = len(report["uncertain"])
    open(a.report, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=1))
    if a.batch_dir:
        for pack, b in batches.items():
            fp = os.path.join(a.batch_dir, f"uuid-fix-{pack}")
            json.dump(b, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"batch {fp}  entries={len(b)}")
    print(f"leaves={report['leaves']} fixed_leaves={report['n_fixed']} "
          f"uncertain_leaves={report['n_uncertain']} -> {a.report}")


if __name__ == "__main__":
    main()
