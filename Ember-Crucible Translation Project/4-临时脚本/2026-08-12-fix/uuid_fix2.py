# -*- coding: utf-8 -*-
"""@UUID[目标]{标签} 错位：以**文档名**为权威的检测 + 修复批次生成。

为什么不用「全库多数写法」：错位是成批复制出来的，多数表本身会被污染。
实测 `Scene.emberVistaRedrak` 的多数中文标签是「土石环阵」(6/10) —— 那正是错的那个。

权威链条（全部是直接证据，不投票）：
  目标 id --(LevelDB packs)--> 英文文档名 --(compendium/en ↔ cn 同构 walk)--> 中文文档名
  某个中文标签属于哪个目标，就看它与该目标文档名对不对得上。

判定：
  fit(标签, 目标) = 标签与该目标的「中文名 / 中文名+英文名 / 英文名」有包含关系。
  · 位置 j 的标签 fit 它当前挂着的目标        -> 合规
  · 位置 j 的标签只 fit 同叶另一个目标         -> 错位
  · 全体错位位置所需目标的多重集 == 它们当前持有目标的多重集 -> 纯轮转，可机械修复
  · 其余（需要动到合规位置、fit 不唯一、无 fit 证据）-> UNCERTAIN，交人工

修复动作：**中文可见文字一个字都不动**，只把整段 `@UUID[...]` 的方括号内容在同叶内
换位（每个目标串逐字未改，多重集不变，标记闸照旧通过）。
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}
UUID_RX = re.compile(r"@UUID\[([^\]]*)\](\{([^}]*)\})?")
CJK = re.compile(r"[一-鿿]")


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
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
    return [(m.group(1), m.group(3), m.start(), m.end()) for m in UUID_RX.finditer(s or "")]


def key_of(target):
    core = target.split()[0] if target.split() else target
    if "#" in core:
        head, frag = core.split("#", 1)
        return head.split(".")[-1] + "#" + frag
    return core.split(".")[-1]


def bare(k):
    return k.split("#")[0]


def cn_core(name):
    if not name:
        return None
    m = list(CJK.finditer(name))
    return name[:m[-1].end()].strip() if m else name.strip()


class Authority:
    def __init__(self, ids, namemap):
        self.ids, self.nm = ids, namemap

    def en_name(self, k):
        r = self.ids.get(bare(k))
        return r.get("name") if r else None

    def forms(self, k, en_label=None):
        """该目标（以及该英文标签）可接受的中文/英文写法集合。"""
        out = set()
        for src in (self.en_name(k), en_label):
            if not src:
                continue
            out.add(src)
            for cn in self.nm.get(src) or []:
                out.add(cn)
                c = cn_core(cn)
                if c:
                    out.add(c)
        return {x for x in out if x}

    def fit(self, label, k, en_label=None, strong=False):
        """strong=True 只认「标签 == 文档名的某个写法」这种硬证据。

        松匹配（包含关系）只用来确认「这个位置本来就是对的」——用它来判定
        错位会误伤：`树篱迷宫入口` 含 `树篱迷宫`、`马尔斯通晚会` 含 `马尔斯通`、
        `裂谷蜡烛` 含 `蜡烛`，三处都是本来正确的链接。
        """
        if not label:
            return False
        for f in self.forms(k, en_label):
            if label == f:
                return True
            if not strong and len(label) >= 2 and (label in f or f in label):
                return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--packs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--batch-dir")
    a = ap.parse_args()

    auth = Authority(json.load(open(a.ids, encoding="utf-8")),
                     json.load(open(a.names, encoding="utf-8")))
    packs = [p.strip() for p in a.packs.split(",") if p.strip()]

    report = {"packs": packs, "leaves": 0, "with_links": 0,
              "fixed": [], "uncertain": []}
    batches = defaultdict(dict)

    for pack in packs:
        for r in load_pairs(a.repo, pack):
            if not r["cn"]:
                continue
            report["leaves"] += 1
            el = [(t, l) for t, l, _, _ in links(r["en"]) if l]
            cl = [(t, l, s, e) for t, l, s, e in links(r["cn"]) if l]
            if not cl:
                continue
            ck = [key_of(t) for t, _, _, _ in cl]
            if len(set(ck)) < 2:
                continue
            report["with_links"] += 1

            # 每个中文位置对应的英文标签（按目标分组、按出现顺序配）
            en_by_key = defaultdict(list)
            for t, l in el:
                en_by_key[key_of(t)].append(l)

            avail = Counter(ck)
            fits, loose = [], []   # 硬证据集合 / 松证据集合（后者只用于「本来就对」）
            for j, (t, l, _, _) in enumerate(cl):
                s, w = set(), set()
                for k in avail:
                    for elab in (en_by_key.get(k) or [None]):
                        if auth.fit(l, k, elab, strong=True):
                            s.add(k)
                        if auth.fit(l, k, elab):
                            w.add(k)
                fits.append(s)
                loose.append(w)

            # 错位 = 硬证据指向别的目标，且松证据也不支持它现在挂着的目标
            mis = [j for j in range(len(cl))
                   if fits[j] and ck[j] not in fits[j] and ck[j] not in loose[j]]
            entry = {
                "pack": pack, "path": r["path"], "batch_path": r["batch_path"],
                "links": [{"pos": j, "label": cl[j][1], "target": cl[j][0],
                           "cur_key": ck[j], "cur_doc": auth.en_name(ck[j]),
                           "fits": sorted(fits[j]),
                           "fit_docs": [auth.en_name(x) for x in sorted(fits[j])]}
                          for j in range(len(cl))],
                "en_links": [{"t": t, "label": l, "doc": auth.en_name(key_of(t))}
                             for t, l in el],
            }
            if not mis:
                continue

            # 三种状态：FIT(标签对得上自己的目标，确认无误，绝不动)
            #           MIS(标签对得上同叶别的目标、对不上自己)
            #           UNK(标签与本叶任何目标的文档名都对不上，无证据)
            unk_all = [j for j in range(len(cl))
                       if j not in mis and ck[j] not in loose[j]]

            def solve(pool_idx):
                """在 pool_idx 这些位置之间重排目标；返回 (new, why)。"""
                need = []
                keys = set(ck[x] for x in pool_idx)
                for j in mis:
                    cand = fits[j] & keys
                    if len(cand) != 1:
                        return None, (f"位置 {j}「{cl[j][1]}」应归属的目标不唯一或落在已确认无误的"
                                      f"位置上：fits={sorted(fits[j])}")
                    need.append(next(iter(cand)))
                have = Counter(ck[x] for x in pool_idx)
                if not Counter(need) <= have:
                    return None, (f"错位位置所需目标不在可移动集合内："
                                  f"need={dict(Counter(need))} have={dict(have)}")
                left = have - Counter(need)
                extra = [x for x in pool_idx if x not in mis]
                if extra and len(set(left.elements())) > 1:
                    return None, f"剩余 {dict(left)} 要分给 {len(extra)} 个无证据位置，分法不唯一"
                pool = defaultdict(list)
                for j in pool_idx:
                    pool[ck[j]].append(cl[j][0])
                nw = [t for t, _, _, _ in cl]
                for j, k in zip(mis, need):
                    nw[j] = pool[k].pop(0)
                for j, t in zip(extra, [pool[k].pop(0) for k in left.elements()]):
                    nw[j] = t
                return nw, None

            # 先只在明确错位的位置之间解；解不出来再把「无证据」位置拉进来，
            # 但只拉进持有被需要目标的那几个，免得把整叶的链接都搅动。
            new, why = solve(mis)
            if new is None:
                want = set().union(*(fits[j] for j in mis)) if mis else set()
                extra = [j for j in unk_all if ck[j] in want]
                new, why2 = solve(sorted(mis + extra)) if extra else (None, why)
                if new is None:
                    entry["why"] = why if not extra else why2
                    report["uncertain"].append(entry)
                    continue
            # 收尾自检：任何位置若其标签在本叶里明明能对上某个目标，而算出来的
            # 结果没给它那个目标，说明这一叶还没解干净（多半是三元轮转只解开了一环）
            touched = set(mis) | {j for j in range(len(cl)) if new[j] != cl[j][0]}
            bad = [j for j in sorted(touched)
                   if loose[j] and key_of(new[j]) not in loose[j]]
            if bad:
                entry["why"] = (f"解不完整：位置 {bad} 的标签能对上本叶某个目标，"
                                f"但排出来的结果没给它")
                report["uncertain"].append(entry)
                continue
            mis = [j for j in range(len(cl)) if new[j] != cl[j][0]]
            if not mis:
                continue

            s, out, prev = r["cn"], [], 0
            for (t, l, st, en_), nt in zip(cl, new):
                out.append(s[prev:st])
                out.append(f"@UUID[{nt}]{{{l}}}")
                prev = en_
            out.append(s[prev:])
            fixed = "".join(out)
            vis = lambda x: UUID_RX.sub(lambda m: (m.group(3) or ""), x)
            assert vis(fixed) == vis(s), r["path"]          # 可见文字逐字不变
            assert Counter(x[0] for x in links(fixed)) == \
                   Counter(x[0] for x in links(s)), r["path"]  # 目标多重集不变
            entry["moved"] = [{"pos": j, "label": cl[j][1],
                               "from": cl[j][0], "from_doc": auth.en_name(ck[j]),
                               "to": new[j], "to_doc": auth.en_name(key_of(new[j]))}
                              for j in mis]
            entry["new_cn"] = fixed
            # 人工通读用的窗口（原文 / 改后 / 英文），避免整叶几万字
            entry["cn_before_win"] = s[max(0, min(cl[j][2] for j in mis) - 260):
                                       min(len(s), max(cl[j][3] for j in mis) + 260)]
            fl = [x for x in links(fixed) if x[1]]
            entry["cn_after_win"] = fixed[max(0, min(fl[j][2] for j in mis) - 260):
                                          min(len(fixed), max(fl[j][3] for j in mis) + 260)]
            movedkeys = {ck[j] for j in mis} | {key_of(new[j]) for j in mis}
            enl = [x for x in links(r["en"]) if x[1] and key_of(x[0]) in movedkeys]
            if enl:
                entry["en_win"] = r["en"][max(0, min(x[2] for x in enl) - 260):
                                          min(len(r["en"]), max(x[3] for x in enl) + 260)]
            report["fixed"].append(entry)
            batches[pack][r["batch_path"]] = fixed

    report["n_fixed_leaves"] = len(report["fixed"])
    report["n_fixed_links"] = sum(len(f["moved"]) for f in report["fixed"])
    report["n_uncertain"] = len(report["uncertain"])
    open(a.report, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=1))
    if a.batch_dir:
        for pack, b in batches.items():
            fp = os.path.join(a.batch_dir, f"uuid-fix-{pack}")
            json.dump(b, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"batch {fp}  entries={len(b)}")
    print(f"leaves={report['leaves']} with_links={report['with_links']} "
          f"fixed_leaves={report['n_fixed_leaves']} fixed_links={report['n_fixed_links']} "
          f"uncertain={report['n_uncertain']} -> {a.report}")


if __name__ == "__main__":
    main()
