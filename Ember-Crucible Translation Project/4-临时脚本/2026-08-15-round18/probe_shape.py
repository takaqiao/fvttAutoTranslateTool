# -*- coding: utf-8 -*-
"""探针：核对「CN 的标签块结构 = 新英文的标签块结构」这条硬证据到底成不成立。

反空转：必须报出扫了多少叶、多少块。三方（旧英文/新英文/中文）各切一次块。
"""
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/3-常用脚本/qa")
import scan_dropped_terms as S

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
TAG = re.compile(r"<[^>]+>")


def sig(s):
    return [m.group().lower() for m in TAG.finditer(s)]


def nblocks(s):
    return len(TAG.split(s))


def main(report, repo, baseline):
    rep = json.load(open(report, encoding="utf-8"))
    packs = S.baseline_packs(baseline)
    olds, news, cns = {}, {}, {}
    for pack, op in packs.items():
        cur = os.path.join(repo, "compendium", "en", pack)
        if not os.path.exists(cur):
            continue
        o, n, c = {}, {}, {}
        S.leaves(S.load_json(op).get("entries", {}), [], o)
        S.leaves(S.load_json(cur).get("entries", {}), [], n)
        cnp = os.path.join(repo, "compendium", "cn", pack)
        if os.path.exists(cnp):
            S.leaves(S.load_json(cnp).get("entries", {}), [], c)
        olds[pack], news[pack], cns[pack] = o, n, c

    st = collections.Counter()
    detail = []
    for f in rep["findings"]:
        pk, pa = f["pack"], f["path"]
        oe, ne, cn = olds[pk][pa], news[pk][pa], cns[pk][pa]
        st["leaf"] += 1
        st["blk_new"] += nblocks(ne)
        st["blk_cn"] += nblocks(cn)
        st["blk_old"] += nblocks(oe)
        same_new = sig(ne) == sig(cn)
        same_old = sig(oe) == sig(cn)
        cnt_new = nblocks(ne) == nblocks(cn)
        cnt_old = nblocks(oe) == nblocks(cn)
        st[f"sig_new={same_new}"] += 1
        st[f"sig_old={same_old}"] += 1
        st[f"cnt_new={cnt_new}"] += 1
        st[f"cnt_old={cnt_old}"] += 1
        detail.append((pk, pa, nblocks(oe), nblocks(ne), nblocks(cn), same_old, same_new))
    print(f"扫了 {st['leaf']} 叶；块数合计 旧英文 {st['blk_old']} / 新英文 {st['blk_new']} / 中文 {st['blk_cn']}")
    for k in sorted(st):
        if k.startswith(("sig_", "cnt_")):
            print(f"  {k}: {st[k]}")
    print("\n=== 块数不等的叶 ===")
    for pk, pa, no, nn, nc, so, sn in detail:
        if nn != nc:
            print(f"  {pk} :: {pa[-70:]}  旧{no} 新{nn} 中{nc}  sig_old={so} sig_new={sn}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
