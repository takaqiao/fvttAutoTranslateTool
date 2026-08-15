# -*- coding: utf-8 -*-
"""S4 · 分片 finding `npc-translit-split`：逐个音译对做「英文闸 + 叶级计数」。

只读。对每组 (canonical, variant)：
  - 统计两侧各自命中的叶数与出现次数，并标出该叶是不是 name / tokenName 字段
  - 英文闸：命中 variant 的每一片叶，其同路径英文必须含该英文专名，否则不算
"""
import sys, re, os
import s4_lib as L

PAIRS = [
    ("Nathira", "纳希拉·杰索斯", "纳西拉·杰索斯"),
    ("Gedron", "格德隆·塔斯", "盖德隆·塔斯"),
    ("Gurty", "古蒂·霍尔德斯通", "格蒂·霍尔德斯通"),
    ("Tyrwar", "奥基娅·提尔沃", "奥基娅·提尔瓦"),
    ("Kraddok", "希耶尔·克拉多克", "西耶尔·克拉多克"),
    ("Terracini", "佩妮·特拉奇尼", "彭妮·特拉奇尼"),
    ("Faviyos", "加斯特恩·法维约斯", "加斯特恩·法维奥斯"),
    ("Kohle", "科代恩·科尔", "科代恩·科勒"),
    ("Chess", "瓦索洛缪·切斯", "沃索洛缪·切斯"),
    ("Verocorrt", "布琳娜·维罗科尔特", "布琳娜·维罗科特"),
    ("Sarinland", "阿科斯·萨林兰德", "阿科斯·萨林兰"),
]


def side(word):
    """返回 [(repo, pack, path, cn, en, n_hits)]"""
    out = []
    for repo in ("1", "2"):
        for pack in L.packs(repo):
            if not os.path.exists(os.path.join(L.REPO[repo], "compendium", "en", pack)):
                continue
            cn = dict(L.all_leaves(repo, pack, "cn"))
            en = dict(L.all_leaves(repo, pack, "en"))
            for p, v in cn.items():
                if word in v:
                    out.append((repo, pack, p, v, en.get(p, ""), v.count(word)))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name_en, canon, var in PAIRS:
        c = side(canon)
        v = side(var)
        # variant 计数要减去「variant 是 canonical 的前缀」的重复（如 萨林兰 ⊂ 萨林兰德）
        v = [r for r in v if r[3].count(var) > r[3].count(canon)] if canon.startswith(var) or var in canon else v
        print(f"### {name_en}  canon={canon}({sum(r[5] for r in c)} 次 / {len(c)} 叶)"
              f"  variant={var}({sum(r[5] for r in v)} 次 / {len(v)} 叶)")
        for repo, pack, p, cnv, env, n in c:
            if p.endswith(".name") or p.endswith(".tokenName"):
                print(f"    [canon-NAME] {repo}|{pack}|{p} = {cnv!r}  EN={env!r}")
        for repo, pack, p, cnv, env, n in v:
            gate = name_en in env
            tail = p.rsplit(".", 1)[-1]
            print(f"    [var {'GATE-OK ' if gate else 'GATE-FAIL'}] {repo}|{pack}|{p}  ({tail}, x{n})")
            if not gate:
                print(f"          EN={env[:160]!r}")
        print()


if __name__ == "__main__":
    main()
