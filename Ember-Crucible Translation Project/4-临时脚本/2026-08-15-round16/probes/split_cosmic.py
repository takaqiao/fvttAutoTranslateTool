"""T-2：`Cosmic` / `Cosmological` / `Cosmology` → 宇宙；`Cosmos` → 寰宇（专名，一个都不动）。

## 判据

`Cosmos` 是余烬世界那个寰宇本身的**专名**（`余烬寰宇 Ember Cosmos` / `寰宇地图 Cosmos Map`），
保持「寰宇」；其余三个同词根的泛指词一律「宇宙」，锚点是页名「**宇宙观 Cosmology**」
（`Players' Guide.pages.User Interface.text` 里 `@UUID[...]{Cosmology}` → `{宇宙观}`，已对）。
`universe` / `multiverse` 本来就译「宇宙 / 多元宇宙」，与本条同向，一并作为对齐的锚。

## 闸：**只碰英文含 cosmic/cosmological/cosmology 的叶**

全库 432 叶含「寰宇」，其中英文含 `cosmolog*`/`cosmic*` 的是 265 叶；
**只含 `Cosmos` 的那 300 余叶一个都不许动**，本脚本的闸从源头保证了这一点
（`if not re.search(COSMO_ONLY, en)` 直接跳过）。同一叶两者并存的，靠下面的逐块对齐分开。

## 做法：按 HTML 标签切块 + 块内逐位配对

整叶计数太粗（一叶里 `Cosmos` 与 `cosmic` 常并存），所以按标签切块——标签是机械，
两侧逐字节相同，块数必然相等（实测 265 叶全部相等）。块内把英文出现按
S=`Cosmos` / U=`cosmic|cosmological|cosmology|universe|multiverse` 排成序列，
中文按 S=寰宇 / U=宇宙 排成序列，**长度相等才逐位配对**；配到 EN=U 而 CN=S 的位置，
就把那一处「寰宇」改成「宇宙」。长度不等、或落在 SKIP 表里的块，跳过并列出来人看。

⚠ 逐位配对是筛子不是判据：中文会调换语序。已知两处就是这么翻车的，见 SKIP。

## 英文闸带 re.IGNORECASE

本项目为漏这个栽过三次。这里 `Cosmos`/`cosmos`、`Cosmic`/`cosmic` 都大量存在，
`Cosmological Attunement` 与 `cosmological forces` 也大小写混用，闸必须不区分大小写；
S/U 的分类**只看词形不看大小写**（`cosmos` 一律 S，其余一律 U）。

## 扫描结论

265 叶闸下，逐块全对 204 叶；不对的 61 叶里，25 处是干净的「单个 cosmic → 单个寰宇」，
9 处是多词块但能逐位配对，其余 27 处人看过全部判定不动（理由记在 SKIP 里），
另有 1 处（Spectra）逐位替换会写出「宇宙对宇宙中…」这样的重复病句，走 MANUAL 改写。
"""
import argparse
import bisect
import collections
import json
import os
import re

COSMO_ONLY = re.compile(r"\bcosmolog\w*\b|\bcosmic\w*\b", re.I)
EN_TOK = re.compile(r"\bcosmos\b|\bcosmolog\w*\b|\bcosmic\w*\b"
                    r"|\bmultiverses?\b|\buniverses?\b", re.I)
CN_TOK = re.compile("寰宇|宇宙")
CN_CLS = {"寰宇": "S", "宇宙": "U"}

MASKP = [re.compile(r"@[A-Za-z]+\[[^\]]*\]"), re.compile(r"\[\[[^\]]*\]\]")]
TAG = re.compile(r"<[^>]+>")


def en_cls(x):
    return "S" if x.lower() == "cosmos" else "U"


def mask(s):
    """@UUID[...] 的目标段与 [[/cmd ...]] 涂成等长空格；花括号里的标签是散文，保留。"""
    for p in MASKP:
        s = p.sub(lambda m: " " * len(m.group()), s)
    return s


def toks(s, tok_re, cls):
    """返回 [(块号, match)]，块号 = 该位置之前已闭合的 HTML 标签数（与 TAG.split 的下标一致）。"""
    ms = mask(s)                       # 等长替换，偏移与原串一致
    ends = [m.end() for m in TAG.finditer(ms)]
    return [(bisect.bisect_right(ends, m.start()), m, cls(m.group()))
            for m in tok_re.finditer(ms)]


def nblocks(s):
    return len(TAG.split(mask(s)))


# ── 人看过、判定「本来就对，不动」的块（**别再重查**）─────────────────────────
SKIP = {
    ("Cosmos.pages.Attunement.contentOverview", 1):
        "语序调换：EN「thirteen known cosmological forces within the cosmos」→ "
        "CN「在寰宇之中，已知存在十三种宇宙学力量」——两个词都译对了，逐位配会配反",
    ("Cosmos.pages.Metaphysics.text", 24):
        "语序调换：EN「channeling the tangible cosmic forces and beings that populate the cosmos」→ "
        "CN「引导遍布寰宇之中的那些有形的宇宙力量与存在」——都译对了",
    ("Players' Guide.pages.Setting Overview.text", 5):
        "中文多出的「宇宙」来自 multiverse→多元宇宙；两处 Cosmos 都是寰宇，本来就对",
    ("Players' Guide.pages.Attunement.text", 341):
        "「Cosmological Attunement」被 <sup> 拆到两块，中文「某个宇宙同调」搬到了前一块，"
        "与 343 块合起来两侧相等",
    ("Players' Guide.pages.Attunement.text", 343):
        "同上",
    ("Players' Guide.pages.User Interface.text", 23):
        "英中语序相反（EN 先地图名后图标，CN 先图标后地图名），标签落在了相邻块。"
        "实际三个标签全对：World Map=世界地图 / Cosmos Map=寰宇地图 / Region Map=地区地图",
    ("Players' Guide.pages.User Interface.text", 25):
        "同上",
    ("The Winding Trail.pages.Sheltered Campsite.text", 270):
        "EN「the upcoming cosmological zenith」译作「即将到来的诸月同抵天顶」，"
        "是按世界观展开的意译，不是把 cosmological 写成了寰宇，不动",
    ("Introduction.pages.Foreword.text", 6):
        "EN「in an in-universe voice」→「以世界内角色的口吻」；in-universe 是元叙述用语，非本条术语",
    ("Deities.pages.Sha-Xotha.contentGamemaster", 4):
        "EN「A more in-universe name」→「在世界内更常见的称呼」，同上",
    ("Deities.pages.Outer Gods.contentGamemaster", 4):
        "同上",
    ("actors.Pale Whisperer.biography.private", 11):
        "EN「the cosmic fabric of the universe」中文合译成「宇宙的织构」一个词，没有寰宇误用",
    ("actors.Writhing Whisperer.biography.private", 11):
        "同上（「宇宙结构」）",
}

# ── 逐块人裁改写：逐位替换会写出病句的 ───────────────────────────────────────
MANUAL = {
    # EN「a direct cosmic response to the disparate forces of the universe」。
    # 逐位替换会得到「是宇宙对宇宙中诸多分异力量的直接回应」，重复且不通；改写成下面这句。
    "Deities.pages.Spectra.contentGamemaster": [
        ("她的诞生似乎是寰宇对宇宙中诸多分异力量的直接回应",
         "她的诞生似乎是宇宙对自身诸多分异力量的一次直接回应", 1)],
}


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def apply_manual(path, cn):
    for suf, ops in MANUAL.items():
        if not path.endswith(suf):
            continue
        new = cn
        for old, rep, exp in ops:
            n = new.count(old)
            if n != exp:
                print(f"  ⚠ 人裁表失效：{suf[-44:]} 的「{old[:12]}…」实测 {n} 处、表里写 {exp} 处"
                      f" —— 整叶跳过，请重新人看")
                return None
            new = new.replace(old, rep)
        return new if new != cn else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, action="append")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    st = collections.Counter()
    review = []
    for repo in args.repo:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        if not os.path.isdir(en_dir):
            continue
        for fname in sorted(os.listdir(en_dir)):
            if not fname.endswith(".json"):
                continue
            cn_path = os.path.join(cn_dir, fname)
            if not os.path.exists(cn_path):
                continue
            en_doc = json.load(open(os.path.join(en_dir, fname), encoding="utf-8"))
            cn_map = dict(walk(json.load(open(cn_path, encoding="utf-8"))))

            batch = {}
            for path, en in walk(en_doc):
                if not COSMO_ONLY.search(en):      # 只含 Cosmos 的叶：一个都不碰
                    continue
                cn = cn_map.get(path)
                if not cn:
                    continue
                st["gate"] += 1

                man = apply_manual(path, cn)
                if man is not None:
                    batch[path.removeprefix("entries.")] = man
                    st["manual"] += 1
                    continue

                if nblocks(en) != nblocks(cn):
                    st["shape"] += 1
                    review.append((fname, path, -1, "标签块数两侧不等", "", ""))
                    continue

                et = toks(en, EN_TOK, en_cls)
                ct = toks(cn, CN_TOK, lambda g: CN_CLS[g])
                eby = collections.defaultdict(list)
                cby = collections.defaultdict(list)
                for b, m, k in et:
                    eby[b].append(k)
                for b, m, k in ct:
                    cby[b].append((m, k))

                fixes = []          # (原串起点, 原串终点) -> 宇宙
                clean = True
                for b in sorted(set(eby) | set(cby)):
                    ee = eby.get(b, [])
                    cc = [k for _, k in cby.get(b, [])]
                    if ee == cc:
                        continue
                    clean = False
                    if any(path.endswith(s) and b == i for (s, i) in SKIP):
                        st["skip_ok"] += 1
                        continue
                    if len(ee) != len(cc):
                        st["review"] += 1
                        review.append((fname, path, b, "块内词数不等",
                                       "".join(ee), "".join(cc)))
                        continue
                    bad = [(x, y) for x, y in zip(ee, cc) if x != y]
                    if any(x == "S" and y == "U" for x, y in bad):
                        # 中文写成「宇宙」而英文是专名 Cosmos —— 反向缺陷，不在本条裁决内，报出来
                        st["review"] += 1
                        review.append((fname, path, b, "反向：EN=Cosmos 而 CN=宇宙",
                                       "".join(ee), "".join(cc)))
                        continue
                    for (m, k), want in zip(cby[b], ee):
                        if k == "S" and want == "U":
                            fixes.append(m.span())
                    st["blocks"] += 1
                if fixes:
                    new = cn
                    for s0, s1 in sorted(fixes, reverse=True):
                        assert new[s0:s1] == "寰宇", (path, new[s0:s1])
                        new = new[:s0] + "宇宙" + new[s1:]
                    batch[path.removeprefix("entries.")] = new
                    st["changed"] += 1
                elif clean:
                    st["clean"] += 1

            if batch:
                out = os.path.join(args.out_dir, f"r16-T2-cosmic.{fname}")
                json.dump(batch, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"{fname:32s} {len(batch):3d} 叶 -> {out}")

    print(f"\n闸下 {st['gate']} 叶 · 逐块全对 {st['clean']} 叶 · 改写 {st['changed']} 叶"
          f"（共 {st['blocks']} 块）· 人裁改写 {st['manual']} 叶 · 已判定不动 {st['skip_ok']} 块"
          f" · 待人看 {st['review']} 块 · 标签结构异常 {st['shape']} 叶")
    if review:
        print("\n=== 需人看 ===")
        for f, p, i, why, a, b in review:
            print(f"  [{f[:24]}] {p[-56:]} 块{i} {why}  EN={a} CN={b}")


if __name__ == "__main__":
    main()
