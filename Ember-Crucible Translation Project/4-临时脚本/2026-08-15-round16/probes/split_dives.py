"""T-1：把散文里 `the Dives` / `Arcturel` / `Arcturian` 三个词的中文写法与英文逐位对齐。

## 判据

§8 已裁：`the Dives`=矿渊（区）· `Arcturel`=阿克图瑞尔（城）· `Arcturian`=阿克图里安（定语／
文化标签）／阿克图里安人（指人的名词）· `Arcturel Dives`=阿克图瑞尔矿渊（**整体专名，不拆**）。
name 侧 8 叶本来就对（矿渊 The Dives / 阿克图瑞尔矿渊 Arcturel Dives ×2 / 地表：矿渊 /
过渡：抵达矿渊 / 过渡：下行至矿渊 / 黑暗：矿渊 / 矿渊矿井效应），本脚本不碰它们。

## 为什么不能整叶替换，也不能整叶计数

同一叶里三个词同时出现，最多的一叶（`An Old Friend.pages.Unhappy Accidents.text`）英文 33 处。
整叶计数会被两种东西糊掉：中文的语序调换（「in the Dives below Arcturel」→「阿克图瑞尔下方
矿渊中」），和中文的代词还原（「which itself lies a mile below」→「而矿渊本身又在…」）。

所以本脚本**按 HTML 标签切块再逐块对齐**：标签本身是机械，两侧逐字节相同，因此
`TAG.split()` 出来的块数两侧必然相等（实测 51 叶全部相等，一叶没漏）。块比整叶细得多，
剩下的假阳性就只有「同一块内语序调换」和「跨 <sup>/<span> 边界搬动的从句」两类，量小到能人看。

## 英文闸：这里**只有 Dives 一个词区分大小写**，其余全 IGNORECASE

⚠ 本项目为漏 `re.IGNORECASE` 栽过三次，本脚本第一版又栽了第四次：
`Storming the Consortium.text` 的 `Any other arcturians in the area` 是小写，
大小写敏感的 `\bArcturians?\b` 漏掉它，于是把一处**完全正确**的「阿克图里安人」报成了缺陷。
`Dives` 之所以必须大小写敏感，是因为小写 `dives` 是动词，且已**逐处枚举确认**：
全库 `(?i)\bdives\b` 命中 112 叶，其中 102 叶含大写 `Dives`，另 10 叶只有小写，
逐处看过全是动词（「a tiny red-beaked bird which dives down」「the tar gripping their legs
dives back into the earth」「successfully dives for cover」「a sleeping Leviathan and it dives
into the ocean」「the loremistress dives back into her book」）；且**没有任何一叶同时含大小写**，
所以大小写就是一条干净的分界线，不是偷懒。

英文闸另外要覆盖上游三处拼写事故（不改英文，只是别让它们伪装成缺陷）：
  · `Acturel`（漏一个 r，`A Peculiar Encampment.text`「the people of Acturel」）
  · `Arctural`（`Poolside Predicaments.text`「her superiors in Arctural」）
  · `DivesArcturel`（缺空格，`Chessmen Homecoming.text`；主控已裁：照英文原样理解、中文写对即可）

## 扫描结论：51 条路径 ×2 包 = 102 叶，逐块核完只有 2 处真缺陷

其余 9 处块级不符**全部人看过，全部判定不动**，理由记在 KNOWN_OK 里。别再重查。
"""
import argparse
import collections
import json
import os
import re

# ── 英文闸 ───────────────────────────────────────────────────────────────────
# (?i:...) 内联大小写不敏感；裸 \bDives\b 保持大小写敏感（判据见 docstring）
EN_TOK = re.compile(
    r"(?i:\bDivesArcturel\b|\bArcturel\s+Dives\b)"
    r"|\bDives\b"
    r"|(?i:\bArcturians?\b|\bArcturel\b|\bActurel\b|\bArctural\b)")
CN_TOK = re.compile("阿克图瑞尔矿渊|阿克图里安人|阿克图里安|阿克图瑞尔|矿渊")
CN_CLS = {"阿克图瑞尔矿渊": "AD", "阿克图里安人": "I", "阿克图里安": "I",
          "阿克图瑞尔": "E", "矿渊": "D"}

MASKP = [re.compile(r"@[A-Za-z]+\[[^\]]*\]"), re.compile(r"\[\[[^\]]*\]\]")]
TAG = re.compile(r"<[^>]+>")


def en_cls(x):
    xl = x.lower()
    if xl == "divesarcturel":
        return "DE"
    if "dives" in xl and xl.startswith("arcturel"):
        return "AD"
    if x == "Dives":
        return "D"
    if xl.startswith("arcturian"):
        return "I"
    return "E"


def mask(s):
    """把 @UUID[...] 的目标段与 [[/cmd ...]] 整条涂成等长空格。

    ⚠ 必须涂：`[[/culture Arcturian]]` 的方括号内部照抄英文、中文侧没有对应词，
    不涂的话 `Rock Bottom.text` 会凭空多出一个 EN 侧 I，整叶报错。
    `@UUID[...]{标签}` 只涂方括号，花括号里的标签是散文、要参与对齐。
    """
    for p in MASKP:
        s = p.sub(lambda m: " " * len(m.group()), s)
    return s


def blocks(s):
    return TAG.split(mask(s))


# ── 逐叶人裁 ─────────────────────────────────────────────────────────────────
# 键是叶路径后缀，值是 [(旧串, 新串, 预期出现次数)]；次数对不上就整叶跳过并告警。
MANUAL = {
    # EN「journeys to the sinkhole city of Arcturel and then down to the Dives district」
    # 中文把「down to the Dives district」整句漏译了 —— 这是全库唯一一处 EN 有 Dives
    # 而 CN 一个「矿渊」都没有的叶。「矿渊城区」沿用同库 A Troubled Tradeway.summary 的写法。
    "The Book Of Tales.pages.The Crystal Quarryman.overview": [
        ("队伍前往天坑之城阿克图瑞尔，去了解佐迪·特拉斯克的故事。",
         "队伍前往天坑之城阿克图瑞尔，再下行至矿渊城区，去了解佐迪·特拉斯克的故事。", 1)],
    # EN「Alternately, a nearby bystander could have something to say」—— 是「附近的」，
    # 不是「矿渊的」。这处「矿渊」在英文里没有依据，是凭空多出来的地名。
    "Glitter in the Dark.pages.Presenting the Evidence.text": [
        ("或者，矿渊的一名旁观者也可能说出一些话", "或者，附近的一名旁观者也可能说出一些话", 1)],
}

# ── 人看过、判定「本来就对，不动」的块（**别再重查**）─────────────────────────
# 键：(叶路径后缀, 块号)；值：不动的理由。
KNOWN_OK = {
    ("Arctus Plateau Gazetteer.pages.Arcturel.text", 255):
        "中文多出的「人们常说阿克图瑞尔是他们的衣食之本」译的是英文的 the city，不是多译的地名",
    ("Arctus Plateau Gazetteer.pages.Arcturel.text", 259):
        "英文「Politically, Arcturian is ruled by」是上游把城名误写成了族名（同段后文写的是 "
        "everything in Arcturel）；中文写「阿克图瑞尔由…统治」语义正确，不按字面改成阿克图里安",
    ("Arctus Plateau Gazetteer.pages.The Dives.text", 33):
        "语序调换：EN「Located nearly a mile below Arcturel, The Dives is…」→ CN「矿渊位于阿克图瑞尔下方…」",
    ("Arctus Plateau Gazetteer.pages.Rock Bottom.text", 9):
        "中文多出的第二个「矿渊」译的是 which itself（代词还原），不是多译的地名",
    ("An Old Friend.pages.Unhappy Accidents.text", 417):
        "语序调换：EN「distillery in the Dives below Arcturel」→ CN「阿克图瑞尔下方矿渊中那座…酿酒厂」",
    ("Glitter in the Dark.pages.Storming the Consortium.text", 78):
        "从句跨 <sup> 边界搬到了前一块（中文把「都需要一次成功的…检定」拆开），与 84 块合起来两侧相等",
    ("Glitter in the Dark.pages.Storming the Consortium.text", 84):
        "同上，与 78 块合起来两侧相等",
    ("Glitter in the Dark.pages.Overview.text", 15):
        "中文多出的「阿克图瑞尔下层」是对 the Dives 位置的补足说明，属加译语境，非术语错",
    ("Glitter in the Dark.pages.Chessmen Homecoming.text", 4):
        "英文是脏数据 the DivesArcturel（缺空格）；主控已裁照原样理解，中文「在矿渊」写对了，不动",
}

# 另有两处 EN 侧拼写事故已并进 EN_TOK，不再报警，记在这里备查：
#   · A Peculiar Encampment.text「the people of Acturel」→ CN 阿克图瑞尔民众（对）
#   · Poolside Predicaments.text「her superiors in Arctural」→ CN 阿克图瑞尔的上司（对）
# 还有一处 EN 标签过期但中文更对、也不动：
#   · crucible-adventure 的 The Device.items.Powered Effect.description
#     EN 标签写 {Lower Arcturel Mine Effect}，指向的 RollTable 现名 The Dives Mine Effects，
#     中文写「矿渊矿井效应」跟的是表的现名，比英文标签更准，不回改。


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
                print(f"  ⚠ 人裁表失效：{suf[-46:]} 的「{old[:14]}…」实测 {n} 处、表里写 {exp} 处"
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
                if not re.search(r"\bDives\b", en):
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

                eb, cb = blocks(en), blocks(cn)
                if len(eb) != len(cb):
                    # 标签结构两侧不同 —— 该由 scan_markup_drift 管，这里只报不改
                    st["shape"] += 1
                    review.append((fname, path, -1, f"标签块数 {len(eb)}≠{len(cb)}", "", ""))
                    continue
                clean = True
                for i, (e, c) in enumerate(zip(eb, cb)):
                    ee = [en_cls(m.group()) for m in EN_TOK.finditer(e)]
                    cc = [CN_CLS[m.group()] for m in CN_TOK.finditer(c)]
                    if ee == cc:
                        continue
                    clean = False
                    if any(path.endswith(s) and i == b for (s, b) in KNOWN_OK):
                        st["known_ok"] += 1
                        continue
                    st["review"] += 1
                    review.append((fname, path, i, "块内不对齐",
                                   "".join(ee), "".join(cc)))
                if clean:
                    st["clean"] += 1

            if batch:
                out = os.path.join(args.out_dir, f"r16-T1-dives.{fname}")
                json.dump(batch, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"{fname:32s} {len(batch):3d} 叶 -> {out}")

    print(f"\n闸下 {st['gate']} 叶 · 逐块全对 {st['clean']} 叶 · 人裁改写 {st['manual']} 叶"
          f" · 已判定不动 {st['known_ok']} 块 · 待人看 {st['review']} 块 · 标签结构异常 {st['shape']} 叶")
    if review:
        print("\n=== 需人看 ===")
        for f, p, i, why, a, b in review:
            print(f"  [{f[:24]}] {p[-56:]} 块{i} {why}  EN={a} CN={b}")


if __name__ == "__main__":
    main()
