"""第十六轮：把英文 `rank` 的**游戏机制义**统一成「阶位」，普通名词义原样不动。

## 为什么需要这个探针，而不能 unify_terms.py 一把梭

英文闸 `\branks?\b`（**必须不区分大小写** —— 大写 `Rank` 只有 34 处，小写 `rank` 有 200+，
第一版漏了小写，把 `crucible.rules` 的 Skill Checks 误判成「EN×1 : CN×6 计数不等」；
同样的坑 `split_region_area_map.py` 顶部已经警告过一次，我又踩了）下，
库内中文是 **等级 223 叶 : 阶位 47 叶**。但 `rank` 在本库里是**两个词**：

- **游戏机制义** —— `You gain the Novice rank in the Arcana skill` / `Attunement Rank 1` /
  `Soulbound Rank` / `training rank` / `Bonus Rank Scale`。§8 2026-08-14d 已裁 **`Rank`=阶位**
  （与 `Tier`=阶 / `level`=等级 三分）。
- **普通名词义** —— `denote their civic rank`（公民地位）/ `within the ranks of Shard Gods`（行列）/
  `stripped of his rank within the order`（教团职位）/ `the social rank of individuals`（社会地位）/
  `attained the full rank of Sage`（位阶）。**这些不归那条裁决管。**

所以做法沿用本项目在 `@UUID` 标签与 `Region Map`/`Area Map` 上验证过的**逐位对齐**：
先把每一处英文 `rank` 按上下文分类成 GAME / COMMON，再把中文里「等级」的出现顺序与
**GAME 那些**配对。只有两侧计数相等时才动手；不等的整叶跳过并列出来人看。

⚠ 逐位对齐是**筛子不是判据**：计数相等不等于语序也对得上（中文会把从句提前）。
所以跳过的要人看，落下的也要抽样复核。
"""
import argparse
import json
import os
import re

RANK = re.compile(r"\branks?\b", re.I)
# 英文里同时出现 tier/level 时，中文的「等级」可能是在译 level，整叶跳过交人看
OTHER = re.compile(r"\b(tiers?|levels?)\b", re.I)
CN_TOKEN = "等级"
CN_WANT = "阶位"

# 机制义的判据：`rank` 前后 60 字符内出现这些
# ⚠ `Rank\s*\d` 与 `Bonus…Scale` 这两条是踩过坑补的：表格里的裸「Rank 1」～「Rank 5」
# 和被 `<td>` 拆开的「Bonus | Rank | Scale」，按整串匹配一个都命中不到。
GAME = re.compile(
    r"(Novice|Journeyman|Adept|Master|Untrained|training|skill|Attunement|Attuned"
    r"|Soulbound|Soulmark|talent|progress|superior to|exhaustion|Scale"
    r"|Rank\s*\d|\bBonus\b)",
    re.I,
)
# 普通名词义的判据（优先级高于 GAME —— 命中就判 COMMON）
COMMON = re.compile(
    r"(civic|social|nobil|noble|militar|clerical|within the order|of the order"
    r"|stripped of|full rank of|ranks of|swell(ing)? the ranks|rank[- ]and[- ]file"
    r"|hierarch|station|office|through the ranks|within [^.]{0,30}\branks\b"
    r"|beyond its ranks|rose to ranks|\branks (after|and)\b|rank as an? )",
    re.I,
)

_TAG = re.compile(r"<[^>]+>")


def classify(en, m):
    # ⚠ 必须先剥标签再判：`Bonus Rank Scale` 在原文里是被 `<td>` 拆开的三格，
    # 不剥标签的话上下文窗口里根本凑不出这个串（第一版 16 叶判 UNKNOWN 全栽在这）。
    ctx = _TAG.sub(" ", en[max(0, m.start() - 90): m.end() + 90])
    if COMMON.search(ctx):
        return "COMMON"
    if GAME.search(ctx):
        return "GAME"
    return "UNKNOWN"


# ── 逐叶人裁：探针判「计数不等」或「英文同时有 tier/level」而跳过的那些，人看过之后的定论 ──
# 键是叶路径后缀，值是 [(旧串, 新串, 预期出现次数 | None=不限)]。
# 预期次数对不上就整叶跳过并告警 —— 上游一改措辞，这张表就该重新人看，不能悄悄改错。
#
# 判过「本来就对、不动」的（**别再重查**）：
#   · A Sage Welcome —「等级」译的是 advancing them in level
#   · Railen / Veiled Chain — 译的是组织内部位阶（rank among the Ekat / rise to the rank）
#   · Grave Assignments —「等级」译的是 dungeon level
#   · Flame Guard — 分类器把「ranks depending on experience and skill」误判成 GAME（因为近旁有
#     skill 一词），实为火焰卫队的内部位阶，不改
#   · Sage Advice —「等级」译的是 advance in level
#   · Tethra Shùl.archetype / Necromancer — EN 的 `threat rank` 是散文写法，Crucible 的定名是
#     `Threat Level`＝威胁等级（见 Adversaries.Overview），所以「威胁等级」本就是对的
MANUAL = {
    "Players' Guide.pages.Attunement.text": [("等级", "阶位", 9)],
    "Players' Guide.pages.Death and Soulbound.text": [("等级", "阶位", 3)],
    # 这一叶 6 处「等级」里只有 2 处译的是 rank，其余 4 处是 character level / Challenge Rating
    "Gamemaster's Guide.pages.Death and Soulbound.text": [("等级晋升", "阶位晋升", 1),
                                                          ("适当等级的", "适当阶位的", 1)],
    "Gamemaster's Guide.pages.Patch 0.5.2.text": [("等级", "阶位", 6)],
    "Gamemaster's Guide.pages.Patch 0.5.5.text": [("等级", "阶位", 2)],
    # Bonus|Rank|Scale 表头那一格；同叶另一处「角色等级」译的是 character level，不动
    "Character Mechanics.pages.Talents.text": [("<p>等级</p>", "<p>阶位</p>", 1)],
    "Adversaries.pages.Overview.text": [("训练等级", "训练阶位", None), ("更高等级", "更高阶位", 1)],
    # EN 的 `levels of society` 中文没译成「等级」，所以唯一那处「等级」就是 Novice rank
    "Society Novice.description": [("等级", "阶位", 1)],
    # 三处全是 Soulbound rank；同叶的 `per Level` 中文写作「每级」，不受影响
    "entries.Soulbound.description": [("等级", "阶位", 3)],
}


def apply_manual(path, cn_val):
    """返回人裁后的新值；没有对应规则或预期次数对不上时返回 None。"""
    for suf, ops in MANUAL.items():
        if not path.endswith(suf):
            continue
        new = cn_val
        for old, rep, exp in ops:
            n = new.count(old)
            if exp is not None and n != exp:
                print(f"  ⚠ 人裁表失效：{suf[-40:]} 的「{old}」实测 {n} 处、表里写 {exp} 处 —— 整叶跳过，请重新人看")
                return None
            new = new.replace(old, rep)
        return new if new != cn_val else None
    return None


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, action="append")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats = {"changed": 0, "manual": 0, "skip_mismatch": 0, "skip_other": 0, "skip_unknown": 0, "clean": 0}
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
            for path, en_val in walk(en_doc):
                if not RANK.search(en_val):
                    continue
                cn_val = cn_map.get(path)
                if not cn_val or CN_TOKEN not in cn_val:
                    continue
                # 人裁表优先：探针判不了的那些，按人看过的定论改
                manual = apply_manual(path, cn_val)
                if manual is not None:
                    batch[path.removeprefix("entries.")] = manual
                    stats["manual"] += 1
                    continue
                if OTHER.search(en_val):
                    stats["skip_other"] += 1
                    review.append((repo[:1], fname, path, "英文同时有 tier/level", "-", "-"))
                    continue
                kinds = [classify(en_val, m) for m in RANK.finditer(en_val)]
                if "UNKNOWN" in kinds:
                    stats["skip_unknown"] += 1
                    review.append((repo[:1], fname, path, "有 rank 无法分类",
                                   str(len(kinds)), str(cn_val.count(CN_TOKEN))))
                    continue
                n_game = kinds.count("GAME")
                n_cn = cn_val.count(CN_TOKEN)
                if n_game == 0:
                    stats["clean"] += 1  # 全是普通名词义，中文的「等级」另有来源，不动
                    continue
                if n_game != n_cn:
                    stats["skip_mismatch"] += 1
                    review.append((repo[:1], fname, path,
                                   f"GAME {n_game} ≠ 等级 {n_cn}（另有 COMMON {kinds.count('COMMON')}）",
                                   str(n_game), str(n_cn)))
                    continue
                new = cn_val.replace(CN_TOKEN, CN_WANT)
                if new != cn_val:
                    batch[path.removeprefix("entries.")] = new
                    stats["changed"] += 1

            if batch:
                out = os.path.join(args.out_dir, f"rank.{fname}")
                json.dump(batch, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"{fname:36s} {len(batch):4d} 叶 -> {out}")

    print(f"\n改写 {stats['changed']} 叶 + 人裁 {stats['manual']} 叶 = {stats['changed'] + stats['manual']} 叶"
          f" · 跳过(计数不等) {stats['skip_mismatch']} "
          f"· 跳过(含 tier/level) {stats['skip_other']} · 跳过(无法分类) {stats['skip_unknown']} "
          f"· 全普通名词义不动 {stats['clean']}")
    if review:
        print("\n=== 需人看 ===")
        for r, f, p, why, a, b in review:
            print(f"  [{r}|{f[:26]}] {p[-64:]}  {why}")


if __name__ == "__main__":
    main()
