#!/usr/bin/env python3
"""往 RESOLUTIONS.assertions.json 追加 R-readaloud-coverage，并更新 meta.updated。

⚠ 本规则里**没有任何正则**（本类型不吃 en_tokens/cn_tokens），所以不存在
「JSON 把 \\b 吃成退格符」那一类风险（R-catwalk 形态）。真有正则时不许走这条路，
要直接编辑文件。幂等：已存在就原地替换。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
RULES = os.path.join(P, '5-其他内容', 'RESOLUTIONS.assertions.json')

WHY = (
    "**这一条补的是第二十一轮留下的半个洞：`@Embed[… readaloud=\"…\"]` 里那整段朗读正文"
    "上一轮被纳入了 `scan_content_coverage` 的正文口径（EN 48 段 / 16966 字符、"
    "CN 48 段 / 5392 字符，落在 30 叶），却没有任何判据能判它。** "
    "第二十二轮逐条复核实测（现读复算同值）：① 这 48 段英文里**阿拉伯数字 0 个**，"
    "而 `scan_content_coverage` 的默认判据是数字多重集 ⇒ 对这 16966 字符的产出**必然是 0 命中**；"
    "② 把全库最长的一段中文朗读正文（`ember.adventure :: The Winding Trail / Dusktide Destruction`，"
    "311 字）整段删空，默认口径**仍报 0 条**；③ `--with-terms` 不成立 —— 把该叶单独喂进去，"
    "**干净时**就报缺定译 39 条、删空后 43 条，两侧都远远非零，没有判别力。"
    "也就是说那 16966 字符处在「已进统计口径、却无闸」的状态，比全盲更坏：口径行会让人以为查过了。 "
    "判据是**段级**四支，互相独立、各报各的（阈值取值与取证见 `a_enricher_text_coverage` 上方的块注释）："
    "**A 缺失**（中文槽整个没有 / 没有一个汉字，绝对判据）· "
    "**B 段级中英字符比** < 0.20（现读 48 段实测区间 [0.263, 0.406]、中位 0.313，"
    "与全库正文中英比中位数 0.31 同分布；阈值距实测最小值留 31% 富余）· "
    "**C 本段范围内的定译专名**（现读 36 段命中 102 条、缺 0 条 —— **范围限定在这一段是这一支唯一能用的理由**，"
    "放回叶级就是 39 条起步的噪声）· "
    "**D 句数**：中文句末标点数 ≥ floor(英文句数 × 0.75)（现读 48 段的中／英句数比只取 "
    "{1.0, 1.17, 1.25, 1.33, 2.0}，一段都不低于 1.0，留余量是为了不把「两句英文合成一句中文」判红）。 "
    "**双向回测（第二十二轮实测，探针落盘在 `4-临时脚本/2026-08-16-round22/`）**："
    "特异度 —— 当前库 48 段合计 **0 条违规**；灵敏度 —— 逐段把中文删空 **48/48 报**，"
    "逐段按字符砍掉后一半 **46/48 报**。 "
    "⚠ **判据边界（别把绿读成全覆盖）**：跑掉的那 2 个槽是同一段的孪生两份 —— "
    "`Gamemaster's Guide / Main Quest Overview` 的 EN 219 字 / **1 句**、CN 89 字"
    "（全库中文最丰的一段，比值 0.406），砍一半后 0.201 卡在阈值上方 0.001，英文只有 1 句所以句数支也够不着；"
    "同一段砍 51% 就报。把 `min_ratio` 提到 0.22 能收进来（实测 48/48），"
    "代价是距实测最小值只剩 20% 富余 —— 本轮**选择留富余**：这条闸守的是「整段没跟上」，不是「少译一句」。 "
    "⚠ 另有两支已经实现但**现在无从查起**，detail 行会把这个 0 明说出来："
    "含阿拉伯数字的英文段 **0 段**、含段内标记（`<tag>` / `@X[…]`）的英文段 **0 段**。"
    "上游哪天往朗读框里塞一句「DC 18」，它们自动开始有信号 —— 这个 0 是「没东西可查」，不是「查过了没问题」。 "
    "⚠ 与 `enricher_slot_gate` 那 7 条**不重复**：那 7 条只在英文槽命中某个已裁术语类时才有话说，"
    "一段没有已裁术语的朗读正文它们完全沉默；本条判的是「中文有没有跟上」，与具体术语无关，两者正交。 "
    "⚠ 不限 scope：现读 48 段全在 ember，crucible 侧一段都没有 —— 但闸下着，上游哪天往 crucible 加朗读框会自动纳入。"
)

RULE = {
    "id": "R-readaloud-coverage",
    "title": "增强器朗读正文（`@Embed[… readaloud=\"…\"]`）的**段级覆盖闸**：中文必须跟上英文",
    "decision": "2026-08-16（第二十二轮；上一轮 2026-08-15 第二十一轮只做到「纳入口径」，没有判据）",
    "kind": "enricher_text_coverage",
    "why": WHY,
    "slots": ["param:readaloud"],
    "glossary": "5-其他内容/glossary/glossary_ec.json",
    "min_en_chars": 60,
    "min_ratio": 0.20,
    "min_sentence_frac": 0.75,
    "min_leaves": 4500,
    "min_slots": 40,
    "min_gated": 40,
    "min_anchor_terms": 7000,
    "min_anchor_slots": 30,
    "min_anchor_hits": 80,
    "max_unpaired": 0,
}

NOTE = ("2026-08-16（第二十二轮：新增断言类型 `enricher_text_coverage` 与 1 条 "
        "`R-readaloud-coverage` —— 上一轮把 `@Embed[… readaloud=\"…\"]` 的整段朗读正文纳入了 "
        "`scan_content_coverage` 的正文口径，但**纳入 ≠ 判得到**：那 48 段英文里阿拉伯数字 0 个，"
        "默认的数字多重集判据必然 0 命中，实测把最长的一段中文删空也一声不响；`--with-terms` "
        "干净时就报 39 条、删空 43 条，两侧非零没有判别力。本轮改用**段级**四支判据"
        "（缺失／中英字符比／段内定译专名／句数），现读 48 段 0 违规，删空回测 48/48 报、"
        "删一半 46/48 报，判据边界写在 R-readaloud-coverage 的 why 里）｜ ")


def main():
    d = json.load(open(RULES, encoding='utf-8'))
    before = len(d['assertions'])
    ids = [a['id'] for a in d['assertions']]
    if RULE['id'] in ids:
        d['assertions'][ids.index(RULE['id'])] = RULE
        act = '替换'
    else:
        d['assertions'].append(RULE)
        act = '追加'
    if '第二十二轮' not in d['meta']['updated']:
        d['meta']['updated'] = NOTE + d['meta']['updated']
    json.dump(d, open(RULES, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{act} {RULE["id"]}；断言数 {before} -> {len(d["assertions"])}')


if __name__ == '__main__':
    main()
