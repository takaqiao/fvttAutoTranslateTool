# -*- coding: utf-8 -*-
"""第十九轮 Y6 新增的 7 条 `enricher_slot_gate` 断言的**唯一定义处**。

`--trial` 写出一个只含新条的规则文件（试跑／量数用）；
`--merge` 把它们并进 `5-其他内容/RESOLUTIONS.assertions.json`（同 id 覆盖，不重复追加）。

⚠ **并入之后，权威定义就是 RESOLUTIONS.assertions.json，不是这里。**
本轮并入后又往 `R-arcturel-arcturian-labels` 的 why 里补了回测口径与边界说明，
所以此处的 `why` 已经比库里那份短。再跑一次 `--merge` 会把那些补充**覆盖掉** ——
要改规则请直接改 RESOLUTIONS.assertions.json。本文件只保留下来当「这 7 条当初是怎么来的」的取证。
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(P, "5-其他内容", "RESOLUTIONS.assertions.json")

COMMON_WHY = (
    "**本条与另外 6 条 `enricher_slot_gate` 同族，共同的取证与判据说明写在 "
    "`R-arcturel-arcturian-labels` 的 why 里，先读那条。** 一句话：`split_blocks` 把 "
    "`@X[…]{标签}` 连花括号一起涂空，标签里的译名整条不进任何块级闸；本族断言按 "
    "**(动词, 目标)** 把两侧增强器配对，逐个「英文槽 ↔ 中文槽」比既定译名。")

RULES = [
 {
  "id": "R-arcturel-arcturian-labels",
  "title": "Arcturel/Arcturian 的**增强器标签级**闸：补块级闸涂空标签留下的 189 处无闸",
  "decision": "2026-08-15h（第十九轮 Y6）",
  "kind": "enricher_slot_gate",
  "why": "**这条是 `R-arcturel-arcturian-blocks` 自己 why 里量化过的那片盲区的正解。** "
         "`split_blocks` 把 `@X[…]{标签}` 连花括号一起涂空（取舍理由在脚本注释里，是对的），"
         "代价是标签里的译名整条不进块级闸。取证：全库中文「阿克图瑞尔|阿克图里安」共 **2218 处，"
         "其中 278 处（12.5%）落在增强器内**。第十八轮把这 278 处**逐处改错、每处重跑全部读库断言**，"
         "只有 **89 处**会让某条断言变红（且全是叶级间接覆盖：只有该叶里这一处是唯一带该词的地方时才响），"
         "**其余 189 处改错后全套依旧全绿**。第十九轮重做同一次回测，本条上线后 **189 → 4**"
         "（详见本条末尾「剩下的 4 处」）。⚠ **第十八轮原写「标签另有 R-arcturian-split 的 "
         "`{Arcturians}` 域 / R-arcturian-actor-card / scan_uuid_swap 看着」这句话经逐处变异回测证伪**："
         "`R-arcturian-actor-card` 只钉 `actors.Arcturian` 的 name/tokenName 四叶、那四叶里根本没有增强器，"
         "一处都盖不到；`scan_uuid_swap` 根本不在断言套里。**教训：why 里写「另有 X 闸看着」是可证伪断言，"
         "写之前必须逐处变异回测。** "
         "⚠ **与 `scan_label_vs_name` 分工明确、不是它的重复**：那条比的是「`@UUID{标签}` 的中文 ↔ "
         "**目标文档的中文 name**」，且只报「英文标签本来就等于目标英文 name」的；本条比的是"
         "**标签里的术语与既定译名**，标签本身允许与目标 name 不同。`{Arcturel Dives}` 译成"
         "「阿克图里安矿渊」时，scan_label_vs_name 一声不吭（英文标签本来就不等于目标 name），本条报红。 "
         "⚠ **配对不按出现序号，按 (动词, 目标) 分组后组内取序** —— 这是实测改的：按整叶序号配对时 "
         "**30650 对里 1388 对（4.5%）配歪**（中文「定语在前」把增强器整个搬位），造出 "
         "`EN=Arcturel/CN=杰夫赫尔家族`、`EN=Arcturian/CN=奥尔丹`、`EN=Arcturian Liquor/CN=烧瓶` "
         "这四类**幻影缺陷**；改按目标分组后全库 30650 个增强器 **0 个配不上**，四类幻影同时消失。 "
         "⚠ **槽位不只花括号标签**：`@Embed[… label=\"…\" readaloud=\"…\"]` 的这两个参数里塞的是"
         "**整段朗读正文**（实测 46+46 处，最长 300+ 字），同样被 split_blocks 整段涂空，是本洞里最大的单块。 "
         "⚠ **中文独有槽**（英文侧是裸 `@UUID`、Foundry 渲染目标名，中文补了标签；实测 582 个）"
         "退回反向闸：中文槽里的类必须在「本叶英文 ∪ 同一目标在全库别处的英文标签」里出现过。"
         "**正向回退试过，当场 70 条假阳性**（一叶英文里提到 Arcturian，不代表这叶每个中文标签都得带族名）；"
         "只认本叶英文也不行，会误报 `{阿克图里安小饰品}`（本叶英文无 Arctur 字样，但同一个 RollTable "
         "全库别处 28 次写着 `{Arcturian Trinkets}`）。 "
         "⚠ **剩下的 4 处**（回测里改错后仍然全绿的）：`Delivered from Evil`／`The Ballad of Dres` 两页的"
         "中文独有槽，孪生共 4 槽 —— 它们的中文标签整串等于目标文档名，改错后反向闸仍然放行，"
         "因为「同一目标别处的英文标签」两类都出现过。要抓这 4 处得引入目标文档的**中文 name**作仲裁，"
         "那正是 `scan_label_vs_name` 的判据，本条不重复造。 "
         "⚠ 大小写：**IGNORECASE**。`Arcturel`／`Arcturian` 是专名，本库没有同形的小写普通词"
         "（与 `R-dives-mine` 的小写 `dives`＝动词正相反）；上游的五种拼写变体 "
         "`Acturel`/`Arctural`/`Arcurel`/`Arturel`/`Acturian` 一并进类，不进就是假阳性。 "
         "⚠ `Arcturelian` 派生自 `Arcturel` 归 E 类（2026-08-15 主控补裁），本条按集合判不按序列判，"
         "所以不必像 `R-arcturel-arcturian-blocks` 那样为它单列 `L` 类钉死搭配。 "
         "⚠ `except_slots` 的 2 槽是**已裁的合法例外**，与 `R-arcturel-vs-arcturian` 的 "
         "`except_paths` 同一处、同一理由：上游标签与目标错配（英文标签写 `{Lower Arcturel Mine Effect}`，"
         "它指向的 RollTable 名字却是 `The Dives Mine Effects`），中文跟着**目标名**译成「矿渊矿井效应」。"
         "只在 ember.crucible-adventure 一包，不是漏钉孪生。",
  "case_sensitive": False,
  "forbid_absent": True,
  "cn_only_leaf_fallback": True,
  "en_tokens": [
   {"re": "\\b(?:Arcturel|Acturel|Arctural|Arcurel|Arturel)\\w*", "cls": "E"},
   {"re": "\\b(?:Arcturian|Acturian)s?\\b", "cls": "I"}
  ],
  "cn_tokens": [{"re": "阿克图瑞尔", "cls": "E"}, {"re": "阿克图里安", "cls": "I"}],
  "except_slots": [
   {"path": "actors.The Device.items.Powered Effect.description",
    "en": "Lower Arcturel Mine Effect", "cn": "矿渊矿井效应"},
   {"path": "actors.The Device.items.Powered Effect.actions.poweredEffect.description",
    "en": "Lower Arcturel Mine Effect", "cn": "矿渊矿井效应"}
  ],
  "min_leaves": 5000, "min_slots": 20000, "min_gated": 250, "max_unpaired": 0
 },
 {
  "id": "R-dives-labels",
  "title": "the Dives = 矿渊 的**增强器标签级**闸",
  "decision": "2026-08-15h（第十九轮 Y6；裁决本体见 R-dives-mine，2026-08-16d）",
  "kind": "enricher_slot_gate",
  "why": COMMON_WHY +
         " 本条守的是 `R-dives-mine`（the Dives＝矿渊）在标签里的那一半：`R-dives-mine` 是**叶级**的，"
         "一叶里提到 5 次只错标签那 1 次它不会响。实测英文槽命中 28 个（`{The Dives}`／"
         "`{Arcturel Dives}`／`{The Dives Mine Effects}`），28/28 含「矿渊」。 "
         "⚠ **必须 case_sensitive，与 R-dives-mine 同一理由且这条理由是逐条单判出来的**："
         "小写 `dives` 是动词（`Jasper dives for cover`），而本条的槽位里**包含 readaloud 整段朗读正文**，"
         "动词形态在那里出现的概率比在标签里高得多，按 IGNORECASE 判会直接造出假阳性。 "
         "⚠ **不开 `forbid_absent`**：中文槽里出现「矿渊」而英文槽没有 `Dives` 是**合法**的 —— "
         "`{Lower Arcturel Mine Effect}`→「矿渊矿井效应」是跟着目标名译的（见 R-arcturel-arcturian-labels "
         "的 except_slots）。 ⚠ scope 只有 ember：crucible 系统侧没有这个地名。",
  "case_sensitive": True,
  "scope": ["ember"],
  "en_tokens": [{"re": "\\bDives\\b", "cls": "D"}],
  "cn_tokens": [{"re": "矿渊", "cls": "D"}],
  "min_leaves": 4500, "min_slots": 19000, "min_gated": 24, "max_unpaired": 0
 },
 {
  "id": "R-kessia-labels",
  "title": "Kessia=凯西亚 / Kessian=凯西安 的**增强器标签级**闸",
  "decision": "2026-08-15h（第十九轮 Y6；裁决本体见 R-kessia-vs-kessian，2026-08-16e）",
  "kind": "enricher_slot_gate",
  "why": COMMON_WHY +
         " 本条是「专名 vs 文化形容词」四组二分里 Kessia 那组的标签级版本 —— "
         "`R-kessia-vs-kessian` 的 why 自陈「叶级判据分不出**一叶之内**某一处该写专名还是写形容词」，"
         "标签正是「一处」的最小单位。实测英文槽命中 52 个（`{Kessia}` 26 · `{Kessian}` 18 · "
         "`{Kessians}` 6），除登记的 1 槽（孪生 2 处）外全部一致。 "
         "⚠ `except_slots` 的 `Deities.pages.Sunalin.text` 是**已裁的合法形态不是欠账**："
         "英文 `{Kessian} continent`、中文「凯西亚大陆」—— `R-kessia-vs-kessian` 的 why 里"
         "**逐字点名** `Kessian continent` 属于「散文里作定语表来源时写凯西亚…是对的」那一支"
         "（凯西亚商人＝来自凯西亚的商人）。孪生两包各一槽。 "
         "⚠ 大小写 IGNORECASE：`Kessia`/`Kessian` 是专名，无小写同形普通词。",
  "case_sensitive": False,
  "en_tokens": [{"re": "\\bKessians?\\b", "cls": "KI"}, {"re": "\\bKessia\\b", "cls": "KE"}],
  "cn_tokens": [{"re": "凯西安", "cls": "KI"}, {"re": "凯西亚", "cls": "KE"}],
  "except_slots": [
   {"path": "journals.Deities.pages.Sunalin.text", "en": "Kessian", "cn": "凯西亚"}
  ],
  "min_leaves": 5000, "min_slots": 20000, "min_gated": 45, "max_unpaired": 0
 },
 {
  "id": "R-ordain-labels",
  "title": "Ordain=奥尔丹 / Ordani=奥尔达尼 的**增强器标签级**闸",
  "decision": "2026-08-15h（第十九轮 Y6；裁决本体见 R-ordain-vs-ordani，2026-08-16e）",
  "kind": "enricher_slot_gate",
  "why": COMMON_WHY +
         " 「专名 vs 文化形容词」四组二分里 Ordain 那组的标签级版本，与 `R-kessia-labels` 同型。"
         "实测英文槽命中 348 个，348/348 一致，零豁免。 "
         "⚠ 大小写 **IGNORECASE**，与 `R-ordain-vs-ordani` 同一理由（上游有 8 叶把城名写成小写 "
         "`ordain`，全是城名不是动词）—— 与同族的 `R-dives-labels` 必须 case_sensitive **正相反**，"
         "**每条闸都要单独判一次「大小写是不是判据的一部分」，没有全局默认可抄**。 "
         "⚠ `\\bOrdani\\w*` 写在 `\\bOrdain\\w*` 前面：交替式最左优先，顺序即优先级。"
         "`Ordan's Key` 两条都匹配不到（词边界不成立），不必也不该动它。",
  "case_sensitive": False,
  "en_tokens": [{"re": "\\bOrdani\\w*", "cls": "OI"}, {"re": "\\bOrdain\\w*", "cls": "OE"}],
  "cn_tokens": [{"re": "奥尔达尼", "cls": "OI"}, {"re": "奥尔丹", "cls": "OE"}],
  "min_leaves": 5000, "min_slots": 20000, "min_gated": 300, "max_unpaired": 0
 },
 {
  "id": "R-shard-god-labels",
  "title": "Shard God 三分（之神／诸神／女神）的**增强器标签级**闸",
  "decision": "2026-08-15h（第十九轮 Y6；裁决本体见 R-shard-god，2026-08-15e）",
  "kind": "enricher_slot_gate",
  "why": COMMON_WHY +
         " `R-shard-god` 的 why 写着它两处判据边界，本条补的是其中一处："
         "叶级的负向先行断言（只查「整叶只含单数／只含复数」的叶）把**同叶单复数并存的 70 叶**整叶排除，"
         "而标签是逐处的、没有这个问题。实测英文槽命中 115 个：`{Shard Gods}` 75→碎片诸神 · "
         "`{Shard God}` 28→碎片之神 · `{Shard Goddess}` 6→碎片女神 · 小写 `{shard god}` 2 · "
         "带尾空格 2 · `{Shard God Devotee}` 2，**115/115 一致**。 "
         "⚠ 这里能对单复数逐处上闸，而块级的 `R-shard-god-blocks` 明确**不把单复数进类**（那里 779 块"
         "有 46 块不齐、逐条看过全是合法中文，是判据不成立）—— 差别在于**标签是专名式的短串**，"
         "中文不会把 `{Shard Gods}` 铺开成「三位火焰之碎片之神和四位战斗之碎片之神」。"
         "**同一个术语在不同粒度上判据强度可以不同，这一条要记住。** "
         "⚠ 类的顺序 Goddess → Gods → God，交替式最左优先；中文侧同理女神 → 诸神 → 之神。 "
         "⚠ 大小写 IGNORECASE：库里实有小写 `{shard god}` 2 处，按 case_sensitive 判会漏看。",
  "case_sensitive": False,
  "en_tokens": [
   {"re": "\\bShard Goddess(?:es)?\\b", "cls": "GS"},
   {"re": "\\bShard Gods\\b", "cls": "GP"},
   {"re": "\\bShard God\\b", "cls": "G1"}
  ],
  "cn_tokens": [
   {"re": "碎片女神", "cls": "GS"}, {"re": "碎片诸神", "cls": "GP"}, {"re": "碎片之神", "cls": "G1"}
  ],
  "min_leaves": 5000, "min_slots": 20000, "min_gated": 100, "max_unpaired": 0
 },
 {
  "id": "R-moon-epithet-labels",
  "title": "三个月亮尊号（空洞／焦灼／风暴）的**增强器标签级正向闸**",
  "decision": "2026-08-15h（第十九轮 Y6；裁决本体见 R-moon-hollow/charred/tempest，2026-08-15g）",
  "kind": "enricher_slot_gate",
  "why": COMMON_WHY +
         " 三条月亮裁决现在只有 `cn_absent`（守废写法不回潮），**一条正向闸都没有** —— "
         "`R-moon-hollow` 的 why 自己写着「抓不到上游换了尊号而中文没跟」。本条在标签这一格补上正向："
         "英文槽出现 `Hollow/Charred/Tempest Moon` → 中文槽必须是对应的空洞／焦灼／风暴之月。"
         "实测英文槽命中 12 个（`{Aura - The Hollow Moon}` 4 · `{Ragen - The Charred Moon}` 4 · "
         "`{Mayis - The Tempest Moon}` 4），12/12 一致。 "
         "⚠ 只有 12 个槽，`min_gated` 定在 10 —— **这条的反空转护栏比同族其它条紧得多**，"
         "因为基数小、正则一写坏就直接归零而不会「掉一点」。 "
         "⚠ 另三个月亮（`Shattered`／`Blasted`／`Spirit`）标签里也有（破碎／爆裂／灵魂之月），"
         "但 §8 只成组裁了三个，本条只钉已裁的三个 —— **判据不替裁决做主**。",
  "case_sensitive": False,
  "en_tokens": [
   {"re": "\\bHollow Moon\\b", "cls": "H"},
   {"re": "\\bCharred Moon\\b", "cls": "C"},
   {"re": "\\bTempest Moon\\b", "cls": "T"}
  ],
  "cn_tokens": [
   {"re": "空洞之月", "cls": "H"}, {"re": "焦灼之月", "cls": "C"}, {"re": "风暴之月", "cls": "T"}
  ],
  "min_leaves": 5000, "min_slots": 20000, "min_gated": 10, "max_unpaired": 0
 },
 {
  "id": "R-token-rank-labels",
  "title": "Token=指示物 / Rank N=阶位 的**增强器标签级正向闸**",
  "decision": "2026-08-15h（第十九轮 Y6；裁决本体见 R-token-foundry-ui 2026-08-15f 与 R-rank-sense-compendium）",
  "kind": "enricher_slot_gate",
  "why": COMMON_WHY +
         " 两个术语合一条，因为两者在标签里的实测面都只有 4 个槽，各自单列一条的"
         "`min_gated` 会低到失去反空转意义。 "
         "① **Token**：`R-token-foundry-ui` 是 `cn_absent`，只守「令牌／代币」不回潮，"
         "**没有任何正向闸**要求 `Token` 一定译成「指示物」。本条在标签里补上：实测 "
         "`{Dynamic Token}`→「动态指示物」4 槽，4/4。 "
         "② **Rank**：`R-rank-sense-compendium`／`R-rank-sense-blocks` 都要先做机制义／普通名词义分类，"
         "而标签里两种义项都真实存在 —— `{Rank 1 Soulmark}`／`{Cora Attunement (Rank IV)}` 是机制义（4 槽），"
         "而 readaloud 槽里的 `join the ranks of the benevolent Sindaric sages` 是普通名词义（4 槽，"
         "中文正确地写成「行列」不是「阶位」）。**本条只钉形态明确的机制义** "
         "`\\bRank\\s+(?:\\d+|[IVXLC]+)\\b`（Rank 后面直接跟数字或罗马数字），"
         "普通名词义那 4 槽根本不进类，不判 —— **收窄判据，不是放宽闸**。 "
         "⚠ 不开 `forbid_absent`：中文槽里出现「指示物」「阶位」而英文槽没有对应词的情形没核过，"
         "开之前要先在当前库实跑一遍。",
  "case_sensitive": False,
  "en_tokens": [
   {"re": "\\bTokens?\\b", "cls": "TK"},
   {"re": "\\bRank\\s+(?:\\d+|[IVXLC]+)\\b", "cls": "RK"}
  ],
  "cn_tokens": [{"re": "指示物", "cls": "TK"}, {"re": "阶位", "cls": "RK"}],
  "min_leaves": 5000, "min_slots": 20000, "min_gated": 6, "max_unpaired": 0
 },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.trial:
        p = os.path.join(HERE, "new_rules.trial.json")
        json.dump({"meta": {"note": "第十九轮 Y6 新条试跑"}, "assertions": RULES},
                  open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("写出", p, len(RULES), "条")
    if a.merge:
        d = json.load(open(RES, encoding="utf-8"))
        by = {r["id"]: i for i, r in enumerate(d["assertions"])}
        added = replaced = 0
        for r in RULES:
            if r["id"] in by:
                d["assertions"][by[r["id"]]] = r
                replaced += 1
            else:
                d["assertions"].append(r)
                added += 1
        json.dump(d, open(RES, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"并入 {RES}：新增 {added} 条 / 覆盖 {replaced} 条 / 现共 {len(d['assertions'])} 条")


main()
