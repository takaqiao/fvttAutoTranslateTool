"""把 Y2 的三条块对齐断言插进 RESOLUTIONS.assertions.json（各自紧跟其父条）。

幂等：已存在同 id 就整条覆盖，不重复插入。
"""
import json
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\5-其他内容\RESOLUTIONS.assertions.json"

SHARD = {
    "id": "R-shard-god-blocks",
    "title": "Shard God 的**叶内**闸：按块比「女神 vs 神」，并禁止块内整处漏译",
    "decision": "2026-08-15f（第十八轮 Y2：补 R-shard-god 自己在 why 里写死的覆盖洞）",
    "kind": "block_aligned_gate",
    "mode": "count_ge",
    "why": "R-shard-god 的 why 白纸黑字写着它的死角：三支 `en_gate` 用负向先行 `(?s)^(?!.*\\bShard Gods\\b).*\\bShard God\\b`，所以**同叶单复数并存的 70 叶结构上就不进闸**。本条按块级标签把中英切块后逐块判，那 70 叶连同其余叶一起被覆盖：闸下 **502 叶 / 694 有词块**（叶级三支合计 455 叶，且那 455 叶里没有一叶是混排的）。⚠ **先说清一件事：单／复数逐位对齐这个判据本身不成立，不是译文错。** 本轮实测按 S/P 逐位比，779 块里 46 块不齐，逐条看过全部是合法中文 —— 中文不标复数，且惯于把 `the Shard God {Aythorn}` 译成「碎片诸神之一的{艾索恩}」、把 `three Shard Gods of Fire and four of Battle` 拆成「三位火焰之碎片之神和四位战斗之碎片之神」、把 `Shard Gods are mortal ascendants` 的类名义写成「碎片之神」（这一条 R-shard-god 早已登记为 except_paths）。所以本条**不把单复数当类**，只留中文真正承载得住的那个区分：**女神 vs 神**。判据是「中文各类计数不得少于英文」＋「F 类（女神）反向存在闸」。⚠ **为什么是「可多不可少」而不是逐位相等**：放过的那一格是**代词还原**（英文 they/them → 中文点名「碎片诸神」），实测 13 块残差**无一例外是中文多、英文少**。抓得住的是：块内漏译一处、把「碎片女神」并进「碎片之神」、把某一支整体改名。实测 694 块**全部通过、零违规**。⚠ 大小写必须 IGNORECASE：英文对白里大量写小写 `a shard god` / `of all the shard gods`（`Planting a Seed` 一叶 4 处），大小写敏感版会造出 79 块假阳性 —— 本项目第五次栽在大小写上。⚠ 上游拼写事故 `Shards Gods`（Shard 上多一个 s，`Deities.pages.Shard Gods.text` 2 处）已进 token 表，不进就是假阳性",
    "leaf_gate": "\\bShards? God",
    "en_tokens": [
        {"re": "\\bShards? Goddess(?:es)?\\b", "cls": "F"},
        {"re": "\\bShards? Gods?\\b", "cls": "G"}
    ],
    "cn_tokens": [
        {"re": "碎片女神", "cls": "F"},
        {"re": "碎片诸神", "cls": "G"},
        {"re": "碎片之神", "cls": "G"}
    ],
    "backward_classes": ["F"],
    "min_leaves": 450,
    "min_blocks": 600,
    "max_shape_mismatch": 0
}

ARCT = {
    "id": "R-arcturel-arcturian-blocks",
    "title": "Arcturel（城）/ Arcturian（族）的**叶内**逐块对齐：抓叶内单处串行",
    "decision": "2026-08-15f（第十八轮 Y2：补 R-arcturel-vs-arcturian 自己在 why 里写死的覆盖洞）",
    "kind": "block_aligned_gate",
    "mode": "sequence",
    "why": "R-arcturel-vs-arcturian 的 why 写着它抓不到的东西：「叶级判据分不出**一叶之内**某一处该写城名还是族名 —— 96% 的叶两个词都有，这条能抓的是整支被改名，抓不到叶内单处串行」。本条把判据下沉到块：闸下 **905 叶 / 1714 有词块**，两侧类别序列必须逐位相等。**这一沉真的抓出了两处叶级判据看不见的缺陷**（见 except_blocks 里标 ESCALATED 的两条）。⚠ 上游有六种拼写变体，全部实测存在，不进 token 表就变成假阳性：城名 `Acturel`(漏 r) / `Arctural` / `Arcurel` / `Arturel`；族名 `Acturian`(漏 r，Golden Flats / Talei / Brevin 各 1) / `Arcturelian`（Chessman / Woven Construct / Constructed Companion，＝「Arcturel 的」的另一种构词，按**族名／文化形容词**归类，与 `Arcturian` 同支）。⚠ `Arcturel Dives` 是整体专名「阿克图瑞尔矿渊」，中文含「阿克图瑞尔」，两侧各出一个 E，天然对齐，不必单列一类。⚠ **判据边界**：① 富文本增强器的 `{标签}` 两侧都涂掉了（理由见脚本里 split_blocks 的注释），标签另由 R-arcturian-split 的 `{Arcturians}` 域 / R-arcturian-actor-card / scan_uuid_swap 看着；② 逐位相等对**语序调换**敏感，实测 1714 块里只有 1 块是这一类（Vartholomew Chess），已登记；③ 中文的**代词化**（第三次提到时写「它」）也会不齐，实测 1 块，已登记",
    "leaf_gate": "\\b(?:Arcturel|Arcturel?ians?|Arcturians?|Acturel|Acturians?|Arctural|Arcurel|Arturel)\\b",
    "en_tokens": [
        {"re": "\\bArcturel?ians?\\b", "cls": "I"},
        {"re": "\\bActurians?\\b", "cls": "I"},
        {"re": "\\bArcturians?\\b", "cls": "I"},
        {"re": "\\bArcturel\\b", "cls": "E"},
        {"re": "\\bActurel\\b", "cls": "E"},
        {"re": "\\bArctural\\b", "cls": "E"},
        {"re": "\\bArcurel\\b", "cls": "E"},
        {"re": "\\bArturel\\b", "cls": "E"}
    ],
    "cn_tokens": [
        {"re": "阿克图里安人", "cls": "I"},
        {"re": "阿克图里安", "cls": "I"},
        {"re": "阿克图瑞尔", "cls": "E"}
    ],
    "except_paths": ["actors.The Device.items.Powered Effect"],
    "except_blocks": [
        {"path": "Arctus Plateau Gazetteer.pages.Arcturel.text", "block": 205,
         "en": "IIE", "cn": "EIE",
         "why": "上游把城名写成了族名：`Politically, Arcturian is ruled by…`，而同段后文写的是 `everything in Arcturel`。中文写「阿克图瑞尔由…统治」语义正确，**不按字面回改**。split_dives 的 KNOWN_OK 里已有同一条"},
        {"path": "Glitter in the Dark.pages.Overview.text", "block": 15,
         "en": "", "cn": "E",
         "why": "中文加译：英文只说 `in The Dives district`，中文补成「阿克图瑞尔下层{矿渊}」，是对位置的补足说明，属加译语境不是多译地名。split_dives 的 KNOWN_OK 里已有同一条"},
        {"path": "Glitter in the Dark.pages.Concerned Merchants.text", "block": 179,
         "en": "EEE", "cn": "EE",
         "why": "中文代词化：英文第三次仍写 `Arcturel's relatively lawless nature`，中文写「了解**它**相对无法无天的性质」。前两处都点了名，语义完整"},
        {"path": "actors.Vartholomew Chess.biography.private", "block": 1,
         "en": "IE", "cn": "EI",
         "why": "纯语序调换：EN `a workshop known as Arcturian Automations in the upper Arcturel Tradeway` → CN「在阿克图瑞尔上层贸易道经营着一间名为“阿克图里安自动机”的工坊」。两个词都在、都对"},
        {"path": "actors.Sadri Zhalimorne.biography.private", "block": 21,
         "en": "I", "cn": "II",
         "why": "**ESCALATED 真缺陷（Y2 升报，译文不归本条改）**：中文把 `the western reaches of the Plateau` 写成「阿克图里**安**高原」，而这片高原是 Arctus Plateau＝阿克图**斯**高原。全库计数 阿克图斯高原 1048 : 阿克图里安高原 **1** —— 就是这一处。这正是叶级判据抓不到、块级才抓得出的「叶内单处串行」"},
        {"path": "actors.Constructed Companion.biography.private", "block": 7,
         "en": "II", "cn": "I",
         "why": "**ESCALATED 真缺陷（Y2 升报，译文不归本条改）**：英文 `by the famed Arcturelian artificer Vartholomew Chess`，中文写成「名匠瓦索洛缪·切斯」，**漏掉了族名**。同一段英文在孪生条目 `actors.Woven Construct.biography.private` 块3 的中文里写的是「著名阿克图里安工匠瓦索洛缪·切斯」—— 两处该一致"}
    ],
    "min_leaves": 850,
    "min_blocks": 1600,
    "max_shape_mismatch": 0
}

RANK = {
    "id": "R-rank-sense-blocks",
    "title": "compendium 侧 Rank 的**块级**义项闸：块内单一义项时上正向闸",
    "decision": "2026-08-15f（第十八轮 Y2：补 R-rank-sense-compendium 自己在 why 里写死的覆盖洞）",
    "kind": "block_sense_gate",
    "why": "R-rank-sense-compendium 的 why 写死了两块不查的地方：「**故意不做**『GAME → 中文必须含阶位』的正向闸（纯 GAME 的 230 叶里有 13 叶中文正当地没有阶位）」与「混合叶 8 / 无法分类叶 57 同样不查」。两块的根因是同一个：**分类窗口是整叶**，叶里只要混了别的义项，整叶就判不动。本条把窗口收到块内：闸下 **391 叶 / 555 含 rank 的块**，其中机制义 369 块 / 普通名词义 109 块 —— 那 369 块**全部上了正向闸**，这是叶级版明确放弃的方向。混合与无法分类掉到 0 / 72 **块**（不是 57 **叶**），同叶其它段落照判。判据修正两处，都是**收紧分类、不是放宽闸**：① `strong_game`——块小了之后 COMMON 的 `ranks of` 会咬到 `Ranks of attunement progression`（叶级时那一叶别处还有 GAME、落进 MIX 桶所以从没暴露过），同调／魂印／`Rank N` 是本系统机制专名，优先级必须高于 COMMON 的泛化措辞；② `exempt`——`rank of exhaustion`（＝层）· `close ranks`（＝并肩结阵）· `join their ranks`（＝加入他们）是**第三个义项**，本来就不归「阶位」那条裁决管，块内出现即整块不判。⚠ 另有一处纯粹是**切块粒度**的教训：第一版按 split_dives 的写法切**所有**标签，正向闸报出 42 处，其中成片是中文「定语在前」把词搬过 `<strong>` 边界（`Cora Attunement.description` 英文块「damage equal to 2 times your attunement rank」的「同调阶位」搬到了前一块）。改成只按块级标签切、行内标签剥成空格之后掉到 12 处。⚠ **判据边界**：块内混合义项与无法分类仍然不判（0 / 72 块）；`min_common_blocks` 守着反向闸没被判空",
    "occ": "\\branks?\\b",
    "leaf_gate": "\\branks?\\b",
    "cn": "阶位",
    "window": 90,
    "sense": {
        "strong_game": "(attunement|attuned|soulbound|soulmark|Rank\\s*\\d)",
        "game": "(Novice|Journeyman|Adept|Master|Untrained|training|skill|Attunement|Attuned|Soulbound|Soulmark|talent|progress|superior to|exhaustion|Scale|Rank\\s*\\d|\\bBonus\\b)",
        "common": "(ranks? (depending|based) on|civic|social|nobil|noble|militar|clerical|within the order|of the order|stripped of|full rank of|ranks of|swell(ing)? the ranks|rank[- ]and[- ]file|hierarch|station|office|through the ranks|within [^.]{0,30}\\branks\\b|beyond its ranks|rose to ranks|\\branks (after|and)\\b|rank as an? )",
        "exempt": "(ranks? of exhaustion|close ranks|join(ing)? their ranks)"
    },
    "except_blocks": [
        {"path": "Gamemaster's Guide.pages.Patch 0.3.3.text", "block": 143,
         "why": "更新日志被整段改写：EN `Add preliminary mechanics and automation support for Soulbound (rank 1 only)` → CN「为魂缚（仅 1 级）加入初步机制与自动化支持」。R-rank-sense-compendium 的 why 里已把这一处列为**正当地没有「阶位」**的 13 叶之一"},
        {"path": "Unfinished Business.pages.Shine On.text", "block": 4,
         "why": "**ESCALATED 真缺陷（Y2 升报，译文不归本条改）**：`Rank 1 Soulmark` → 「1 级魂印」。§8 已裁 Rank＝阶位、Level＝等级，这里用「级」与 Level 撞车。全库实测 级魂印 10 : 魂印阶位 2、级魂缚 1 : 魂缚阶位 7 —— **同一个机制两种写法并存**，需要主控裁一次再统一，不是本条能自行改的"},
        {"path": "Unfinished Business.pages.Shine On.text", "block": 14,
         "why": "同上，`this Rank 1 Soulmark` → 「这个 1 级魂印」"},
        {"path": "Unfinished Business.pages.Shine On.text", "block": 31,
         "why": "同上，`Agraband's Rank 1 Soulmark` → 「阿格拉班德 1 级魂印」"},
        {"path": "Unfinished Business.pages.The Old Flame.text", "block": 264,
         "why": "**ESCALATED 真缺陷（Y2 升报，译文不归本条改）**：中文把 `Rank 1` **原样留成了英文** ——「拥有[[/attunement primordis]] Rank 1的角色…」。这不是术语选择问题，是英文残留（全库共 4 处：本叶与 268 块 ×孪生两包）"},
        {"path": "Unfinished Business.pages.The Old Flame.text", "block": 268,
         "why": "同上，`Rank 2 or higher` 原样留成英文「Rank 2或更高」"}
    ],
    "min_leaves": 350,
    "min_blocks": 500,
    "min_game_blocks": 300,
    "min_common_blocks": 80,
    "max_shape_mismatch": 0
}

NEW = [("R-shard-god", SHARD), ("R-arcturel-vs-arcturian", ARCT), ("R-rank-sense-compendium", RANK)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    doc = json.load(open(P, encoding="utf-8"))
    rules = doc["assertions"]
    for parent, new in NEW:
        rules[:] = [r for r in rules if r["id"] != new["id"]]
        idx = next(i for i, r in enumerate(rules) if r["id"] == parent)
        rules.insert(idx + 1, new)
        print(f"  插入 {new['id']}（紧跟 {parent}，位置 {idx + 1}）")
    doc["meta"]["updated"] = (
        "2026-08-15f（第十八轮 Y2：新增 block_aligned_gate / block_sense_gate 两个**叶内**断言类型，"
        "补上 R-shard-god / R-arcturel-vs-arcturian / R-rank-sense-compendium 三条各自在 why 里写死的"
        "覆盖洞；三条新断言合计现读 502+905+391 叶、694+1714+555 块）｜ "
        + doc["meta"]["updated"])
    json.dump(doc, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写回 {P}：共 {len(rules)} 条断言")


if __name__ == "__main__":
    main()
