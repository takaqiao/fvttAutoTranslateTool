# -*- coding: utf-8 -*-
"""Write the per-finding verdict table for unit U2."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SW = r'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json'
OUT = r'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/findings/U2.md'
F = json.load(open(SW, encoding='utf-8'))['findings']

D = {
76: ("Blood Barons", "不改", "英文同叶对同一目标用了两个标签 {Moiran}x2 / {Blood Barons}x3，中文 莫伊兰x2 / 血男爵x3 逐位对齐。finding 的 en_label 只取叶内首个标签，此处不可信"),
77: ("Blood Barons", "不改", "同 76"),
78: (" Blood Barons", "不改", "同 76（英文标签自带前导空格 { Blood Barons}，中文照抄）"),
79: ("Book", "改→书籍", "dnd5e 外部条目，本库无 name；同一英文标签 Book→书籍 16 次 : 书 2 次"),
80: ("Temple of Ku'arta Key", "改→库阿尔塔神殿钥匙", "物品 name 字段＝库阿尔塔神殿钥匙（两包各一）；同标签 4 : 2"),
81: ("Corpse Loot (Arctus Plateau)", "改→尸体战利品（阿克图斯高原）", "同标签 8 次全角括号 : 4 次半角+空格；库内标点规范为全角"),
82: ("Wandren Watcher", "改→万德伦注视者", "英文闸 \\bWandren\\b：万德伦 264 : 流浪 5（Wandren 是专名，另有 NPC Juro Wandren）；同标签 5 处均作万德伦注视者。该 actor 的 name/tokenName 仍是「流浪注视者」，建议另案一并改"),
83: ("The Nineteen", "改→十九人", "目标页 name＝十九人 The Nineteen；同标签 54 : 10"),
84: ("Pit Trap Bouncers", "改→陷坑保镖", "场所 journal name＝陷坑 Pit Trap；同标签 4 处作陷坑保镖。「坑陷阱弹跳器」是把 bouncer 当弹跳装置的机翻"),
85: ("Starshield", "改→星之盾", "目标页 name＝星之盾 The Starshield；星之盾 14 : 星盾 4"),
86: ("Chiaroscuran Beast", "改→明暗野兽", "英文闸 Chiaroscuran：明暗 14 : 基亚罗斯库兰 3（英文里指 chiaroscuro 明暗画法——粉笔与木炭之躯、Redraw Form，不是地名）；同标签 明暗野兽 7 : 明暗兽 4，name 字段的「野兽」也在这一边"),
87: ("Chiaroscuran Beast", "改→明暗野兽", "同 86"),
88: ("Amelia Naxan", "改→阿梅莉亚·纳克桑", "actor name＝阿梅莉亚·纳克桑；同标签 15 : 6 : 1"),
89: ("Amelia Naxan", "改→阿梅莉亚·纳克桑", "同 88（同一叶第二处，一并替换）"),
90: ("Amelia Naxan", "改→阿梅莉亚·纳克桑", "同 88"),
91: ("Crumbling Sanctuary", "改→崩塌圣所", "目标 journal name＝崩塌圣所 Crumbling Sanctuary；同标签 8 : 2 : 2"),
92: ("Encounter Details", "不改", "英文同叶用 {Encounter Details} 与 {Searching the Aftermath} 两个标签指向同一页，中文逐位对齐；多数派「主层」来自别处拿页名当标签的叶子"),
93: ("Searching the Aftermath", "不改", "同 92"),
94: ("Mutagist Kaftor Brenk", "改→突变学派卡夫托尔·布伦克", "同标签 4 处无空格 : 2 处有空格，纯排版不一致"),
95: ("Ordani Ruffian", "不改", "多数派是错的一边：英文闸 \\bOrdani\\b 全库 奥尔达尼 644 : 奥达尼 16，而那 16 处全部集中在这一个 actor（含它自己的 name 字段）。本处「奥尔达尼恶棍」才是与全库一致的写法，要改的是 actor name + 另 4 叶标签"),
96: ("Corpse Loot (Arctus Plateau)", "改→尸体战利品（阿克图斯高原）", "同 81"),
97: ("Crumbling Sanctuary", "改→崩塌圣所", "同 91"),
98: ("Milestone Progression", "不改", "英文同叶两个标签 {Milestone}/{Milestone Progression}，中文 里程碑/里程碑进阶 对齐；且「里程碑进阶」正是目标页 name"),
99: ("Climber's Kit", "改→攀爬者工具包", "dnd5e 外部条目、本库无 name；同标签 4 : 2 取多数。（dnd5e 汉化补丁作「攀爬工具」，按 PROJECT.md 它只是低优先级建议源）"),
100: ("Writhing Tendrils", "不改", "英文同叶两个标签 {Writhing Tendrils}/{Writhing Whisperer Tendril}，中文 蠕动触须/蠕动低语者触须 对齐"),
101: ("(英文无标签)", "不改", "多数派是错的一边：目标物品 name＝爆炸火药桶 Barrel of Blast Powder，英文该段小标题就是 Barrel of Blast Powder、正文写「桶上画着爆炸瓶的图案」，另有独立物品 爆炸瓶 Blast Flask。全库 9 处把它标成「爆炸瓶箱」，那 9 处才是缺陷"),
102: ("(英文无标签)", "不改", "同 101（同叶第二处）"),
103: ("Rask Juvenile", "不改", "actor 的 name 与 tokenName 均为 拉斯克幼体（两包共 4 个字段）；多数派「幼年拉斯克」4 处与 name 不符，建议另案统一"),
104: ("Ordani Cityfolk", "改→奥尔达尼市民", "同一英文标签 Ordani Cityfolk 全库 7 叶作奥尔达尼市民（Ordani Citizens 另 2 叶亦然），仅此处作城镇居民"),
105: ("Abyssal Horrors", "改→深渊恐魔", "目标页 name＝深渊恐魔 Abyssal Horrors；同标签 17 : 5 : 1"),
106: ("Amelia Naxan", "改→阿梅莉亚·纳克桑", "同 88（此处还错成「纳克珊」）"),
107: ("Abyssal Horrors", "改→深渊恐魔", "同 105"),
108: ("(英文无标签)", "改→点燃的肯瑞斯火炬", "物品 name＝点燃的肯瑞斯火炬 Lit Kynryth Torch，其余 4 处标签同名。英文本处无标签，中文自加的标签与物品名不符（火把/火炬 + 括号后置）"),
109: ("Grand Kalion Stadium", "不改", "英文同叶两个标签 {Grand Kalion Stadium}/{Arena Ridge} 指向同一页，中文 宏大卡利昂体育场/竞技岭 对齐"),
110: ("Grand Kalion Stadium", "不改", "同 109"),
111: ("Ordani Cityfolk", "改→奥尔达尼市民", "同 104"),
112: ("(英文无标签)", "改→采石场奔道", "目标页 name＝采石场奔道 Quarry Run；同目标 10 处标签均为采石场奔道。「采石场之门」把 Run 读成了门"),
113: ("Rune-Marked Arrowhead", "不改", "「符文标记」与两件物品的 name 同源（箭头 name＝符文-已标记箭头、绶带 name＝符文标记绶带）。全族另有 符文印记 8 次 / 符文刻印 6 次，需要一次统一裁决（见 notes），但按 name 字段这一处不算错"),
114: ("Rune-Marked Arrowhead", "不改", "同 113"),
115: ("(英文无标签)", "改→符文标记绶带", "物品 name＝符文标记绶带 Rune-Marked Sash；本处「符文刻印腰带」两个维度都偏（刻印/标记、腰带/绶带）"),
116: ("The Mote Chase", "改→微粒追逐战", "目标页 name＝微粒追逐战 The Mote Chase；同标签 3 : 1"),
117: ("Metal Crumbler", "不改", "多数派是错的一边：物品 name＝碎金者 Metal Crumbler（两包各一）。全库另有 4 处标签作「金属碎解剂」、物品自身描述作「金属粉碎剂」，要统一的是那些"),
118: ("Metal Crumbler", "不改", "同 117"),
119: ("Metal Crumbler", "不改", "同 117"),
120: ("Metal Crumbler", "不改", "同 117"),
121: ("Draconic", "不改", "英文同叶两个标签 {Draconic}/{Dragons} 指向同一页，中文 龙类/巨龙 对齐"),
122: ("Shattering", "改→大破裂", "目标页 name＝大破裂 The Shattering；同目标标签 大破裂 106 : 破碎 2，库内该事件专名就是大破裂"),
123: ("Big Liz", "不改", "英文同叶 {Big Liz}/{Baradom} 指向同一 actor（克恩给那只巨蜥起的绰号，英文原文 \"Big Liz\" which is actually a very large, very old Baradom），中文 大莉兹/巴拉多姆 逐位对齐"),
124: ("Cook's Utensils", "改→厨师工具", "同标签 3 : 1；dnd5e 汉化补丁亦作「厨师工具」，两方一致"),
125: ("Playing Cards Set", "改→扑克牌", "同标签 3 : 1 取多数（dnd5e 补丁作「整副纸牌」，仅参考）"),
126: ("Chasm Candles", "不改", "物品 name＝裂隙蜡烛 Chasm Candle，本处与 name 一致；多数派「裂谷蜡烛」4 处与 name 不符，建议另案统一"),
127: ("Emelyn Arvoda", "不改", "该通用 actor（Arcturian 市民卡）在同一叶被英文标成 5 个人名 {Emelyn Arvoda}/{Arcturians}/{Eolas Hathwick}/{Calandra}/{Hob Korell}，中文逐位对齐"),
128: ("Eolas Hathwick", "不改", "同 127"),
129: ("Calandra", "不改", "同 127"),
130: ("Hob Korell", "不改", "同 127"),
131: ("Lockpicks", "改→开锁工具", "crucible.equipment 条目 name＝开锁工具 Lockpicks（另有 3 处同名 name 亦然）；同标签 10 : 2"),
132: ("Shard Gods", "改→碎片诸神", "目标页 name＝碎片诸神 Shard Gods；复数标签 Shard Gods→碎片诸神 69 : 碎片之神 6。与 2026-08-12b 决议不冲突：单数 Shard God 仍作碎片之神（28 处）"),
133: ("Arcturian Respirator", "改→阿克图里安呼吸器", "2026-08-12b 决议 Arcturian→阿克图里安；该物品 name 已改为阿克图里安呼吸器，这两处标签是决议漏执行的残留"),
134: ("Arcturian Respirator", "改→阿克图里安呼吸器", "同 133（同一叶第二处）"),
135: ("Reviled Magic", "改→遭憎魔法", "英文闸 \\bReviled\\b：遭憎 10 : 受憎 2；同标签 8 : 2"),
136: ("Kessia", "不改", "Kessia（国名）与 Kessian（族名）是两个英文词：英文闸 Kessia→凯西亚 65/65、Kessian(s)→凯西安 57。多数派「凯西安」来自 en_label=Kessian 的叶子，本处英文是 Kessia"),
137: ("Ordain", "不改", "英文同叶 {Ordain}x1 与 {Ordani}x3 指向同一文化页，中文 奥尔丹/奥尔达尼 逐位对齐，正合既有决议"),
138: ("Kessia", "不改", "同 136"),
139: ("Aura", "改→奥拉", "2026-08-12b 决议：Aura 作月亮专名→奥拉，手势 Gesture: Aura 才作灵气。此处上下文是「气之月」，目标就是 cosmos 的月亮页（name 已是 奥拉 Aura）"),
140: ("Nineteen Nights in Haxim", "不改", "英文闸 \\bHaxim\\b：哈克西姆 6 : 哈西姆 4，且物品 name＝哈克西姆的十九个夜晚，本处音译与 name 一致。多数派 4 处漏了「克」，该改的是那 4 处"),
141: ("(英文无标签)", "改→{识别施法}天赋", "英文是 uses the @UUID[...] talent——标签在英文里根本不存在、talent 是链接外的词；中文把标签写成「天赋」等于抹掉了天赋名。同目标另有 6 处标签作识别施法。改为 {识别施法}天赋，与英文结构一致"),
142: ("Shard Gods", "改→碎片诸神", "同 132"),
143: ("The Eternal Soul", "不改", "英文同叶 {The Eternal Soul}/{Soul Transference} 指向同一页，中文 永恒灵魂/灵魂转移 对齐"),
144: ("Soul Transference", "不改", "同 143"),
145: ("Spiritlands", "不改", "英文同叶 {Eternas}/{Spiritlands}，中文 艾特纳斯/灵界荒土 对齐；finding 把 en_label 记成了 Eternas"),
146: ("her journal", "不改", "英文就是小写代词式 {her journal}，同叶另一处 {Avwynn's Journal}→阿芙温的日志。中文跟英文走"),
147: ("Evesso's Note", "改→埃维索的便笺", "物品 name＝埃维索的便笺 Evesso's Note；同标签 4 : 2"),
148: ("Sin", "不改", "英文同叶 {Sin Marmot}/{Sin} 指向同一 actor，中文 辛·旱獭/辛 对齐"),
149: ("Sunalins", "不改", "英文是复数 Sunalins（该万神殿的诸神），目标页 name＝苏纳林 Sunalin，「苏纳林诸神」既保留 name 词干又译出复数。同标签全库三写（苏纳林斯 6 / 苏纳林 4 / 苏纳林诸神 2），真正该改的是凭空多出「斯」的 6 处，建议另案统一"),
150: ("Aura", "改→奥拉", "同 139，上下文是「某轮月亮渐盛」"),
}

HDR = [
"# U2 —— `@UUID` 标签命名不一致逐条裁决（findings[76:151]，共 75 条）",
"",
"**方法**：finding 里的 `en_label` 只取叶内该目标的**第一个**英文标签；一旦同一叶对同一目标用了两个英文标签",
"（`{Moiran}`/`{Blood Barons}`、`{Big Liz}`/`{Baradom}`、`{Draconic}`/`{Dragons}`…）它就张冠李戴。",
"所以每条都先用 `4-临时脚本/2026-08-12-audit3/u2_align.py` 把该叶英文侧与中文侧的同目标标签**按出现次序逐位对齐**再判 ——",
"仅这一步就把 22 条「看着像错」的判成了不改。",
"依据阶梯：目标文档 `name` 字段（`u2_name.py` 直读 compendium/en↔cn）> 同一英文标签的全库分布（`u2_census.py`）> 英文闸计数（`term_gate.py`）。",
"",
"**判定：改 39 条 / 不改 36 条。**",
"批次：`audit3/batches/U2__ember__ember.adventure.json`（16 叶）、`audit3/batches/U2__ember__ember.crucible-adventure.json`（20 叶）；",
"`apply_translations.py --force --dry` 两包均 **0 拒绝**，标记/内联命令/标签多重集逐叶比对无漂移。",
"",
"| 序号 | 目标 | 英文标签（逐位对齐后的真值） | 原中文 | 判定 | 依据 |",
"|---|---|---|---|---|---|",
]

rows = []
for i in range(76, 151):
    x = F[i]
    en, verd, why = D[i]
    cn = x['cn_label']
    rows.append(f"| {i} | `{x['target']}` | {en} | {cn} | {verd} | {why} |")

TAIL = [
"",
"## 本单元查出、但落在 findings 范围外的缺陷（多数派本身是错的，扫描器永远不会报）",
"",
"| 词条 | 库内多数写法 | 正确写法（证据） | 波及 |",
"|---|---|---|---|",
"| `Barrel of Blast Powder` | 爆炸瓶箱 9 处 | **爆炸火药桶**（物品 name；英文小标题 Barrel of Blast Powder，且 `Blast Flask` 是另一件物品 爆炸瓶） | 9 叶标签 |",
"| `Metal Crumbler` | 金属碎解剂 4 处 + 描述里的 金属粉碎剂 | **碎金者**（物品 name，两包各一） | 4 叶标签 + 1 条描述 |",
"| `Chasm Candle` | 裂谷蜡烛 4 处 | **裂隙蜡烛**（物品 name） | 4 叶标签 |",
"| `Rask Juvenile` | 幼年拉斯克 4 处 | **拉斯克幼体**（name + tokenName，两包共 4 字段） | 4 叶标签 |",
"| `Nineteen Nights in Haxim` | 《哈西姆的十九夜》4 处 | **哈克西姆**（英文闸 6:4，物品 name 亦作哈克西姆） | 4 叶标签 |",
"| `Sunalins` | 苏纳林斯 6 处 | **苏纳林/苏纳林诸神**（页 name＝苏纳林，「斯」是把复数 -s 也音译了） | 6 叶标签 |",
"| `Ordani Ruffian` | 奥达尼恶棍（含 actor name 2 处） | **奥尔达尼恶棍**（英文闸 Ordani：奥尔达尼 644 : 奥达尼 16，后者全部集中在这一个 actor 上） | 2 个 name + 4 叶标签 |",
"| `Wandren Watcher` | actor name/tokenName＝流浪注视者 | **万德伦注视者**（英文闸 264:5，Wandren 是专名） | 4 个 name/tokenName 字段 |",
"| `Chiaroscuran Beast` | actor name＝基亚罗斯库兰野兽 | **明暗野兽**（英文闸 明暗 14 : 基亚罗斯库兰 3） | 2 个 name |",
"| `Rune-Marked *` | 符文印记 8 次 / 符文刻印 6 次 / 符文标记 4 次 | 需一次统一：箭头 name＝`符文-已标记箭头`（带连字符，本身就该修），绶带 name＝符文标记绶带 | 2 件物品 name + 约 12 处标签 |",
"",
"上面这些**不在**本单元的批次里 —— 它们不属于 findings[76:151]，且逐条都要改 `name` 字段或跨包多叶，",
"应由主控单独排一批（改 `name` 的那几条同时会影响 token 名与合集列表，属于另一类改动）。",
]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write("\n".join(HDR + rows + TAIL) + "\n")
print("wrote", OUT, len(rows), "rows")
