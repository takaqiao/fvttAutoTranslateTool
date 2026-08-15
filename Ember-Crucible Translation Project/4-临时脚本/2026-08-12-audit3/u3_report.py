# -*- coding: utf-8 -*-
"""U3: emit the per-finding verdict table (findings/U3.md)."""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SC = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

# idx -> (new_label or None, 依据)
D = {
151: ("深渊恐魔", "页 name＝深渊恐魔 Abyssal Horrors；同标签 EN 闸 23:5；孪生包 #43 同错"),
152: ("十九人", "页 name＝十九人 The Nineteen；EN 'The Nineteen' 标签 54:10；本页下文自称「二十人」"),
153: ("萨克茨", "第 8 节 2026-08-12b 决议 Sockets→萨克茨；页 name＝萨克茨 Sockets；标签 53:6"),
154: (None, "**多数派是错的一边**：EN 闸 \\bMazira\\b 马兹拉 10 : 马齐拉 7，且 Maziran 的 name＝马兹兰（同一音译词根 兹）。本条写法正确，另 7 处「马齐拉」才是要扫的"),
155: ("十九人", "同 152"),
156: (None, "该处英文是单数 'Elder God'（页 name 是复数 Elder Gods＝上古诸神），中文跟着英文走"),
157: (None, "该处英文是 'Elder Goddess'（扫描器只报了该目标的第一个英文标签），中文「上古女神」忠实"),
158: ("萨克茨", "同 153"),
159: ("十九人", "同 152"),
160: ("瑟洛克", "页 name＝瑟洛克 Theroch；标签 10:2；EN 闸 \\bTheroch\\b 瑟洛克 14 : 瑟罗克 8"),
161: ("碎片诸神", "页 name＝碎片诸神 Shard Gods；EN 复数 'Shard Gods' 标签 67:8（决议：单数 Shard God 才作碎片之神，同叶后一处单数不动）"),
162: ("奥拉", "决议 2026-08-12b：月亮 Aura→奥拉（手势才留灵气）；页 name＝奥拉 Aura；标签 17:8"),
163: ("深渊恐魔", "同 151"),
164: ("卡西娅", "页 name＝卡西娅 Casia；标签 32:6"),
165: ("卡西娅", "同 164"),
166: ("萨克茨", "同 153"),
167: ("卡西娅", "同 164"),
168: (None, "锚点 #failing-to-escape，该处英文就是 'Failing to Escape'；多数「摧毁高塔」是同目标无锚点时的页名，不可比"),
169: (None, "锚点 #the-veiled-chain-trap，该处英文是 'The Veiled Chain Trap'；扫描器的 en_label 取了该目标的首个英文标签，属误报"),
170: (None, "锚点 #using-the-locator-rod，该处英文是 'Using the Locator Rod'；多数「庄园通行权限」来自 #manor-access，属误报"),
171: ("马尔石", "页 name＝马尔石 Marlstone；标签 17:1；决议 2026-08-09「裸 Marlstone 是街区名 马尔石，马尔斯通只用于 Marlstone Manor」"),
172: ("画廊", "页 name＝画廊 Gallery；标签 8:4"),
173: ("画廊", "同 172（同叶第二处）"),
174: (None, "标签＝actor name「加斯特恩·法维约斯 Gastern Faviyos」，本条正确；另 4 处「加斯滕·法维约斯」才是偏离（孪生 #65 同为加斯特恩，不要跟着多数改）"),
175: ("华服", "dnd5e 外部条目，无本库 name；同条目同英文标签 EN 闸 17:2"),
176: ("旅行服", "同 175，9:2"),
177: ("与卡夫托尔战斗", "目标小节自身的中文标题就是「与卡夫托尔·布伦克战斗」；actor name＝卡夫托尔·布伦克；同锚点另一处标签已是「与卡夫托尔战斗」；EN 闸 卡夫托尔 37 叶"),
178: (None, "同锚点、写法与小节标题一致，正确（多数「发光蓄水池」是无锚点时的页名）"),
179: ("一再坠落", "JournalEntry name＝一再坠落 To Fall and Fall Again；标签 8:2"),
180: ("辉耀", "actor name＝辉耀 Lucent；标签 20:2；EN 闸 辉耀 20 : 卢森特 5"),
181: ("回避侦测术", "dnd5e 法术，本库同条目标签 4:2"),
182: ("制图工具", "dnd5e 工具，本库同条目标签 3:1"),
183: ("网", "内嵌 item name＝网 Net（16 处）；孪生包同一页同一句就写「网」"),
184: ("突进强攻", "内嵌 item name＝突进强攻 Lunging Assault；标签 3:1"),
185: ("总管", "actor name＝总管 Chamberlain；标签 18:1:1；孪生包同一页此处正是「总管」。「张伯伦」是把职衔当姓氏的误译"),
186: (None, "标签＝actor name「银光束仆从」；EN 闸 \\bSilver Beam Servitors?\\b 仆从 13 叶 : 仆役 7 叶——本条在正确一边，另 7 叶才要扫"),
187: (None, "同 186"),
188: (None, "同 186"),
189: ("奥拉", "同 162（此处原写「灵光」，是第三种写法）"),
190: ("星之盾", "页 name＝星之盾 The Starshield；EN 闸 星之盾 55 : 星盾 4"),
191: (None, "该处英文是 'Blood Barons'，中文「血男爵」忠实；扫描器 en_label 取的是该目标首个英文标签 'Moiran'，属误报"),
192: (None, "同 191"),
193: (None, "同 191；且中文保留了英文标签的前导空格，逐字节对齐"),
194: ("书籍", "dnd5e 条目，本库同条目标签 30:2（含 16 处单数 Book 也作书籍）"),
195: ("库阿尔塔神殿钥匙", "item name＝库阿尔塔神殿钥匙；标签 4:2"),
196: ("尸体战利品（阿克图斯高原）", "同英文标签 8:4，差别仅半角括号；中文排版用全角"),
197: ("明暗野兽", "孪生包同一页（Local Color/Matters of Perspective）就写「明暗野兽」；EN 单数标签 明暗野兽 7 : 明暗兽 4 : 基亚罗斯库兰野兽 1"),
198: ("十九人", "同 152"),
199: ("水妖精", "actor name＝水妖精 Water Sprite；标签 20:1；同句 水微灵/水访客 同族"),
200: ("陷坑保镖", "场景与 journal name＝陷坑 Pit Trap；同英文标签 4:2。「弹跳器」是 bouncer 的机翻"),
201: ("星之盾", "同 190"),
202: (None, "EN 闸 明暗兽 8 叶 : 明暗野兽 7 叶，同一 journal 内 3:3 分裂，孪生包同路径写法相同——属需要一次全库统一（连 actor name 一起）的术语项，不宜逐叶单改"),
203: (None, "同 202"),
204: ("阿梅莉亚·纳克桑", "actor name＝阿梅莉亚·纳克桑；标签 15:6:1；EN 闸 阿梅莉亚 20 叶 : 阿米莉亚 9 叶"),
205: ("阿梅莉亚·纳克桑", "同 204（同叶第二处）"),
206: ("物件定位术", "dnd5e 法术，本库同条目标签 3:1"),
207: ("阿梅莉亚·纳克桑", "同 204"),
208: ("崩塌圣所", "JournalEntry name＝崩塌圣所 Crumbling Sanctuary；标签 8:2:2"),
209: (None, "锚点 #encounter-details，该处英文就是 'Encounter Details'；多数「主层」是无锚点时的页名，属误报"),
210: (None, "锚点 #searching-the-aftermath，该处英文是 'Searching the Aftermath'，属误报"),
211: ("突变学派卡夫托尔·布伦克", "同英文标签 4:2，差别仅中间那个空格；中文名与前缀之间不加空格"),
212: (None, "**多数派是错的一边**：EN 闸 \\bOrdani\\b 奥尔达尼 644 叶 : 奥达尼 16 叶，文化页 name 也是「奥尔达尼 Ordani」。本条正确；该 actor 的 name「奥达尼恶棍」与另 12 处标签才是要扫的"),
213: (None, "标签＝actor name「西瑟里安 Sitherian」（英文侧该链接本来没有标签，是译者补的）；另 3 叶写「西希拉之子」——那是 EN 'Child of Sitheera' 的译法，用在此处才是偏离"),
214: ("尸体战利品（阿克图斯高原）", "同 196"),
215: ("杰克罗卡的账本", "item name＝杰克罗卡的账本 Jekeroka's Ledger；标签 3:1；孪生包同一页此处正是「杰克罗卡」"),
216: ("拉斯特·索恩", "EN 闸 \\bRaster Thorn\\b 拉斯特·索恩 43 叶 : 栅格荆棘 3 叶（3 处里 2 处是 actor name 本身）；孪生包同一页此处正是「拉斯特·索恩」；该 NPC 有 Raster's Quarters / Raster's Throne 两页，Raster 是人名。**此条与 actor name 相反，是 name 字段自己错**"),
217: ("崩塌圣所", "同 208"),
218: (None, "该处英文是 'Milestone Progression'（页 name 同名），中文「里程碑进阶」正确；多数「里程碑点数」对应英文 'Milestone' point，属误报"),
219: ("攀爬者工具包", "dnd5e 条目，本库同条目标签 4:2"),
220: (None, "该处英文是短式 'Writhing Tendrils'（actor name 是 Writhing Whisperer Tendril＝蠕动低语者触须），英文本来就不同"),
221: (None, "标签＝actor name「拉斯克幼体 Rask Juvenile」；另 4 处「幼年拉斯克」里有 4 叶英文是小写 'juvenile Rask'（普通名词，本来就该那样译）"),
222: ("水妖精", "actor name＝水妖精 Water Sprite；标签 20:1"),
223: ("水访客", "actor name＝水访客 Water Visitor；标签 13:1"),
224: ("霍伦多尔", "决议 2026-08-09：Horrendor→霍伦多尔（「惊惧者」是 Harrower）；actor name＝霍伦多尔；标签 23:1"),
225: ("角色创建", "crucible.rules 的 JournalEntry name＝角色创建 Character Creation；标签 4:1"),
}

d = json.load(open(SC + "/uuid_swap.json", encoding="utf-8"))
lines = ["# U3 · `@UUID` 标签命名不一致逐条裁决（findings[151:226]，75 条）", "",
         "判据阶梯：**目标文档 `name` 字段 > 孪生包同路径写法 > 同英文标签的全库计数（EN 闸）> 该目标的全库多数**。",
         "「多数」一栏是 `scan_uuid_swap` 按 (目标, 英文标签) 算的；它按目标 id 聚合，**锚点链接（`#xxx`）会被算进同一个桶**，",
         "所以 `en_label` 字段给的是该目标的第一个英文标签、不是本处的——本单元 9 条误报全部出自这里。", "",
         "| # | 目标 | 英文标签(本处实际) | 原中文 | 判定 | 依据 |", "|---|---|---|---|---|---|"]
n_ch = 0
for i in range(151, 226):
    f = d["findings"][i]
    new, why = D[i]
    verdict = f"**改 → `{new}`**" if new else "不改"
    if new:
        n_ch += 1
    en = f["en_label"]
    m = re.search(r"该处英文(?:就)?是 '([^']+)'", why)
    if m:
        en = m.group(1) + "（扫描器报 " + str(f["en_label"]) + "）"
    lines.append(f"| {i} | `{f['target']}` | {en} | {f['cn_label']} | {verdict} | {why} |")
lines += ["", f"合计：改 {n_ch} 条 / 不改 {75 - n_ch} 条。落盘批次见 `audit3/batches/U3__*.json`（45 个叶，`apply_translations --force --dry` 三包全 0 拒绝）。"]
os.makedirs(SC + "/findings", exist_ok=True)
open(SC + "/findings/U3.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("changed:", n_ch, "-> ", SC + "/findings/U3.md")
