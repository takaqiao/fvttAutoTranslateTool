# 05 — L4 全本深度 QA 进度

**会话**：2026-05-27 s13（接续 L2 QA 收尾后）+ 可能持续多会话

**深度**：逐句中英对齐 + 术语查 + 仅修明显错误（术语/偏差/漏译/原文不匹配）。语义/风格问题仅记录不改。

**术语优先级**：pf2_cn > pf2e_compendium > 地狱破灭 PDF（仅冲突时查）> wiki

**批次策略**：每批 3-5 页，每 10-15 页 commit 一次

## 总览

| 指标 | 数 |
|---|---|
| 总页数 | 258 |
| 已完成 | 100 |
| 进行中 | 0 |
| 待处理 | 158 |
| 最后批次 | Batch 20 pp.096-100 |
| 下次起始页 | 101 |

## 处理顺序

按页码 page_001 → page_258 顺序（按 PDF 自然顺序，与 01_progress.md 章节边界对齐）。

## 每页状态（紧凑视图）

图例：✅ done / 🔄 in_progress / ⏸ pending / ⚠ issues_recorded（仅记录，未改全部）

### pp.001-049（前言 + Ch1-2）
```
001:✅ 002:✅ 003:✅ 004:✅ 005:✅ 006:⚠ 007:✅ 008:✅ 009:✅ 010:✅
011:✅ 012:✅ 013:✅ 014:✅ 015:✅ 016:✅ 017:✅ 018:✅ 019:✅ 020:✅
021:✅ 022:✅ 023:✅ 024:✅ 025:✅ 026:✅ 027:✅ 028:✅ 029:✅ 030:✅
031:✅ 032:✅ 033:✅ 034:✅ 035:✅ 036:✅ 037:✅ 038:✅ 039:✅ 040:✅
041:✅ 042:✅ 043:✅ 044:✅ 045:✅ 046:✅ 047:✅ 048:✅ 049:✅
```

### pp.050-099（Ch3-4 + Ch5 开篇）
```
050:✅ 051:✅ 052:✅ 053:✅ 054:✅ 055:✅ 056:✅ 057:✅ 058:✅ 059:✅
060:✅ 061:✅ 062:✅ 063:✅ 064:✅ 065:✅ 066:✅ 067:✅ 068:✅ 069:✅
070:✅ 071:✅ 072:✅ 073:✅ 074:✅ 075:✅ 076:✅ 077:✅ 078:✅ 079:✅
080:✅ 081:✅ 082:✅ 083:✅ 084:✅ 085:✅ 086:✅ 087:✅ 088:✅ 089:✅
090:✅ 091:✅ 092:✅ 093:✅ 094:✅ 095:✅ 096:✅ 097:✅ 098:✅ 099:✅
```

### pp.100-149（Ch5-7 + Ch8 开篇）
```
100:✅ 101:⏸ 102:⏸ 103:⏸ 104:⏸ 105:⏸ 106:⏸ 107:⏸ 108:⏸ 109:⏸
110:⏸ 111:⏸ 112:⏸ 113:⏸ 114:⏸ 115:⏸ 116:⏸ 117:⏸ 118:⏸ 119:⏸
120:⏸ 121:⏸ 122:⏸ 123:⏸ 124:⏸ 125:⏸ 126:⏸ 127:⏸ 128:⏸ 129:⏸
130:⏸ 131:⏸ 132:⏸ 133:⏸ 134:⏸ 135:⏸ 136:⏸ 137:⏸ 138:⏸ 139:⏸
140:⏸ 141:⏸ 142:⏸ 143:⏸ 144:⏸ 145:⏸ 146:⏸ 147:⏸ 148:⏸ 149:⏸
```

### pp.150-199（Ch8-10）
```
150:⏸ 151:⏸ 152:⏸ 153:⏸ 154:⏸ 155:⏸ 156:⏸ 157:⏸ 158:⏸ 159:⏸
160:⏸ 161:⏸ 162:⏸ 163:⏸ 164:⏸ 165:⏸ 166:⏸ 167:⏸ 168:⏸ 169:⏸
170:⏸ 171:⏸ 172:⏸ 173:⏸ 174:⏸ 175:⏸ 176:⏸ 177:⏸ 178:⏸ 179:⏸
180:⏸ 181:⏸ 182:⏸ 183:⏸ 184:⏸ 185:⏸ 186:⏸ 187:⏸ 188:⏸ 189:⏸
190:⏸ 191:⏸ 192:⏸ 193:⏸ 194:⏸ 195:⏸ 196:⏸ 197:⏸ 198:⏸ 199:⏸
```

### pp.200-258（Ch11 + 战役之外 + 寇兰廷续 + 贵族 + 工具箱）
```
200:⏸ 201:⏸ 202:⏸ 203:⏸ 204:⏸ 205:⏸ 206:⏸ 207:⏸ 208:⏸ 209:⏸
210:⏸ 211:⏸ 212:⏸ 213:⏸ 214:⏸ 215:⏸ 216:⏸ 217:⏸ 218:⏸ 219:⏸
220:⏸ 221:⏸ 222:⏸ 223:⏸ 224:⏸ 225:⏸ 226:⏸ 227:⏸ 228:⏸ 229:⏸
230:⏸ 231:⏸ 232:⏸ 233:⏸ 234:⏸ 235:⏸ 236:⏸ 237:⏸ 238:⏸ 239:⏸
240:⏸ 241:⏸ 242:⏸ 243:⏸ 244:⏸ 245:⏸ 246:⏸ 247:⏸ 248:⏸ 249:⏸
250:⏸ 251:⏸ 252:⏸ 253:⏸ 254:⏸ 255:⏸ 256:⏸ 257:⏸ 258:⏸
```

## 已发现 issues 与修复（按批次）

### Batch 1 (pp.001-005, 2026-05-27 s13)
- p001/002: 仅作者名页，无 issues
- p003: ToC 「卡里之眼」→「卡利之眼」(Khari = 卡利)
- p004: ToC vs stat block 对齐修正 8 处
  - 阿沃达扎 → 阿德沃达扎 / 多塔里折磨者分队 → 多塔里刑罚分队
  - 梦溺者 → 梦境溺者 / 龙骑兵小队 → 龙骑兵分队 / 猎人小队 → 猎手分队
  - 地狱潮 → 地狱浪潮 / 梅尔兹托雷贡 → 梅尔托雷贡
  - 誓嚎者 → 嚎天精 / 奥基亚斯提斯 → 奎伦特
- p005: 仅页码标识，无 issues

### Batch 2 (pp.006-010, 2026-05-27 s13)
- p006: ⚠ "恶魔教派的斯戎家族" 英文 "diabolical House Thrune" — 翻译略偏（diabolical 形容词 vs 恶魔教派 名词化），但属于判断而非明显错误，**记录不改**。其余术语 C2 已修。
- p007: 1 处 ToC「潘戈莱斯」→「邦戈莱」(Pangolais 与 s12 term_map 同步)
- p008/010: C2 阶段已修主要术语，深度复核无新 issues
- p009: 空章节插页

### Batch 2 出栏 — page_202/207 多处 map 术语统一（pp.7/207/202 map 区一并修复）
- p007: 已含上文
- **p207**：6 处大幅修正
  - 旺达（Vyre）→ 维里 (cf NPC 旺达·喷毒鞭 Vanda Spitelash 同形异义，仅 page_207 这一处是 Vyre 城；其他 page_074/075/086 的「旺达」是 NPC 名 KEEP)
  - 艾里迪尔（Elidir）→ 艾利迪尔 + Elidir 重新归入依斯嘉（依斯嘉首府，原误置 RAVOUNEL）
  - Pangolais 重复出现于 RAVOUNEL 和 NIDAL 两段（原文翻译错误分组）— 移除 RAVOUNEL 段重复；NIDAL 段 庞戈莱 → 邦戈莱
  - 里德万（Ridwan）→ 瑞德万
  - 阿斯波丹山脉 → 阿斯珀德尔山脉 (Aspodell Mountains)
  - 巴罗伍德 → 巴罗林 (Barrowood，s11 wiki canonical)
  - 妖林 → 乌斯克森林 (Uskwood)
- **p202**：1 处「阿斯波丹之墙」→「阿斯珀德尔长城」(Aspodean Wall)

### Batch 3 (pp.011-015, 2026-05-27 s13)
- p011-013: C1/C2 已修主要术语，深度复核无新 issues
- p014: term_map 中 vordine 与实际译文不一致——term_map 旧设「沃迪内」（s10 note），实际译文 23x 用「军团魔」、仅 2 页 4x 用「沃迪内」。**判定**：按实际使用 canonical = 军团魔，update term_map + 改 pp.182/242 4 处 沃迪内 → 军团魔
- p015: C1 已修

### 跨页统一（与 Batch 3 同步）
- vordine：term_map 改回 军团魔；page_182 (1) + page_242 (3) 共 4 处 沃迪内 → 军团魔

### Batch 4 (pp.016-020, 2026-05-27 s13)
- p016/017/019/020: 深度复核 clean
- p018: NPC 名「内尔德·哈瓦萨乌」→「奈尔德·哈瓦萨乌」(term_map canonical, p218 已用 奈尔德)
- p020: 「异端魁首」保留（vs Ch11/城邦志「大异教官」的全本不一致问题暂记录不改，需用户判断后统一）

### Batch 5 (pp.021-025, 2026-05-27 s13)
- 全 5 页深度复核 clean，无新 issues
- p021/024/025 新读，p022/023 C3 阶段已修
- 术语 (异端魁首/卡蒂利娅/塞贝尼奥·杜·托林/乌贝诺·埃诺斯/埃默汀·欧恩/晨曦银/九芒星 等) 全部与 term_map 一致

### Batch 6 (pp.026-030, 2026-05-27 s13)
- p026/029/030 C2/C1 已修，clean
- p027: 「圣记官」（signifer）保留——与 p208「旗使」不一致，但 term_map「Hellknight signifer」= 地狱骑士圣记官，**记录跨页统一问题待决**
- p028: 「阴间登录官」→「地狱注册官」(Infernal registrar, term_map canonical; pp.148/208 已用 地狱注册官)

### Batch 7 (pp.031-035, 2026-05-27 s14)
- p031/032/034: 深度复核 clean，术语全部与 term_map 一致（傲世号 Imperious/Impervious、卡蒂利娅、铃花会/碎冠者/锁链骑士团/细缕者/破狱者、自审 Reckoning、Khari contingent 卡利联络组、Vidric revolution 维德瑞克革命、Kintargan Revolution 金塔戈革命）
- **p033**：1 处「地狱骑士遗骨骑士」→「地狱骑士死墓骑士」(graveknight canonical 死墓骑士；全本 7 文件 11 处用死墓骑士，仅 p033 一处用遗骨骑士)
- **p035**：1 处「阈空之门」→「阈限门径」(Liminal Doorway, pf2e_compendium + wiki canonical)

### 跨批次额外修复（与 Batch 7 同步）
- **p198 line 45**：「邪秽不死墓骑士」→「邪秽不死的死墓骑士」（缺一个「死」字 typo，全本死墓骑士 canonical 单字漏字修正）

### Batch 8 (pp.036-040, 2026-05-27 s14)
- p036/037: 漫长攀登障碍 + 地面层地图。深度复核 clean
- **p038**：2 处「警视画像」→「警惕肖像」(Watchful Portrait, pf2e_compendium canonical)
- **p039**：3 处修正
  - 「谜港」→「里德尔波特」(Riddleport, term_map line 524 + wiki canonical)
  - 「护手缠绕」→「缠手带」(handwraps, pf2e_compendium canonical)
- **p040**：1 处「强力出击护手缠绕」→「重拳缠手带」(Handwraps of Mighty Blows, pf2e_compendium canonical)
- p038/p039 records 衰弱 1（enfeebled 1/drained 1）+ 疲倦（fatigued）使用——与 pf2_cn/pf2e_compendium/wiki canonical（力竭/流失/疲乏）不一致，全本广泛使用，加入待决议题

### Batch 9 (pp.041-045, 2026-05-27 s14)
- p042/043/044/045 深度复核 clean，多数 stat block 与 NPC 名一致（乌罗·阿登、看守魔、路登·马尔迪努斯、特尼尔/卡沃斯、雅泽奇、宗-库山）
- **p041**：5 处法术/药水/法杖术语统一为 pf2e_compendium canonical
  - 清心术 → 清神醒脑（Clear Mind）
  - 强身术 → 身健体康（Sound Body）
  - 稳足术 → 稳固脚步（Sure Footing）
  - 高级治愈药水 → 高等治疗药水（Healing Potion, Greater）
  - 大型治疗法杖 → 上等治疗法杖（Staff of Healing, Major）
- **p042**：3 处「葛斯雷格」→「高斯瑞格」（Gosreg，pf2_cn re-zh_Hans canonical）
- 「改良擒拿」（Improved Grab）保留——pf2e_compendium canonical 为「精通擒获」，但项目「擒拿」用法广泛（23 文件 38 处），属项目惯例，不在 batch 级修

### 跨页统一（与 Batch 9 同步）
- **p120**：2 处「戈斯雷格」→「高斯瑞格」（与 p042 同步 Gosreg canonical）

### 已记录但不改的跨页不一致
- **Dominion of the Black**：p042 用「黑暗主宰」（sf2e_compendium canonical），p118/p120/p121 用「暗域之主」（共 5 处）——两译文皆有源头，加入待决议题

### Batch 10 (pp.046-050, 2026-05-27 s14)
- p046/047/048/049/050 深度复核 clean，绝大多数装备/法术/生物术语与 canonical 一致（高等强击/圣洁/邪秽符文石、高等瓶装闪电、燃疫松香、双足飞龙毒、高等生命灵药、万用制剂、海藻怨灵、误生尸、复合短弓、战用连枷、神圣形态、敕令打击、服从之触、神圣狂怒、鲜血铁棺、吸血鬼之宴、冻寒黑暗、信念崩塌、战斗祭司）
- 弃誓者（Foresworn）/ 厄登·砾掌 / 泽拉米尔 / 利亚布里斯·科尔博拉 / 拉西德·伊利托 全部与 term_map 一致
- 死墓骑士、不死、邪秽、措手不及 全部 canonical
- 萨格拉贡保留（AP 内部一致，与 pf2_cn 冲突已知）

### 跨页修复（与 Batch 10 同步：Necril canonical = 死灵语，pf2_cn）
- **p198 line 34**：「死语」→「死灵语」
- **p201 line 9**：「死语」→「死灵语」

### Batch 11 (pp.051-055, 2026-05-27 s14)
- p051-053 深度复核 clean（盐源庄园 + Ch3 开篇 + 地图）
- p054/055（围攻+亵典+异端讲座）深度复核 clean，多数术语 canonical：玛门 ✓、依斯嘉 ✓、仆魔（Gylou）✓、琳奈塔·海凿 ✓ term_map、Brinemidden Manor「盐源庄园」project canonical、安加沃尔魔（Ayngavhaul）/ 契约魔（Phistophilus）/ 莱瓦洛克（Levaloch）/ 科阿提魔（Coarti）保留——皆 AP 内部一致 transliteration（与 pf2_cn canonical 堕天魔/狱战魔 冲突，按 sarglagon 原则 AP-internal 优先）
- p055「恐惧 1/2/3」（frightened）使用——canonical「惊惧」（pf2e_compendium），项目 9 文件 46 处广泛使用，加入待决议题 #3

### Batch 12 (pp.056-060, 2026-05-27 s14)
- 全 5 页深度复核 clean：亵典/沃蕊/雷玛斯/灰萎/雷纳德·豪克/尼弗拉斯/邪魔螳螂/雾犬/灰萎/渐隐之风
- 多数装备/船只/法术 canonical 一致：狱火战靴、大海蛇、帆船、登船擒拿、高等治疗药水、高等游泳药水
- 「真实之语药剂」（potion of truespeech）保留——canonical「实言药水」但项目其他页面 truespeech 能力一律用「真实之语」，AP 内部一致优先
- 安弗里塔石柱群 / 御风者灯塔 / 加勒鲁斯 / 雷玛斯 / 灰萎 / 雾犬 等专名 project-specific OK

### Batch 17 (pp.081-085, 2026-05-27 s15)
- **p081**：2 处 heading 英文 difficulty 漏译修正
  - 「MODERATE 13」→「**严峻 13**」（D5 恐惧花园；项目 de facto pattern）
  - 「LOW 13」→「**低 13**」（D6 支配礼拜堂）
  - p081 其他术语 canonical：仙德拉/成年珊瑚龙 / 奈米拉/木林仙后变体 / 蝰蛇藤 / 支配矩阵 / 痛感珊瑚 作祟 / 系于海
- **p082**：4 处修正
  - 「MODERATE 13」→「**严峻 13**」（D8 苦痛使徒）
  - 「大型解毒剂（major antidote）」→「上等解毒剂」（pf2e_compendium equipment-srd canonical）
  - 「高级生命药水（greater elixir of life）」→「高等生命灵药」（canonical Elixir of Life Greater）
  - 「死帽菇粉（deathcap powder）」→「死帽粉」（canonical）
  - p082 其他 canonical：奎伦特 Quelaunt / 奥克腾 Aucturn / 黑色支配者（待决议题 #4 第 4 种变体）/ 巨章鱼/支配术杖
- **p083**：3 处修正
  - 「SEVERE 13」→「**严峻 13**」（D10 哀痛之厅 heading）
  - 「纱幕尊主级阿果尔苏」→「幕后尊主级阿果尔苏」（Veiled Master, pf2e_compendium B1 canonical）
  - 「化形（Change Shape）」→「变身（Change Shape）」（pf2e_compendium canonical, 与 p064 蜜鬼婆 stat block 同步）
- **p084**：12 处法术名 + heading + 装备 canonical 同步
  - 「LOW 13」→「**低 13**」（D11 梦境宝库 heading）
  - Xoxren stat block spell list 11 处 canonical 同步（与 p070 Oourax stat block 一致）：
    - 滑行术 → 黑缠蛇（Slither）
    - 死亡视界 → 魅影杀机（Vision of Death）
    - 致盲术 → 目盲术（Blindness）
    - 迅捷术 → 加速术（Haste）
    - 察形辨象 → 识破无形（See the Unseen）
    - 心灵之手 → 念动之手（Telekinetic Hand）
    - 心灵投射物 → 念动射弹（Telekinetic Projectile）
    - 祸难术 → 绝望术（Bane）
    - 偏执 → 狂乱偏执（Paranoia）
    - 盾击术 → 护盾术（Shield）
    - 高级生命药水 ×2 → 高等生命灵药 ×2（Elixir of Life Greater）
  - p084 Xoxren 整个 spell list 此前用了旧 D&D / 非 PF2 通译；与 Oourax (p070) 已是 canonical 形成对比；现统一为 canonical
- **p085**：2 处修正
  - 「多塔里行刑队」×2 → 「多塔里折磨者分队」（与 5 页 7+ 实例 canonical 同步）
  - 「阿斯摩蒂斯神选者」→「阿斯摩蒂斯神卫」（pf2e_compendium classes.json Champion=神卫 canonical, 与 p075/p127 项目内一致）
  - p085 其他术语 canonical：警觉点/优势点/游骑要员/眼之仆从/沉眠者护身符/萨格拉贡魔/元素海啸/法术冲击箭/斯戎特工/不死灯塔/泰莎被俘 等

### 跨页统一（与 Batch 17 同步）
- **p004 ToC**：「多塔里刑罚分队」→「多塔里折磨者分队」（Batch 1 误改纠正：当时声称对齐 stat block，但实际 stat block pp.012/018/030 全部用「折磨者分队」；现项目 7 页 8+ 实例统一）
- **p086**：3 处「多塔里行刑队/神选者」→「多塔里折磨者分队/神卫」（Batch 18 之前先做掉，避免再次发现）

### Batch 16 (pp.076-080, 2026-05-27 s15)
- p076 深度复核 clean（罗特罗维奥 14 级 stat block；spell list 全部 canonical：神圣裁决/噬星爆发/处死一指/伤害术/圣刃护球/召唤魔族/真视术/清神醒脑/神圣魂焰/飞行/法术解除结界/行动无碍/束缚死灵/目盲术/水中呼吸/水面行走/侦测魔法/圣枪术/指路术/传讯术/虚能噬/激现骸骨/将军令/卡利步兵/纯净军团小队/纯净之披风/船员）
  - 唯一保留：「真视术」（Truesight, project 4 页用此；pf2_compendium canonical「真景术」）——待决议题 #9
- p077 深度复核 clean（厄登催促/卡利获释/瑞玛奖励/水下武器属性符文/折磨之殿背景/沉眠者之眼/奥拉夫鲁/阿尔古苏/心能者/索克斯伦/神殿三圈结构/游荡巡逻）；术语 canonical：阿尔古苏 alghollthu / 心能者 Psychic / 命源 Vitalizing / 震波 Shockwave
- p078 深度复核 clean（折磨邪教徒/巨章鱼/审讯/外齐利亚尼 stat）；术语 canonical：阿扎克缇 / 巨齿鲨 Megalodon / 鬼火 Will-o-Wisp
  - 注：「严峻 13」为 MODERATE 13 误译（待决议题 #6）
- p079 仅地图 clean
- **p080**：2 处修正（heading 英文 difficulty 漏译）
  - 「TRIVIAL 13」→「**轻微 13**」（D3 试炼之室；与 p068 轻微 12 同样 canonical 一致）
  - 「MODERATE 13」→「**严峻 13**」（D4 绝望之礁；遵循项目 de facto MODERATE→严峻 pattern，待决议题 #6 解决后批量重译）
- p080 其他术语 canonical：乌拉卡尼侦察兵 stat / 海兽骑乘 / 血染海水 / 试炼之室 / 痛感珊瑚 作祟 stat
- 跨页发现：项目自 p080 起多页 heading 英文 difficulty 漏译（pp.080/081/082/083/084/093/095/097/100-105/112-115/119-120 等 25+ 页）。已批内修 p080，其他随各 batch 推进时修

### Batch 15 (pp.071-075, 2026-05-27 s15)
- p071/p073 深度复核 clean（结语续/Ch4 开篇/海盗与谋划/泰莎登场/3 地图）
- p072 深度复核 clean，多数术语 canonical 一致（罗特罗维奥/沃蕊/泰莎/咸眼邪教/阿斯摩蒂斯/折磨之殿/沉眠者之眼）；但首次发现：
  - **「咸涡之眼邪教」vs「咸眼邪教」**：p070 用咸涡之眼（2x），p072 用咸眼（4x）。后者更频繁、更字面接近 Brine-Eye。
- p074 深度复核 clean（三叉戟之吻战 / 塔洛尔潮汐巨人 / 折磨束缚者 / 乌拉卡尼绑架者 / 龙龟 / 纯净军团 / 锈铁魔 等术语 canonical 一致）
  - 注：「低 13 或 严峻 13」为 LOW OR MODERATE 13 误译（属待决议题 #6）
- **p075**：1 处修正
  - 「伊欧梅黛神卫变体」→「艾欧曼狄神卫变体」（Iomedae project canonical 艾欧曼狄；p180/p181/p184/p185 共 6+ 处用艾欧曼狄；仅 p075 一处误用伊欧梅黛）
- 其他 stat block 术语 canonical：旺达·喷毒鞭 / 神卫灵光 / 虚能之触（Touch of the Void）/ 锈铁魔 Ferrugon / 弓箭兵团 / 精英骷髅步兵 / 硫磺军团 Brimstone Corps / 高等梦魇
- p075 城墙之战 严峻 13 = SEVERE 13 ✓（正确，对应待决议题 #6 中 SEVERE 译法一致）

### 跨页统一（与 Batch 15 同步）
- **page_070**：2 处「咸涡之眼邪教」→「咸眼邪教」（与 p072 4 处 canonical 同步；Brine-Eye Cult 项目 6 处全部归一）

### 已记录但不改（Batch 15）
- Iomedae「艾欧曼狄」（项目用法 7+ 页）vs pf2e_compendium canonical「艾奥梅黛」——加入待决议题 #8

### Batch 14 (pp.066-070, 2026-05-27 s15)
- p068 深度复核 clean（伊格拉拉/地狱矛号/活体投石机/遗产号/混淆背叛作祟/F 被遗忘之泪之井开篇）；术语 canonical：阈值/段/小规模冲突遭遇/Edge Point 优势点 一致
- p069 深度复核 clean（F 区事件 / 灵魂撕裂者作祟 / 精灵浪骑分队 / 塞达克西人战团 / 冥河海蛇）
- **p066**：4 处条目修正（B3 canonical 同步紊传畸体异能术语）
  - 「失稳之场」→「失衡之域」（Destabilizing Field, pf2e_compendium B3 canonical）
  - 「失稳生物」→「失衡生物」
  - 「失稳状态」→「失衡状态」
  - 「失稳」（其他）→「失衡」
  - 项目仅 p066/p067 出现失稳系列共 2 页，localized 修复
- **p067**：2 处修正
  - 「失稳生物」→「失衡生物」（Transpose 异能内）
  - 「地狱潮汐分队」→「地狱浪潮分队」（infernal tide troop, p004 + p241 canonical 地狱浪潮）
- **p070**：2 处修正
  - 「凶悍（Ferocity）」→「凶猛（Ferocity）」（pf2e_compendium ability glossary canonical, 项目仅 1 处用凶悍）
  - 5 环 法术「滑溜」→「黑缠蛇」（Slither, pf2e_compendium spells-srd canonical）
- 其他 spell list 全部 canonical 一致：洞烛机先（True Target）/心灵扭曲（Warp Mind）/闪电链/解除魔法/放逐术/唤起魂灵/飞行/行动无碍/火墙术/火球术/加速术/回避帷幕（Veil of Privacy）/识破无形/料敌机先（Sure Strike）/侦测魔法/引火术（Ignition）/护盾术/念动之手/虚能噬（Void Warp）/瞬时洞见（Flash of Insight）
- 其他 stat block 术语 canonical：紊传畸体异能名 Reposition 移位 / Transpose 移调转置 / Displace 位移（B3 用调换，feats-srd 用移位/移调转置；项目沿用 feats-srd canonical，与 B3 名词级差异保留——已记录待决议题）

### 跨页统一（与 Batch 14 同步）
- **page_069**：3 处「地狱潮汐」→「地狱浪潮」（title summary + 主文 + stat block 名）
- **page_128**：1 处「地狱潮汐」→「地狱浪潮」（infernal tide troop 主文引用）
- **page_129**：1 处「地狱潮汐」→「地狱浪潮」（stat block heading）
- p186 line 39「这地狱潮汐」描述性比喻保留（非 troop 名引用）

### Batch 13 (pp.061-065, 2026-05-27 s15)
- p061/062 深度复核 clean（B5 续/雷玛斯豆罐边栏/纳罗纳废墟/14 道门/煤烬姊妹/奥拉克斯背景）；术语 canonical：副执棒官卡琳娜·寇尔、Order of the Pyre 焚烧骑士团、Abrogail I 阿波罗盖一世、Recall Knowledge 回忆知识
- p065 仅地图标签，clean
- **p063**：4 处修正
  - **「鬼婆三联 14 级」→「鬼婆三联 13 级」**（英文 LEVEL 13，clear typo 数字 off by 1）
  - 「成年余烬龙」→「成年灰烬龙」（adult cinder dragon, pf2e_compendium MC2 line 11108 canonical）
  - 「砒霜」→「蜜糖与砒霜」（Poisoned Candy, pf2e_compendium MC 189 canonical 双关名）
  - 「鬼婆团」→「巫团」（Coven, pf2e_compendium MC 189 canonical, 与 page_149 一致）
- **p064**：2 处修正
  - 「鬼婆团（Coven）」stat block ability + 「鬼婆团法术列表」→ 「巫团」（同上 canonical）
  - 「砒霜（Poisoned Candy）」→「蜜糖与砒霜」（同 p063）
- 其他 sweet hag 异能（背叛之触/变身/昏睡之触）+ NPC 名（煤姊妹/烬姊妹/灰姊妹）+ 红帽精/失智术/虫颚打击/拐杖糖爪 全部 canonical 一致
- 记录但不改：p061「察觉检定察觉动机」省连接词读起来略生涩但不算错；p063/064「拐杖糖爪 candy cane claw」vs pf2_compendium「棒糖爪」——「拐杖糖」更字面，保留

### Batch 18 (pp.086-090, 2026-05-27 s16)
- p086/087/088 深度复核 clean（Ch4 最终突击 stat block / 街区表 / 派系机遇）
- **p089**：1 处「艾格利安」→「埃戈里安」（Egorian term_map line 215 canonical）
- **p090**：2 处修正
  - 「严重 14」→「**严峻 14**」（SEVERE 14 canonical）
  - 「安菲利塔之柱」→「安弗里塔石柱群」（Pillars of Anferita，p057 canonical）
- 记录但不改：厄尔芬（Ulfen，无 TM canonical，2 页限定，保留）/ 巨像（juggernaut，p090/p091 12 处建立 AP-internal canonical，与 TM「钢铁之躯」冲突）

### Batch 19 (pp.091-095, 2026-05-27 s16)
- **p091** (8 处)：3x 探索者→探路者（书名）、2x 化物罗刹→**忿怒罗刹**（Raja-Krodha wiki canonical）、2x 奥奇阿斯提斯→**奎伦特**（term_map line 444 self canonical，待决议题 #10 见下）、1x 尼斯洛切湾→尼斯洛克湾（项目 p007/p207 majority canonical）
- **p092** (9 处)：严重 14→**严峻 14**、1x 探索者→探路者、3x 化物罗刹→忿怒罗刹（含 stat block）、4x 奥奇阿斯提斯→奎伦特（含 stat block + 引文）
- **p093** (6 处)：3x 探索者→探路者、1x 奥奇阿斯提斯→奎伦特、1x「强击回掷匕首」→「强击**回力**匕首」（returning rune canonical 回力 pf2_cn）、1x 螺旋钟→**螺旋风铃**（spiral chimes canonical）
- **p094** (2 处)：「回想知识」→「**回忆知识**」（Recall Knowledge canonical，项目 9 文件 10 处用「回忆」，仅 p094/p112 用「回想」）、「艾布罗盖尔」→「**阿波罗盖**」（Abrogail，1 处与项目 240+ canonical 同步）
- p095 深度复核 clean（MODERATE 难度 14 属待决议题 #6，记录不改）
- 记录但不改：「拜尼卢斯」（Jilia Bainilus，wiki canonical「贝尼勒斯」，AP 内部 5 处一致保留）/「远转」（translocate，TM canonical「次元移位」，19 文件项目惯例保留）/「多米娜」（Domina，TM canonical「统御者」，项目音译保留）/「塔洛斯」（talos，TM canonical「金元素裔」，项目音译保留）

### 跨页统一（与 Batch 19 同步）
- **p105**：1 处「+1 强击回掷长矛」→「+1 强击**回力**长矛」（与 p093 returning rune 同步，项目 2 处全清）
- **p112**：1 处「回想知识」→「**回忆知识**」（与 p094 同步，项目 2 处全清）

### Batch 20 (pp.096-100, 2026-05-27 s16)
- **p096** (5+ 处)：「探索者战役设定」→「**探路者战役设定**」、「探索者公会」→「**探索者协会**」（Pathfinder Society wiki canonical，公会 vs 协会区别 in-game org vs brand）、3x 探索者→探路者（NPC 核心 + 野性嚎啸）、「野性嚎啸」→「**荒野之嚎**」（Howl of the Wild canonical）、「精良抗火护符」→「**上等**抗火护符」（major = 上等，与 Batch 17 解毒剂同步）
- **p097** (7 处)：3x 探索者→探路者、严重 14→**严峻 14**、「高级反制护符」→「**高等**反制护符」（greater = 高等）、「+2 高级回弹冒险者袍服」→「+2 **高等**回弹冒险者袍服」（同上）、「处决术」→「**处死一指**」（execute canonical pf2_cn）
- **p098** (5 处)：1x 探索者→探路者、2x 处决术→处死一指、1x 高级反制护符→高等反制护符、1x「+2 高级强击朽冻」→「+2 **高等**强击朽冻」（同上 rune tier）
- p099 深度复核 clean（米纳多山脉 / 诡谎魔 / 闪光岩堡 / 梅兰迪亚 等 AP-specific term 全部 project-internal canonical）
- **p100** (3 处)：2x 探索者→探路者、「探路者野性嚎啸」→「**探路者荒野之嚎**」
- 记录但不改：「真言术」（truespeech canonical「真实之语」，项目 8 文件 12 处广泛使用，加入待决议题 #11）/「神职」（cleric canonical「牧师」，项目惯例保留）/「高阶祭司」（high priest canonical「大祭司」，项目用法保留）/「反制护符」（charm canonical「咒符」，项目「护符」与抗火护符一致保留）

### 跨页统一（与 Batch 20 同步）
- **p117**：1 处「**处决术**」→「**处死一指**」（execute 法术，与 p097/p098 同步，项目 3 文件全清）
- **p102**：1 处「《探索者：野性嚎啸》187」→「《**探路者荒野之嚎**》187」（缓步生物群 stat block）
- **p114**：1 处「《**Pathfinder 野性嚎啸**》139 页」→「《**探路者荒野之嚎**》139 页」（斯戎暴龙 stat block）

## 已知待决议题（需用户决策后再批量改）

1. **「异端魁首」vs「大异教官」**（Archheathen 同英文异译）
   - 异端魁首：8 文件 19 处（Ch1-Ch5）
   - 大异教官：6 文件 12 处（寇兰廷城邦志 pp.212-218）
   - 同人 Kettermaul Charthagnion 的官职头衔。 page_213 etymology 段落明确用「大异教官（arch-heathen）」做来源解释——若改成「异端魁首」需重写该段。需用户判断。

2. **「signifer」单用**（与 Hellknight signifer 复合用法关系）
   - p027 用「圣记官」
   - p208 用「旗使」
   - term_map：Hellknight signifer = 地狱骑士圣记官，signifer 单用 = 旗使（B 阶段所设，可能需对齐）

3. **条件状态译名**（pf2_cn / pf2e_compendium / wiki canonical vs 项目实际用法）
   - **Enfeebled** canonical=「力竭」，项目用「衰弱」（18 文件含 enfeebled 上下文的多数）
   - **Drained** canonical=「流失」，项目用「衰弱」（p039/p120/p178/p250 等，与 enfeebled 同字撞名）
   - **Fatigued** canonical=「疲乏」，项目用「疲倦」（7 文件）
   - **Frightened** canonical=「惊惧」，项目用「恐惧」（9 文件 46 处）
   - 涉及范围广（衰弱 18 + 疲倦 7 + 恐惧 9 文件）；项目早期翻译惯例与官方 canonical 偏离。需用户判断是否批量统一为 canonical。

4. **Dominion of the Black 译名**（项目内部 5 处不一致）
   - 「黑暗主宰」：p042 用（与 sf2e_compendium canonical 一致）
   - 「暗域之主」：p118/p120/p121 用，共 5 处（Ch6 主线 Verlar Bretan 巫妖背景）
   - pf2_cn / pf2e_compendium 未收录此术语，sf2e_compendium 为唯一权威源——但项目主流（Ch6）用「暗域之主」也有合理性。需用户判断。

5. **Improved Grab 译名**（pf2e_compendium canonical vs 项目惯例）
   - canonical=「精通擒获」（pf2e_compendium bestiary-ability-glossary）
   - 项目用「改良擒拿」（p043/p044）+ 「擒拿」/ 「擒抱」（23 文件 38 处）
   - 项目早期已定型，全面替换风险高。需用户判断是否统一为 canonical。

6. **难度等级 MODERATE / SEVERE 译名撞名**（项目大规模问题，Batch 14 发现）
   - 项目实际使用：MODERATE → 「严峻」 + SEVERE → 「严峻」（两个等级被压扁到同一中文词）
   - canonical：MODERATE = 中等 / SEVERE = 严峻
   - 已确认 MODERATE 被译为「严峻」的页面：p012, p022 (x2), p028, p029, p030, p040, p042, p044, p046, p049, p050, p054 (x2), p056, p057 (x2), p058, p060 (x2), p067 (D 召唤圆环), p074
   - 已确认 SEVERE 被译为「严峻」的页面（实际正确）：p048, p064 (C 倾塌之塔), p068 (F 被遗忘之泪之井) 等
   - 影响范围：~20+ 页。需用户判断：是否批量改 MODERATE 系列为「中等」（保留 SEVERE 为「严峻」）

7. **Amalgamite「融合体」vs 紊传畸体（B3 canonical）**
   - 项目 pp.066/067/068 用「融合体」13 处，建立 project-internal canonical
   - pf2e_compendium B3 canonical=「紊传畸体」（Pathfinder Bestiary 3）
   - 异能名同步差异：项目「失稳之场/失稳生物」→ Batch 14 已改为「失衡之域/失衡生物」（B3 一致）
   - 但「融合体」生物名本身未改（避免破坏项目内部统一）；其他异能（位移/移位/移调转置 vs B3 调换/移形/易位）也保留项目用法（feats-srd canonical 一致）
   - 需用户判断：是否将 13 处「融合体」改为「紊传畸体」（B3 canonical）

8. **Iomedae 神祇译名**（项目惯例 vs canonical）
   - 项目用「艾欧曼狄」（p180/p181/p184/p185 等 7+ 处，已确立 project canonical）
   - pf2e_compendium 神祇 deities.json canonical=「艾奥梅黛」
   - Batch 15 已修正 p075 单处「伊欧梅黛」（项目第三种变体）→「艾欧曼狄」（project canonical）
   - 需用户判断：是否将 7+ 处「艾欧曼狄」改为「艾奥梅黛」（pf2e_compendium canonical）

9. **Truesight 法术译名**（项目惯例 vs canonical）
   - 项目用「真视术」（p076/p176/p199/p201/p202 共 4 页 5 处）
   - pf2e_compendium spells-srd canonical=「真景术」
   - 「真视术」是 D&D/老 PF1 通译惯例；「真景术」是 PF2 SRD 的新定译
   - 需用户判断

10. **Ochiastis 译名混淆**（项目内部 term_map 错误，B19 暴露）
    - 项目 term_map line 444 设：Ochiastis → 奎伦特（self source）
    - 但 Quelaunt（MC2 异怪/类星魔，p082 用）与 Ochiastis（MC2 刺客魔鬼，p243-244 真身）在 PF2 是**两种不同生物**
    - 项目 p004 ToC / p082 / p244 / p091 / p092 全部以「奎伦特」统称二者
    - B19 已按项目 canonical 同步（p091/p092 奥奇阿斯提斯 → 奎伦特），但底层混淆未解
    - 需用户判断：是否拆分（奎伦特=Quelaunt 异怪，奥奇阿斯提斯=Ochiastis 刺客魔鬼）

11. **Truespeech 持续法术译名**（项目惯例 vs canonical，B20 暴露）
    - 项目用「真言术」（p098/p174/p175/p176/p191/p196/p201/p202 共 8 文件 12 处）
    - pf2_cn TM canonical=「真实之语」（仅 p056/p064 共 2 文件 3 处用 canonical）
    - 备注: 之前 B12 记忆「项目其他页面 truespeech 能力一律用真实之语」实际有误——实测主流是「真言术」
    - 需用户判断：是否批量改为「真实之语」

---
