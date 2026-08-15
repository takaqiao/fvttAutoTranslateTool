# -*- coding: utf-8 -*-
"""Round-22: splice the generated ARRANGEMENTS body into ember-hardcoded-cn.mjs.

Does an exact-anchor replacement (both anchors must be found exactly once,
otherwise exit 2 — no fuzzy patching), and refuses to run if the generated body
does not contain the expected number of lines.

Anti-空转: prints the byte size before/after and the key count it wrote.
"""
import io, os, re, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
MJS = os.path.join(BASE, r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

body = io.open(os.path.join(HERE, "arrangements_body.txt"), encoding="utf-8").read().rstrip("\n")
lines = body.split("\n")
print(f"generated body lines: {len(lines)}")
if len(lines) != 204:
    print("UNEXPECTED BODY SIZE")
    sys.exit(2)
body = body.rstrip(",")  # last entry must not end with a comma

src = io.open(MJS, encoding="utf-8").read()
before = len(src)

DOC_OLD = """/**
 * 音景「编排」名。arrangement.label 是 ember.mjs 里的硬编码常量
 * （5694 / 5787 / 12393 / 14064 / 14748 各 var 块），babele 与 i18n 两条通道都够不到。
 *
 * ⚠ 这张表**只装编排名**（外加 `Reset`，理由见下）。第二十一轮以前它一张表同时兜两档
 * （9 键里 4 个其实是 optgroup 组名），混着长下去迟早出误命中，已拆开：组名一律进
 * SOUNDSCAPE_GROUPS。实测 42 个组名 / 212 个去重编排名里有 **7 条同串**，同串只写一次、
 * 写在组名表；本表现在的 4 条编排名（`Shent Ruins` / `Shent Ruins Tension` /
 * `The Pit Trap - Intense` / `The Pit Trap - Relaxed`）都**只是编排名、不是组名**，与组名表零重叠。
 * 覆盖面没变：编排名 212 条里翻了 11 条（本表 4 ＋ 组名表里同串的 7），恒英文 201 条。
 */"""

DOC_NEW = """/**
 * 音景「编排」名。arrangement.label 是 ember.mjs 里的硬编码常量
 * （5694 / 5787 / 12393 / 14064 / 14748 各 var 块），babele 与 i18n 两条通道都够不到。
 *
 * ⚠ 这张表**只装编排名**（外加 `Reset`，理由见下）。第二十一轮以前它一张表同时兜两档
 * （9 键里 4 个其实是 optgroup 组名），混着长下去迟早出误命中，已拆开：组名一律进
 * SOUNDSCAPE_GROUPS。实测 42 个组名 / 212 个去重编排名里有 **7 条同串**，同串只写一次、
 * 写在组名表（`Ancient Ruins` / `Ankarist Theme` / `Lyla Theme` / `Marlstone Gala` /
 * `Ordain` / `Sin Theme` / `The Pit Trap`），本表**一条都不重复写** —— 重复写会让
 * MOOD_PANEL 的 `{...SOUNDSCAPE_GROUPS, ...ARRANGEMENTS}` 里后者把组名译文顶掉。
 *
 * **第二十二轮：编排名补完**（项目所有者要求；第二十一轮「暂不补」的裁决到此为止）。
 * 本表 = `Reset` ＋ **204 条编排名**；加上组名表里那 7 条同串，212 条去重编排名里
 * 覆盖 **211**、恒英文 **1**（`Seven Sails`，理由见文末）。编排名恒英文 **201 → 1**。
 * 全集不是估计：`4-临时脚本/2026-08-16-round22/probe_groups_r22.mjs` 重跑第二十一轮那套
 * 双方法互校探针（手写大括号配对正则 ／ node:vm 真解析器 ＋ `obj.id === 注册表键`），
 * 自报 `A=44 B=44 BAD=0`、两法逐条相同、`267 条 / 去重 212 条 / 同串 7 条`；
 * 表体由 `gen_arrangements.py` 从该探针产物生成，生成器**要求 212 条各自落进
 * 「同串／留英／本表」三桶之一**，剩一条就 exit 2，落不进桶的不会被悄悄漏掉。
 *
 * ── 结构词统一译法（先定表再逐条套，不要一条一个样）──────────────────
 *   时段    `Day`白天 · `Night`夜晚
 *   情绪    `Calm`平静 · `Tension`紧张（＝ MOODS 表，两条通道同译）· `Quiet`静谧 ·
 *           `Chaos`混乱 · `Sad`哀伤 · `Intense`激烈 · `Relaxed`舒缓 · `test`测试
 *   曲式    `Main`主段 · `Section N`第 N 段 · `Interlude`间奏 · `Interval`间歇 ·
 *           `Rises`渐强 · `Verse`主歌 · `Chorus`副歌 · `Bridge`桥段 · `Melody`旋律 ·
 *           `Rhythm`节律（glossary_ec `Rhythm`＝节律）
 *   曲风    `Heroic`英勇 · `Atonal`无调性 · `Spooky`阴森 · `Weird`诡谲 · `Dramatic`戏剧性
 *   战斗    `X Fight` / `X Combat` 一律作「X战斗」，与 SOUNDSCAPE_GROUPS 的战斗组名同调
 *   分隔符  一律 ` · `（沿用组名表的 `元素战斗 · 火` / 本表原有的 `陷坑 · 激烈`）
 *   ⚠ `Chaos`＝混乱 与 glossary_ec 的 `Chaos`＝混沌 **有意不同**：这里是实验室里乱作一团的
 *     环境音，不是宇宙学意义的「混沌」。同理 `Weird`＝诡谲 **不取** glossary_ec 的
 *     `Weird`＝怪影杀手（那是生物名，另一个域）。
 *
 * ── 专名依据（一律先过 compendium 英文闸，大小写逐条单独判；括号内为命中的英文串）──
 * 探针 `gate_arr.py`（两仓库 en/cn 配对 43115 条英文叶 ＋ 两份 lang）与
 * `exact_arr.py`（拿 212 条标签整串去撞已译英文叶）自报扫描量，`gloss_probe.py`
 * 再把每条标签的各级子串拿去查 glossary_ec（7974 条）。三者一致的才写进来：
 *   · 整串命中已译英文叶（最强）：`Burial Grounds`墓地 · `Fogbound Caverns`雾缚洞窟 ·
 *     `Inkaro Pools`因卡罗水潭 · `Kaleidoscope Caverns`万花筒洞窟 · `Mycelian Expanse`菌丝旷野 ·
 *     `Pathways`通路 · `Primordial Bastion`原初堡垒 · `Signara`西格纳拉 ·
 *     `Spellbreaker Tower`破法者之塔 · `The Ballad of Dereth Erekos`德雷斯·埃雷科斯之歌 ·
 *     `Yakoshta Mine`雅科什塔矿井 · `Ember Cosmos`余烬寰宇（lang 里 1 条整串）。
 *   · 去掉结构词后整串命中：`Amerasp Grove`阿梅拉斯普林地 · `Arcturel`阿克图瑞尔 ·
 *     `Corpin Sanctuary`科尔平庇护所 · `Dripstones`滴石笋 · `Ember's Bounty`余烬的恩赐 ·
 *     `Forest of Stone`石之森林 · `Golden Flats`金色平原 · `Helkas`赫尔卡斯 · `Nain`奈因 ·
 *     `Ocean`海洋 · `Ordain Docks`奥尔丹船坞 · `Ordain Flats`奥尔丹平原 ·
 *     `Ordain Spires`奥尔丹尖塔区 · `Redrak Fields`雷德拉克原野 ·
 *     `Rustvar Valleys`鲁斯特瓦尔山谷 · `Sarin Strand`萨林海滨 · `Seawall`海堤 ·
 *     `Skybrush`天刷镇 · `Splinter Canyons`碎裂峡谷 · `Steed's Point`斯蒂德角 ·
 *     `The Teeth`卡迪索斯之牙 · `Tidal Pools`潮汐池 · `Verdant Paths`翠绿径 ·
 *     `Wedgelands`楔地 · `Yakoshta`雅科什塔 · `Graven's Rest`格雷文之憩。
 *   · 命中同族短名，按同一构词法套用：`The Bleak Archive`黯淡秘库 → `Bleak Archive` ·
 *     `The Cauldron`坩埚湖 → `Cauldron` · `Volcanic Bluffs`火山峭壁 → `Bluffs`峭壁 ·
 *     `Clouded Jungle`迷雾丛林 → `Jungle`丛林 · `Mountains of the Sun`太阳群山 →
 *     `Mountains`群山 · `The Broken Tower`破碎之塔 · `The Scrapyard`废料场 ·
 *     `The Waterworks`水务工程 · `Kalion Stadium Underworks`卡利昂竞技场地下工事 →
 *     `Stadium Underworks`竞技场地下工事 · `Overwatch Garrison`守望驻军营 →
 *     `Garrison`驻军营 · `Redrak Farm`雷德拉克农场 ＋ `Ooze Pools`软泥池 →
 *     `Ooze Farm`软泥农场 · `Toothbreaker Hideout`碎牙帮藏身处 ＋ `Raiders`劫掠者 →
 *     `Raiders' Hideout`劫掠者藏身处 · `Noxious Spit`剧毒唾液 → `Noxious Cave`剧毒洞穴 ·
 *     `Clockwork Feather`发条羽毛 ＋ GM 指南 `Dungeon`地下城 → `Clockwork Dungeon`发条地下城 ·
 *     `Writhing Grave`蠕动之墓 → `Kaleidoscope Grave`万花筒之墓 ·
 *     `Brevin Festival`布雷文庆典 → `Helkas Festival`赫尔卡斯庆典 ·
 *     `Vineyard Attack`葡萄园袭击 → `Helkas Attack`赫尔卡斯袭击 ·
 *     `Vista: X`＝远景：X → `Camp Vista`营地远景 ·
 *     `A Song for Lady Stonecraft`献给石艺女士的一首歌 → `Lady Stonecraft`石艺女士 ·
 *     `Shrine to Spectra`斯佩克特拉圣祠（Shrine ＋ 神名）→ `Shrine of Nite`奈特圣祠
 *     （`Nite` 是碎片之神，合集 `pages/Nite/name` 整串作「奈特 Nite」）·
 *     `Signara Water`西格纳拉水域 → `Golden Flats Water`金色平原水域 ·
 *     `Ordain Interior vista`奥尔丹室内远景（合集正文逐字）→ `Ordain Interior`奥尔丹室内 ·
 *     `Mutagist X`突变学派X ＋ `Empty Laboratory`空实验室 → `Mutagist Laboratory`突变学派实验室 ·
 *     `Bandit`强盗（`Carmin the Bandit`强盗卡尔敏）· `Drake`龙兽 · `Rejarh`雷贾尔
 *     （合集正文「浮空之城雷贾尔沉入了海浪之下」）→ `Sunken Rejarh`沉没的雷贾尔 ·
 *     `Seydiri`塞迪里（合集正文「塞迪里文化」，与组名 `Seydiri Theme`＝塞迪里主题 同调）。
 *   · glossary_ec 定稿的单词：`Bloodletter`放血者 · `Spires`尖塔 · `Ancient`远古 ·
 *     `Giants`巨人 · `Upper`上层 / `Lower`下层 · `Water`水域 · `Vista`远景 · `Folk`民谣
 *     （＝三条 `X Folk` 组名的既定译法）。
 *   · 依据只到「同族词 ＋ 构词」这一档的 4 条，写明以免下轮误当定稿：
 *     `Blood Woods`血色森林（`Woods` 两仓库 0 命中；按同组 `Golden Flats`＝金色平原 的
 *     「颜色词＋地貌」构词，`Forest`＝森林 取自 `Forest of Stone`＝石之森林）·
 *     `Rock Spires`岩石尖塔（`Rock Spires` 整串 0 命中，`Spires`＝尖塔 有据）·
 *     `Shipwreck`沉船（整串 0 命中，通用名词）·
 *     `Ocean Ship`海洋 · 船上（`Ship` 无已译叶；同组另两条是 `Ocean Day/Night`，
 *     故按「海洋 ＋ 变体」处理，不当成一个地名）。
 *
 * ⚠ `Marlstone Gala Tension` 取组名表的「马尔石晚会」，**不取** glossary_ec 的
 *   「马尔斯通晚会」：英文闸下 compendium 三处（`The Marlstone Gala` / `Vista: Marlstone Gala`）
 *   全作「马尔石晚会」，组名表也已定「马尔石晚会」。词表那条是孤例，本文件不跟。
 * ⚠ `Shent Water Temple` 作「申特水之神殿」（跟组名 `Water Temple`＝水之神殿），
 *   **不写成**合集里那座建筑的场景名「申特月神殿」（`Shent Moon Temple`）—— 英文闸按键判，
 *   本表的键是 `Shent Water Temple`，上游对同一栋楼有两种写法，各译各的。
 * ⚠ 有 4 条译文一串对两个键，全部是**上游拼写变体指同一处地方**，不是撞名：
 *   `Teeth Day/Night` ↔ `The Teeth Day/Night`、`Rustvar Valley Day/Night` ↔
 *   `Rustvar Valleys Day/Night`。生成器专门把这一项打印出来核对过（只此 4 条）。
 * ⚠ `Seven Sails` **故意留英**：两仓库 43115 条英文叶里 `Seven Sails` en-hits=0、
 *   连 `Sails` 单词都 0 命中，glossary_ec 也没有，模块自带语料里查不到这个名字指什么
 *   （酒馆？船？曲名？）。宁可露英文也不猜 —— 下一轮的英文残留扫描若报到这 1 条，
 *   属**预期内**，按本条驳回；等上游正文出现这个名字再补。
 */"""

if src.count(DOC_OLD) != 1:
    print(f"DOC anchor count = {src.count(DOC_OLD)} (expected 1)")
    sys.exit(2)
src = src.replace(DOC_OLD, DOC_NEW)

TABLE_START = 'const ARRANGEMENTS = {'
i = src.index(TABLE_START)
j = src.index("\n};", i) + len("\n};")
old_table = src[i:j]
if old_table.count('"Reset": "重置"') != 1:
    print("Reset anchor missing")
    sys.exit(2)
head = old_table.split('"Reset": "重置",')[0] + '"Reset": "重置",\n\n'
new_table = head + body + "\n};"
src = src[:i] + new_table + src[j:]

io.open(MJS, "w", encoding="utf-8", newline="\n").write(src)
print(f"file bytes {before} -> {len(src)}")
print(f"wrote {len(lines)} arrangement keys + Reset")
