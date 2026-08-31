# 书2 —— 肆季鬼志 第二幕《任叶飘落》补充曲库

模组自带的 8 个播放列表（原声带 / 循环播放的配乐 / 氛围 Ambiance 等）不在此列，
这里只放**社区补充层**，对应 书1 那种逐场景／逐 NPC 的选曲。

## 部署状态

**已部署**（2026-08-29）：21 个 .ogg 已上传到
`/root/fvtt14-data/Data/assets/SoG/Book2/`（VPS `CN` = 139.196.146.211）。
MD5 全量比对一致，播放列表 29 条路径在服务器上逐条解析全部命中。

剩下的一步：在 Foundry 里导入 `fvtt-Playlist-书2.json`。

重新部署时：把 21 个 .ogg 放进 `Data/assets/SoG/Book2/`（目录名必须是 `Book2`，
播放列表路径写死了），然后导入 JSON。

## 排序

按 Book 2 叙事顺序编号。`sorting="a"` 是按 name 排，每条 name 以两位序号开头，
所以清单顺序 == Foundry 侧栏顺序。

| 序号 | 段落 | 内容 |
|---|---|---|
| 01–02 | **第1章 四季轮转** | 神藏（第3周首访 p18／第10周建议 p24）、无脸鬼（第6周 p21）|
| 03–05 | **第2章 · 穿过鬼打墙** p29–32 | 进入段 → A1-A3 与食心秃鹫战 → 种下魂种触发苦迦菩提的愤怒 |
| 06–11 | **第2章 · 朝圣之径** p33–43 | 行军床垫 ×2 → 第一日 B1 伊奥加卡 → 硬仗 → 第三日 D1/D2 恶灵 → 抵近寺院 |
| 12–19 | **第3章 · 谭杉寺** p45–58 | 探索／悬疑 → 苦迦菩提武僧 ×2 → 通用苦迦菩提遭遇 → 智惠现身／长谈 → 转生 |
| 20–25 | **第3章 · E16 苦迦菩提之墓** p59–61 | 心月两阶段（+3 条备选）→ 英雄揭示收尾 |
| 26–29 | **通用** | 险径探索 / 山行 / 战鼓 / 夜幕 |

播放列表共 **29 条**：其中 21 条指向 `assets/SoG/Book2/`，
8 条复用库里已有文件的**原路径**（NPC / Book1 / Pandaria / Suspense / Special Scenes），
不搬动也不复制任何旧文件 —— 上传前已逐条确认这 8 个路径在服务器上存在。

`folder` / `sorting` / `channel` / `mode` 与 书1 完全一致（同一目录、按名排序、音板模式）。

## 曲目来源

| 来源 | 条目 |
|---|---|
| **Arch1v3** 的 Book 2 精选单（PF2E Discord，经 WolfBrother 汇编文档转录）| 04 05 07 08 09 10 12 20 21 25 |
| **lilithilu**《九日 Nine Sols》方案（2025-11-21）| 18 22 23 |
| **Auroroth**《黑神话：悟空》等推荐（2026-08-08）| 14 15 16 24 26 27 28 29 |
| 库里早有、原先归在 书1 / Pandaria / 悬疑 / 特殊场景 / NPC 下的 Book 2 内容 | 01 02 03 06 11 13 17 19 |

WolfBrother 汇编文档（公开）：
<https://docs.google.com/document/d/1nmU7wyrcQkekdCcVM8q93Npb3ggIAOTeESmqNJZEkiQ/edit>

## 关于 Arch1v3 原单里的 4 条死链

原单 10 条里有 4 条 YouTube 链接已失效。处理如下：

| 槽位 | 原链接 | 处理 |
|---|---|---|
| 05 苦迦菩提前奏 | `NkYejDiG9xg`（转私密）| **已复原**。Wayback 快照（20260414135402）读出曲名 `Drakengard OST - Chapter XIII ~ Closing`（Nobuyoshi Sano，152s），改用 Nobuyoshi Sano 官方 Topic 频道同长度版本 `Y8dObNK8w4c`。与 Arch1v3 在 Book 3 两次选用 Drakengard「第十三章 Closing」互为佐证。|
| 04 穿透鬼打墙／食心秃鹫 | `bniffIOGNgI` | **不可考**，用替补 |
| 07 朝圣之径氛围 | `qmESLRxus14` | **不可考**，用替补 |
| 10 朝圣之径恶灵 | `qBaele3z2lU` | **不可考**，用替补 |

这三条查过：Wayback（无快照，或快照是已删除页）、CDX 全时间轴、i.ytimg 缩略图（404）、
archive.ph、全网搜索 video id —— 都没有。替补是从同批社区推荐里按用途挑的，
不是原曲，`_manifest.tsv` 的「来源标记」列标了 `替补`。

真要还原，只能去 **Pathfinder on Foundry VTT** 服务器翻 Arch1v3 的原帖
（那个频道不在 `SoG/SoGChat/` 存下的 PF2E 存档里）：
<https://discord.com/channels/613968515677814784/1159170386404053084/1288548001614135356>

## 译名依据

人名地名一律取模组翻译文件 `SoG/pf2e-season-of-ghosts.adventures.json` 的
`entries['Season of Ghosts']` 下 `actors` / `scenes` / `folders`：

心月 Xin Yue · 智惠 Zhi Hui · 伊奥加卡 Iogaka · 谭杉寺 Tan Sugi Monastery ·
苦迦菩提 Kugaptee · 苦迦菩提之树 Kugaptee's Tree · 苦迦菩提的愤怒 Kugaptee's Anger ·
苦迦菩提武僧 Monk of Kugaptee · 鬼打墙 Wall of Ghosts · **食心秃鹫 Heart-Eating Vulture** ·
恩科 Enko · 河童 Kappa · 神藏 Shinzo · 小暗 Yami · 灵堕魔 Nindoru · 蛋脸鬼 Dalgyal Gwishin

章名取 `folders`：第1章 四季轮转 / 第2章 踏险斩祸一道清 / 第3章 在智慧的废墟中。

「朝圣之径 Pilgrim's Path」取自模组自带原声带播放列表第 11 轨的官方译名，
与库里已有的 `朝圣之径 - WoW - Krasarang Wilds B.ogg` 一致。

槽位 04 原文 "the vulture" 查实为 A3 夺魂者巢穴的 **食心秃鹫**（模组 actor `Heart-Eating Vulture`，
Book 2 p32）。槽位 05 原文 "the tree hazard with the heart" 查实为 p32「种下魂种」段：
拿走心脏后触发 **苦迦菩提的愤怒**（complex haunt，倒树声＋逼近的巨影），
所以归在第 2 章鬼打墙内，不是第 3 章。

## 文件

| 文件 | 说明 |
|---|---|
| `_manifest.tsv` | 单一事实源：序号 / 章节 / 中文名 / 英文曲名 / kind / 来源标记 / ref / 用途。`kind=new` 的 ref 是 YouTube id，`kind=reuse` 的 ref 是编码好的 VPS 路径 |
| `_download.sh` | 只下载 `kind=new` 的行，转 ogg，可重跑（已存在的跳过）|
| `_build_playlist.py` | 由清单原样保序生成 playlist JSON |
| `fvtt-Playlist-书2.json` | 成品，直接导入 Foundry |
| `_manifest.old.tsv` | 重排前的清单，留档 |

## 重跑注意

本机 yt-dlp 是 2026.03.17，对默认 web 客户端会 **HTTP 403**。
脚本里已固定 `--extractor-args "youtube:player_client=android"`。
哪天 android 客户端也被封，先 `pip install -U yt-dlp` 再说。

音频参数 `libvorbis -q:a 5`，与全库一致（实测输出 140–150 kb/s）。
