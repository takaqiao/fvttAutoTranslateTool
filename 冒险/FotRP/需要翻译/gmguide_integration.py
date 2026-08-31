"""
FotRP GM 指南整合脚本

把外部 GM 指南内容作为新 page 追加到 4 个第一本 JSON 文件。
strict: 既有 page 的 text.content 用 SHA256 校验不变。

usage:
  python gmguide_integration.py book1 dry_run
  python gmguide_integration.py book1 apply
"""
import argparse
import copy
import hashlib
import html.parser
import json
import re
import sys
import time
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(r"C:\Users\Taka\Desktop\fvtt\FotRP")
NEW = ROOT / "需要翻译" / "NEW"
QA = ROOT / "_qa_reports"
QA.mkdir(parents=True, exist_ok=True)

BOOK_FILES = {
    "book1": {
        "ch1": NEW / "第一本" / "fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json",
        "ch2": NEW / "第一本" / "fvtt-JournalEntry-chapter-2-PpZKICROB7B08r53.json",
        "ch3": NEW / "第一本" / "fvtt-JournalEntry-chapter-3-jy3Vrm5yA3jrrG5W.json",
        "bm":  NEW / "第一本" / "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json",
    },
    "book2": {
        "ch1": NEW / "第二本" / "fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json",
        "ch2": NEW / "第二本" / "fvtt-JournalEntry-chapter-2-FTJT1CRdfjQzy3Ek.json",
        "ch3": NEW / "第二本" / "fvtt-JournalEntry-chapter-3-SPXqld4nww21Orrg.json",
        "bm":  NEW / "第二本" / "fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json",
    },
    "book3": {
        "ch1": NEW / "第三本" / "fvtt-JournalEntry-chapter-1-xClvGtftweJDu3vX.json",
        "ch2": NEW / "第三本" / "fvtt-JournalEntry-chapter-2-fwmZr935hxQLlBus.json",
        "ch3": NEW / "第三本" / "fvtt-JournalEntry-chapter-3-VPgzvXimMH8NKzBk.json",
        "bm":  NEW / "第三本" / "fvtt-JournalEntry-back-matter-1ylYqjGKvevX3BgC.json",
    },
}

# ---------- _id 生成 ----------
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def base62(n: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(BASE62[n % 62])
        n //= 62
    return "".join(reversed(out))

def make_id(book: str, slug: str) -> str:
    """生成 16 字符 _id: 'gmg' + 13 base62 字符基于 SHA1。"""
    h = hashlib.sha1(f"{book}::{slug}".encode("utf-8")).digest()
    n = int.from_bytes(h[:10], "big")
    return "gmg" + base62(n, 13)

# ---------- HTML 校验 ----------
class TagBalance(html.parser.HTMLParser):
    VOID = {"br", "hr", "img", "meta", "link", "input"}
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack = []
        self.errors = []
    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)
    def handle_endtag(self, tag):
        if not self.stack:
            self.errors.append(f"</{tag}> with empty stack")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"</{tag}> mismatch with <{self.stack[-1]}>")
        else:
            self.stack.pop()
    def report(self):
        if self.stack:
            self.errors.append(f"unclosed: {self.stack}")
        return self.errors

def validate_html(content: str) -> list:
    p = TagBalance()
    p.feed(content)
    return p.report()

# ---------- @UUID 格式校验 ----------
UUID_RE = re.compile(
    r"@UUID\[Compendium\.[^.\]]+\.[^.\]]+\.(Actor|Item|JournalEntry|JournalEntryPage)\.[A-Za-z0-9]{16}\](\{[^}]*\})?"
)
UUID_FIND = re.compile(r"@UUID\[[^\]]*\](\{[^}]*\})?")

def validate_uuids(content: str) -> list:
    errors = []
    for m in UUID_FIND.finditer(content):
        s = m.group(0)
        if not UUID_RE.fullmatch(s):
            errors.append(f"bad UUID: {s}")
    return errors

# ---------- page 模板 ----------
def build_page(name: str, content: str, page_id: str, ts_ms: int) -> dict:
    return {
        "name": name,
        "flags": {},
        "text": {"content": content, "format": 1},
        "src": None,
        "sort": 0,
        "type": "text",
        "_id": page_id,
        "system": {},
        "title": {"show": True, "level": 1},
        "image": {},
        "video": {"controls": True, "volume": 0.5},
        "category": None,
        "_stats": {
            "coreVersion": "14.360",
            "systemId": "pf2e",
            "systemVersion": "8.0.3",
            "createdTime": ts_ms,
            "modifiedTime": ts_ms,
            "lastModifiedBy": "ZFWDpCkLkUnxBrCr",
            "compendiumSource": None,
            "duplicateSource": None,
            "exportSource": None,
        },
        "ownership": {"default": -1},
    }

# ---------- 各书 page 的 HTML 内容 ----------
# 命名: book / file_key / page_slug
PAGES_BOOK1 = {
    "ch1": [
        {
            "slug": "book1-gm-overview",
            "name": "GM 指南：第一本总评 GM GUIDE: BOOK 1 OVERVIEW",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。原始 PDF 翻译保持不变；以下为 GM 实战经验汇总，可与现有章节配合使用。</p>
<h2>节奏与难度评估</h2>
<ul>
<li><b>场次预估</b>：约 16 场（第 1 章约 4 场、第 2 章约 7 场、第 3 章约 5 场，按 2 小时/场计）</li>
<li><b>游戏内时间</b>：仅 4 天（第 1-3 日资格赛 + 第 4 日颁奖与离岛）</li>
<li><b>真实游戏时长</b>：三本中最长。游戏内时间过得飞快，但桌前实际时间最长</li>
<li><b>难度</b>：显著偏低。满配队伍准备增益时多数遭遇会被秒杀</li>
</ul>
<blockquote title="GM 提示"><p>整本只 4 天的设定违背了锦标赛叙事结构 —— 玩家来到时可能期待"每天 6-8 场战斗的疯狂连战"，但实际是地下城清剿 + 六角格探索。开跑前务必先用 <b>玩家预期校准</b>（详见「三分钟速读」页）与桌上玩家对齐这一节奏。</p></blockquote>
<h2>抵邦木岛之前</h2>
<h3>船上提前接触队伍</h3>
<p>这是最重要的桌前预备之一。在抵岛前的航程中，安排 2-3 场 NPC 队伍社交或友谊赛，让玩家提前认识千野最硬、寒霜之吼等团队。</p>
<p>经典桥段：千野最硬队员请玩家吃甜咖喱（鹰虎是甜食党）后印象转变 —— 玩家："这帮家伙又友善又乐观，他们肯定要死。"船上提前建立的人物关系，会让第二本剧情戏剧死亡的冲击力倍增。</p>
<h3>过峡抵达 + 购物日</h3>
<p>强烈建议：邦木岛资格赛前给 1-2 天过峡自由活动时间，玩家逛城市、买装备、社交。过峡是居住地等级 20（等同阿布萨隆），14 级以下常见与罕见物品都可购得。</p>
<blockquote title="GM 提示"><p>玩家指南暗示这段是有的，但实际书面没安排 —— GM 自加。具体推荐装备清单见「开跑前必做」页。</p></blockquote>""",
        },
        {
            "slug": "book1-temple-patches",
            "name": "GM 指南：义洛理神庙补丁 GM GUIDE: TEMPLE PATCHES",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。义洛理神庙（A1-A15）的房间细节平衡问题与战斗修补建议。</p>
<h2>总体节奏</h2>
<ul>
<li>3 天清完；房间可任意顺序</li>
<li>全清 = +3 凤凰羽毛 + 足够经验值升 12 级</li>
</ul>
<h2>A14 北侧水井：困难地形提醒</h2>
<p>文字写明 <i>thick undergrowth is difficult terrain</i>（茂密灌木丛是困难地形），但地图上没标。</p>
<p><b>GM 解读</b>：整片深绿区域当困难地形处理。</p>
<h2>毒蛇藤（重大平衡问题）</h2>
<p>义洛理神庙最危险且最容易团灭的遭遇。问题：</p>
<ul>
<li>大半径催眠花粉（DC 高，几乎全队中招）</li>
<li>中招后角色不能做精神集中动作</li>
<li>无法预备攻击等毒蛇藤出招</li>
<li>近战角色基本陷死循环</li>
</ul>
<blockquote title="修补共识"><p>三选一：<b>① 缩小花粉半径</b>；<b>② 成功豁免后整场战斗免疫</b>；<b>③ 替换为精英化的德苏隆（Dezullon，11 级）</b>。</p></blockquote>
<p><b>铺垫建议</b>：把哥利亚蜘蛛（带毒）放在毒蛇藤之前，让玩家自觉"该买解毒剂了"，毒蛇藤一战的体验会缓和很多。</p>
<h2>卡托布莱帕斯（可谈判）</h2>
<p>关键设计：可以选择不打。玩家与卡托布莱帕斯（Catoblepas）谈交易（"守庙换尸体喂它"），这是创造性玩家最爱的桥段。GM 应主动透露它"似乎在等什么交易"的暗示。</p>
<h2>哥利亚蜘蛛</h2>
<ul>
<li>8 小时瘫痪能让玩家彻底瘫一晚，节奏建议放在白天开打</li>
<li>喜剧桥段：蛮族用背包扛瘫痪角色到处走</li>
<li>处理伤口可用来"并消 1 stupefied/瘫痪持续时间"（GM 裁量）</li>
</ul>
<h2>多重鬼魂</h2>
<p>同时遇到 2-3 个鬼是可能的，通常是玩家失误（主动攻击了可选鬼）。</p>
<blockquote title="跨 AP 提示"><p>如果队伍从《憎恶秘库》（Abomination Vaults）链接到本 AP，玩家会有"鬼魂疲劳"。本 AP 神庙鬼 + 后续邦木岛哨塔鬼 + 锦标赛鬼怪叠加，建议把神庙鬼或哨塔鬼缩减一组。</p></blockquote>
<h2>铁/石巨像</h2>
<ul>
<li>10 ft 体型，巷道刚好放下</li>
<li>铁巨像在废弃监狱区域</li>
<li>抗法、近战为主</li>
</ul>
<h2>扁虱群 + 黏土巨像（建议跳过）</h2>
<p>低价值经验，多重抗性/弱点烦琐，没有有趣能力。多数 GM 直接跳过。</p>
<h2>A6 宿舍：6 个木制神圣符号</h2>
<p>A6 宿舍内有 <b>6 个木制义洛理神圣符号</b>。这是给 <b>老衲像谜题</b>（在邦木岛 §30 触发）用的，<b>很多 GM 漏掉</b>。请把它放在显眼处描述，玩家会自然带走。</p>""",
        },
        {
            "slug": "book1-day1-danger",
            "name": "GM 指南：第一日危险事件警告 GM GUIDE: DAY 1 DANGER ALERT",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第一日 / 资格赛中最危险的两个事件预警，GM 必读。</p>
<h2>⚠ 第一日 4 名内家好手资格队</h2>
<p>第一日上午会触发 4 名内家好手（Ki Adept）资格队事件。<b>这队带『解离术』（Disintegrate）</b>。</p>
<blockquote title="致命问题"><p>玩家 11-12 级 HP 仅 40-60，<b>凤凰项链未激活时会被秒杀</b>。即使豁免成功也可能 9d6 直伤秒人。</p></blockquote>
<h3>强制修补方案</h3>
<ul>
<li><b>方案 A（推荐）</b>：让执行者（Enforcer）阿米尔塔在事件触发前正式激活全员凤凰项链，激活动作明确出现在场景里</li>
<li><b>方案 B</b>：赛前明确警告玩家"这队有解离术"</li>
<li><b>方案 C</b>：换成低阶法术（如力场飞弹）</li>
</ul>
<h2>凤凰项链规则要点回顾</h2>
<ul>
<li><b>激活后</b>：所有伤害变非致命，包括法术、炸弹、投射物、攻城武器（如炮弹）</li>
<li><b>不阻止</b>：项链不阻止法术其他效果。解离术仍可"解离"你（不死，但承受其他效果）</li>
<li><b>必须戴身</b>：不许塞入无尽袋；项链通过心电感应与执行者通信</li>
<li><b>分配</b>：5 人队每人一条（避免战斗中"谁倒地谁还活"的歧义）</li>
</ul>
<h2>其他第一日注意事项</h2>
<h3>阿姆扎双胞胎"第一缕光"事件</h3>
<p>第一日黎明必触发。<b>+2 凤凰羽毛</b>，难度可控，是良好的"建立比赛氛围"事件。曼亚拉·阿姆扎（火妹）+ 丽嘉娜·阿姆扎（冰姐）双胞胎兄妹的对战风格本身就是教学桥段。</p>
<h3>炽焰余烬 / 云间小丑随机遭遇</h3>
<p>第一日及之后会有这些游荡队伍登场。它们的功能是 <b>介绍其他游荡队伍存在</b>，不要打出团灭难度 —— 节奏化处理，让玩家看到"还有别人也在抢凤凰羽毛"即可。</p>""",
        },
    ],
    "ch2": [
        {
            "slug": "book1-prison-lobby-fix",
            "name": "GM 指南：监狱大厅勘误 GM GUIDE: ABANDONED PRISON LOBBY FIX",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。废弃监狱 R3 区域的官方勘误与修补方案。</p>
<h2>问题</h2>
<p>书面 <b>R3 警卫宿舍</b>提及守卫监视一个"大厅"（Lobby），但 <b>地图上没有大厅</b>。</p>
<h2>原因</h2>
<p>开发期间该建筑是阿巴达大钱庄（Grand Bank of Abadar）/ 神庙，R2 是银行大厅，改成监狱时漏改了引用。</p>
<h2>三种修补方案</h2>
<ul>
<li><b>方案 ①</b>：把 R2 登记处当大厅处理 —— 守卫监视 R2 入口处的开放区域</li>
<li><b>方案 ②</b>：在 R3 警卫宿舍与 R1-R2 之间加窗户或箭眼 —— 守卫透过窗户/箭眼监视</li>
<li><b>方案 ③</b>：直接忽略书面表述 —— 守卫只巡逻自己的 R3 区域，不监视任何"大厅"</li>
</ul>
<blockquote title="GM 建议"><p>方案 ① 改动最小，最不破坏现有地图战术布局。方案 ② 给玩家潜入路径多一些可玩性。方案 ③ 最省事但牺牲一些细节。</p></blockquote>""",
        },
        {
            "slug": "book1-bonmu-economy",
            "name": "GM 指南：邦木岛沙盒经济 GM GUIDE: BONMU HEXPLORATION ECONOMY",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第 2 章邦木岛六角格探索（Hexploration）的运行经济与判定规则建议。</p>
<h2>六角格活动经济</h2>
<p>书面规则较模糊。<b>社区惯例</b>：</p>
<ul>
<li>每天约 <b>32 次活动</b> = 8 小时休息 + 16 小时活动 × 2</li>
<li><b>移动到相邻六角格</b> = 1 activity</li>
<li><b>探勘</b>（Reconnoiter，找站点 + 访问 1 个）= 1 activity</li>
<li><b>已探勘后访问额外站点</b> = 10 分钟（部分 activity）</li>
</ul>
<p><b>追踪工具</b>：Foundry Boss Bar 模块 / Party Resources 模块 / 党 token elevation +32 递减计数。</p>
<h2>团队"在家"判定</h2>
<p>书面无明确规则。<b>实测建议</b>：投 50/50：</p>
<ul>
<li><b>在家</b>：触发遭遇/社交</li>
<li><b>不在</b>：空跑，玩家可侦察据点</li>
</ul>
<p>5 次据点访问平均 2.5 个团队遇到。RNG 太差时加额外随机遭遇补凤凰羽毛节奏。</p>
<h2>海岸六角不打折</h2>
<p>海岸六角虽仅 1/4 陆地（地图视觉），<b>不打折</b>（不要把它们当成"半个六角"）。用"困难地形"叙事代替：海浪、礁石、潮汐让移动更慢。</p>
<h2>游荡队伍机制</h2>
<ul>
<li>用 Patrol 模块：每 PC activity 移动一次游荡队伍</li>
<li>队伍可挑战 PC，也可被 PC 挑战</li>
<li><b>智能 NPC 视角</b>：他们也想最大化凤凰羽毛赌注，会主动挑选有 5+ 羽毛的玩家挑战</li>
</ul>
<blockquote title="GM 提示"><p>NPC 游荡队伍主动挑战 PC 时，可以让 PC 拒绝（损失一点声誉/凤凰羽毛 0.5 罚）或接受。让玩家知道"主动出击 = 高赌注高回报"，可大幅提升沙盒探索的策略层次。</p></blockquote>""",
        },
        {
            "slug": "book1-bonmu-locations",
            "name": "GM 指南：邦木岛地点参考 GM GUIDE: BONMU LOCATIONS REFERENCE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。邦木岛资格赛规则、必触发事件、巨兽巢穴 / 迷音鸟修补，以及 60+ 命名站点的完整中英对照表。</p>
<h2>资格赛规则</h2>
<ul>
<li><b>凤凰羽毛上限 10</b>；超过时去找执行者兑换</li>
<li><b>500 gp / 每根羽毛</b>赌注奖励（可选规则，激励高赌注）</li>
<li><b>光辉使者输了会还羽毛</b>（书面可选规则；增加戏剧性）</li>
</ul>
<h2>必触发事件</h2>
<ul>
<li><b>第 1 日黎明</b>：阿姆扎双胞胎"第一缕光"（强制；+2 凤凰羽毛；难度可控）</li>
<li><b>第 1 / 2 日</b>：4 名内家好手资格队 —— <b>⚠ 危险，详见「第一日危险事件警告」页</b></li>
<li><b>第 1+ / 2+ 日</b>：炽焰余烬 / 云间小丑 —— 介绍其他游荡队伍</li>
<li><b>玩家有 5 凤凰羽毛后</b>：工崴挑战（醉拳大师 Gomwai）；自动触发；+2 凤凰羽毛</li>
<li><b>第 2 日夜</b>：兵马俑大军突袭 —— 关键事件，详见「兵马俑夜袭改写」页</li>
<li><b>第 3 日</b>：收尾，少事件，可压缩</li>
</ul>
<h2>创伤野兽巢穴 G1-G10 修补</h2>
<p><b>重大问题</b>：书面没有地图、没有战利品。</p>
<p><b>GM 通用补丁</b>（金额按队伍等级与章节财富曲线自定）：</p>
<ul>
<li><b>巨兽部位</b>（鳞、牙、皮）：每件按章节财富比例分配</li>
<li><b>蛋</b>（如有筑巢）：约半件部位价值</li>
<li><b>单巢穴</b>：1 件魔法物品</li>
</ul>
<p><b>已知巨兽 10 种</b>：黑蝎、恐惧大鹏、巨螳螂、帝皇暴龙、洞穴虫、死墓棘龙、奔鸟、酸蚀王蜥、避日蛛、猛犸象龟。</p>
<p><b>地图</b>：必须自找 —— r/battlemaps、Patreon、自做。</p>
<h2>迷音鸟（社区最痛恨遭遇）</h2>
<p>问题：</p>
<ul>
<li>数据卡没有"巧手"技能，但要求"偷走"项链</li>
<li>玩家普遍痛恨</li>
</ul>
<p><b>修补方案</b>：</p>
<ul>
<li><b>改为追逐序列</b>（最常用）</li>
<li><b>直接跳过</b>（多数 GM 选择）</li>
<li><b>只让迷音鸟偷"戴在身上"的项链</b>（不许塞无尽袋）</li>
</ul>
<blockquote title="净光守护者借势"><p>让迷音鸟受净光守护者雇佣偷 PC 的凤凰羽毛 —— 失去 5 羽毛后玩家"纯粹恨意"被建立，为第二本/第三本净光守护者大反派铺路。</p></blockquote>
<h2>净光守护者据点</h2>
<ul>
<li>邦木岛上有据点，但 <b>无明显标识</b></li>
<li>GM 决定是否明显（书面留余地）</li>
<li>玩家通常不在这里直接战斗（决战还没到位）</li>
</ul>
<h2>哨塔 + 鬼魂</h2>
<ul>
<li>多座；附鬼魂遭遇</li>
<li>⚠ 与神庙鬼合并疲劳；如果玩家从《憎恶秘库》链接来更糟</li>
</ul>
<p><b>改写选项</b>：</p>
<ul>
<li>把鬼魂改成"魔加鲁 vs 别的怪兽"的视景幻象（同时铺魔加鲁伏笔，社区实战方案）</li>
<li>16 级女妖换成 14 级噪鬼（12 级队伍可应对）</li>
</ul>
<h2>老衲像</h2>
<ul>
<li>与义洛理神圣符号谜题相关</li>
<li>谜题：A6 宿舍的 6 个木制符号在这里用得上 —— 神庙阶段就要给玩家留意</li>
<li>替换战斗：可让依格妲尼（Ingdani）当辅助 NPC</li>
</ul>
<h2>邦木岛全 60+ 命名站点对照表</h2>
<p>来自 Foundry Addons 模块的完整中英文译名表，方便 GM 现场检索。</p>
<h3>抵港（B 系列）</h3>
<ul>
<li>B1 抵港码头 Arrival Dock</li>
<li>B2 南码头 South Dock</li>
<li>B3 东北码头 North East Docks</li>
</ul>
<h3>比武场地（C 系列）</h3>
<ul>
<li>C1 海滩场地 Beach Site（×5 变体）</li>
<li>C2 森林场地 Forest Site（×4 变体）</li>
<li>C3 山地场地 Mountain Site（×4 变体）</li>
<li>C4 河流场地 River Site（×3 变体）</li>
<li>C5 废墟场地 Ruins Site（×4 变体）</li>
</ul>
<h3>传送塔（D）/ 石市（E）</h3>
<ul>
<li>D1-D5 传送塔 1-5 号 Transport Tower 1-5</li>
<li>E 石桌集市（主） Stone Market + E2-E5 子集市</li>
</ul>
<h3>F 系列：30 个固定命名站点</h3>
<ul>
<li>F1 古鲁哈斯塔图书馆 Library of Gruhastha</li>
<li>F2 南方灯塔 Southern Lighthouse</li>
<li>F3 戈兹瑞神龛 Shrine of Gozreh</li>
<li>F4 祭司宅邸 Priest's Estate</li>
<li>F5 盐矿场 Salt Quarry</li>
<li>F6 贵族宅邸 Noble's Estate</li>
<li>F7 奥术图书馆 Arcane Library</li>
<li>F8 恐龙牧场 Dinosaur Ranch</li>
<li>F9 冰屋 Icehouse</li>
<li>F10 造船厂 Shipyard</li>
<li>F11 酿酒厂 Distillery</li>
<li>F12 德鲁伊环 Druid Circle</li>
<li>F13 西方灯塔 Western Lighthouse</li>
<li>F14 船屋 Boathouse</li>
<li>F15 木匠工棚 Carpenter's Shed</li>
<li>F16 静月之庙 Temple of Shizuru</li>
<li>F17 海盗湾 Pirate Cove</li>
<li>F18 山顶观测站 Mountain Observatory</li>
<li>F19 原牛牧场 Aurochs Ranch</li>
<li>F20 雨棚 Rain Shelter</li>
<li>F21 东方灯塔 Eastern Lighthouse</li>
<li>F22 蔗田 Cane Farm</li>
<li>F23 风蚀古碑 Weatherworn Monument</li>
<li>F24 墓园 Cemetery</li>
<li>F25 校舍 Schoolhouse</li>
<li>F26 月夜神龛 Shrine of Tsukiyo</li>
<li>F27 黑风神庙 Temple of Hei Feng</li>
<li>F28 制革匠村 Tanners' Village</li>
<li>F29 捕鲸哨塔 Whalers' Lookout</li>
<li>F30 北方灯塔 Northern Lighthouse</li>
</ul>
<h3>G 系列：10 个巨兽巢穴</h3>
<ul>
<li>G1 黑蝎兽穴 Black Scorpion Lair</li>
<li>G2 恐惧大鹏兽穴 Dread Roc Lair</li>
<li>G3 螳螂兽穴 Mantis Lair</li>
<li>G4 猛犸象龟兽穴 Mammoth Turtle Lair</li>
<li>G5 暴龙兽穴 Tyrannosaurus Lair</li>
<li>G6 洞穴虫兽穴 Cave Worm Lair</li>
<li>G7 棘龙兽穴 Spinosaurus Lair</li>
<li>G8 奔鸟兽穴 Dromornis Lair</li>
<li>G9 酸蚀王蜥兽穴 Caustic Monitor Lair</li>
<li>G10 避日蛛兽穴 Solifugid Lair</li>
</ul>
<h3>H 系列：8 个陶玛塔神龛（对应 8 个祝福）</h3>
<ul>
<li>H1 安国纳神龛 Shrine of Ahngonar</li>
<li>H2 巴布纳彼神龛 Shrine of Babbunabi</li>
<li>H3 金牙婆神龛 Shrine of Jinya-Por</li>
<li>H4 坎提亚尼神龛 Shrine of Kantiyani</li>
<li>H5 妮萨野神龛 Shrine of Ni-Sa-Yei</li>
<li>H6 拉米加神龛 Shrine of Ramijav</li>
<li>H7 史汉迪瓦拉神龛 Shrine of Shihandivara</li>
<li>H8 乌蛮塔神龛 Shrine of Umantar</li>
</ul>
<h3>I 系列：4 个哨塔</h3>
<ul>
<li>I1-I4 西南/东南/西北/东北哨塔 SW/SE/NW/NE Watchtower</li>
</ul>
<h3>J 系列：10 个次要站点</h3>
<ul>
<li>J1 宝箱 Treasure Chest</li>
<li>J2 高树 Tall Tree</li>
<li>J3 废弃商铺 Abandoned Shop</li>
<li>J4 废弃小屋 Abandoned Hut</li>
<li>J5 兽径 Beast Trail</li>
<li>J6 被遗忘的小神龛 Forgotten Small Shrine</li>
<li>J7 河畔狐穴 Riverside Fox Den</li>
<li>J8 毁坏的神龛 Ruined Shrine</li>
<li>J9 废弃前哨 Abandoned Outpost</li>
<li>J10 不祥的枯树 Ominous Dead Tree</li>
</ul>
<h3>大写字母关键地点</h3>
<ul>
<li>A 义洛理神庙 Temple of Irori</li>
<li>A5 灯笼小舍画廊 Lantern Lodge Gallery</li>
<li>K 执行者基地 Enforcer Base</li>
<li>L 净光守护者基地 Lightkeeper Base</li>
<li>M 塔米坎之穴 Tamikan's Den</li>
<li>N 斑足村 Mottlefoot Village</li>
<li>O 贾班之穴 Jaiban's Den</li>
<li>P 哈米那布山 Mount Haminabu</li>
<li>Q 破碎蛋壳 Broken Eggshell</li>
<li>R 废弃监狱 Abandoned Prison</li>
<li>S 空池塘 Empty Pond</li>
<li>T 曼南迦尔村 Mananggal Village</li>
</ul>""",
        },
        {
            "slug": "book1-terracotta-raid",
            "name": "GM 指南：兵马俑夜袭改写 GM GUIDE: TERRACOTTA ARMY NIGHT RAID",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第 2 日夜兵马俑大军夜袭事件的团灭风险与改写建议。</p>
<h2>机制</h2>
<ul>
<li><b>Wave 1 + 2 强制</b>，Wave 3 + 4 可选</li>
<li><b>起源</b>：光辉使者 / 雕匠辛达拉的代理（玩家不会知道辛达拉，但 GM 心里有数）</li>
<li><b>叙事钩</b>："玩家会立刻怀疑光辉使者" —— 这是设计意图</li>
</ul>
<h2>⚠ 团灭风险</h2>
<p>Wave 1 就能剃光资源，Wave 2 几乎团灭。书面遭遇按"满血满资源"调难度，但夜袭设计预设玩家是疲惫状态。</p>
<h2>三种改写方案</h2>
<h3>方案 ① Wave 2 后给 Boon</h3>
<p>让依格妲尼（Ingdani，或替代友军 NPC）出现给增益/治疗。理由："守护神注意到玩家的努力，给一次喘息"。叙事上避免突兀，可以铺成"邦木岛原住民暗中支持你们"。</p>
<h3>方案 ② 逃跑 = 给传送塔</h3>
<p>允许玩家逃跑，但仍施加追兵压力。让玩家逃到附近的 D1-D5 传送塔避险。<b>设计原则</b>："越接近死亡的感觉而无实际伤害 = 最有趣"。</p>
<h3>方案 ③ 全混蛋模式</h3>
<p>玩家允许复活 / 不计死亡。"努力杀人无后果"。把光辉使者的危险性压力给到位 —— 让玩家深刻记得"光辉使者真的会杀我们"。</p>
<blockquote title="GM 提示"><p>方案 ① 最常见，方案 ② 最有戏剧张力，方案 ③ 适合喜欢硬核战斗的桌子。三选一即可，不要同时用（玩家会感觉太过保护或太过粗暴）。</p></blockquote>""",
        },
    ],
    "ch3": [
        {
            "slug": "book1-ch3-wrap",
            "name": "GM 指南：第 3 章收尾与过峡抵达 GM GUIDE: CHAPTER 3 WRAP & GOKA ARRIVAL",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第一本第 3 章收尾的演绎建议与过峡抵达环节的伏笔铺设。</p>
<h2>渡轮事件</h2>
<p>书面描述较略。GM 可加：</p>
<ul>
<li>船上颁奖仪式：执行者把凤凰羽毛兑换为正式资格证书</li>
<li>邦木岛-过峡航程间的 NPC 队伍互动：让千野最硬等友军在船上重逢，巩固关系</li>
<li>可选：海上小冲突，让玩家展示资格赛中学到的技巧</li>
</ul>
<h2>魔加鲁伏笔（第一次预兆）</h2>
<p><b>这是 GM 主动添加的伏笔</b>。在第 3 章中段或末尾，让玩家在哨塔/瞭望台/船上视野中第一次看见远方的异象：</p>
<blockquote title="伏笔建议"><p>玩家在瞭望塔或航行中看见远处海面浮现 <b>巨大的鳞片影子</b>，或听到深沉的咆哮声。即使是身经百战的水手也称从未见过。这是魔加鲁（Mogaru）的远景预兆 —— 第二本末出现时玩家不会"突然冒出"，而是"那个东西终于来了"。</p></blockquote>
<p>实战变体：把第一本邦木岛哨塔鬼魂改成"魔加鲁 vs 别的怪兽"的视景幻象（同时解决哨塔鬼疲劳问题）。</p>
<h2>过峡抵达 + 郝金仪式</h2>
<p>过峡上岸 + 郝金（Hao Jin）初次见面是第一本的高潮场景之一。</p>
<h3>郝金的演绎</h3>
<p>关键基调：<b>分心而非恶意</b>。玩家应该觉得她可爱而不是不可信任。</p>
<ul>
<li>"喝椰子（带伞）" —— 标志性桥段，戴小伞的椰子杯</li>
<li>童趣感：对玩家做事像看戏剧，不太 care 比赛结果</li>
<li>综艺主持感：把仪式做成"开场秀"</li>
<li>偶尔走神 —— 半神级法师注意力被各种小东西吸引</li>
</ul>
<blockquote title="演绎提示"><p>郝金性格选择多样（详见 GM 指南 §50）：自然之力 / 综艺主持 / 冷漠导师 / 混乱大女主 / 无性恋（canon）。第一本只露一两面，选一种基调贯穿到底即可。</p></blockquote>""",
        },
    ],
    "bm": [
        {
            "slug": "book1-3min-quickread",
            "name": "GM 指南：三分钟速读 GM GUIDE: 3-MINUTE QUICK READ",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。整套《赤凰斗士》战役的开跑前要点速读，GM 拿到 AP 后第一时间该读的内容。</p>
<h2>这是什么 AP？</h2>
<ul>
<li><b>11 → 20 级</b> / 三本 / 天夏背景的动漫风武术大赛</li>
<li><b>实际结构</b>：大量地下城 + 探索，少量锦标赛对决</li>
<li>不是"每天一场比赛"，玩家预期要早早校准</li>
</ul>
<h2>开跑前最重要的五件事</h2>
<ul>
<li><b>① 三本通读</b>，尤其第三本（雕匠辛达拉铺垫必须从第一本开始铺）</li>
<li><b>② 装好四件套</b>：FotRP Expanded · Kalnix 地图 · Chasarooni Addons · PF2e Troops Helper</li>
<li><b>③ 桌前白板讲清五条桌规</b>：凤凰项链 / 预增益 / 复活 / 偷盗 / 致死（详见「开跑前必做」页）</li>
<li><b>④ 备好至少 6 支 NPC 队伍数据卡</b></li>
<li><b>⑤ Challonge.com 对阵图 + 锦标赛音乐 playlist</b></li>
</ul>
<h2>全 AP 三个最致命的坑</h2>
<blockquote title="⚠ 必读"><p><b>① 第一日 4 名内家好手资格队带『解离术』</b><br>11 级 HP 40-60 必死，必须强制激活凤凰项链或赛前预警。</p>
<p><b>② 蓝蝮蛇『梅花滂沱 + 死之泪』组合（DC 47 / 1 分钟瘫痪）</b><br>几乎稳定团灭，多数 GM 直接不用『死之泪』。</p>
<p><b>③ 第三本仅 4 张地图</b><br>必装 Kalnix 地图模块，否则一半遭遇靠『心中战棋』。</p></blockquote>
<h2>第三本最大设计 bug：雕匠辛达拉缺乏铺垫</h2>
<p>雕匠辛达拉（Syndara the Sculptor）在第一、二本几乎完全没出场，第三本直接登场当反派。</p>
<blockquote title="→ 跨本提示"><p>至少要从第一本开始铺一条暗线。第三本附录的 §48 提供 8 种铺垫方案，其中部分（如方案 ②化名赞助人、③ "shadowy figure"、⑧ 玩家背景钩）需要在第一本/第二本就执行。本卷起即需埋线，详见第三本 back-matter 中「辛达拉铺垫方案」页。</p></blockquote>
<h2>玩家预期校准</h2>
<p><b>风格定位</b>：动漫冒险（龙珠 / 拳愿阿修罗 / 灼眼之魂）。</p>
<p><b>真实结构 ≠ 期望</b>：第一本全是地下城/六角探索，第二本中段才进入锦标赛主体。玩家若抱"每天 1-2 场锦标赛"预期来，第一本全是六角探索 + 地下城会大失所望。开跑前用 10 分钟讲清这一点。</p>""",
        },
        {
            "slug": "book1-pre-campaign-prep",
            "name": "GM 指南：开跑前必做 GM GUIDE: PRE-CAMPAIGN PREPARATION",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。GM 在第一场前必须完成的准备清单。</p>
<h2>必装四件套</h2>
<ul>
<li><b>《赤凰斗士 Expanded》PDF</b> —— 13 支额外队伍 + 净光守护者/马菲卡替代法术清单 + 蓝蝮蛇毒液修订 + 观众影响力规则</li>
<li><b>Kalnix Foundry 地图重制模块</b> —— 第三本自带地图严重不足，必需品，非可选</li>
<li><b>Chasarooni《赤凰: Addons》</b> —— 重做的 NPC 数据卡 + 宏 + 转场动画序列</li>
<li><b>PF2e Troops Helper</b> —— 跑《花之百人众》（44 人小兵团）必备 Foundry 模块</li>
</ul>
<h2>11 级角色构筑</h2>
<h3>必备职业位</h3>
<ul>
<li><b>专职治疗位</b>：神职 / 医学专家。单战斗医术救不回锦标赛高强度遭遇</li>
<li><b>范围伤害位</b>：团队遭遇经常 4-6 个 NPC，纯单挑队伍很痛苦</li>
</ul>
<h3>强烈警告</h3>
<ul>
<li><b>召唤师</b>：与凤凰项链非致命规则的兼容性差</li>
<li><b>元素能师</b>：巅峰能力兼容性问题</li>
<li><b>纯技能型</b>：探索阶段没问题，但锦标赛里没用武之地</li>
</ul>
<h3>推荐方向（天夏主题契合）</h3>
<ul>
<li>武僧 / 圣武士（须搭配"自由迈步"奥义）/ 法刃</li>
<li>武术造物师 / 武器熟练战士</li>
<li>天夏种族（狐妖、河童、巡梦人）</li>
</ul>
<h2>自由原型 / 自动加成</h2>
<h3>自由原型</h3>
<ul>
<li><b>主流 GM 都开</b>，但 <b>必须 ban 圣武士（Champion）</b>（「自由迈步」奥义在锦标赛非致命环境下效率太高）</li>
<li>开自由原型后所有遭遇要重新调难度（+1 难度补偿）</li>
<li><b>天夏限定方案</b>：允许天夏 book 原型 + 武术家 + 龙裔（折中方案）</li>
</ul>
<h3>自动加成进度（ABP）</h3>
<p>本 AP 影响很小：高级锦标赛主要消耗品（药水、卷轴、毒液），永久魔法物品权重不大。</p>
<h3>⚠ 关键警告</h3>
<p>不要同时开自由原型 + 传奇（Mythic）。</p>
<h2>桌前五条规则</h2>
<p>开第一场前白板写明：</p>
<ul>
<li><b>① 非致命默认</b>：官方比赛中凤凰项链激活后，全部伤害（含法术、炸弹、攻城武器、破片箭）非致命化。0 HP 倒地 = 被判出局，不死人</li>
<li><b>② 凤凰项链必须戴在身上</b>：不许塞入无尽袋，否则迷音鸟之类怪物盗窃时玩家会钻空子</li>
<li><b>③ 预增益红线</b>：8 小时 / 当日增益保留；10 分钟增益每人最多预备 1 个；1 分钟增益不许预备</li>
<li><b>④ 复活规则</b>：每队入场附送 1 次免费"死者复生"名额（项链效力下死亡也能复活）</li>
<li><b>⑤ 道具偷窃</b>：净光守护者在场外可能用"湮没光环"偷东西 —— GM 节制使用</li>
</ul>
<h2>推荐前置 AP</h2>
<ul>
<li><b>《法外之徒：阿尔肯斯塔》</b>（1-11，最丝滑）：灯塔意象回调，GM 自加</li>
<li><b>《憎恶秘库》</b>（1-11，最常见）：加探索者协会介入引出过峡邀请；⚠ 玩家会鬼魂疲劳</li>
<li><b>《群鬼之季》</b>（1-10，较麻烦）：改写第四本，让 Ren 答应"赢锦标赛就给钥匙"</li>
<li><b>独立 11 级开场</b>：玩家自由组建主题队伍，无前置包袱</li>
</ul>
<h2>首场前一周清单</h2>
<ul>
<li>三本读完</li>
<li>《FotRP Expanded》PDF 读一遍</li>
<li>装好 Kalnix 模块 + Chasarooni Addons</li>
<li>至少 5-6 支主要 NPC 队伍数据卡备好（千野最硬 / 净光守护者 / 风语者 / 刺骨蔷薇 / 格拉利昂最强 / 千野第一日资格队等）</li>
<li>玩家 11 级角色 + 起始购物清单</li>
<li>Challonge.com 锦标赛对阵图设好</li>
<li>音乐 playlist 准备好</li>
<li>桌前白板五条规则讲清楚</li>
</ul>
<h2>11 级起始购物清单（主动推荐给玩家）</h2>
<p>过峡是居住地等级 20（等同阿布萨隆），14 级以下常见+罕见物品都买得到。<b>不要让玩家自己摸</b> —— 主动塞清单：</p>
<ul>
<li><b>解毒</b>：解毒剂数支（神庙蜘蛛毒 + 蓝蝮蛇后期都用得上）</li>
<li><b>反隐身</b>：妖火术魔棒</li>
<li><b>侦测</b>：侦测魔法 / 鉴定魔棒</li>
<li><b>持续治疗</b>：治疗卷轴 / 治疗法杖</li>
<li><b>环境谜题</b>：万能溶剂</li>
<li><b>存物</b>：无尽袋（但凤凰项链不能塞进去）</li>
</ul>
<p><b>符文应用</b>：商店允许 1-2 小时即时镶嵌 +1/+2 符文，不要让镶嵌占探索时间。</p>
<h2>动漫开场感</h2>
<ul>
<li>玩家做剧情回顾（recap）给 1 个英雄点</li>
<li>用动漫开场曲调回顾，可额外给另一名玩家 1 个英雄点</li>
<li>循环圣斗士星矢开场曲增添氛围</li>
</ul>
<h2>玩家背景集成钩</h2>
<p>在第一场之前问玩家：</p>
<ul>
<li>"你为什么参加？" → 抽角色钩</li>
<li>"你认识其他参赛者吗？" → 给某 NPC 队员埋成识别</li>
<li>"你想避免/找谁？" → 让某资格队员变剧情人</li>
</ul>
<p><b>典型范例</b>：</p>
<ul>
<li>PC 有"假定已死的哥哥"背景 → 直接绑成净光守护者成员</li>
<li>PC 父亲在前届锦标赛"神秘失踪" → 绑成雕匠辛达拉抓走的挂毯受害者（这是辛达拉铺垫方案 ⑧）</li>
</ul>""",
        },
        {
            "slug": "book1-resources-tools",
            "name": "GM 指南：资源与工具清单 GM GUIDE: RESOURCES & TOOLS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。跑《赤凰斗士》全 AP 的资源、模块、地图、音乐、工具完整清单。</p>
<h2>必读社区 PDF</h2>
<h3>《赤凰斗士 Expanded》【必装】</h3>
<ul>
<li>来源：Pathfinder Infinite 平台（搜索 "Fists of the Ruby Phoenix Expanded"）</li>
<li>内容：13 支新队伍 + 净光守护者/马菲卡法术清单修复 + 蓝蝮蛇毒液修订 + 观众影响力规则</li>
</ul>
<h3>《赤凰 Team: OriGami / OriGome》</h3>
<ul>
<li>5 PC 队伍补员神器：可拆分加入千野最硬 / 寒霜之吼 / 净光守护者</li>
</ul>
<h2>Foundry VTT 模块</h2>
<h3>Kalnix 地图重制【必装】</h3>
<ul>
<li>GitHub：搜索 "fists-of-the-ruby-phoenix-map-remake-by-kalnix"</li>
<li>内含：碎裂之螺、琉璃光屋、大道场（远古森林/午夜湖/灼热沙漠多套主题）、雕匠辛达拉第二阶段</li>
<li>主要 150px，部分 200px</li>
</ul>
<h3>Chasarooni《赤凰: Addons》</h3>
<ul>
<li>GitHub：搜索 "ruby-phoenix-addons" 作者 ChasarooniZ</li>
<li>内容：重做的 NPC 数据卡 + Sequencer 宏 + 转场动画 + 凤凰资源动画</li>
</ul>
<h3>PF2e Troops Helper【跑花之百人众必装】</h3>
<ul>
<li>GitHub：搜索 "pf2e-troops-helper" 作者 reyzor1991</li>
</ul>
<h3>其他常用模块</h3>
<ul>
<li><b>World Explorer + SimpleFog</b>：第一本六角揭示</li>
<li><b>Patrol Module</b>：游荡队伍随机巡逻</li>
<li><b>Boss Bar / Party Resources</b>：跟踪每日 32 hex 活动</li>
<li><b>JB2A</b>：动画素材</li>
<li><b>Sequencer</b>：转场动画依赖</li>
<li><b>Token Magic FX</b>：水面、灯光等环境效果</li>
<li><b>Scene Transitions</b>：自定义转场视频</li>
<li><b>Vehicles &amp; Mechanisms</b>：追逐序列与移动危害控制</li>
<li><b>Pickyadventurer</b>：手动从冒险包选物品</li>
</ul>
<h2>地图与艺术资源</h2>
<h3>免费 / 社区</h3>
<ul>
<li><b>Dommilles Wondrous Worlds (DWW)</b>：免费包</li>
<li><b>Mikweaka's Ring of Wonders</b>：免费图，跨维度场景</li>
<li><b>r/battlemaps</b>（Reddit）：东亚主题主要来源</li>
<li><b>Magical Cartography</b>（Twitter @CartosTom）</li>
<li><b>Eightfold Paper</b>（Patreon）</li>
<li><b>TehoxMaps</b>（Patreon，部分免费）</li>
</ul>
<h3>Paizo 官方付费</h3>
<ul>
<li><b>Battle Cards PDF</b> ★ 强烈推荐 —— 含第三本版本净光守护者 + 千野最硬全员高清艺术</li>
<li><b>Pawn Collection</b>：代币提取来源</li>
<li><b>NPC Token Pack</b>：郝金、奈燕妃女皇的代币唯一来源</li>
</ul>
<h2>音乐播放清单（按场景）</h2>
<ul>
<li><b>过峡都市日常</b>：东亚 jazz YouTube playlist</li>
<li><b>义洛理神庙灵异</b>：Sekiro OST</li>
<li><b>超自然/琉璃光屋</b>：Ghostwire Tokyo OST</li>
<li><b>魔加鲁出场</b>：原子吐息 SFX</li>
<li><b>二胡哀曲</b>：YouTube 搜索"erhu sad"</li>
<li><b>锦标赛 1v1</b>：太鼓 YouTube</li>
<li><b>锦标赛战斗（拳愿）</b>：Kengan Ashura / 罪恶装备 / 街霸 6</li>
<li><b>千野最硬主题</b>：JoJo 系列 OST（社区多人推荐）</li>
<li><b>净光守护者主题</b>：Apashe Lacrimosa / 街霸 Juri</li>
<li><b>雕匠辛达拉主题</b>：宿傩主题</li>
<li><b>歌剧院遭遇</b>：YouTube 戏剧 OST</li>
<li><b>过峡风景</b>：药屋少女的呓语 OST / 降世神通 OST / FFXIV 鞍部</li>
<li><b>日系战斗</b>：火焰之纹章 if 白夜 OST</li>
<li><b>观众笑声/嘲讽 SFX</b>：Roll20 论坛搜索"crowd tokens"</li>
</ul>
<h2>关键 JSON 资源（直接粘到 Foundry）</h2>
<h3>祝福（Blessings）</h3>
<ul>
<li>fvtt-Item-ahngonars-blessing-WSvMd0VPAkh8Bkm1.json</li>
<li>fvtt-Item-babbunabis-blessing-3xxBeOrkt0d5Uc5o.json</li>
<li>fvtt-Item-kantiyanis-blessing-HYgcZnWLxAIVnq3D.json</li>
<li>fvtt-Item-jinya-pors-blessing-ab33TlURf8b4dk5v.json</li>
<li>fvtt-Item-blessing-of-{ahngonar,babbunabi,jinya-por,kantiyani,ni-sa-yei,ramijav,shihandivara,umantar}.json</li>
</ul>
<h3>观众系统 / 变形 / 六角探索</h3>
<ul>
<li>fvtt-Item-effect_-rally-the-crowd-{,critical}.json</li>
<li>fvtt-Item-aura_-crowd-approval-A7NudaEdg6FPPpsH.json</li>
<li>fvtt-Item-effect_-crowd-approval{-greater,}.json</li>
<li>fvtt-Actor-crowd-favor-RrlZSRD2Z3C8izXU.json</li>
<li>fvtt-Item-spell-effect_-{boar,tiger}-polymorph-*.json</li>
<li>fvtt-Actor-hexploration-activities.json</li>
</ul>
<h3>Cac-Lee（千野最硬第 5 人）</h3>
<ul>
<li>9 级：fvtt-Actor-cac-lee-mon.json</li>
<li>13 级：fvtt-Actor-cac-lee-mon_1.json</li>
<li>18 级『彼岸花』形态：fvtt-Actor-cac-lee-mon-higanbana-form.json</li>
</ul>
<h2>工具</h2>
<ul>
<li><b>Challonge.com</b>：锦标赛对阵图（免费）</li>
<li><b>Photopea</b>：免费浏览器版 Photoshop，拼合 NPC 立绘</li>
<li><b>Demiplane Nexus</b>：PF2e 怪物在线数据库</li>
</ul>""",
        },
        {
            "slug": "book1-combatant-enhancements",
            "name": "GM 指南：通用格斗者增强 GM GUIDE: BASE COMBATANT ENHANCEMENTS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。本附录已有 6 个通用格斗者数据卡介绍 page（基础对手 / 武器大师 / 灵巧战士 / 内家好手 / 箭道精英 / 巧变法师），本页提供平衡修补建议。</p>
<h2>背景</h2>
<p>书面通用格斗者数据卡偏弱，特别是 14-15 级以上的资格队/挑战队使用时，玩家会感觉打击伤害与玩家自身相去甚远。社区共识：对应玩家等级适当增强。</p>
<h2>通用增强表</h2>
<ul>
<li><b>灵巧战士（十手/手里剑）</b>：十手 3d4+14；手里剑 3d4+17；偷袭 3d6</li>
<li><b>箭道精英</b>：弓 3d10+16；拳 3d4+11（替代书面荒谬的 1d4+7）</li>
<li><b>巧变法师</b>：拳 +24（3d6+10），长剑 +25（3d6+11）；加打击伤害</li>
<li><b>武器大师（刀）</b>：运动 +27；刀 3d10+16；甩石器 3d6+11</li>
</ul>
<h2>巧变法师法术清单替换</h2>
<p>具体替换列表无固定标准，GM 自定原则：</p>
<ul>
<li>默认法术清单偏弱 → 替换为更高强度的直伤法术</li>
<li>克敌机先（重铸版削弱）→ 换为其他法术（如急动法术、力场飞弹）</li>
</ul>
<h2>财富 / 装备调整</h2>
<p><b>战利品短缺问题</b>：避战队伍可能第一本结束时落后约 20,000 gp。建议补救：</p>
<ul>
<li><b>巨兽巢穴加战利品</b>：巨兽部位 + 物品（按章节财富自定）</li>
<li><b>哨塔加宝藏</b>：鬼魂相关物 + 钱币 + 艺术品</li>
<li><b>神龛加 200-500 gp</b> + 1-2 药水（每个 H 系列神龛）</li>
<li><b>赞助人在第二本加额外资金</b>（如第二本开始时给玩家 +5,000 gp 装备津贴）</li>
</ul>
<p><b>早期物品奖励</b>：第二本末让玩家获得破天弓（Sky-Piercing Bow）之外的更多宝库物品。</p>
<blockquote title="GM 提示"><p>增强通用格斗者的同时，记得相应增加战利品 —— 难度增加但回报不增加会让玩家觉得"白挨揍"。简单原则：每个增强敌人在战利品中增加 1 个等级对应消耗品。</p></blockquote>""",
        },
        {
            "slug": "book1-phoenix-necklace-rules",
            "name": "GM 指南：凤凰项链规则手册 GM GUIDE: PHOENIX NECKLACE COMPLETE RULES",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。凤凰项链完整规则手册，从激活机制到玩家钻空漏洞处理。第一本资格赛开始时就要用到，跨三本通用参考。</p>
<h2>规则要点</h2>
<ul>
<li><b>激活后所有伤害变非致命</b>：包括法术、炸弹、投射物、攻城武器（如炮弹）、破片箭。0 HP 倒地 = 被判出局，不死人</li>
<li><b>不阻止法术其他效果</b>：解离术（Disintegrate）仍可"解离"你（不死，但承受其他效果）；麻痹、流失、束缚、迷惑等效果照常生效</li>
<li><b>项链由执行者激活</b>：让执行者的激活动作明确出现在场景里 —— 玩家应该看见这个仪式动作，建立"激活 = 比赛开始"的仪式感</li>
<li><b>通过心电感应通信</b>：项链与执行者保持心电感应连接，因此必须戴在身上才能正常工作</li>
</ul>
<h2>常见漏洞处理</h2>
<ul>
<li><b>玩家想塞无尽袋让迷音鸟偷不到</b> → <b>强制戴在身上</b>。GM 解释："项链通过心电感应跟执行者通信，离开身体几寸就失联"</li>
<li><b>玩家不激活就用致命伤害</b> → 场外（非比赛）允许，但其他队伍会以同样方式回敬。"光辉使者会很乐意按你给定的标准对付你"</li>
<li><b>5 人队怎么分项链</b> → <b>一人一条</b>。避免战斗中"谁倒地谁还活"的歧义。RAW 模糊，但所有 5 人队伍都用 OriGami 补员时有官方 5 项链处理</li>
<li><b>项链激活时机由谁判断</b> → 默认执行者激活；只有当 PC 主动要求"以致命模式打"时不激活</li>
<li><b>项链能不能给 NPC 队伍用</b> → 所有正式比赛参赛者都有。NPC 也是非致命。这就是为什么"刚被你打倒的对手下一日还活蹦乱跳"是合理的</li>
</ul>
<h2>⚠ 第一日危险事件回顾</h2>
<p>第一日 4 名内家好手资格队带『解离术』。详见 <b>「第一日危险事件警告」</b>页。<b>必须赛前明确激活项链</b>。</p>
<h2>项链与重要法术效果的交互</h2>
<ul>
<li><b>解离术（Disintegrate）</b>：不死，但伤害仍计入"非致命"且解离条件正常应用</li>
<li><b>炫彩之球（Prismatic Sphere）</b>：项链不阻止"被困在球内"</li>
<li><b>阴影狂风 / 阴影突袭</b>：伤害非致命化</li>
<li><b>毒液</b>：项链不阻止毒效（瘫痪、虚弱、流失），但毒蚀的"持续伤害"非致命</li>
<li><b>解构 / 死亡效应法术</b>：典型如死亡之指 —— 书面争议；多 GM 裁定"死亡效应"不被项链阻挡（毕竟项链不是无敌），所以"每队 1 次免费复活"是给这类情况留余地的</li>
<li><b>麻痹 / 瘫痪</b>：完全正常生效，只是结束战斗时不会真死</li>
</ul>
<blockquote title="GM 建议"><p>开第一场前白板讲清五条桌规时，把这一条放在最显眼位置：「项链激活 = 非致命，但解离术 / 毒 / 瘫痪 / 控制效果照常生效」。开局前 10 分钟讲清，避免半场后玩家"想钻空"。</p></blockquote>
<h2>跨书提示：哪些战斗项链失效</h2>
<ul>
<li><b>第一本前期</b>：项链未发放/未激活的非比赛战斗（义洛理神庙清剿全程项链未激活）</li>
<li><b>第二本魔加鲁连战</b>：怪兽袭击不是比赛 —— <b>项链失效</b>，可以死人</li>
<li><b>第三本琉璃光屋决战</b>：半位面内项链效力如何？多 GM 选择 <b>失效</b>（强化终局感）；少数 GM 保留以减少 PC 死亡</li>
</ul>""",
        },
        {
            "slug": "book1-showcase-presentation",
            "name": "GM 指南：综艺化演绎建议 GM GUIDE: SHOWCASE PRESENTATION",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。强烈推荐的 GM 演绎技巧 —— 锦标赛综艺化。建议从第一本就开始铺垫，第二本进入锦标赛主体时已经成熟。</p>
<h2>核心原则</h2>
<p>赤凰武道会本质是 <b>动漫式格斗综艺</b>（参考拳愿阿修罗 / 龙珠武道大会 / 街霸格斗大赛）。把它当摔角秀 / 综艺节目来跑，比当严肃军事策略博弈跑更对味。</p>
<h2>多贺台·艾米提前到第一本出现</h2>
<p>多贺台·艾米（Tagada Emmi）书面是第二本登场的解说员。<b>多 GM 推荐提前到第一本</b>：</p>
<ul>
<li><b>巴铭采访式片段</b>：在邦木岛资格赛日间多贺台来"采访"玩家，问"今天的策略是什么？""上一场感觉怎么样？""你对千野最硬有什么看法？"</li>
<li><b>船上预露面</b>：玩家航行到邦木岛时多贺台已在场，做开赛预热</li>
<li><b>第一日开赛仪式</b>：多贺台站执行者旁边宣告参赛队伍 —— 给比赛一个综艺开场</li>
<li><b>过峡抵达环节</b>：第一本结尾多贺台主持赛季正式开幕，与郝金一起做开幕仪式</li>
</ul>
<h2>魔法海螺当麦克风</h2>
<p>多贺台用 <b>魔法海螺当麦克风</b>（综艺感的核心道具），声音传到远在过峡的观众席。这意味着：</p>
<ul>
<li>玩家做事时 <b>远方观众实时观看</b> → 玩家自觉地"为镜头表演"</li>
<li>玩家做帅动作 → 多贺台现场宣布、播给观众听</li>
<li>玩家做蠢事 → 多贺台也喷（综艺解说不会保护选手脸面）</li>
<li>观众席给玩家做出回应（欢呼、嘘声、口号），通过海螺传回比武场</li>
</ul>
<h2>反派被嘘 +1 加成（摔角逻辑）</h2>
<p>反派招式被观众嘘的时候反而 <b>+1 状态加成</b>（"你越恨我我越强"）。这给反派比赛带来戏剧张力：</p>
<ul>
<li>蓝蝮蛇用毒 → 观众"嘘！" → 蓝蝮蛇 +1 攻击 + 伤害（成为更可怕的反派）</li>
<li>玩家用合理但"无聊"招（如纯持续治疗 + 后排支援）→ 观众也会嘘</li>
<li>玩家做"帅但不利"招（如英雄性救队友、单挑大反派）→ 观众喝彩 +1 加成</li>
<li>玩家"耻辱败北"也是好戏 —— 多贺台可以做戏剧化解说</li>
</ul>
<blockquote title="GM 提示"><p>这套机制可以与 §16 观众影响力子系统（详见第二本 back-matter）联动。简化版：1-5 阶代币，每"观众喜欢的动作"+1，每"反派招"-1。第二本观众影响力满 3 时全员 +1 状态攻击/伤害，5 时 +2。</p></blockquote>
<h2>各队伍主题音乐</h2>
<ul>
<li><b>千野最硬</b>：JoJo OST 全套</li>
<li><b>净光守护者</b>：Apashe Lacrimosa / 街霸 Juri 主题</li>
<li><b>魔加鲁出场</b>：原子吐息 SFX（哥斯拉主题感）</li>
<li><b>雕匠辛达拉主题</b>：宿傩主题</li>
<li><b>玩家队伍</b>：开赛前问玩家"你们的主题音乐？" —— 让玩家选自己的入场曲</li>
<li><b>观众支持转向时音乐变化</b>：观众影响力数值变化时主题音乐切歌（用 Foundry 的 Scene Transitions 模块）</li>
</ul>
<p>详细音乐推荐表见 <b>「资源与工具清单」</b>页。</p>
<h2>第一本如何铺综艺感</h2>
<ul>
<li><b>船上预热</b>：玩家航行到邦木岛途中，让千野最硬等友军队伍一起在船上 —— 多贺台采访片段植入</li>
<li><b>资格赛日间</b>：多贺台·艾米偶尔出现做赛后采访 —— 1-2 次即可，不喧宾夺主</li>
<li><b>巨兽巢穴 / 邦木探索</b>：让玩家想象"探索片段也有摄像头" → 表演式互动</li>
<li><b>兵马俑夜袭</b>：把 Wave 2 改成"突然出现的反派环节"，多贺台远程报道"邦木岛突发袭击！全员撤离！"（戏剧张力 +1）</li>
<li><b>过峡抵达 + 郝金仪式</b>：第一本结尾的"赛季开幕式" → 用综艺感的隆重仪式 + 多贺台主持</li>
</ul>
<h2>注意：先问玩家口味</h2>
<blockquote title="GM 提示"><p>综艺化演绎是本 AP 最能提升桌前体验的技巧之一，但要看玩家口味。开第一场前问玩家"你们想跑严肃武侠还是动漫综艺？" —— 60%+ 玩家会选综艺。如果玩家选"严肃"，不要硬塞综艺感，专注规则修补即可。</p></blockquote>""",
        },
        {
            "slug": "book1-errata-decisions",
            "name": "GM 指南：第一本勘误与决策 GM GUIDE: BOOK 1 ERRATA & DECISIONS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第一本《绝岛之殇》全部已知勘误、平衡问题、决策点汇总表。</p>
<h2>第一本勘误汇总</h2>
<ul>
<li><b>废弃监狱 R3</b>：书面提及"大厅"但地图无 —— 详见「监狱大厅勘误」页</li>
<li><b>A14 困难地形</b>：地图未标 —— 整片深绿当困难地形</li>
<li><b>毒蛇藤</b>：花粉太大 + 行动锁死 —— 缩小半径 / 成功豁免后免疫，详见「义洛理神庙补丁」页</li>
<li><b>迷音鸟</b>：无巧手 + 玩家痛恨 —— 改追逐 / 跳过 / 强制项链戴身上，详见「邦木岛地点参考」页</li>
<li><b>巨兽巢穴</b>：无地图无战利品 —— 按章节财富自加部位 + 自找地图</li>
<li><b>花之百人众</b>：无小兵团处理 —— 用 PF2e Troops Helper 模块</li>
<li><b>千野最硬 20HP 弱点</b>：高级时几乎不可能触发 —— 忽略</li>
<li><b>资格队（八臂调和等）无数据卡</b>：用基础格斗者模板</li>
<li><b>「正义的」亚彬急动</b>：无 1/日限制 —— 加 1/日限制</li>
<li><b>女妖 16 级</b>：12 级队伍太强 —— 换 14 级噪鬼</li>
<li><b>第 1 日 4 名内家好手解离术</b>：PC 11 级 HP 不够 —— 强制激活凤凰项链，详见「第一日危险事件警告」页</li>
</ul>
<h2>第一本时间压力</h2>
<p><b>第一本六角探索有时间帽：不可能全清岛</b>。这是设计意图。</p>
<ul>
<li>3 日资格赛 = 玩家必须取舍：哪些据点访问，哪些跳过</li>
<li>第 3 日可压缩（玩家提前完成 10 凤凰羽毛即可）</li>
<li>不要让玩家产生"我必须打完所有遭遇"的错觉，否则节奏拖死</li>
</ul>
<h2>第一本主要决策点</h2>
<ul>
<li><b>① 凤凰项链怎么激活？</b> 强制戴在身上；执行者激活</li>
<li><b>② 毒蛇藤怎么改？</b> 缩花粉半径或成功豁免后免疫</li>
<li><b>③ 迷音鸟跑还是不跑？</b> 大多跳过或改追逐</li>
<li><b>④ 巨兽巢穴战利品？</b> 加部位 + 1 件物品/巢（金额按章节财富自定）</li>
<li><b>⑤ 第 1 日 4 名内家好手解离术？</b> 强制激活项链或预警</li>
<li><b>⑥ 处理伤口频率？</b> 1 小时/人无限次；禁预增益（除当日）</li>
<li><b>⑦ 升级时机？</b> 节奏化（章节里程碑）</li>
<li><b>⑧ 死亡如何处理？</b> 除"死亡效应"外都模糊处理；每队 1 次免费复活</li>
</ul>""",
        },
    ],
}

PAGES_BOOK2 = {
    "ch1": [
        {
            "slug": "book2-overview-goka",
            "name": "GM 指南：第二本总评与抵过峡 GM GUIDE: BOOK 2 OVERVIEW & GOKA ARRIVAL",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第二本《比赛开始！》（Ready? Fight!）的节奏建议与第 1 章砍刀建议。</p>
<h2>总评</h2>
<ul>
<li><b>核心事件</b>：5 天锦标赛 + 第 5 日连战（决赛 + 净光守护者 + 魔加鲁中断）</li>
<li><b>管理玩家预期</b>：依旧是大量地下城+探索，少量锦标赛对决</li>
<li><b>节奏</b>：第 1 章是"过渡章"，第 2 章是锦标赛主体</li>
</ul>
<blockquote title="GM 提示"><p>玩家如果在第一本期待"满章锦标赛"会失望，第二本中段才真正进入锦标赛节奏。开第二本前再次校准玩家预期。</p></blockquote>
<h2>第 1 章：抵达过峡（节奏砍刀章）</h2>
<p><b>多 GM 共识：砍掉第 1 章部分事件</b>，让玩家尽快进第 2 章锦标赛。书面第 1 章事件多且节奏分散，全跑会让玩家觉得"什么时候才开始比赛？"</p>
<h3>重要第 1 章事件（建议保留）</h3>
<ul>
<li><b>曹丸调查</b>：犯罪调查支线（书面较虚，GM 自填）。可压缩成 1-2 场关键情报场景</li>
<li><b>灯笼小舍 / 玛莱卡·陶</b>（Malaika Tao）：必备雕匠辛达拉铺垫场所，<b>不要砍</b></li>
<li><b>歌剧院（殷后戏院）</b>：少数有官图的遭遇 —— 视觉精彩，建议保留</li>
</ul>
<h3>可砍/降难度的第 1 章事件</h3>
<ul>
<li><b>日蚀随机遭遇</b>：近团灭难度 —— <b>可砍或降难度</b>。如果保留，把"日蚀之光"伤害降一档</li>
<li>部分赞助人争取支线 —— 不要全跑，挑 2-3 个玩家感兴趣的赞助人聚焦</li>
</ul>
<h2>4 个核心第 1 章事件</h2>
<ol>
<li><b>抵达过峡 + 赞助人争取</b>：玩家与 5 个可争取赞助人（玛莱卡·陶 / 飞田卡索 / 阿达纳·乌马尔 / 邱美莎 / 夏之芽）社交</li>
<li><b>灯笼小舍画廊</b>：雕匠辛达拉铺垫 + 玛莱卡·陶任务线</li>
<li><b>殷后戏院遭遇</b>：地下犯罪 + 优秀官图战斗场景</li>
<li><b>日蚀随机遭遇</b>：可选；如果保留要降难度</li>
</ol>
<h2>跨本提示：辛达拉铺垫</h2>
<blockquote title="→ 跨本提示"><p>灯笼小舍 + 玛莱卡·陶是第二本主要的雕匠辛达拉铺垫场所。如果在第一本已经用了 §48 铺垫方案（详见第三本附录「雕匠辛达拉铺垫方案」页），第二本的灯笼小舍要呼应已铺好的暗线（化名赞助人 / shadowy figure / 玩家背景钩等）。</p></blockquote>""",
        },
    ],
    "ch2": [
        {
            "slug": "book2-tournament-schedule",
            "name": "GM 指南：锦标赛 8 日日程 GM GUIDE: TOURNAMENT 8-DAY SCHEDULE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第 2 章锦标赛的 8 日完整日程与第 5 日资源压力。</p>
<h2>8 日完整日程</h2>
<ul>
<li><b>Day 1</b>：玩家 vs 寒霜之吼；陷阱（浮动喷火器/寒冰地砖）</li>
<li><b>Day 2</b>：表演赛 vs 八臂调和（护城河迷宫）+ 表演赛"龙兽竞速"追逐</li>
<li><b>Day 3</b>：玩家 vs 风语者</li>
<li><b>Day 4</b>：败方组第一轮 + 表演赛"赤玉塔塔楼战斗"</li>
<li><b>Day 5</b>：败方组第二轮 + 表演赛（迷宫 + 寒林巨鬼）</li>
<li><b>Day 6</b>：玩家 vs 净光守护者；卡熙莱"黄金盟入室盗窃"事件</li>
<li><b>Day 7</b>：败方组最终轮 + 库比亚赌场"雷管"地下格斗</li>
<li><b>Day 8</b>：净光守护者决赛 + 可能复赛 + 魔加鲁怪兽袭击</li>
</ul>
<h2>对阵图揭示时机</h2>
<p><b>书面 RAW 模糊。改良方案</b>：Day 1 早晨揭示完整对阵图，让玩家做"放水/真打"的策略选择 —— 玩家可实时看自己排名、对手、上下半区。用 Challonge.com 做免费在线锦标赛括号。</p>
<h2>第 8 日流程</h2>
<p>决赛 → 魔加鲁攻城 → 拉祖追逐 → 转第三本。<b>第 8 日是节奏最紧的一天</b>，准备充分。</p>
<h2>第 5 日资源压力（设计意图）</h2>
<p>第 5 日是资源紧缺的设计意图日。两派应对：</p>
<h3>援助派（推荐用于 5 PC 队伍）</h3>
<ul>
<li>风语者给治疗术</li>
<li>盛阳漫步给重新专注</li>
<li>友军 NPC 提供消耗品</li>
</ul>
<h3>严苛派（用于 6 PC 满配队伍）</h3>
<ul>
<li>让玩家烧资源</li>
<li>让郝金牺牲式复活倒地 PC</li>
<li>不予援助，让玩家深刻感受"锦标赛是消耗战"</li>
</ul>
<blockquote title="GM 提示"><p>选择派别取决于队伍构成：5 PC + 偏综艺向 → 援助派；6 PC 满配 + 偏硬核 → 严苛派。两派可以混用，例如前几日严苛、第 5 日给一次援助让玩家喘息。</p></blockquote>""",
        },
    ],
    "ch3": [
        {
            "slug": "book2-mogaru-interrupt",
            "name": "GM 指南：魔加鲁中断转第三本 GM GUIDE: MOGARU INTERRUPT TO BOOK 3",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第二本第 8 日决赛后魔加鲁怪兽袭城事件，是 AP 唯一真实的时间压力点。</p>
<h2>事件流程</h2>
<p>第 5 日（书面）或 Day 8 决赛后第 1 场（多 GM 修改），<b>魔加鲁（Mogaru）出现</b>。详细机制与数据卡修补见 <b>第三本「魔加鲁连战」</b>页。</p>
<h2>设计意图</h2>
<p>魔加鲁连战是 AP 的标志性场景之一 —— 从锦标赛的"格斗运动"切换到"灾难片"的氛围转换。要充分使用"哥斯拉式"演绎：</p>
<ul>
<li>原子吐息 SFX</li>
<li>母子重构：把魔加鲁性别改写为母亲保护幼崽（更有情感张力）</li>
<li>平民死亡机制：玩家可用欺骗/表演检定分散魔加鲁注意力</li>
<li>哥斯拉电影感的镜头切换：远景 → 近景 → 怪兽视角</li>
</ul>
<h2>跨本提示</h2>
<blockquote title="→ 跨本提示"><p>魔加鲁数据卡平衡问题（AC、HP、喷吐 DC）详见第三本「魔加鲁连战」页。第二本只是触发场景，实际战斗在第三本开篇。这一段是"节奏切换页"，让 GM 知道接下来该翻到第三本第 1 章。</p></blockquote>
<h2>第 5 / 8 日玩家应对</h2>
<ul>
<li>玩家烧资源穿越魔加鲁狂暴抵拉祖（Razu）</li>
<li>拉祖追击 = 第二本最后一战，过渡到第三本</li>
<li>如果玩家试图直接打魔加鲁 —— 17 级队伍打不动，应让玩家明白"逃才是正确选择"</li>
</ul>
<h2>音乐与氛围</h2>
<p>魔加鲁出场配乐：原子吐息 SFX（哥斯拉主题感）。第 8 日决赛结束 → 一段太鼓 OST → 突然切到原子吐息 SFX —— 让玩家在听觉上立即感受到"出大事了"。</p>""",
        },
    ],
    "bm": [
        {
            "slug": "book2-tournament-rules",
            "name": "GM 指南：锦标赛核心规则与观众影响力 GM GUIDE: TOURNAMENT RULES & CROWD INFLUENCE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。锦标赛运作的核心房规：对阵图、治疗、升级、预增益红线、观众影响力子系统。第二本锦标赛主体期间反复使用。</p>
<h2>对阵图与 Challonge.com</h2>
<p>用 Challonge.com 做免费在线锦标赛括号。玩家可实时看自己排名、对手、上下半区；每场结果即时填入，自动绘出后续配对。</p>
<p><b>对阵图揭示时机</b>：RAW 模糊。<b>改良方案</b>：第 1 日早晨揭示完整对阵图，让玩家做"放水/真打"的策略选择。</p>
<h2>比赛之间治疗与休息</h2>
<p>最常见折中方案：</p>
<ul>
<li>✓ <b>允许处理伤口</b>（1 小时/人，无限次）</li>
<li>✗ <b>禁止预增益</b></li>
<li>✗ <b>禁止准备新法术</b></li>
</ul>
<p><b>死亡处理</b>：除"死亡效应"外都模糊处理；每队 1 次免费复活。</p>
<h2>升级时机</h2>
<p><b>节奏化升级</b>（章节里程碑）：避免锦标赛里"某玩家 16 级、某玩家 17 级"的尴尬。<b>书面建议</b>：玩家 17 级进入魔加鲁中断；锦标赛结束前不升级。</p>
<h2>预增益红线</h2>
<ul>
<li><b>8 小时 / 当日增益</b>：保留</li>
<li><b>身体涂层和饰品状的当日增益</b>：默认允许</li>
<li><b>10 分钟以下增益</b>：所有人到场后才能放</li>
<li><b>1 分钟以下增益</b>：禁止</li>
</ul>
<h2>观众影响力子系统</h2>
<h3>完整版</h3>
<p>用 FotRP Expanded PDF 的完整观众子系统（详细规则在 PDF 中）。</p>
<h3>简化版（无 Foundry 或不想跑完整版）</h3>
<p>用 1-5 张计数标记代表观众情绪：</p>
<ul>
<li>每场起始 <b>3</b></li>
<li>每"观众喜欢的动作" <b>+1</b></li>
<li>每"反派招" <b>-1</b></li>
<li><b>3 以上</b>：全员 +1 状态攻击/伤害</li>
<li><b>5</b>：全员 +2 状态攻击/伤害</li>
</ul>
<p>详细综艺化演绎方案见 <b>第一本附录「综艺化演绎建议」</b>页。</p>
<h2>迷宫 / 困境法术禁用</h2>
<p>锦标赛中禁用迷宫术、困境术 —— 不好玩，GM 与玩家都烦。开第二本前白板加上这条房规。</p>""",
        },
        {
            "slug": "book2-team-tinos-toughest",
            "name": "GM 指南：千野最硬队伍档案 GM GUIDE: TINO'S TOUGHEST PROFILE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。剧情主角队伍千野最硬（Tino's Toughest）的成员演绎、平衡问题、戏剧高潮。</p>
<h2>成员名册</h2>
<ul>
<li><b>千野·董</b>（Tino Tung）—— 武僧 + 圣武士兼修</li>
<li><b>「正义的」亚彬</b>（Yabin the Just）—— 龙系狂热者</li>
<li><b>鹰虎</b>（Takatorra）—— 厨师/支援，<b>爱甜食</b></li>
<li><b>纪幽</b>（Ji-Yook）—— 狐影流武僧</li>
<li><b>Cac-Lee</b> —— 5 人队加员，含 9/13/18 级角色卡 JSON</li>
</ul>
<p><b>Cac-Lee 背景</b>：元（千野哥哥）的旧对手，元死后转向追随千野。这给 Cac-Lee 加入提供了自然剧情入口。</p>
<h2>千野性格</h2>
<p>奉行义洛理（Irori），自我提升，少年漫主角式无视追求者（即使其他角色暗恋他他也意识不到）。</p>
<p><b>音乐</b>：JoJo OST 全套。</p>
<h2>常见问题：太弱</h2>
<ul>
<li>AC 偏低，玩家会感觉"千野居然这么菜"</li>
<li>多 GM 升 1-2 级 + 加装备</li>
<li>给千野加圣武士的"惩罚打击"反应</li>
</ul>
<h2>戏剧亮点</h2>
<h3>苏塔奴的惊骇假面</h3>
<p>用惊骇假面（Mask of Terror）让苏塔奴对千野施法，让千野看见死去的弟弟元，挣扎不愿战斗。这是第 5 日决赛的高潮设计。</p>
<h3>第 5 日戏剧死亡</h3>
<p>千野被解离术击中，郝金复活他（千野试图救郝金时被打中）。这是 AP 标志性场景之一。</p>
<h2>⚠ 解离术郝金戏的真值魔王</h2>
<p>书面让郝金被解离术失败救（20 级根本不可能失败那个豁免）。<b>多数 GM 改写</b>：郝金故意挨解离术来嘲讽 NPC ——"Do you really think that will hurt me?"</p>
<blockquote title="GM 演绎"><p>千野最硬的"友善+乐观"基调要在第一本船上互动中建立（详见第一本「第一本总评」页的船上互动建议）。第二本玩家与千野最硬的对战是情感最复杂的一场 —— 玩家既要赢但又不想真伤他们。</p></blockquote>
<h2>跨本提示：第三本恶鬼形态</h2>
<p>第三本千野最硬有腐化形态：</p>
<ul>
<li>千野（恶鬼形态 Demon Form）</li>
<li>鹰虎（大天狗形态 Great Tengu Form）</li>
<li>纪幽（九尾狐形态 Nine-Tailed Fox Form）</li>
<li>亚彬（白蛇形态 White Snake Form）</li>
</ul>
<p>第二本玩家死前还能见到他们的"原本面貌"，第三本会变形。这种"被腐化的友军"是 AP 的情感核心之一。</p>""",
        },
        {
            "slug": "book2-team-lightkeepers",
            "name": "GM 指南：净光守护者队伍档案 GM GUIDE: LIGHTKEEPERS PROFILE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。最终大反派队伍净光守护者（Lightkeepers）的 4 成员详细修补方案、毒液平衡、法术清单改写。<b>第二本最重要的备课内容</b>。</p>
<h2>成员名册</h2>
<ul>
<li><b>捣达</b>（Ran-To）—— 武僧 / 擒抱专家</li>
<li><b>白谷</b>（Hakusa）—— 游荡者 / 忍者（剧透：白谷真野 Shino Hakusa）</li>
<li><b>蓝蝮蛇</b>（Blue Viper）—— 炼金术士 / 用毒师</li>
<li><b>苏塔奴</b>（Syu Tak-Nwa）—— 神秘巫女</li>
</ul>
<h2>§37.2.1 捣达（破口袋专门）</h2>
<h3>⚠ 力量值极高问题</h3>
<p>借机攻击触发擒抱、力量 +35。千野的强韧 DC 是 35：</p>
<ul>
<li>捣达投出 2 就成功</li>
<li>投出 11 就大成功</li>
<li>50% 概率束缚千野 = 等于跳过千野这个回合</li>
</ul>
<p><b>修补</b>：用旧版"擒抱"规则，或力量 -2。</p>
<p><b>风格选择</b>：静默无对话 vs 多说两句。</p>
<h2>§37.2.2 白谷（白谷真野）</h2>
<p><b>真野静默化共识</b>：只用动作和眼神，不说话。</p>
<h3>增益推荐（高难度版）</h3>
<p><b>多消耗品</b>：</p>
<ul>
<li>卡之冻火（Frozen Lava of Ka）</li>
<li>石身变体药剂（Stone Body Mutagen）</li>
</ul>
<p><b>单次法术</b>：</p>
<ul>
<li>阴影狂风（Tempest of Shades）—— 设定：眼纹身储存她杀过的灵魂</li>
<li>点穴（Pressure Points）DC 7 平稳检定</li>
</ul>
<p><b>石化毒</b>：被捣达用毒武器击中可以石化，玩家瞬间"卧槽"时刻。</p>
<p><b>新法术添加</b>：阴影突袭（Shadow Raid）+ 2× rank 7 幽影爆破（Shadow Blast）。</p>
<p><b>重生剧本钩子</b>：让 PC 与白谷有亲缘（如"PC 父亲转世为净光守护者成员"），最终战让白谷倒戈。</p>
<h2>§37.2.3 蓝蝮蛇（最危险的成员）</h2>
<p><b>历史评价两极</b>：早期像"无效笑话" → 增益后"几乎一回合干掉整队"。</p>
<h3>毒液问题（核心修补点）</h3>
<p><b>梅花滂沱 + 死之泪</b>（Plum Deluge + Tears of Death）= 几乎稳定的团灭触发器：</p>
<ul>
<li>DC 47 强韧豁免，20 ft 爆发</li>
<li>失败：1 轮瘫痪</li>
<li>大失败：1 <b>分钟</b>瘫痪 + 毒蚀</li>
</ul>
<h3>修补共识</h3>
<ul>
<li><b>不用死之泪</b>（多数 GM 选择）：改用黑莲汁或惑心迷雾</li>
<li><b>改接触毒 → 伤口毒</b>：让 DC 自由</li>
<li><b>拉满极端 DC + 全免起效时间</b>（队伍自带解药情况下）</li>
<li><b>加惑心迷雾 + 梅花滂沱组合</b>（中等危险）</li>
</ul>
<blockquote title="GM 提示"><p>重铸版（Remaster）取消起效时间后基本所有传统毒液无效，梅花滂沱与死之泪几乎无法平衡。除非队伍硬核要求，否则简单粗暴：<b>不用死之泪</b>。</p></blockquote>
<h3>其他增益</h3>
<ul>
<li>装备：变形蜘蛛颈圈、银汞变体药剂、更好的毒液</li>
<li>戒指反应可连发：每回合 1 次，无需额外蓄毒</li>
</ul>
<h2>§37.2.4 苏塔奴（法师，<b>法术清单必改写</b>）</h2>
<p><b>默认法术清单公认垃圾</b> —— 必须改写。</p>
<h3>增益方案（高难度版 6 PC 队伍）</h3>
<p><b>必加</b>：</p>
<ul>
<li>无影无踪（Disappearance）—— 开局必出，免费传送 + 急动</li>
<li>急动法术物品 —— 让她第 1 轮出 6 阶减速术 + 无影无踪</li>
</ul>
<p><b>加</b>：</p>
<ul>
<li>永悲圣咏（Canticle of Everlasting Grief）—— 控制</li>
<li>7 阶木偶替身（Wooden Double）—— 转移</li>
</ul>
<p><b>替换</b>：</p>
<ul>
<li>把使无法行动焦点法术换成直接伤害</li>
</ul>
<p><b>剧情高潮神器</b>：用惊骇假面让千野看见死去的弟弟元。</p>
<h2>§37.2.5 净光守护者综合战术（高难度版）</h2>
<h3>Turn 1</h3>
<ol>
<li><b>苏塔奴</b>：无影无踪 → 免费传送进入 → 急动减速术（6 阶）</li>
<li><b>捣达</b>：进入扰乱姿态 → 抓最近施法者</li>
<li><b>蓝蝮蛇</b>：戒指反应 → 给施法者惑心迷雾</li>
<li><b>白谷</b>：阴影狂风 / 准备法术</li>
</ol>
<h3>Turn 2+</h3>
<ol>
<li><b>苏塔奴</b>：群体加速给队友 / 进攻法术</li>
<li><b>白谷</b>：点穴 / 卡之冻火</li>
<li><b>蓝蝮蛇</b>：龙胆汁（无起效时间）给脆皮 / 邪术耀斑给法师</li>
<li><b>捣达</b>：持续欺负施法者</li>
</ol>
<h2>§37.2.6 净光守护者作弊机制（推荐采用）</h2>
<h3>执行者操控</h3>
<p>净光守护者用魅惑魔法控制执行者。原执行者已死，新执行者因裙带关系上位 —— 玩家应该察觉到比赛"不太对劲"。</p>
<h3>噩梦法术骚扰</h3>
<p>对决前一天对玩家施法噩梦（Nightmare）。不 RAW，但完美剧情。</p>
<h3>第三方挑战</h3>
<p>等玩家刚打完一场未恢复时立刻挑战 —— 趁势杀凤凰羽毛 / 杀士气。</p>
<h2>跨本提示：第三本最终战</h2>
<p>第三本净光守护者最终战全员精英状态，详见第三本「终局战斗与尾声」页。</p>""",
        },
        {
            "slug": "book2-team-speakers-to-winds",
            "name": "GM 指南：风语者队伍档案 GM GUIDE: SPEAKERS TO THE WINDS PROFILE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。玛甘比学院 6 人队风语者（Speakers to the Winds），由马菲卡·阿尤瓦里（Mafika Ayuwari）carry。</p>
<h2>成员名册</h2>
<ul>
<li><b>马菲卡·阿尤瓦里</b>（Mafika Ayuwari）—— 导师</li>
<li><b>阿基拉·暴风踵</b>（Akila Stormheel）</li>
<li><b>无境·蜂鸟</b>（Boundless Hummingbird）</li>
<li><b>普希·努瓦</b>（Phuthi Nuware）</li>
<li><b>苏吉特·哈梅兰</b>（Surjit Hamelan）</li>
<li><b>乌木巴西</b>（Umbasi）</li>
</ul>
<p><b>6 人</b> —— 唯一超过 4 人的队，5 PC 队不需补员。</p>
<h2>马菲卡的真危险</h2>
<ul>
<li><b>炫彩之球无法应对</b> → 1 轮让队伍投降的情况真实发生过</li>
<li><b>大成功"极冰射线"1 轮干掉 PC</b> → 极冰射线大成功 + 风暴爆发大失败 = 60 ft 内冲锋的蛮族秒杀</li>
<li><b>英雄气概 +2 / 克敌机先滥用</b></li>
</ul>
<h2>增益推荐</h2>
<p>默认法术清单偏弱（社区共识）→ 用 FotRP Expanded 或自定清单。具体属性数值由 GM 自定（无标准）。</p>
<h2>应对（玩家创意）</h2>
<ul>
<li><b>持续伤害 + 近战推开/绊倒把他物理弄出球</b></li>
<li><b>回旋扔（Whirling Throw）把队友扔进球</b>（与马菲卡近距对抗）</li>
<li><b>"效果类似命名法术"可抵消球壁</b>（桌规可允许创意 dispel）</li>
</ul>
<blockquote title="GM 提示"><p>炫彩之球是 GM 工具箱里"会让玩家投降"的少数法术。如果玩家队伍构成对抗不了它，提前给玩家提示"准备持续伤害"或允许创意应对。完全不破解的话玩家会非常挫败。</p></blockquote>
<h2>马菲卡演绎</h2>
<ul>
<li><b>BAMF + 名人感</b>：他是《千圣力量》（Strength of Thousands）AP 中的传奇导师，FotRP 是他的另一个露面</li>
<li><b>战斗中让他站后排</b>（他是导师，不喜欢硬碰硬）</li>
<li>战败后会"风度地认输"，不像净光守护者那样阴险</li>
</ul>
<h2>跨本提示</h2>
<p>风语者属于"友军反派" —— 第 5 日资源紧缺时，风语者给玩家治疗术援助（详见「锦标赛 8 日日程」页的援助派方案）。</p>""",
        },
        {
            "slug": "book2-team-biting-roses",
            "name": "GM 指南：刺骨蔷薇队伍档案 GM GUIDE: BITING ROSES PROFILE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。刺骨蔷薇（Biting Roses）双精神导师队伍 + 迷宫展赛的修补建议。</p>
<h2>成员名册（参见 back-matter 中既有详细介绍）</h2>
<ul>
<li>雅利卡·穆兰德兹（Yarrika Mulandez）—— 灵媒，配幻灵</li>
<li>阿托斯·罗德里万（Artus Rodrivan）—— 灵箭手</li>
<li>兰通多（Lantondo）—— 木仆占塔者</li>
</ul>
<h2>⚠ 重大勘误：法术清单重复</h2>
<p>书面 <b>双"闪烁图案"</b>（两个法术清单完全一样）—— 几乎团灭 14 级队伍。</p>
<h3>修补</h3>
<p>FotRP Expanded PDF 给两个不同清单。强烈推荐采用。</p>
<h2>迷宫展赛</h2>
<h3>问题</h3>
<ul>
<li>没火伤源 → 打不动 Taiga Yai 的再生</li>
<li>飞行未禁 → 迷宫意义大减</li>
</ul>
<h3>修补方案</h3>
<ul>
<li>改追逐序列 + Taiga Yai 战 + 中央对决</li>
<li>加倍地图比例 + 重做墙壁 + 禁止飞行</li>
<li>给玩家暗示"墙是不可飞越的次元壁"（半位面边界感）</li>
</ul>
<blockquote title="GM 提示"><p>迷宫展赛书面体验差是社区共识。如果不修补，玩家会怀疑设计本身。最简单方案：开赛前直接说"这是禁飞迷宫" + FotRP Expanded 双不同法术清单。</p></blockquote>
<h2>演绎要点</h2>
<ul>
<li><b>雅利卡的幻灵</b>：以无声的、形似人类却长着螳螂头与前肢的身影显形。雅利卡相信这只幻灵是她祖父的灵魂 —— 她最初的格斗导师</li>
<li><b>阿托斯射出由自身灵魂能量化作的箭矢</b>：战斗中必须格外小心，避免对自己不朽的灵魂造成不可逆的伤害</li>
<li><b>兰通多的占卜牌组</b>：开赛前会聚在一起让兰通多为团队进行一次占卜读牌，并据此规划战斗策略、洞察对手能力 —— GM 可以让兰通多预测玩家招式（增加难度）</li>
</ul>""",
        },
        {
            "slug": "book2-teams-qualifying",
            "name": "GM 指南：资格队伍档案合集 GM GUIDE: QUALIFYING TEAMS COMPENDIUM",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。其他次要队伍档案合集：花之百人众 / 格拉利昂最强 / 盛阳漫步 / 寒霜之吼 / 八臂调和 / 灰日狂舞者 / 其他资格队。</p>
<h2>§37.5 花之百人众（44 人小兵团 Hana's Hundreds）</h2>
<ul>
<li><b>Foundry</b>：必装 PF2e Troops Helper 模块</li>
<li><b>代币找图</b>：搜"巨型小兵军团"主题</li>
<li><b>设计原则</b>：<b>喜剧效果第一，难度第二</b></li>
</ul>
<p>把这场打成"动漫军团战" —— 玩家被 44 个小兵围攻，但靠范围伤害和创意机动横扫。GM 描述应该夸张化（小兵互相撞、被一脚踢飞 N 个）。</p>
<h2>§37.6 格拉利昂最强（街霸 II 致敬 Golarion's Finest）</h2>
<p><b>真名披露</b>：整队是<b>街头霸王 II 的致敬</b> —— 知道这个梗后所有 NPC 都对得上号。</p>
<h3>成员名册</h3>
<ul>
<li>布拉托克（Brartork）/ 韩（Han）/ 俊（Jun）/ 库拉克丝（Krankkiss）</li>
<li>明玉（Mingyu）/ 路莫里兹（Numoriz）/ 保宁玛（Paunnima）/ 拉贾娜（Rajna）</li>
</ul>
<p><b>危险等级</b>：极端（320 经验 @ 12 级）；8 实体 + 各人独有招数。</p>
<p><b>修补</b>：内家好手打击伤害太低 → 加到 3d8+16 / 真气波 12d6+13d6。</p>
<h3>GM 用法</h3>
<ul>
<li><b>心情检测器（Vibe check）</b>：玩家太过自信时丢给他们试试</li>
<li><b>神操作</b>：让光辉使者在资格赛最后一天的比赛中直接杀掉他们 —— 早期建立"光辉使者必死"反派情绪</li>
</ul>
<h2>§37.7 盛阳漫步（武术队 Steps of the Sun）</h2>
<ul>
<li>改用扇子作战 + 大成功 = 玩家瞬间"卧槽"时刻</li>
<li><b>赤凰斗扇给敌方</b>：装备未列，GM 决定</li>
<li><b>魔加鲁援助</b>：让他们给玩家重新专注（第 5 日援助派方案）</li>
</ul>
<h2>§37.8 寒霜之吼（北欧主题 Winter's Roar）</h2>
<ul>
<li>维京 / 巫女 / 盾女 / 拳手主题</li>
<li><b>数据卡不存在</b> —— 基础格斗者不适合，GM 自做</li>
<li><b>后期出现</b>：魔加鲁连战时帮玩家对付林诺姆</li>
</ul>
<h2>§37.9 八臂调和（资格赛队 Arms of Balance）</h2>
<p>资格赛队伍；第二本出现时已是 14 级（会过强但仍在）。</p>
<h3>成员名册</h3>
<ul>
<li>兰雅·什瓦纳特丝（Ranya Shibhatesh）</li>
<li>吉瓦蒂·罗瓦特（Jivati Rovat）</li>
<li>乌斯瓦利（Usvani）</li>
<li>普拉万·马吉纳普蒂（Pravan Majinapti）</li>
</ul>
<h2>§37.10 灰日之下狂舞者（Under the Pale Sun Dervishes）</h2>
<ul>
<li>仅 Foundry 怪物图鉴出现：使用武器大师数据卡的"灰日之下"流派</li>
<li>以"舞动剑刃"为主题的资格队</li>
</ul>
<h2>§37.11 其他资格队</h2>
<ul>
<li>苍穹追星（Spectacular Skyseekers）</li>
<li>永恒探询（Eternal Inquirers）/ 伊瑞瑟（Erethel）</li>
<li>黑雪绒（Black Edelweiss）/ 哈特科普（Hatkop）</li>
<li>黄铜矮人（Brass Dwarves）/ 贝尔丹（Beldam）</li>
<li>绞刑队（Gallowed）/ 棺木莉希娅（Coffin Lyssia）</li>
<li>深渊吞噬者（Devourers from Below）</li>
<li>蜀之英杰（Champion of Shu）/ 四魂将（Yotsubatari）</li>
<li>阿姆扎双胞胎</li>
<li>云间小丑（Cloud Jesters）</li>
</ul>
<p><b>通用处理</b>：用通用格斗者模板 + 主题换皮（详见第一本「通用格斗者增强」页的数据卡修补建议）。</p>""",
        },
        {
            "slug": "book2-fill-skip-stealing",
            "name": "GM 指南：5 PC 补员、跳过重做、道具偷窃 GM GUIDE: FILL/SKIP/STEALING POLICIES",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第二本运行中的几个常见房规决策：5 PC 队伍补员、跳过/重做某些遭遇、净光守护者偷窃事件。</p>
<h2>§38 5 PC 队伍补员方案</h2>
<ul>
<li><b>Cac-Lee 加入千野最硬</b>：含 9/13/18 级现成角色卡 JSON（详见第一本「资源与工具清单」页）</li>
<li><b>OriGami 队拆分加入多队</b>：拆 3 人分到寒霜之吼 / 千野最硬 / 净光守护者</li>
<li><b>自做新成员</b>：给两个核心队各做一个 5v5 NPC</li>
<li><b>加精英模板</b>：不加人，精英化一名现有成员</li>
</ul>
<blockquote title="GM 选择"><p>OriGami 是社区主流方案，剧本完整。如果赶时间用 Cac-Lee（仅限千野最硬）。最不推荐"全队精英化"（缺乏剧情联动）。</p></blockquote>
<h2>§39 跳过 / 重做</h2>
<h3>跳过</h3>
<ul>
<li><b>扁虱群</b>（低价值，多重抗性烦琐）—— 详见第一本神庙补丁页</li>
<li><b>迷音鸟</b>（多数 GM）—— 详见第一本邦木岛地点参考页</li>
<li><b>某些资格队随机遭遇</b>（节奏调控）</li>
</ul>
<h3>重做</h3>
<ul>
<li><b>赤凰挑战赛</b>（5 个）→ 改成完整战斗（详见第三本附录「挑战赛改写」页）</li>
<li><b>光辉使者战斗 + 发条竞技场与减速带</b>：让光辉使者死得有戏剧感</li>
</ul>
<h2>§40 迷宫 / 困境法术禁用</h2>
<p>锦标赛中禁用迷宫术（Maze）、困境术（Quandary）—— 不好玩，GM 与玩家都烦。开第二本前白板加上这条房规。</p>
<h2>§65 道具偷窃（湮没光环滥用警告）</h2>
<p>净光守护者用湮没光环（Aura of Unremarkable）可在赛后偷物品而执行者察觉不到 —— <b>节制使用</b>，否则破坏玩家信任。</p>
<h3>建议规则</h3>
<ul>
<li>限制每队伍每次比赛后最多丢 1 件次要物品</li>
<li>玩家觉察 = 隐匿对抗，不要让玩家"完全无法预防"</li>
<li>道具偷窃用于建立"净光守护者无处不在"的氛围，不要变成玩家"我每天检查全部物品"的细务</li>
</ul>
<blockquote title="GM 提示"><p>这条房规存在的意义是：让 GM 知道有这个 NPC 行为，但不要滥用。1-2 次给玩家"那个戒指不见了！"的体验即可。第三本琉璃光屋决战 = 找回被偷物品的机会。</p></blockquote>""",
        },
        {
            "slug": "book2-npc-performance",
            "name": "GM 指南：NPC 演绎要点合集 GM GUIDE: NPC PERFORMANCE NOTES",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第二本主要 NPC 的演绎风格速查。</p>
<h2>§56 千野最硬演绎</h2>
<ul>
<li><b>千野</b>：恭谦义洛理信徒；少年漫主角式无视追求者</li>
<li><b>鹰虎</b>（Takatorra）：甜食党，做菜热情，给场景温度</li>
<li><b>纪幽</b>（Ji-Yook）：狐影流，狡诈式微笑</li>
<li><b>「正义的」亚彬</b>：龙系狂热者（任何人提到"龙"相关都会激动地谈论）</li>
<li><b>Cac-Lee</b>：元死后转向追随，戏剧高潮人物（彼岸花觉醒）</li>
</ul>
<p><b>建议</b>：与 PC 编排浪漫 / 识别钩子（船上预热阶段就开始）。</p>
<h2>§57 净光守护者个体演绎</h2>
<ul>
<li><b>捣达</b>：通常静默或寡言；力量型；不宣告就动手</li>
<li><b>白谷</b>（白谷真野）：静默忍者 —— 共识演绎；只用动作和眼神</li>
<li><b>蓝蝮蛇</b>：花花公子型 / 玩物丧志 / 玩弄毒液艺术</li>
<li><b>苏塔奴</b>：高傲法师 / 暗黑导师 / 腐败者</li>
</ul>
<blockquote title="整体注意"><p>RAW 中"净光守护者除'邪恶'外没什么性格"，需 GM <b>自加性格细节</b>。强烈推荐为每个成员安排"标志性台词"（如苏塔奴每次施法前都引用一句义洛理的语录的反讽版本）。</p></blockquote>
<h2>§58 多贺台·艾米综艺解说</h2>
<ul>
<li>戴海螺当麦克风</li>
<li>风格：摔角解说 + 综艺主持</li>
<li>玩家做帅动作 → 多贺台现场宣布、播给观众听</li>
<li>玩家做蠢事 → 多贺台也喷</li>
<li>效果：玩家自动"为镜头表演"</li>
</ul>
<p>详细综艺化演绎方案见 <b>第一本附录「综艺化演绎建议」</b>页。</p>
<h2>§59 「正义的」亚彬</h2>
<ul>
<li><b>龙系狂热</b>：任何人提到"龙"相关都会激动地谈论</li>
<li><b>白蛇形态</b>：第三本第二章腐化形态（见 NEW 第三本）</li>
<li>第二本玩家与他对战时，可以让他诡异地大谈龙学 —— 玩家会觉得"这家伙是不是有点神经"</li>
</ul>
<h2>§60 马菲卡·阿尤瓦里</h2>
<ul>
<li>BAMF + 名人感</li>
<li>也是《千圣力量》（Strength of Thousands）AP 中的名人</li>
<li>战斗中让他站后排（他是导师，不愿正面硬碰）</li>
<li>战败后会"风度地认输"，不像净光守护者那样阴险</li>
</ul>""",
        },
        {
            "slug": "book2-errata",
            "name": "GM 指南：第二本勘误 GM GUIDE: BOOK 2 ERRATA",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第二本《比赛开始！》全部已知勘误汇总。</p>
<h2>第二本勘误汇总</h2>
<ul>
<li><b>蓝蝮蛇梅花滂沱 + 死之泪</b>：DC 47 + 1 分钟瘫痪 = 稳定团灭 —— <b>不用死之泪</b>，详见「净光守护者队伍档案」</li>
<li><b>苏塔奴默认法术清单</b>：无效 —— 用 FotRP Expanded 或自定，详见「净光守护者队伍档案」</li>
<li><b>捣达力量 +35 + 强化擒抓</b>：100% 抓住 PC —— 旧版擒抱 / 力量 -2</li>
<li><b>马菲卡炫彩之球</b>：1 轮让队投降 —— 允许创意抵消 / 持续伤害拆球，详见「风语者队伍档案」</li>
<li><b>刺骨蔷薇双重复法术清单</b>：几乎团灭 —— FotRP Expanded 修复，详见「刺骨蔷薇队伍档案」</li>
<li><b>迷宫 / 困境法术</b>：烦 —— 禁用</li>
<li><b>5 PC 队伍补员</b>：书面只 4 人 —— OriGami / Cac-Lee / 自做</li>
<li><b>多贺台·艾米缺画像</b>：重要 NPC —— 自找 / Hero Forge</li>
<li><b>赤凰斗扇给敌方</b>：装备未列 —— GM 决定（盛阳漫步使用）</li>
</ul>
<h2>第二本主要决策点</h2>
<ul>
<li><b>① 对阵图何时揭示？</b> 第 1 日早晨完整揭示</li>
<li><b>② 综艺化吗？</b> 强烈推荐：多贺台·艾米提前到第一本（详见第一本「综艺化演绎建议」）</li>
<li><b>③ 观众影响力用什么系统？</b> FotRP Expanded 或 5 阶代币（见「锦标赛核心规则与观众影响力」页）</li>
<li><b>④ 第 5 日援助派 vs 严苛派？</b> 5 PC 援助 / 6 PC 严苛</li>
<li><b>⑤ 净光守护者作弊机制？</b> 推荐采用：执行者操控 + 噩梦法术 + 第三方挑战</li>
<li><b>⑥ 5 PC 补员？</b> OriGami / Cac-Lee / 自做（推荐 OriGami）</li>
<li><b>⑦ 道具偷窃节制？</b> 1-2 次给体验即可，不要滥用</li>
</ul>""",
        },
    ],
}
PAGES_BOOK3 = {
    "ch1": [
        {
            "slug": "book3-overview-map-issue",
            "name": "GM 指南：第三本总评与地图问题 GM GUIDE: BOOK 3 OVERVIEW & MAP CRISIS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第三本《山中之王》（King of the Mountain）的核心问题：地图严重不足 + 节奏压缩。</p>
<h2>总评</h2>
<ul>
<li><b>节奏问题</b>：第 3 章时间压缩剧烈 —— 锦标赛决赛 → 魔加鲁 → 琉璃光屋几乎一天连续</li>
<li><b>巅峰失衡</b>：19-20 级巅峰能力解锁后只剩 2 场战 = 巅峰能力使用机会几乎为零</li>
<li><b>对话稀缺</b>：书面对第 3 章对话与描述非常少 → GM 必须自填</li>
</ul>
<h2>⚠ 最大问题：地图严重不足</h2>
<p>第三本仅 <b>4 张战图</b>，决战双阶段官方建议"心中战棋" —— <b>必装 Kalnix 地图模块</b>。</p>
<h3>地图模块覆盖</h3>
<ul>
<li><b>碎裂之螺</b>（Shattered Spiral）</li>
<li><b>琉璃光屋</b>（Glass Lighthouse）</li>
<li><b>大道场</b>（Grand Dojo）—— 含古林 / 午夜湖 / 灼热沙漠 / 风袭峡谷多套主题</li>
<li><b>雕匠辛达拉第二阶段</b></li>
</ul>
<p>详细资源链接见 <b>第一本附录「资源与工具清单」</b>页。</p>
<h2>第三本核心备课检查清单</h2>
<ul>
<li>✓ 装好 Kalnix 地图模块</li>
<li>✓ 决定雕匠辛达拉铺垫方案（详见「雕匠辛达拉铺垫方案」页）</li>
<li>✓ 准备魔加鲁数据卡修补（详见「魔加鲁连战」页）</li>
<li>✓ 准备净光守护者最终战（详见「终局战斗与尾声」页）</li>
<li>✓ 给玩家 19-20 级巅峰能力使用机会（加遭遇或调时机）</li>
<li>✓ 决定郝金在决战时是否帮玩家</li>
</ul>
<blockquote title="GM 提示"><p>第三本是 AP 中最需要"GM 自己写内容"的一本。如果你跑得拘谨，玩家会感觉"剧情过得太快"。建议第三本启动前再读 §43-§54 一次，重点准备演绎细节。</p></blockquote>""",
        },
        {
            "slug": "book3-wanshou-rai-sho",
            "name": "GM 指南：顽寿调查与来肖修道院试炼 GM GUIDE: WANSHOU & RAI SHO TRIALS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第三本第 1 章顽寿调查相 + 来肖修道院 4 选 3 试炼的失败方案。</p>
<h2>§44 顽寿调查相</h2>
<h3>背景设定</h3>
<ul>
<li>城市被郝金占领 / 魔加鲁危机背景</li>
<li>玩家调查净光守护者 + 隐藏赞助人</li>
<li>智慧之鲸血脉传说与郝金过去相关</li>
</ul>
<blockquote title="⚠ 自填提醒"><p>书面对第 3 章对话 + 描述非常少 —— <b>必须 GM 自己写</b>。准备至少 5-8 个调查场景的对话脚本，否则玩家会觉得"过场太快"。</p></blockquote>
<h2>§45 来肖修道院试炼</h2>
<h3>设置</h3>
<p>修诺大师（Master Senyo / Sifu Xho Nuo，武道会使者）邀请玩家进行 4 项试炼，过 3 项即可通过。</p>
<h3>主要 NPC</h3>
<ul>
<li><b>住持祖建</b>（Zu Jian, Abbot）—— 修道院住持</li>
<li><b>真慧</b>（Zhen Hui）—— 灵魂战士，前仲裁者</li>
<li><b>立彦</b>（Liyan / Leeyan）—— 见习生</li>
<li><b>破云</b>（Break Cloud）—— 麒麟坐骑</li>
</ul>
<h3>⚠ 失败应对（书面没有失败方案）</h3>
<p><b>改写</b>：用立彦（Leeyan，训练员）的训诫不阻断进度，叙事化即可。即使玩家 4 项全失败，让立彦说"你们的态度比成败更重要" + 推动剧情前进。</p>
<blockquote title="GM 提示"><p>来肖修道院的功能是"剧情节奏点" —— 让玩家见到几个关键 NPC + 获得通往天龙仪式的资格。<b>不要让试炼失败阻断剧情</b>。如果玩家真的全失败，叙事处理（立彦帮你们辩护、住持的"内心成长比试炼成果更重要"等）即可。</p></blockquote>
<h3>试炼变体建议</h3>
<ul>
<li>把 4 项试炼之一设计为对应 PC 的背景钩（如圣战士 PC 的"信念检验"）</li>
<li>让某项试炼有"虽然失败但有亮点"的可能（如战斗输了但表演出色）</li>
<li>预留至少 1 项试炼让"不擅战斗的 PC"也能过</li>
</ul>""",
        },
    ],
    "ch2": [
        {
            "slug": "book3-soul-offering",
            "name": "GM 指南：天龙召唤仪式（牺牲点数） GM GUIDE: SOUL OFFERING POINTS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§46 天龙召唤仪式的牺牲点数机制澄清与自愿牺牲机制。</p>
<h2>牺牲点数上限</h2>
<p><b>书面对技能检定是否给点不清</b>。</p>
<h3>标准解读</h3>
<ul>
<li><b>6（初始）</b>+ <b>4（仪式动作）</b>+ <b>10（队伍声誉）</b>= <b>20 基础点</b></li>
<li>每表演检定 = <b>1 点</b></li>
<li>大成功 = <b>2 点</b></li>
</ul>
<blockquote title="GM 提示"><p>书面没有明确"是否给点"的规定。<b>多 GM 采用上述解读</b>：每个表演检定（不论是非战斗也是战斗）都计点，给玩家可触达 30-40 点的空间。20 基础 + 10-20 检定 = 30-40 范围。</p></blockquote>
<h2>自愿牺牲机制</h2>
<p>一个 PC 牺牲给召唤；后果是 <b>PC 死保 + 仪式后复活机会</b>。</p>
<h3>实操建议</h3>
<ul>
<li>让玩家明白这是"剧情死亡 + 注定复活"的设计，避免恐慌</li>
<li>牺牲后的 PC 不是真死，是处于"灵魂状态"被天龙暂时持有</li>
<li>仪式结束后由郝金或天龙本身复活（无消耗组件）</li>
<li>给 PC 在"灵魂状态"期间几个戏剧性的瞬间（如看到天龙的视角、与已故亲人短暂对话）</li>
</ul>
<h2>仪式失败应对</h2>
<p>如果点数不够（如玩家全在战斗，无暇表演）：</p>
<ul>
<li><b>方案 A</b>：让 NPC（郝金 / 立彦 / 真慧）给点数补足，但需要付出代价（如 NPC 失去某种能力）</li>
<li><b>方案 B</b>：仪式部分成功，天龙以"半魂"形态降临，给玩家的战斗加成减半</li>
<li><b>方案 C</b>：直接成功，但代价是仪式后玩家 -1 凤凰羽毛（或其他象征性损失）</li>
</ul>""",
        },
        {
            "slug": "book3-mogaru-gauntlet",
            "name": "GM 指南：魔加鲁连战（含数据卡修补） GM GUIDE: MOGARU GAUNTLET & STAT FIXES",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§47 第 5 日魔加鲁连战的数据卡修补、平民伤亡机制、郝金缺席原因。</p>
<h2>设计意图</h2>
<p>第 5 日（或第 8 日决赛后）一日内不准休息：</p>
<ul>
<li>净光守护者决赛</li>
<li>→ 3 场魔加鲁狂暴战</li>
<li>→ 拉祖追击</li>
<li>→ 转第三本</li>
</ul>
<h2>各派应对</h2>
<ul>
<li><b>援助派</b>：风语者给治疗术 / 盛阳漫步给重新专注</li>
<li><b>严苛派</b>：让玩家烧资源；让郝金牺牲式复活倒地 PC</li>
</ul>
<h2>魔加鲁数据卡调整</h2>
<ul>
<li><b>降 AC（约 -3）+ HP +250</b>：抵 action economy 影响</li>
<li><b>喷吐 DC -4 + 全伤害</b>：让玩家"环境加成"仍有用</li>
</ul>
<blockquote title="GM 提示"><p>魔加鲁是 17-18 级 Boss，原数据卡 AC 极高但 HP 偏低 → 玩家"无法稳定命中但偶尔大成功就削大半"。调整后变成"中等 AC + 高 HP" → 玩家每回合都能造成伤害但不会快速解决。</p></blockquote>
<h2>魔加鲁演绎</h2>
<ul>
<li><b>主题方向</b>：让它像哥斯拉电影</li>
<li><b>音乐</b>：原子吐息 SFX</li>
<li><b>母子重构</b>：把魔加鲁性别改写为母亲保护幼崽（更有情感张力）</li>
<li><b>哨塔预兆</b>：把鬼魂改成魔加鲁 vs 虚空公爵的视景幻象（第一本邦木岛哨塔阶段铺垫，详见第一本附录）</li>
</ul>
<h2>平民伤亡机制</h2>
<p>玩家可用 <b>欺骗 / 表演检定</b> 分散魔加鲁注意力。</p>
<ul>
<li><b>失败 = 魔加鲁尾扫击中平民死亡</b>，加情感重量</li>
<li>大失败 = 多个平民死亡 + 玩家见证</li>
<li>成功 = 平民撤离，但玩家自身受魔加鲁注意</li>
</ul>
<blockquote title="戏剧建议"><p>不要让所有平民都得救。让玩家见证 1-2 次"我救不了所有人"的瞬间 —— 这是 AP 最沉重的桥段之一，但不要全部悲剧化（玩家需要至少 80% 的成功体验，否则心态崩）。</p></blockquote>
<h2>郝金为何不去打魔加鲁</h2>
<p><b>书面</b>：郝金追辛达拉进半位面，没去打魔加鲁。</p>
<h3>解读</h3>
<ul>
<li>(a) <b>阻止辛达拉 = 阻止魔加鲁</b>（多数 GM 选择）</li>
<li>(b) 半位面只暂时通</li>
<li>(c) 单挑怪兽也不行（郝金不是战士型）</li>
</ul>
<p>建议在玩家问"郝金呢？"时，让其他 NPC（如多贺台·艾米 / 立彦）解释 (a)：阻止背后的雕匠就是阻止魔加鲁本身。</p>""",
        },
    ],
    "ch3": [
        {
            "slug": "book3-syndara-foreshadow",
            "name": "GM 指南：雕匠辛达拉铺垫方案（8 种） GM GUIDE: SYNDARA FORESHADOWING (8 OPTIONS)",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§48 雕匠辛达拉缺乏前置铺垫问题（最大设计 bug）的 8 种铺垫方案。<b>必须从第一本开始铺</b>。</p>
<h2>⚠ 辛达拉缺乏前置铺垫（最大设计 bug）</h2>
<p>雕匠辛达拉（Syndara the Sculptor）在第一、二本几乎完全没出场，第三本直接登场当反派。多数玩家会觉得"这反派从哪冒出来的"。</p>
<h2>8 种铺垫方案</h2>
<h3>① 最简方案：保密到底</h3>
<p>让玩家根本无法得知雕匠辛达拉名字 —— 让最终揭示是震撼。适合不愿意 retro-fit 第一本的桌子。</p>
<h3>② 化名赞助人</h3>
<p>辛达拉化名作为锦标赛赞助人（第二本登场）。玩家以为他/她是友军，第三本揭示是大反派。<b>戏剧张力极强</b>。</p>
<h3>③ 第二本传送门场景：玩家瞥见"shadowy figure"</h3>
<p>在某场比赛或事件中让玩家瞥见 shadowy figure（可能过于明显）。</p>
<h3>④ 修道院 PC 闪回</h3>
<p>用气脉时给"辛达拉过去"的闪回（如来肖修道院真慧的灵能片段）。</p>
<h3>⑤ Hwanggot 旅程（如有教士 PC）</h3>
<p>一名戴着 탈짜（taljjae）面具的人自我介绍："我是雕匠辛达拉"。</p>
<h3>⑥ 郝金自述</h3>
<p>郝金对玩家说："他是我数百年前共事的杰出艺术家，我亲手杀了他" —— <b>保留她"辛达拉已死"的信念</b>。这让第三本"辛达拉其实没死，被困琉璃光屋几个世纪"的揭示更震撼。</p>
<h3>⑦ 第二本引入：直接把辛达拉作为赞助人重写情节</h3>
<p>类似 ②，但更激进 —— 把辛达拉作为玩家可争取的赞助人之一（化名）。</p>
<h3>⑧ 玩家背景钩</h3>
<p>让一个 PC 的"消失的父亲/弟弟"是辛达拉抓走的挂毯受害者。这给 PC 提供个人复仇动机。</p>
<h2>跨本铺垫建议表</h2>
<ul>
<li><b>第一本铺什么</b>：邦木岛哨塔可瞥见 shadowy figure（方案 ③）；玩家背景钩可在开跑前就设定（方案 ⑧）</li>
<li><b>第二本铺什么</b>：灯笼小舍 + 玛莱卡·陶画廊（雕匠的旧居）；化名赞助人初露面（方案 ② / ⑦）</li>
<li><b>第三本揭示</b>：所有铺垫汇聚 —— 玩家应该有"原来如此！"的瞬间</li>
</ul>
<blockquote title="GM 提示"><p><b>组合方案 ② + ⑧ 是社区最推荐</b>：化名赞助人 + 玩家背景钩 = 既有客观伏笔又有情感纽带。如果只能选一个 → 选 ②（化名赞助人最容易实施）。</p></blockquote>
<h2>跨本提示</h2>
<p>方案 ②③⑧ 需要在第一本/第二本就执行。本卷 GM 指南是"最终揭示"，但<b>暗线必须从第一本开始</b>。详见第一本附录「三分钟速读」与「开跑前必做」页的辛达拉提醒。</p>""",
        },
        {
            "slug": "book3-syndara-motive-statblock",
            "name": "GM 指南：雕匠辛达拉动机与数据卡修补 GM GUIDE: SYNDARA MOTIVE & STATBLOCK FIXES",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。雕匠辛达拉的真名/性别/动机重写选项 + 数据卡修补与精华镜像问题。</p>
<h2>雕匠辛达拉真名 + 性别 + 动机</h2>
<ul>
<li><b>真名</b>：The Sculptor（雕匠）</li>
<li><b>性别</b>：canonically 不明确；多用"他"（习惯）</li>
<li><b>本质</b>：公理使（axiomite）或星界存在，被郝金关在琉璃光屋几个世纪</li>
</ul>
<h2>郝金失忆背景</h2>
<p>郝金在轴心城（Axis）受审，被迫"放弃自己制作[挂毯]的知识"，连带忘了辛达拉。</p>
<p><b>逃脱方式</b>：模糊（"神雨"过后 / 阿芙力族同谋等改写）。</p>
<h2>辛达拉动机重写（4 种选项）</h2>
<h3>① 抓物质位面建宇宙（推荐）</h3>
<p>偷郝金画笔（挂毯之源），复制其力量 —— <b>纯野心</b>，最简单实施。</p>
<h3>② 哲学伙伴/浪漫</h3>
<p>辛达拉与郝金更深关系；他的悲剧是被关 + 被遗忘。这给最终决战增加情感重量 —— 辛达拉的核心情感是"你忘了我"。</p>
<h3>③ 哲学派</h3>
<p>"宇宙因诸神纷争而危险熵增，凡人被夹在中间" —— 哲学反派，与玩家辩论世界观。</p>
<h3>④ 残忍派</h3>
<p>在半位面困怪兽逼斗，建立残忍形象 —— 适合喜欢"反派必死"基调的桌子。</p>
<blockquote title="GM 推荐"><p>动机 ② + ① 组合最常见：浪漫情感纽带 + 实际行动是抓物质位面。这样辛达拉既有"被遗忘的伴侣"的情感悲剧，又有客观的反派行为。</p></blockquote>
<h2>辛达拉数据卡修补</h2>
<ul>
<li><b>AC 51 → 48</b>；<b>HP +1500</b></li>
<li><b>维度连击 actions</b>：按书面 3 actions（Foundry 导入有时显示 2，需手动修）</li>
<li><b>精华镜像</b>：改为"自由动作 triggered only when Syndara receives harmful effects" —— 避免无穷弹射</li>
<li><b>反射机制</b>：玩家必须先聚焦反射（focused action），否则辛达拉免疫</li>
</ul>
<h2>精华镜像问题详解</h2>
<p>书面"精华镜像"（Essence Reflection）无限制 reaction：</p>
<ul>
<li>问题：每次玩家攻击辛达拉 → 反射伤害到队友 → 无穷链式反应 → 团灭</li>
<li>修补：改为<b>自由动作</b>而非反应，每轮 1 次，只对"造成伤害的法术或攻击"触发</li>
</ul>
<h2>尖晶巨兽形态</h2>
<h3>Lore</h3>
<p>如果辛达拉在败局中被迫变身，他会与半位面融合，可能被永久困住。这是 <b>戏剧结局触发点</b> —— 玩家可以选择"杀辛达拉"vs"困住辛达拉"。</p>
<h3>1 HP 复活机制</h3>
<p>书面："At some point during the fight, Syndara drops to exactly 1 HP, then stands back up"。<b>戏剧高潮，不要漏</b>。</p>
<h3>致命点</h3>
<p>持续伤害比直接伤害更可怕 —— 持续伤害能击杀这种形态。玩家应该领悟到"我们要用毒/焰/酸的持续伤害打"。</p>
<blockquote title="GM 提示"><p>尖晶巨兽形态的 1 HP 复活机制书面 "At some point during the fight" 模糊。建议触发点：<b>当辛达拉血量降到 25% 时</b>，戏剧化复活；或<b>当玩家以为已经赢了时</b>，立即复活制造惊愕。</p></blockquote>""",
        },
        {
            "slug": "book3-lighthouse-leviathan",
            "name": "GM 指南：琉璃光屋与尖晶巨兽形态 GM GUIDE: GLASS LIGHTHOUSE & SPINEL LEVIATHAN",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§48 琉璃光屋半位面的扩展建议 + §49 百鬼夜行等无图遭遇方案。</p>
<h2>琉璃光屋半位面</h2>
<ul>
<li><b>时间膨胀</b>：雇佣兵在数周内获得"数年战斗经验"</li>
<li><b>强弱反转</b>：郝金弱 / 辛达拉强增益 在内部生效</li>
<li><b>位面旅行阻断</b>：玩家无法逃出</li>
<li><b>概念图</b>：高玻璃宝塔被风暴包围</li>
</ul>
<h2>琉璃光屋扩展（书面太短）</h2>
<p>书面琉璃光屋内部过于简略 —— 决战前的"奔向光屋"过程几乎没有内容。建议加：</p>
<ul>
<li><b>guarded treasury</b>（守卫宝库）：含辛达拉偷藏的物品（玩家可在决战前取回被偷物品）</li>
<li><b>collapsing demiplane</b> 末段：FF 真终焉感 —— 半位面在玩家穿越时塌陷</li>
<li><b>weirdness zones</b>：展示辛达拉的创造物（变形空间、扭曲重力、时间倒流等）</li>
</ul>
<blockquote title="GM 提示"><p>琉璃光屋应该感觉像"辛达拉的恐怖博物馆" —— 玩家走过时能看到他过去几个世纪困住的怪物、扭曲的空间实验、未完成的"宇宙画作"等。这给反派 backstory + 视觉冲击。</p></blockquote>
<h2>§49 百鬼夜行等无图遭遇</h2>
<p>书面无地图 —— 户外庭院 / 心中战棋。</p>
<h3>修补方案</h3>
<ul>
<li><b>用 Kalnix 模块的通用大场景</b>（最推荐）</li>
<li>r/battlemaps 搜"yokai night parade"</li>
<li>"心中战棋"模式：用 Foundry 的 fog of war + 大空白地图 + 描述化叙事</li>
</ul>
<h2>尖晶巨兽形态详细机制</h2>
<h3>Lore 整合</h3>
<p>尖晶巨兽（Spinel Leviathan）是辛达拉与半位面融合的最终形态：</p>
<ul>
<li>形态变化触发：辛达拉数据卡 HP 归零 → 自动变形（书面写在 statblock 末段）</li>
<li>战斗等级：Lv 22-24（玩家此时 19-20 级，差距大但有郝金的"宝库奖品"援助）</li>
<li>视觉：水晶巨兽，半位面化的玻璃肌肤，像极尖晶石</li>
</ul>
<h3>1 HP 复活机制（不要漏）</h3>
<p>书面："At some point during the fight, Syndara drops to exactly 1 HP, then stands back up"。这是 <b>戏剧高潮</b>。</p>
<h3>玩家如何处理</h3>
<ul>
<li><b>持续伤害是关键</b>（毒/焰/酸的 D 系列）—— 直接伤害无法稳定击杀</li>
<li>巅峰能力解锁（19-20 级）= 玩家最强招式应在此发动</li>
<li>郝金的宝库奖品（详见「终局战斗与尾声」页）= 关键装备</li>
</ul>
<h2>跨本提示：辛达拉铺垫</h2>
<p>如果按 §48 的 8 种铺垫方案铺好（详见「雕匠辛达拉铺垫方案」页），玩家在琉璃光屋遭遇辛达拉时应该有"啊原来是你"的瞬间。如果没铺垫 → 突兀。</p>""",
        },
        {
            "slug": "book3-final-battles-finale",
            "name": "GM 指南：终局战斗与尾声 GM GUIDE: FINAL BATTLES & EPILOGUE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§51 千野二战（山中之王）+ §52 净光守护者最终战 + §53 终局尾声。</p>
<h2>§51 山中之王（千野二战）</h2>
<p>千野最硬第三本平台战。</p>
<h3>平台战问题</h3>
<ul>
<li>忍者无运动（Athletics）→ 跳不上去</li>
<li>书面平台过高 → 玩家持续掉坑</li>
</ul>
<h3>改写</h3>
<p><b>任天堂大乱斗风格</b> —— "回旋扔（Whirling Throw）把敌人扔进尖刺坑"。让玩家用战术机动而非纯爬升。</p>
<h3>千野最硬腐化形态</h3>
<ul>
<li><b>千野（恶鬼形态 Demon Form）</b></li>
<li><b>鹰虎（大天狗形态 Great Tengu Form）</b></li>
<li><b>纪幽（九尾狐形态 Nine-Tailed Fox Form）</b></li>
<li><b>「正义的」亚彬（白蛇形态 White Snake Form）</b></li>
</ul>
<p>这场比拼最重要的演绎要点：让玩家看到"曾经的朋友被腐化" —— 第二本与他们建立的情感纽带，在这里成为情感重量。</p>
<h2>§52 净光守护者最终最终战</h2>
<ul>
<li><b>全员精英状态</b>（每个成员都精英化）</li>
<li><b>极限案例</b>：用咆哮喝彩卷轴双补倒地 PC（社区评"略显反趣味"，但 RAW 允许）</li>
<li><b>郝金复活机制</b>：在魔加鲁咬之前用郝金复活倒地 PC</li>
</ul>
<h3>战术参考</h3>
<p>详细净光守护者战术见 <b>第二本附录「净光守护者队伍档案」</b>页的 Turn 1 / Turn 2+ 战术表。最终战版本：每个成员的法术清单 +1 阶强化。</p>
<h2>§53 终局 + 尾声</h2>
<h3>郝金宝库奖励</h3>
<p>玩家从郝金宝库选物（书面奖励）。可选物品：</p>
<ul>
<li>破天弓（Sky-Piercing Bow）</li>
<li>赤凰斗扇（Phoenix Fighting Fan）</li>
<li>邦木易石（Bonmuan Swapping Stone）</li>
<li>御日舟（Solar Jian）</li>
<li>枉僧之怒（Wronged Monk's Wrath）专长</li>
<li>其他书面列出的传说物品</li>
</ul>
<h3>后续尾声</h3>
<p>多由 GM 自加。建议元素：</p>
<ul>
<li>千野最硬复活（如果牺牲了）</li>
<li>过峡城邦的重建</li>
<li>玩家成为传奇人物</li>
<li>开放后续 AP 钩子（如国王之诺 Kingmaker）</li>
</ul>
<h3>接续国王之诺</h3>
<p>玩家变过峡城邦统治者 —— 详见「后续 AP 与决策速查」页。</p>
<blockquote title="GM 提示"><p>第三本终局是 AP 高潮，应该用 1-2 场专门的"尾声 session"。不要把"赢决战"当成 AP 终结 —— 给玩家时间消化、与 NPC 道别、安排后续命运。</p></blockquote>""",
        },
    ],
    "bm": [
        {
            "slug": "book3-challenges-skill-replace",
            "name": "GM 指南：挑战赛改写与技能挑战替换 GM GUIDE: PHOENIX CHALLENGES & SKILL REPLACEMENTS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§18 赤凰挑战赛改写 + §68 技能挑战替换战斗方案。</p>
<h2>§18 赤凰挑战赛改写</h2>
<p>第三本（部分在第二本末段）郝金 5 个"宝库挑战"。<b>RAW 是技能挑战</b>（PC 1-2 动作就能解）。</p>
<h3>社区共识：改成完整战斗</h3>
<p>挑战赛书面太短，玩家几乎没参与感。改写为完整战斗后：</p>
<ul>
<li>每个挑战 = 一场 30-60 分钟的战斗</li>
<li>主题战斗：用挑战主题做战斗背景（如"火焰挑战" = 火焰元素遭遇）</li>
<li>玩家有真实的策略选择空间</li>
</ul>
<h2>§68 技能挑战替换战斗</h2>
<p>反过来，某些书面战斗也可以改成技能挑战：</p>
<h3>赤凰挑战赛 5 个 → 完整战斗</h3>
<p>详上节。</p>
<h3>凝固时刻陷阱 → VP 技能挑战</h3>
<p>书面凝固时刻（Frozen Moment）陷阱可以改成"凝固时刻技能挑战"：每个玩家投技能检定（任意），凑够 VP 即可解锁。比单一陷阱救援检定更有团队感。</p>
<h3>恐龙竞速 → 用《湮灭之墓》龙骑赛子系统改造</h3>
<p>恐龙竞速（Dinosaur Race / Drake Race）书面是简化追逐。<b>改写</b>：用《湮灭之墓》（Tomb of Annihilation）龙骑赛子系统 —— 多回合追逐 + 障碍物 + 角色技能交互。</p>
<blockquote title="GM 决策"><p>"挑战赛 → 战斗" 是必做的修补（社区共识）。"陷阱 → 技能挑战" 和 "恐龙竞速 → 改造" 是可选 —— 看 GM 时间和玩家偏好。如果赶时间，跑书面版即可。</p></blockquote>""",
        },
        {
            "slug": "book3-haojin-performance",
            "name": "GM 指南：郝金演绎完整版 GM GUIDE: HAO JIN COMPLETE PERFORMANCE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§50 + §55 郝金（Hao Jin）演绎完整版：性格选项、数据卡问题、决战时是否帮玩家、风格速查。</p>
<h2>性格选择（5 种派别）</h2>
<ul>
<li><b>自然之力</b>："喜欢看戏剧和动作"</li>
<li><b>综艺主持</b>：戏剧化锦标赛宣告</li>
<li><b>冷漠导师</b>：神秘强大但疏离</li>
<li><b>混乱大女主</b>：强大、奇思妙想、任性决策</li>
<li><b>无性恋（canon）</b>：影响"辛达拉浪漫"诠释</li>
</ul>
<blockquote title="GM 推荐"><p>"综艺主持 + 混乱大女主"组合最常见 —— 让郝金成为"既强大又有童趣"的角色，与玩家形成"老师/朋友/无理取闹的姐姐"的复合关系。无性恋 canon 影响第三本辛达拉关系的诠释（哲学伙伴而非情人）。</p></blockquote>
<h2>没数据卡</h2>
<p>Paizo 从未发布郝金数据卡。<b>GM 通用方案</b>：「她的法术列表就是：随便用」（任何法术她都能用）。</p>
<p>这是设计意图 —— 郝金是"超越数据卡的存在"，跑团中不应该把她当成可战斗的 NPC。她的功能是 <b>剧情助手 + 复活机器</b>，不是战斗单位。</p>
<h2>决战时郝金是否帮玩家</h2>
<ul>
<li><b>书面</b>：不帮，被辛达拉困</li>
<li><b>改写</b>：让郝金召唤宝库的传说物给每个玩家</li>
</ul>
<h3>改写建议（推荐采用）</h3>
<p>决战前郝金被困，但她能 <b>通过传送门给玩家递送宝库物品</b>：</p>
<ul>
<li>每个 PC 获得 1 件传说级物品（按 PC 职业匹配）</li>
<li>战胜后郝金随意复活倒地 PC</li>
<li>按宝库挑选物给奖（详见「终局战斗与尾声」页的奖励清单）</li>
</ul>
<h2>§55 郝金风格速查</h2>
<ul>
<li><b>喝椰子（带伞）</b> / 童趣 / 综艺主持感</li>
<li>"她不太 care 比赛结果，主要 care 戏剧"</li>
<li>玩家做"过分"事时把人变鸡</li>
<li>复活 PC 不当回事（第 5 日+）</li>
<li><b>浪漫向</b>：无性恋 canon —— 与辛达拉关系"哲学伙伴"而非情人（多 GM 选择）</li>
</ul>
<h2>第一本 vs 第二本 vs 第三本</h2>
<ul>
<li><b>第一本</b>：初露面，喝椰子，做开赛仪式（详见第一本「第 3 章收尾」页）</li>
<li><b>第二本</b>：作为"郝金天裁" Grand Judge 主持锦标赛；第 5 日决赛后复活 PC；魔加鲁出现时追辛达拉离场</li>
<li><b>第三本</b>：被辛达拉困在琉璃光屋；通过传送门援助玩家；终局战后回归</li>
</ul>""",
        },
        {
            "slug": "book3-villain-performance",
            "name": "GM 指南：魔加鲁与辛达拉演绎 GM GUIDE: MOGARU & SYNDARA PERFORMANCE",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§61 + §62 第三本两大反派的演绎要点。</p>
<h2>§61 魔加鲁演绎</h2>
<ul>
<li>怪兽 = 角色还是自然之力？由 GM 选</li>
<li>郝金与魔加鲁间像母子关系（性别改写后）</li>
<li>别忘原子吐息 SFX</li>
</ul>
<h3>"自然之力"派</h3>
<p>把魔加鲁当成纯粹的灾难 —— 不可沟通、不可推理。玩家只能逃跑或对抗。情感重量来自"无可奈何"。</p>
<h3>"角色"派（推荐）</h3>
<p>魔加鲁有目的（保护幼崽 / 寻找母亲 / 复仇） —— 玩家可以通过表演/欺骗检定影响她的行动。更有戏剧张力。</p>
<h3>母子重构（推荐）</h3>
<p>把魔加鲁性别改写为"母亲保护幼崽" —— 让玩家在战斗中发现幼崽，从而产生"我们到底该不该杀她"的道德困境。</p>
<blockquote title="GM 提示"><p>哥斯拉式怪兽演绎的关键：让玩家既怕她又敬畏她。她不是单纯的怪物，是"被毁掉的母亲" / "被流放的神"。原子吐息 SFX 在背景循环 = 持续提醒玩家这是"灾难片场景"。</p></blockquote>
<h2>§62 雕匠辛达拉演绎</h2>
<ul>
<li><b>哲学派恶人</b>："I'm right, the gods are entropic"</li>
<li><b>浪漫派</b>："你忘了我"是核心情感</li>
<li><b>残忍派</b>：困怪兽逼斗</li>
<li><b>自由派</b>：单纯野心家</li>
</ul>
<h3>声音建议</h3>
<p><b>低沉冷静</b>；<b>不大喊大叫</b>。即使被打中也保持冷静的"我才是对的"基调。</p>
<h3>哲学派详解（最推荐）</h3>
<p>辛达拉的核心理念："宇宙因诸神纷争而危险熵增，凡人被夹在中间。我要建立一个没有诸神的物质位面 —— 即使代价是几个文明的牺牲。"</p>
<p>这给反派一个 <b>可辩论的立场</b>。玩家可以反驳他、被他说服一部分、或干脆暴怒。这种"反派有道理"的设计，是 AP 最哲学化的瞬间。</p>
<h3>浪漫派详解</h3>
<p>辛达拉与郝金有几个世纪的关系，被关入琉璃光屋后她忘了他。最终决战时辛达拉的核心情感："我等了你几个世纪，你来了，却带着杀我的伙伴"。</p>
<p>这种诠释让最终战的情感张力极强，特别是配合 §48 铺垫方案 ⑥（郝金自述"我亲手杀了他"）使用。</p>
<h3>组合演绎（最佳）</h3>
<p>哲学派 + 浪漫派组合：辛达拉既有客观理念（反熵增）又有主观情感（被遗忘的伴侣）。这让他成为 PF2e AP 中最复杂的反派之一。</p>""",
        },
        {
            "slug": "book3-aftermath-decisions",
            "name": "GM 指南：后续 AP 与 30 决策点速查 GM GUIDE: SEQUEL AP & 30 DECISION POINTS",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。§71 后续 1-20 接龙 + 卷十一 全 AP 30 决策点速查。</p>
<h2>§71 后续 1-20 接龙</h2>
<ul>
<li><b>国王之诺</b>（Kingmaker）：玩家变过峡城邦统治者 —— 最自然衔接（玩家在第三本最终战赢得过峡领导权）</li>
<li><b>自做尾声</b>：根据桌子偏好自由发挥</li>
<li><b>重做正义之怒</b>（Wrath of the Righteous）：转传奇队伍 —— 把 20 级玩家转入传奇等级，对抗深渊领主</li>
</ul>
<h2>30 决策点速查表（全 AP）</h2>
<ol>
<li><b>开自由原型？</b> 开，但 ban 圣武士；考虑限定天夏 book + 武术家 + 龙裔</li>
<li><b>升级时机？</b> 节奏化（章节里程碑）；第 5 日前不升级</li>
<li><b>处理伤口频率？</b> 1 小时/人无限次；禁预增益（除当日）</li>
<li><b>凤凰项链怎么戴？</b> 强制戴在身上；执行者激活</li>
<li><b>第 1 日 4 名内家好手解离术怎么办？</b> 强制激活项链或预警</li>
<li><b>毒蛇藤怎么改？</b> 缩花粉半径或成功豁免后免疫</li>
<li><b>迷音鸟跑还是不跑？</b> 大多跳过或改追逐</li>
<li><b>巨兽巢穴战利品？</b> 加部位 + 1 件物品/巢（金额按章节财富自定）</li>
<li><b>5 PC 队伍补员？</b> OriGami / Cac-Lee / 自做</li>
<li><b>花之百人众怎么跑？</b> PF2e Troops Helper 模块</li>
<li><b>净光守护者梅花滂沱+死之泪？</b> 不用死之泪</li>
<li><b>苏塔奴法术？</b> 必改；用 FotRP Expanded 清单</li>
<li><b>捣达力量 +35？</b> 旧版擒抱或 -2</li>
<li><b>马菲卡炫彩之球？</b> 允许创意抵消 + 持续伤害拆球</li>
<li><b>刺骨蔷薇双重法术？</b> FotRP Expanded 改写</li>
<li><b>迷宫 / 困境法术？</b> 锦标赛禁用</li>
<li><b>对阵图何时揭示？</b> 第 1 日早晨完整揭示</li>
<li><b>综艺化吗？</b> 强烈推荐：多贺台·艾米提前到第一本</li>
<li><b>观众影响力用什么系统？</b> FotRP Expanded 或 5 阶代币</li>
<li><b>魔加鲁平衡？</b> 降 AC（约 -3）+ HP +250；喷吐 DC -4 + 全伤害</li>
<li><b>辛达拉怎么铺垫？</b> 保密 + 化名赞助人组合</li>
<li><b>辛达拉动机改写？</b> 抓物质位面建宇宙</li>
<li><b>辛达拉 AC 51 + 数据卡？</b> AC 48 + HP +1500 + 反射限自由动作</li>
<li><b>郝金解离术戏？</b> 改郝金故意挨打嘲讽</li>
<li><b>郝金数据卡？</b> "spell list is Yes"；不数据卡化</li>
<li><b>第 5 日连战援助？</b> 友队 NPC 提供治疗 / 重新专注</li>
<li><b>死亡如何处理？</b> 除"死亡效应"外都模糊处理；每队 1 次免费复活</li>
<li><b>赤凰挑战赛？</b> 改成完整战斗</li>
<li><b>自由原型难度补偿？</b> 全遭遇 +1 难度</li>
<li><b>后续接什么 AP？</b> 国王之诺或自做尾声</li>
</ol>
<blockquote title="GM 提示"><p>这 30 个决策点是跑团前必须想清楚的清单。如果你拿到 AP 还不确定怎么处理这些点，先读对应的 GM 指南页（散布在三本附录中），再做决定。把决策落到桌前白板，开第一场前与玩家对齐。</p></blockquote>""",
        },
        {
            "slug": "book3-errata",
            "name": "GM 指南：第三本勘误 GM GUIDE: BOOK 3 ERRATA",
            "content": """<p><b>本页为外部 GM 指南整合内容</b>。第三本《山中之王》全部已知勘误汇总。</p>
<h2>第三本勘误汇总</h2>
<ul>
<li><b>全书 4 张地图</b>：严重不足 —— <b>Kalnix 模块必装</b>，详见「第三本总评与地图问题」页</li>
<li><b>延续魔棒</b>：无法术名 —— 用最高阶解离术 / 飞弹术</li>
<li><b>来肖试炼 4 选 3</b>：失败无方案 —— 立彦训诫不阻断进度，详见「顽寿调查与来肖修道院试炼」页</li>
<li><b>天龙仪式牺牲点数</b>：上限模糊 —— 6+4+10 基础 + 表演 1/2 点，详见「天龙召唤仪式」页</li>
<li><b>魔加鲁 AC</b>：action 经济难度太高 —— -3 AC + HP +250，详见「魔加鲁连战」页</li>
<li><b>魔加鲁喷吐 DC</b>：太致命 —— DC -4 + 全伤害</li>
<li><b>雕匠辛达拉缺乏铺垫</b>：RAW 大问题 —— 8 种铺垫方案任选，详见「雕匠辛达拉铺垫方案」页</li>
<li><b>辛达拉 AC 51</b>：玻璃大炮 —— AC 48 + HP +1500</li>
<li><b>精华镜像</b>：无穷弹射 —— 自由动作 triggered by harm，详见「雕匠辛达拉动机与数据卡修补」页</li>
<li><b>尖晶巨兽 1 HP 复活机制</b>：RAW 不清何时触发 —— 戏剧高潮时 GM 决定（建议 25% HP 或玩家自信时刻）</li>
<li><b>郝金解离术戏</b>：20 级不可信 —— 改为故意挨打嘲讽</li>
<li><b>千野平台战忍者无运动</b>：跳不上 —— 改图 / 给运动增益，详见「终局战斗与尾声」页</li>
<li><b>百鬼夜行无图</b>：必须自找 —— Kalnix / 心中战棋</li>
<li><b>郝金数据卡</b>：不存在 —— "spell list is Yes"，详见「郝金演绎完整版」页</li>
<li><b>巅峰能力解锁后只剩 2 战</b>：无意义 —— 加遭遇或调时机</li>
</ul>
<h2>第三本主要决策点</h2>
<ul>
<li><b>① 辛达拉怎么铺垫？</b> 保密 + 化名赞助人组合（方案 ② + ⑧）</li>
<li><b>② 辛达拉动机？</b> 抓物质位面建宇宙 + 浪漫派组合</li>
<li><b>③ 魔加鲁演绎？</b> "母亲保护幼崽"重构 + 哥斯拉式镜头</li>
<li><b>④ 郝金性格？</b> 综艺主持 + 混乱大女主</li>
<li><b>⑤ 决战援助？</b> 让郝金通过传送门送玩家传说物品</li>
<li><b>⑥ 来肖试炼失败？</b> 立彦训诫不阻断进度</li>
<li><b>⑦ 牺牲点数？</b> 6+4+10 基础 + 表演每检定 1/大 2 点</li>
<li><b>⑧ 千野平台战？</b> 任天堂大乱斗风格 / 给运动增益</li>
<li><b>⑨ 1 HP 复活触发点？</b> 25% HP 或玩家自信时刻</li>
<li><b>⑩ 后续 AP？</b> 国王之诺（玩家变过峡统治者）</li>
</ul>""",
        },
    ],
}

BOOK_PAGES = {
    "book1": PAGES_BOOK1,
    "book2": PAGES_BOOK2,
    "book3": PAGES_BOOK3,
}

# ---------- 主流程 ----------
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

def baseline_hashes(data: dict) -> dict:
    """计算每个 page 的 text.content SHA256，用于不变性校验。"""
    return {p["_id"]: hashlib.sha256(p["text"]["content"].encode("utf-8")).hexdigest()
            for p in data["pages"]}

def verify_immutability(orig: dict, new_data: dict) -> list:
    """确认所有原有 page 的 text.content 都没改。"""
    errors = []
    orig_hashes = baseline_hashes(orig)
    new_hashes = {p["_id"]: hashlib.sha256(p["text"]["content"].encode("utf-8")).hexdigest()
                  for p in new_data["pages"]}
    for pid, h in orig_hashes.items():
        if pid not in new_hashes:
            errors.append(f"orig page {pid} missing in new")
        elif new_hashes[pid] != h:
            errors.append(f"orig page {pid} content changed")
    return errors

def scan_all_ids() -> set:
    """扫所有 3 本书的全部 page _id，用于跨书碰撞检测。"""
    ids = set()
    for book_dict in BOOK_FILES.values():
        for path in book_dict.values():
            d = load_json(path)
            for p in d["pages"]:
                ids.add(p["_id"])
    return ids

def integrate(book: str, mode: str) -> dict:
    """mode: 'dry_run' | 'apply'; book: 'book1' | 'book2' | 'book3'"""
    existing_ids = scan_all_ids()
    ts_ms = int(time.time() * 1000)
    summary = {"files": {}, "errors": [], "id_skipped": [], "uuid_errors": [], "html_errors": []}

    files_dict = BOOK_FILES[book]
    pages_dict = BOOK_PAGES.get(book, {})

    for file_key, path in files_dict.items():
        orig = load_json(path)
        new_data = copy.deepcopy(orig)
        added_pages = []

        for page_def in pages_dict.get(file_key, []):
            page_id = make_id(book, page_def["slug"])
            if page_id in existing_ids:
                # 已存在 (上次 apply 留下) — 静默跳过，允许增量运行
                summary["id_skipped"].append(f"{file_key}/{page_def['slug']}")
                continue
            existing_ids.add(page_id)

            # 校验 HTML
            html_errors = validate_html(page_def["content"])
            if html_errors:
                summary["html_errors"].append(f"{file_key}/{page_def['slug']}: {html_errors}")

            # 校验 @UUID
            uuid_errors = validate_uuids(page_def["content"])
            if uuid_errors:
                summary["uuid_errors"].append(f"{file_key}/{page_def['slug']}: {uuid_errors}")

            page = build_page(page_def["name"], page_def["content"], page_id, ts_ms)
            new_data["pages"].append(page)
            added_pages.append({"name": page_def["name"], "id": page_id, "chars": len(page_def["content"])})

        # 不变性校验
        immut_errors = verify_immutability(orig, new_data)
        if immut_errors:
            summary["errors"].extend([f"{file_key}: {e}" for e in immut_errors])

        # JSON roundtrip
        try:
            json.loads(json.dumps(new_data, ensure_ascii=False))
        except Exception as e:
            summary["errors"].append(f"{file_key}: JSON roundtrip failed: {e}")

        summary["files"][file_key] = {
            "path": str(path),
            "orig_pages": len(orig["pages"]),
            "new_pages": len(new_data["pages"]),
            "added": added_pages,
        }

        if mode == "apply" and not summary["errors"] and not summary["html_errors"]:
            dump_json(path, new_data)

    return summary

def write_preview(book: str, summary: dict, out_path: Path) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append(f"FotRP GM 指南整合报告 - {book}")
    lines.append("=" * 80)
    lines.append("")
    for fk, info in summary["files"].items():
        lines.append(f"## {fk}: {info['path']}")
        lines.append(f"  原 page 数: {info['orig_pages']}")
        lines.append(f"  新 page 数: {info['new_pages']}")
        lines.append(f"  新增 page:")
        for ap in info["added"]:
            lines.append(f"    [{ap['id']}] {ap['name']}  ({ap['chars']} 字符)")
        lines.append("")

    lines.append("## 校验结果")
    for k in ("errors", "id_skipped", "uuid_errors", "html_errors"):
        if summary[k]:
            lines.append(f"  [!] {k}:")
            for e in summary[k]:
                lines.append(f"    - {e}")
        else:
            lines.append(f"  [OK] {k}: 无")

    lines.append("")
    lines.append("=" * 80)
    lines.append("各 page HTML 内容预览（前 500 字符）")
    lines.append("=" * 80)
    pages_dict = BOOK_PAGES.get(book, {})
    for fk in ("ch1", "ch2", "ch3", "bm"):
        for pdef in pages_dict.get(fk, []):
            lines.append("")
            lines.append("-" * 80)
            lines.append(f"[{fk}] {pdef['name']}")
            lines.append("-" * 80)
            lines.append(pdef["content"][:500] + ("..." if len(pdef["content"]) > 500 else ""))

    out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("book", choices=["book1", "book2", "book3"])
    parser.add_argument("mode", choices=["dry_run", "apply"])
    args = parser.parse_args()

    summary = integrate(args.book, args.mode)
    preview_path = QA / f"gmguide_{args.mode}_{args.book}.txt"
    write_preview(args.book, summary, preview_path)

    print(f"Mode: {args.mode}")
    print(f"Preview: {preview_path}")
    for fk, info in summary["files"].items():
        added_count = len(info["added"])
        print(f"  {fk}: {info['orig_pages']} -> {info['new_pages']} (+{added_count})")

    total_skipped = len(summary["id_skipped"])
    if total_skipped:
        print(f"[i] {total_skipped} pages skipped (already exist)")
    total_errors = sum(len(summary[k]) for k in ("errors", "uuid_errors", "html_errors"))
    if total_errors:
        print(f"[!] {total_errors} errors/warnings - see preview")
        for k in ("errors", "uuid_errors", "html_errors"):
            for e in summary[k]:
                print(f"  [{k}] {e}")
        sys.exit(1)
    else:
        print("[OK] All checks passed")

if __name__ == "__main__":
    main()
