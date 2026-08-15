# -*- coding: utf-8 -*-
"""Build the "中文与英文不符 / 凭空增删" batch for ember.crucible-adventure.json
and its dnd5e twin ember.adventure.json.

Long leaves (Redrak Fields.text, The Challenge Begins.summary) are edited by
exact substring replacement on the CURRENT Chinese so that every byte we did not
intend to touch stays untouched.
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件"
OUT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/4-临时脚本/2026-08-12-fix/batches"
P = "Ember Early Access.journals."

FULL = {
 # ---------- 1.5 blocker ① ----------
 P + "The Expedition Challenge.pages.Closing Ceremonies.overview":
 "<p>队伍在阿加瑟罗斯之室一行之后登上万寓堡，在那里遇到公会会长阿尔科斯·萨林兰德，并领受他的嘉许。阿尔科斯向每名角色颁发一枚阿纳克瑞纽姆徽章——公会成员身份的象征——并提醒他们如今已可以进入绛华档案库，在那里研读阿纳克瑞纽姆的一些珍藏遗物。与此同时，费尔尼斯·奥萨也在场，急切地想与队伍谈谈接下来将要继续的调查。</p>",

 P + "The Expedition Challenge.pages.Closing Ceremonies.summary":
 "<p>由于我们完成了远征挑战，我们已被授予阿纳克瑞纽姆的正式成员身份。费尔尼斯·奥萨对我们的战绩印象深刻，并急切地想要继续调查她已故的导师——公会会长维隆·长矛之死。</p>",

 # ---------- 1.5 blocker ② ----------
 P + "Glitter in the Dark.pages.A Troubled Tradeway.exposition":
 "<p>当你朝阿克图瑞尔的主干道行进时，这座天坑之城的景象令人叹为观止。无数居所悬于高处，又有一部分嵌建在这处巨大地质构造的侧壁之中，它们闪烁的灯火令你惊叹不已。阿克图瑞尔如同一颗无人得见的宝石般潜伏着，光华熠熠却浑然不觉，宛如封存于琥珀之中，坐落在黑暗之上的余烬地表。</p>",

 P + "Glitter in the Dark.pages.A Troubled Tradeway.summary":
 "<p>我们在聚归馆见到了佐迪·特拉斯克；那是阿克图瑞尔的一家酒馆，以热情接待冒险者和寻宝者而闻名。特拉斯克向我们提供了一份工作，报酬为 200 gp：追查那个被指控谋杀了一名阿克图瑞尔市民的失踪构装体，并在必要时将其摧毁。</p>",

 # ---------- 第 3 节：农场设定被改写 ----------
 P + "Arctus Plateau Gazetteer.pages.Redrak Fields.overview":
 "<p>雷德拉克原野是一片生机勃勃的农业腹地，为奥尔丹提供谷物与基本的粮食供应。</p>",

 # ---------- The Expedition Challenge（整卷照旧版英文写） ----------
 P + "The Expedition Challenge.pages.Lightning Strikes Twice.overview":
 "<p>队伍穿行于奥尔丹街头时，遇上了雷纹刃——一支敌对的冒险者队伍；他们不仅是阿纳克瑞纽姆的见习成员，也是远征挑战满怀期待的参赛者。无论角色们此前是否在穿越石之森林的旅途中见过雷纹刃，这次适时的会面都能让队伍对即将到来的比赛有一些基本了解，并让他们与这支对手队伍的关系（无论是好是坏）更进一步。</p>",

 P + "The Expedition Challenge.pages.Lightning Strikes Twice.exposition":
 "<p>繁忙的奥尔丹城在你眼前展开，众多身影在熙熙攘攘的街道上来往穿行。在人群之中，你看见四个引人注目的身影坐在一座大喷泉的边缘。从他们的举止与装备来看，你敢断定这一行四人和你们一样，是一支冒险者队伍。</p>",

 P + "The Expedition Challenge.pages.An Auspicious Acquaintance.overview":
 "<p>队伍穿行于奥尔丹街头时，费尔尼斯·奥萨主动前来搭话；她是阿纳克瑞纽姆的知名成员，也是阿格拉班德·斯威夫特的密友。队伍的事迹已经传到了这位好奇心十足的阿什卡探险家耳中，于是她带着一个互惠互利的机会找上了角色们。她愿意为队伍提供阿纳克瑞纽姆的推荐资格——这将让队伍得以参加远征挑战——条件是队伍协助她重新取得一处名为汇流厅藏身处的地点的通行权；那里曾是一座公会厅，直到一伙凶恶的劫掠者闯了进去。</p>",

 P + "The Expedition Challenge.pages.An Auspicious Acquaintance.summary":
 "<p>我们穿行于奥尔丹街头时，遇见了费尔尼斯·奥萨——一位著名的阿纳克瑞纽姆冒险者，也是阿格拉班德·斯威夫特的好友。费尔尼斯急切地希望我们协助她执行一项任务：从一伙名为碎牙者的凶恶盗贼手中夺回她的汇流厅藏身处。作为交换，费尔尼斯会帮我们取得加入阿纳克瑞纽姆以及参加远征挑战的资格。</p>",

 P + "The Expedition Challenge.pages.An Upcoming Challenge.overview":
 "<p>队伍在探索奥尔丹时偶然来到万寓堡——阿纳克瑞纽姆的显赫总部；这个公会是本地区声名远播的寻宝古物学者组织。在这里，角色们会遇到博闻导师阿德琳·戈斯，她正忙着向一群有意参赛者讲解如何报名参加公会那场著名的远征挑战。只要队伍肯耐心等候，就能亲自与这位博闻导师谈谈这场奇特的比赛，以及自己该如何参加。</p>",

 P + "The Expedition Challenge.pages.An Upcoming Challenge.exposition":
 "<p>万寓堡名副其实。当你穿过前门、踏入入口大厅时，仅仅是这座宏伟古老建筑的体量就令你为之一震。厚重的石墙以历经数百年岁月的巨大木梁为框架。高耸的拱顶在头顶远远弯起，饰有精致的紫色挂毯，其上展示着阿纳克瑞纽姆的象征：一具带刃的罗盘。你的注意力被大厅前台处一场小小的争执吸引过去——一位年长的奥尔达尼女子正不耐烦地打发三名年轻的阿克图里安人；从他们那身探险装备来看，是冒险者。</p>",

 P + "The Expedition Challenge.pages.The Challenge Begins.overview":
 "<p>队伍来到万寓堡参加远征挑战的开幕仪式；各路冒险团体与满怀期待的阿纳克瑞纽姆候选成员都已在此齐聚。公会会长阿尔科斯·萨林兰德站在人群前致辞，给出有用的说明，以及三条将让各队踏上旅程的线索：这些谜语分别暗示了三把挑战钥匙所在的位置，而要参加堡垒下方的寻路者试炼，就必须拿到这些钥匙。</p>",

 P + "The Expedition Challenge.pages.Manufactured Trial.summary":
 "<p>我们在杰刻罗卡别墅击败了阿纳克瑞纽姆的构装体，成功取得了试炼挑战钥匙；这处遗址已被公会改作考验准成员的试炼场。</p>",

 P + "The Expedition Challenge.pages.Amazing Brambles.summary":
 "<p>我们成功穿越了碎裂峡谷深处一座险恶的荆棘迷宫，并在那里找到了迷宫挑战钥匙，它将帮助我们继续推进远征挑战。</p>",

 P + "The Expedition Challenge.pages.Hidden Campsite.overview":
 "<p>为寻找一项基石挑战，队伍在奥尔丹西北方的楔地发现了荒疏已久的克拉托尔大厅。在这里，角色们可以稍作探索，看看这座堡垒内部那唯一的房间——不过它与此地最重要的地标关系不大，那是坐落于附近的一口魔法井。这座令人费解的纪念物被称为欧比斯之井，与灵魂之月有着强烈的寰宇联系，自然也藏着取得水井挑战钥匙的秘密。</p>",

 P + "The Expedition Challenge.pages.Hidden Campsite.exposition":
 "<p>就算门槛上方的阿纳克瑞纽姆徽记还算不上铁证，阿尔科斯·萨林兰德给出的谜语也足以让你相信：这座偏远的居所多半就是他第三条基石挑战线索里提到的“冬日居所”。你瞥见附近有一处更小的地标，正从山谷凹地的阴影中透出柔和的光——那是一汪波光粼粼的水池，四周环绕着发光的石块，构成某种超自然的水井。井面闪耀着柔和的奥术光辉，你不禁猜想它可能藏着怎样的奥秘。</p>",

 P + "The Expedition Challenge.pages.Hidden Campsite.summary":
 "<p>在与欧比斯之井——克拉托尔大厅的核心地标、一座高深莫测的纪念物——几番令人生畏的交锋之后，我们成功取得了水井挑战钥匙。</p>",

 P + "The Expedition Challenge.pages.The Wayfinder's Gauntlet.overview":
 "<p>队伍带着全部三把挑战钥匙返回万寓堡，凭这些钥匙即可进入令人望而生畏的阿加瑟罗斯之室；在那里，他们必须熬过重重考验，才能击败维斯皮安九头蛇——那是疯法师阿加瑟罗斯本人亲手造就的远古怪物。在队伍进入之室之前，星辰法师埃维斯·布莱特斯通会给他们几句鼓励；而一旦进入其中，角色们就会发现，他们不仅要对付之室里的种种危害，还要面对雷纹刃这些对手的野心。</p>",

 P + "The Expedition Challenge.pages.The Wayfinder's Gauntlet.summary":
 "<p>我们在阿加瑟罗斯之室中成功击败了维斯皮安九头蛇，并因此完成了远征挑战。</p>",

 P + "The Expedition Challenge.pages.Scholar in Need.summary":
 "<p>我们遇到了一位名叫洛斯滕·瓦克斯的腼腆的阿纳克瑞纽姆学者，他正埋头研究一根由烬晶构成的古代塞迪里石柱。我们协助洛斯滕·瓦克斯完成了研究，而他也告诉我们，奥尔丹即将举办一场名为“远征挑战”的比赛。</p>",

 # ---------- 其余 journal ----------
 P + "A Brush With Death.pages.The Bard's Trail.summary":
 "<p>我们沿着德雷斯·埃雷科斯的踪迹，追进天刷镇以西的内陆腹地，在那里发现了一片杂草丛生、可能藏有线索的田地。</p>",

 P + "An Old Friend.pages.Traveling with Lyla.overview":
 "<p>当队伍穿越阿克图斯高原时，他们可能会想与莱拉·杰夫赫尔谈谈她的过去、近期发生的事件，以及/或者队伍与莱拉在旅途中遇到的那些人。</p>",

 P + "Thorny Predicaments.pages.Planting a Seed.overview":
 "<p>当队伍的去路被一大团蠕动的藤蔓挡住时，他们发现这是埃迪维尔·斯普劳特的手笔——一名荆芽灵巫师，也是一位有志成为农艺法师的人；他正试图掌握一种魔法藤蔓编织技巧，却在其中遇上了麻烦。</p>",

 P + "Thorny Predicaments.pages.Planting a Seed.summary":
 "<p>我们遇到了一位年轻的荆芽灵，埃迪维尔；他掌握农艺魔法的尝试出了点岔子。我们帮他想出了办法，解决了他弄出来的那些活动荆棘，并送他上路，前往碎裂峡谷寻找一种稀有的法术成分。</p>",

 P + "Thorny Predicaments.pages.Revelers on the Road.summary":
 "<p>我们遇到了两名正在寻找布雷文的迷路旅人。与他们交谈时，我们感到一阵轻微的震动撼动了地面，并得知这类震动在本地很常见。</p>",

 P + "To Fall and Fall Again.pages.Savage Descent.summary":
 "<p>我们发现了一条狭小逼仄、向下通往通路的路线。途中，我们遭遇了种种奇异而超凡的阻碍，它们试图拖住我们的脚步，把我们赶回余烬的地表。</p>",

 P + "To Fall and Fall Again.pages.Strand of Fate.summary":
 "<p>我们抵达了斯托尔萨河滩这座小村庄。当地冒险者阿伯林·洛德失踪，加上笼罩全镇的普遍不安，正让这里备受困扰，值得进一步调查。</p>",

 P + "To Fall and Fall Again.pages.Absolute Destruction.summary":
 "<p>我们发现了一根已被摧毁的申特石柱，它似乎是某种标记，不过其表面刻写的大部分符文都磨损得难以辨认。一位正在查看这根石柱的兰提尔圣武士遭到了怪物袭击；她认为这些怪物是从余烬地表之下的通路中冒出来的。</p>",

 P + "To Fall and Fall Again.pages.Snarled Promises.overview":
 "<p>当队伍朝着雾缚洞窟中那片被腐化区域内的原初堡垒前进时，一个阴险而咆哮的声音侵入了他们的脑海，向他们许诺巨大的力量与财富。若队伍此前曾在“月上之旅”任务中遇到米奥罗斯并站在他那一边，他们会在这里再次遇见这个身影。</p>",

 P + "To Fall and Fall Again.pages.Phantasmal Waters.overview":
 "<p>队伍在试图穿越雾缚洞窟南侧入口与原初堡垒之间的空间时，进入了通路中的一座湖泊。这将被证明是个危险的选择，因为会有长长的触须探出，把队伍成员拖入深水之中。</p>",

 P + "To Fall and Fall Again.pages.Lightless Halls.summary":
 "<p>我们进入了雾缚洞窟中原初堡垒的第一区域，在那里遭遇了可怕的阴影怪物，并解开了一扇通往更深区域的古老坚固大门。</p>",

 P + "Ancient Paths.pages.A Strange New World.exposition":
 "<p>眼前景象的宏伟令安卡里斯特一时看得出神；当他望向下方展开的宏大地下天地时，不经意流露出一丝孩童般的惊叹。</p>"
 "<blockquote><p>我以前只在书里读到过通路。我原以为它顶多是一堆杂乱无章的隧道……而不是余烬之下的一整个世界。绝不是这样的景象。</p></blockquote>"
 "<p>片刻之后，这位龙裔重新集中精神，转身对你们说道。</p>"
 "<blockquote><p>好吧……我们的下一步，是深入那片荒野，搜寻线索。就我所知，那些龙兽就是从这下面某处来的；而且——按莉耶丝特拉的说法——听起来在下面折腾的不止我们。朋友们，通路里有些古怪的事情正在发生。</p>"
 "<p>老实说：我不知道我们会在下面待多久。但我一定要查个水落石出，记住我这句话。</p></blockquote>",

 # 读给玩家的旁白整段照旧版英文写：丢掉了「门上的阿纳克瑞纽姆带刃六分仪招牌」，
 # 又凭空加出「血林的深红色枝叶」「头顶掠过的太阳或月亮」
 P + "Arctus Plateau Gazetteer.pages.Krator's Hall.exposition":
 "<p>一道深邃的峡谷在此切开山间嶙峋的岩壁，四周环绕着陡峭锋利的岩石，为下方的裂隙投下永恒的阴影。谷地凹处，一座巨大的木制建筑高耸于蔓生的枝叶之上，看上去像是某座饱经岁月的旅舍或集会厅。即便远远望去，你也能看见门槛上方有一块饱经风霜的金属招牌，上面是一个一眼就能认出的标志——阿纳克瑞纽姆的带刃六分仪。</p>",

 P + "Ancient Paths.pages.Reuniting with Ankarist.summary":
 "<p>抵达天刷镇这处偏远聚落之后，我们通过安卡里斯特的同伴莉耶丝特拉·格兰与这位盟友重新取得了联系，并了解到调查的最新进展。随后，我们前往神话尖塔天文台，在那里找到一台远古升降机，把我们送入余烬之下的地底通路深处——我们旅程的下一步正在那里等待。</p>",
}

# leaf -> list of (old, new) exact substring replacements on the CURRENT Chinese
PATCH = {
 P + "Arctus Plateau Gazetteer.pages.Redrak Fields.text": [
   ("<p>这些施法者将从@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.iWZG0gUGUL76Dd3r]{奥里亚尔湖}中抽取的泥土与淤泥和牲畜排泄物混合，并运用魔法工艺将其转化为惊人的表土，几乎能让任何作物迅速生长。</p>",
    "<p>这些乡野质朴的法师将从@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.iWZG0gUGUL76Dd3r]{奥里亚尔湖}中抽取的泥土与淤泥同牲畜的粪肥混合，再运用魔法工艺将其转化为惊人的表土，足以种出各式生机勃勃的作物与食粮。</p>"),
   ("<p>这些土壤随后被卖回给本地农民，农民再用它生产出大量粮食，为这片地区——以及奥尔丹——提供食物。这一循环在奥里亚尔湖@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.eNivhO6hTLwtZJEt]的疏浚工、雷德拉克的农艺法师以及雷德拉克原野的农民之间创造出了一种微妙的权力平衡；他们彼此依赖，并因持续的相互合作而获利。</p>",
    "<p>雷德拉克原野的农场通常由劳工议会集体所有并共同经营，议会由在当地耕作土地的所有人组成。依照长期以来的相互约定，这些劳工议会向罗特瓦克的农艺法师，以及奥里亚尔湖@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.eNivhO6hTLwtZJEt]的疏浚工提供粮食与其他必需品，以换取那种经魔法强化的表土。在一种既共生又脆弱的平衡之中，正是这种土壤带来了过剩的农业产出——各议会既借此供养农艺法师与疏浚工，也得以与奥尔丹的莉莉菲尔德家族贸易，换取雷德拉克原野本地无法取得的货物。</p>"),
   ("<p>雷德拉克原野是一片被魔法贯穿的地区，而这股魔法自北方巨大的奥里亚尔湖流出。",
    "<p>雷德拉克原野是一片被魔法贯穿的地区：这股魔法既自北方巨大的奥里亚尔湖流出，也是那自科拉倾泻而下的无尽大地魔法——这要归功于该地区最早的定居者立下的古老盟约。"),
 ],
 P + "The Expedition Challenge.pages.The Challenge Begins.summary": [
   ("<p>我们在万寓堡遇到了其他几支竞争队伍，以及奥尔丹的驻地星辰法师埃维斯·布莱特斯通；他给了我们每人一件魔法物品。之后，阿纳克瑞纽姆的公会会长阿尔科斯·萨林兰现身，并宣布：“第二次远征挑战现已开始！”我们得到了三条线索，要找到三项基石挑战并取回挑战钥匙。</p>",
    "<p>我们在万寓堡与远征挑战的其他参赛者聚在一起，见到了几位值得一提的人物，其中包括奥尔丹的驻地星辰法师埃维斯·布莱特斯通——她给了我们每人一件魔法物品，好让我们在比赛中保住性命。公会会长阿尔科斯·萨林兰德向众人致辞，并向每个人提供了三条线索，每条线索都指向一项基石挑战及其对应挑战钥匙的所在位置。</p>"),
   ("<p“命运之指”", "<p>“命运之指”"),
   ("<p>其他队伍也在搜寻这些基石挑战，因此我们必须尽快取回这些钥匙，情况颇为紧迫。</p>",
    "<p>其他队伍也在搜寻这些基石挑战，因此如果我们真想赢下这场比赛，就必须尽快取回这些挑战钥匙。</p>"),
 ],
 P + "Glitter in the Dark.pages.Surface Matters.overview": [
   ("布里姆镇", "边缘镇"),
 ],
 P + "Glitter in the Dark.pages.Surface Matters.summary": [
   ("灰烬镇", "边缘镇"),
 ],
 P + "Glitter in the Dark.pages.Scene of the Crime.exposition": [
   ("尽管这里的商业活动比阿克图瑞尔上城区慢得多", "尽管这里的商业活动比阿克图瑞尔本城慢得多"),
 ],
 P + "Glitter in the Dark.pages.Scene of the Crime.summary": [
   ("我们冒险深入阿克图瑞尔下城区的矿渊，", "我们冒险深入矿渊城区，"),
   ("那座臭名昭著的天坑底部搜寻，试着找到更多证据值，",
    "那片臭名昭著的天坑深渊底部搜寻，试着找到更多证据，"),
 ],
 # 读给玩家的旁白里整句丢失：Cherish 的自我介绍（连名字一起丢了）
 P + "Local Color.pages.Drawing Attention.exposition": [
   ("<blockquote><p>这对萨尔瓦家族的描绘可真谈不上讨喜，对吧？</p></blockquote>",
    "<blockquote><p>这对萨尔瓦家族的描绘可真谈不上讨喜，对吧？抱歉，我失礼了。我是切瑞什·艾勒里。很高兴认识你们！</p></blockquote>"),
 ],
 # 地名指错：Quarry Run 全库 34 处作「采石场奔道」，此处写成了「采石场异界之门」
 P + "A Brush With Death.pages.Locating Kel Kornan.exposition": [
   ("采石场异界之门", "采石场奔道"),
 ],
 # Traveler's Rest 全库 89 处作「旅者歇脚处」，同一页 exposition 也是；此处音译成了另一个地方
 P + "Unfinished Business.pages.Swift Healing.summary": [
   ("塔维勒歇脚处", "旅者歇脚处"),
 ],
 # 人名指错：Funar Cevher = 富纳尔·杰夫赫尔（家族名已裁决为杰夫赫尔）
 P + "Disgraced House.pages.Finding Funar.overview": [
   ("富纳尔·塞弗方向", "富纳尔·杰夫赫尔方向"),
 ],
 # 人名指错：Falar 全库 117 处作「法拉尔」
 P + "Local Color.pages.A Sketchy Situation.overview": [
   ("调查法拉——", "调查法拉尔——"),
 ],
}


def get(root, dotted):
    node = root
    for k in dotted.split('.'):
        if isinstance(node, dict):
            node = node.get(k)
        else:
            return None
    return node


def build(pack):
    cn = json.load(open(os.path.join(REPO, "compendium", "cn", pack), encoding="utf-8"))
    root = cn["entries"]
    out = dict(FULL)
    misses = []
    for path, reps in PATCH.items():
        cur = get(root, path)
        if not isinstance(cur, str):
            misses.append((path, "no current Chinese"))
            continue
        new = cur
        for old, rep in reps:
            if old not in new:
                misses.append((path, "pattern not found: " + old[:60]))
                continue
            new = new.replace(old, rep, 1)
        if new != cur:
            out[path] = new
    return out, misses


os.makedirs(OUT, exist_ok=True)
for pack, fname in (("ember.crucible-adventure.json", "mismatch.crucible-adventure.json"),
                    ("ember.adventure.json", "mismatch.adventure.json")):
    batch, misses = build(pack)
    json.dump(batch, open(os.path.join(OUT, fname), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{pack}: {len(batch)} entries -> {fname}")
    for m in misses:
        print("   MISS", m)
