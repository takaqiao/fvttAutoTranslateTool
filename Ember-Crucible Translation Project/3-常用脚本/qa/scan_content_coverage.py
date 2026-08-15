#!/usr/bin/env python3
r"""不靠长度、靠「跨语言不变量」找出中文没跟上英文的条目。

（本 docstring 用 r-string：里面引了 `@[A-Za-z]+\[` 这样的正则片段，
不加 r 会被 Python 当成非法转义警告，且下一次抄改时容易被吃掉一个反斜杠。）

⚠ **优先用 `scan_en_drift.py`。** 有旧版英文基准时，`EN_old != EN_new` 是直接证据，
比本脚本的启发式准得多。本脚本是**没有旧基准时的兜底**（例如上游第一次发版就没归档）。

  python scan_content_coverage.py --repo <repo> [--pack <name>] [--out <json>] [--top N]

为什么需要它
------------
现有三种检查各有各的盲区，而且盲区**重叠**：

* `validate_translations.py` —— 路径上有中文就算已译，看不见内容对不对。
* `measure_8c.py` / `measure_stale_extra.py` —— 只比 `<p>`/`<li>` **块数**。
  上游「换掉内容但块数不变」时它们完全沉默（实证：`Lantern Roads/Impromptu Jail`
  标着「应补 1 块」，实际整段场景被换成了完全不同的文字）。
* `scan_markup_drift.py` 的 `TRUNCATED` —— 判据是「中文纯文本 < 英文的 0.22 倍」。
  上游把英文**改写**但长度相当时，比值仍落在正常区间，它一声不响。

于是「中文是照更早的英文写的」这一类，只有在**恰好变短**或**恰好动了标记**时才会被发现。
本项目有大量译文是**移植/继承**来的（孪生包 1.4 万条整条复制自 crucible 侧、
`Arcturel Tradeway` 28 页从改名前的旧路径搬来、v1.0.15 的存量），
这些都没有针对今天的英文逐句核对过 —— 正是这个盲区的高危区。

判据：跨语言不变量
------------------
1. **数字**。英文正文里的阿拉伯数字（DC、尺数、伤害、轮数、次数）在中文里必须还在。
   中文译文照惯例保留阿拉伯数字（「DC 15」「4 英尺」「持续 3 轮」），所以这条极稳。
2. **有定译的专名**。`glossary_ec.json` 里有中文定译的英文词，若英文正文出现了它，
   中文里就应当出现对应的中文（或该专名的其它已知写法）。

两者都**先剥掉标记再比**：`@UUID[...]`、`[[/...]]`、`&Reference[...]`、HTML 标签里的
数字属于机关参数，闸门已经在管，重复计入只会制造噪声。

误报来源（读结果时要知道）
--------------------------
* 英文用英文数词（"four"、"a dozen"）而中文写「四」—— 本脚本只看阿拉伯数字，不会误判；
  反过来英文写 `4` 中文写「四」会被报出来，属**真实**的风格不一致，值得看。
* **专名检查默认关闭**（`--with-terms` 才开）。实测 `glossary_ec` 里混着大量通用词
  （`Shield→盾牌`、`Counter→反制`、`blocked→屏蔽`、`Goblin→地精语`），拿它去扫散文
  命中的几乎全是噪声：crucible 上报 284 条，逐条看没有一条是真的。数字检查则只报 14 条。
* 同一数字在英文里出现多次、中文合并成一次表述 —— 见下面「按次数比对」一节。
  原先靠「整条按集合比」一刀切规避，代价是**任何次数差都看不见**，已改为邻近折叠。

按次数比对（2026-08-12 改）
--------------------------
原先 `cn_nums = set(...)`，只问「这个数字在中文里出现过吗」，不问出现几次。
于是 `3 Talent Points` → 「2点天赋点」被同页里程碑表里**别处的 3** 完全掩盖 ——
全书最核心的成长规则错了一级，本脚本报 0。（审计 2026-08-12 第 1.1 条。）

现在改成**多重集**：英文里出现 c 次的数字，中文侧必须能拿出 c 个可接受写法。
多重集比集合敏感得多，所以原有的宽容规则一条都不能少（1–3），另补三条（4–6）。
**4–6 每一条都是拿全库跑出来的假警报倒推的，不是预防性加的** —— 不要凭想象往这里加规则，
每加一条就少看见一类真缺陷。

1. 中文数字折算（`cn_numerals_to_digits`）—— 不做会逼出「3 层矿井」「第 1 军团」这种坏中文
2. HTML 实体剥离（`ENTITY`）—— `&#x27;` 会被当成 27
3. `decade/dozen/score/century/millennium` 量词换算（`SCALED`）—— 不做会逼出「2 个十年」
4. **单/双**（新，`CN_DIGIT`）—— `2 Hands` / `Balanced 2H` 的正确中文是「双手」，
   `a single …` 是「单次」。不认这两个字会把 `Gesture: Pulse`、`Acrobat` 这类报成缺 2
5. **邻近折叠**（新，`--merge-window`，默认 60 字符）—— 同一个数字在**同一个块内**、
   60 字符内重复出现，算一份信息（`no more than 6 boons and 6 banes` →
   「恩惠骰与祸骰影响都不能超过 6 个」）。**只折英文侧的需求，不折中文侧的供给**，
   且**跨块（表格单元/列表项/段落）不折** —— 两条约束都是踩出来的，见 `demanded()`
6. **斜杠数对**（新，`SLASH_PAIR`）—— `24/7` 是英文成语，中文写「全天24小时」只留一个数；
   分母整体豁免。`1/2` → 「一半」同理

**判据的敏感度**：严格多重集（`--merge-window 0`）在两个仓库共报 9 条，
逐条核对后 8 条是上面 4–6 覆盖的正常中文、1 条是真缺陷（`Level Advancement` 的天赋点）。
加上 4–6 之后 crucible 报 1 / ember 报 0，即真缺陷一条不漏、假警报归零。

增强器里的可见文本参数（2026-08-15 第二十一轮改）
------------------------------------------------
原先 `MARKUP` 把整个 `@X[…]` 一并剥掉，理由是「方括号里是机关参数」。对 `@UUID[…]`、
`@Check[athletics|dc:17]` 这类成立，对 `@Embed[… readaloud="…"]` **不成立** ——
`readaloud=` 里塞的是**整段朗读正文**：EN 侧 48 段 / 16966 字符（最长 951 字）、
CN 侧 48 段 / 5392 字符，散在 **30 叶**（`@Embed[` 46 处 + `@embed[` 2 处，
`@[A-Za-z]+\[` 两种拼写都吃）。这 16966 字符**过去一个字都没进过本脚本的数字闸**。

同一段正文在块级通道那边也是盲的（`assert_resolutions.split_blocks` 把 `@X[…]` 涂成等长
空格），所以它是**两条通道同时看不见**的一块，不是「另有闸看着」。

现在改成：`@X[…]` 整体仍然剥掉，但把其中 `label=` / `readaloud=` 两个参数的**值**留下来
当正文。参数白名单 `ENR_TEXT_PARAMS` 与 `assert_resolutions.py` 的 `_ENR_TEXT_PARAMS`
同源、同值。

⚠ **白名单只能是这两个，不能图省事把方括号里的东西整个留下**：`@Embed[Actor.xxx]` 的
目标 id、`@Check[athletics|dc:17]` 的技能名与 DC、`count=` / `classes=` 都是照抄英文的
机器参数，计入覆盖率会造出一整片假缺口（`dc:17` 会立刻变成「中文缺 17」）。
实测全库参数名只有三种：`readaloud` 96 · `label` 90 · `classes` 8，白名单外的只有 `classes`。

留下来的值两端各补一个 `BOUND` 哨兵：朗读框与它周围的 GM 正文是**两个信息单元**，
不该让 `--merge-window` 的邻近折叠跨过这条边界（折叠只减需求，跨边界折就是掩盖）。

`--no-enricher-text` 恢复旧口径，**只为出「纳入前 / 纳入后」的可比数字**，不要日常用。

⚠⚠ **纳入 ≠ 判得到（2026-08-15 实测，务必读完再下结论）**
实测这 48 段英文朗读正文里**一个阿拉伯数字都没有**（0/48）。也就是说：纳入之后，
本脚本的**默认判据（数字多重集）对这 16966 字符的产出必然是 0 命中** —— 这个 0
是「无从查起」，不是「查过了没问题」。把它当成「查过了，干净」就是第四种空转形态
（闸在跑，但可判集合是空的）。
⚠ **「请加 `--with-terms`」这个建议是错的，第二十二轮复核实测推翻**：把那 30 叶里
最长一段所在的叶单独喂进去，**干净时**本脚本就报缺定译 39 条、把中文删空后 43 条 ——
两侧都远远非零，**没有判别力**。原因是 `--with-terms` 是**叶级**的：它拿 7601 条锚点去扫
整叶散文，命中的绝大多数与朗读框无关（这正是上面「误报来源」写的那件事）。
判别力只在**把范围收到那一段朗读正文之内**时才出现：同一批锚点、限定在段内，
48 段里 36 段命中 102 条、中文缺 0 条。

⇒ 所以这部分的闸**不在本脚本里**，在 `assert_resolutions.py` 的
   `enricher_text_coverage` 类型 / `R-readaloud-coverage` 那一条（第二十二轮新增）：
   按 (动词,目标) 把两侧增强器配对后**逐段**判「缺失 / 段级中英字符比 / 段内定译专名 / 句数」，
   现读 48 段 0 违规，删空回测 48/48 报、砍一半 46/48 报。
   本脚本这一侧保留的只是**口径**（把这 16966 字符算进覆盖率统计），不是判据 ——
   两者的分工必须说清楚，否则下一轮又会把这里的「0 条」读成「查过了」。

反空转：本脚本每次都打印「增强器可见文本：捞到 N 个参数 / 计入 X 字符 /
**其中带阿拉伯数字 K 个** / 落在 Y 叶」。N=0 打印「无从查起」；N>0 而 K=0 打印
「计入了但数字闸咬不到」。「0 条问题」有两种含义，判据有义务把
「查过了没问题」和「压根没东西可查」分开。另有 `--selftest` 做双向变异回测
（合成样本上：删掉中文朗读正文必须报；不删必须不报；旧口径下两种都不报 = 证明洞真的存在）。
真库上的变异回测在 `4-临时脚本/2026-08-15-round21/regress_readaloud.py`；
第二十二轮针对**新闸**的逐段双向回测在 `4-临时脚本/2026-08-16-round22/regress_all48.py`
（48 段逐段删空 48/48 报、砍一半 46/48 报）与 `regress_readaloud_root.py`（磁盘副本 + 主闸入口）。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

# ⚠ 本轮新加的告警行里有 U+26A0（⚠）。Windows 默认控制台是 GBK，编不出这个字符：
#   不加这一行，脚本会在**印完口径行、印出结论行之前**抛 UnicodeEncodeError 退出（实测 exit=1、
#   报告体一行不印）。qa/ 下另外 18 个脚本都带这一行，本脚本此前缺，是因为它以前不印非 GBK 字符。
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
# 块级标签＝信息单元的边界。表格单元、列表项、段落之间的同一个数字是**两份**信息
# （里程碑表的 `<td>3</td><td>3</td>` 是两行的数据），行内标签（strong/em/span/sup）不是。
# 邻近折叠只在同一个块内进行，见 `counted()`。
BLOCK_TAG = re.compile(
    r'<\s*/?\s*(?:p|div|br|hr|table|thead|tbody|tfoot|tr|td|th|caption|'
    r'ul|ol|li|dl|dt|dd|h[1-6]|section|article|aside|header|footer|'
    r'blockquote|pre|figure|figcaption|form|fieldset)\b[^>]*>', re.I)
BOUND = '\x1f'   # 块边界哨兵。与空格等长，不影响 en_len/cn_len 与 \b 词边界
MARKUP = re.compile(r'@[A-Za-z]+\[(?P<enr>[^\]]*)\]|\[\[[^\]]*\]\]'
                    r'|&(?:amp;)?[Rr]eference\[[^\]]*\]')
# 增强器里**玩家能看见的文本参数**，与 assert_resolutions.py 的 `_ENR_TEXT_PARAMS` 同源同值。
# 只认这两个：其余参数（`count=` `classes=`）和方括号里的目标 id / 技能名 / DC 都是照抄
# 英文的机器值，计入覆盖率只会造假缺口。加白名单前请先数一遍全库参数名（见模块 docstring）。
ENR_TEXT_PARAMS = ('label', 'readaloud')
# ⚠ **必须同时吃带引号与不带引号的值。** 上一版只认双引号 —— 库里
# `@Embed[Actor.LUptsqBgGJVWcg9v label=Squish]`（**无引号**，只在 EN 侧、孪生 2 叶）
# 因此永远不进纯文本，而中文侧写的是 `label="挤压"`（带引号）会进，
# 造成 EN 44 : CN 46 的不对称假象。
# ⚠ 顺带订正「全库参数名只有三种」那句 —— **不全**：还有无引号的
# `count` 166 · `rollable` 20 · `cite` 4 · `label` 2（第二十一轮全库复算）。
# 前三个是机器参数、被白名单排掉是对的，但 `label=Squish` 是**可见文本**。
ENR_PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*(?:"([^"]*)"|([^\s\]"]+))')
# 只防「数字被切成两半」，**不能**用 \w 做边界：中文里数字紧贴汉字（「其AC为17，HP为25」），
# 而汉字在 Python 正则里算 \w，`(?<![\w.])` 会把中文侧的每个数字都挡掉 ——
# 结果英文侧数字全被判成「缺失」，首版就是这么跑出 49% 命中率的假警报。
NUM = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)')

# HTML 实体里带数字：`Jeweler&#x27;s` 的 `&#x27;` 会被当成数字 27 报缺失。
ENTITY = re.compile(r'&#?\w{1,8};')

# 单 / 双：`2 Hands`→「双手」、`Balanced 2H`→「平衡双手」、`a single section`→「单 1 区段」。
# 这两个字在中文里就是一和二的量词形态，不认它们会把正确译文报成缺数字
# （实测 crucible `Acrobat` / `Gesture: Pulse`、ember `Gesture: Cone` 三条全是这一类）。
CN_DIGIT = {'零': 0, '〇': 0, '一': 1, '单': 1, '二': 2, '两': 2, '双': 2, '三': 3,
            '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
CN_NUM = re.compile(r'[零〇一二三四五六七八九十百千两单双]+')


def cn_numerals_to_digits(s: str) -> str:
    """中文数字折算成阿拉伯数字后再比对。

    **不做这一步，本检查就会逼出坏中文。** 中文侧本来就该写「三层矿井」（全库 16 处）、
    「第一军团」（1st Host）、「二十年」（2 decades）；只认阿拉伯数字的话，
    这些全都会被报成「英文里的数字中文没有」，逼着译者改成「3 层矿井」「第 1 军团」。
    是规则错了，不是译文错了 —— check_catchup.py 上一次就是这么踩的坑，同源修在这里。

    只做粗折算（够用即可）：十=10、二十=20、三层=3、第一=1。
    """
    def conv(m):
        t = m.group()
        if t == '十':
            return '10'
        if len(t) == 2 and t[0] == '十':          # 十一 .. 十九
            return str(10 + CN_DIGIT.get(t[1], 0))
        if len(t) == 2 and t[1] == '十':          # 二十 .. 九十
            return str(CN_DIGIT.get(t[0], 0) * 10)
        if len(t) == 3 and t[1] == '十':          # 二十五
            return str(CN_DIGIT.get(t[0], 0) * 10 + CN_DIGIT.get(t[2], 0))
        if len(t) == 1:
            return str(CN_DIGIT.get(t, ''))
        return t
    return CN_NUM.sub(lambda m: ' ' + conv(m) + ' ', s)


def _markup_repl(m, keep_enr: bool, stats):
    """`MARKUP` 的替换体：`@X[…]` 里的可见文本参数留下来当正文，其余一律涂成空格。

    `stats` 是可选的 `{参数名: [个数, 字符数]}`，用来支撑「我这次捞到多少」那行反空转输出。
    """
    inner = m.group('enr')
    if inner is None or not keep_enr:
        return ' '
    vals = []
    for pm in ENR_PARAM.finditer(inner):
        name = pm.group(1).lower()
        if name in ENR_TEXT_PARAMS:
            val = pm.group(2) if pm.group(2) is not None else pm.group(3)
            vals.append(val)
            if stats is not None:
                s = stats.setdefault(name, [0, 0, 0])
                s[0] += 1
                s[1] += len(val)
                # 第三格＝这段文本里有没有阿拉伯数字。**默认判据只看数字**，
                # 所以「捞到了」不等于「判得到」：捞到 100 段全无数字，数字闸依旧是 0 命中。
                s[2] += 1 if NUM.search(val) else 0
    if not vals:
        return ' '
    # 两端补 BOUND：朗读框 / 标签与周围正文是不同的信息单元，邻近折叠不该跨过去。
    return BOUND + BOUND.join(vals) + BOUND


def strip_markup(s: str, keep_enr: bool = True, stats=None) -> str:
    """剥标记。留下来的参数值里理论上可能再嵌标记（当前库实测 0 处：`[[` `<` `@` 全 0），
    嵌了就再剥一层 —— `re.sub` 不会回扫自己的替换结果，所以这里显式跑到不动点（至多 3 趟）。
    """
    for _ in range(3):
        out = MARKUP.sub(lambda m: _markup_repl(m, keep_enr, stats), s)
        if out == s:
            return out
        s, stats = out, None   # 第二趟起不再计数，免得同一个参数被数两遍
    return s


def plain(s: str, cn: bool = False, keep_enr: bool = True, stats=None) -> str:
    """剥掉标记与标签，只留给人读的正文。cn=True 时顺带折算中文数字。

    块级标签换成 `BOUND` 哨兵而不是空格（同样是 1 个字符，长度统计不变），
    这样 `counted()` 能分清「同一句里说了两遍」和「表格两行各有一个」。

    `keep_enr=True`（默认）时，`@Embed[… readaloud="…"]` 这类增强器的可见文本参数
    会被留下来算作正文 —— 见模块 docstring「增强器里的可见文本参数」。
    """
    out = ENTITY.sub(' ', strip_markup(s, keep_enr, stats))
    out = TAG.sub(lambda m: BOUND if BLOCK_TAG.fullmatch(m.group()) else ' ', out)
    return cn_numerals_to_digits(out) if cn else out


# 带倍数的量词：英文的「2 decades」在中文里无论写「二十年」还是「20 年」，
# 那个独立的 2 都保不住 —— 这是语言差异，不是漏译。硬要让 2 出现，
# 只能逼出「2 个十年」这种坏中文（库里真的出现过，已订正）。
# 所以英文侧遇到这类量词时，把换算后的值也算作可接受写法。
SCALED = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)\s+(decades?|dozens?|scores?|centur(?:y|ies)|'
                    r'millennia|millenniums?)\b', re.I)
SCALE = {'decade': 10, 'dozen': 12, 'score': 20, 'centur': 100,
         'millennia': 1000, 'millennium': 1000}


def acceptable_forms(pe: str):
    """英文正文里每个数字 -> 中文侧可接受的写法集合。"""
    forms = {n: {n} for n in NUM.findall(pe)}
    for num, unit in SCALED.findall(pe):
        u = unit.lower().rstrip('s')
        mult = next((v for k, v in SCALE.items() if u.startswith(k)), None)
        if mult:
            scaled = float(num) * mult
            forms.setdefault(num, {num}).add(
                str(int(scaled)) if scaled == int(scaled) else str(scaled))
    return forms


# 斜杠数对：`24/7`（英文成语，中文作「全天24小时」）、`1/2`（中文作「一半」）。
# 中文只会留下其中一个数，分母整体豁免 —— 实测 Spellbreaker Tower 的
# 「patrolled 24/7」就是这么被报成缺 7 的（同页另有真正的 `Level 7`，中文有）。
SLASH_PAIR = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\d)')


def demanded(pe: str, window: int):
    """英文侧**折叠后**的数字需求：同一块内、`window` 字符内的重复算一份信息。

    英文一句里把同一个数字说两遍（`no more than 6 boons and 6 banes`），
    中文合成一句说完（「恩惠骰与祸骰影响都不能超过 6 个」）是正常中文，不是漏译。

    ⚠ **只折英文侧，且必须认块边界。** 两条都踩过：
    * 中文侧也折 —— 中文长度只有英文的 0.35 倍，同一个 window 在中文侧等于放宽 3 倍，
      crucible 一下从 2 条涨到 50 条假警报。折叠只减需求、不减供给才是安全方向。
    * 不认块边界 —— 里程碑表 `<td>3</td><td>3</td>` 会被折成一份需求，
      `3 Talent Points`→「2点天赋点」这条真缺陷立刻被重新掩盖，等于本次白修。
    """
    last, out = {}, Counter()
    for m in NUM.finditer(pe):
        n, at = m.group(1), m.start()
        if n not in last or at - last[n] > window or BOUND in pe[last[n]:at]:
            out[n] += 1
        last[n] = at
    return out


def number_multiset(pe: str, pc: str, window: int = 60):
    """多重集比对：英文里出现 c 次的数字，中文侧要拿得出 c 个可接受写法。

    返回 `[(数字, 缺几次), ...]`。宽容规则见模块 docstring 的 1–6。
    """
    en_counts = demanded(pe, window)
    for _, denom in SLASH_PAIR.findall(pe):      # 规则 6：斜杠分母豁免
        if en_counts.get(denom):
            en_counts[denom] -= 1
    en_counts = +en_counts                       # 丢掉计数归零的项
    if not en_counts:
        return []

    forms = acceptable_forms(pe)
    pool = Counter(NUM.findall(pc))              # 中文侧按原样计数，不折叠

    # 先分配「可接受写法最少」的数字：宽容项（decade 换算等）有备选，
    # 让它先抢会把稀缺的中文数字用掉，制造假缺失。
    missing = []
    for n in sorted(en_counts, key=lambda x: (len(forms.get(x, {x})), -len(x))):
        need, got = en_counts[n], 0
        for f in sorted(forms.get(n, {n}), key=lambda f: (f != n, -pool[f])):
            take = min(pool[f], need - got)
            pool[f] -= take
            got += take
            if got == need:
                break
        if got < need:
            missing.append((n, need - got))
    return missing


def selftest() -> bool:
    """双向变异回测（不读库，秒回）。

    证明三件事，缺一不可：
    1. **能报** —— 中文朗读正文里的数字被删掉，新口径必须报出缺失；
    2. **不误报** —— 中文朗读正文完整时，新口径必须一声不响；
    3. **洞是真的** —— 同样两个样本喂给旧口径（`--no-enricher-text`），**两次都报 0**。
       第 3 条是本轮存在的理由：它把「本来就没问题」和「本来就看不见」区分开。
    另加两条护栏：方括号里的机器参数（目标 id / `dc:17` / `count=` / `classes=`）
    绝不能被计入，否则会造出一整片假缺口。
    """
    en = ('<p>The vault door is sealed.</p>'
          '<p>@Embed[JournalEntryPage.abc123 readaloud="Three sigils burn along the rim, '
          'and the lock demands 4 keys turned within 12 seconds." label="Vault Warden"]</p>')
    cn_ok = ('<p>金库大门已被封死。</p>'
             '<p>@Embed[JournalEntryPage.abc123 readaloud="轮缘上燃着三道符文，'
             '这把锁要求在 12 秒内转动 4 把钥匙。" label="金库守卫"]</p>')
    cn_bad = ('<p>金库大门已被封死。</p>'
              '<p>@Embed[JournalEntryPage.abc123 readaloud="轮缘上燃着三道符文。" '
              'label="金库守卫"]</p>')

    def miss(e, c, keep):
        return number_multiset(plain(e, keep_enr=keep),
                               plain(c, cn=True, keep_enr=keep))

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(('  PASS  ' if cond else '  FAIL  ') + msg)
        ok = ok and bool(cond)

    print('scan_content_coverage --selftest')
    # 先证明「探针真的在验」：新口径下英文正文里必须真的出现朗读文本，否则后面全是空转
    pe = plain(en)
    check('12 seconds' in pe and 'Vault Warden' in pe,
          f'新口径下 readaloud/label 的正文进了纯文本（EN 纯文本 {len(pe)} 字）')
    check('12 seconds' not in plain(en, keep_enr=False),
          '旧口径下同一段正文确实不在纯文本里（洞的直接证据）')

    m_bad = miss(en, cn_bad, True)
    check([n for n, _ in m_bad] and set(n for n, _ in m_bad) >= {'4', '12'},
          f'① 删掉中文朗读正文 → 新口径报缺失 {m_bad}')
    check(miss(en, cn_ok, True) == [],
          f'② 中文朗读正文完整 → 新口径不报（实得 {miss(en, cn_ok, True)}）')
    check(miss(en, cn_bad, False) == [] and miss(en, cn_ok, False) == [],
          '③ 旧口径对①②**都报 0** —— 证明这个洞过去是真的全盲，不是「另有闸看着」')

    # 机器参数护栏：dc:17 / 目标 id 里的数字 / count= / classes= 都不许进覆盖率
    mech_en = '<p>Force the door with @Check[athletics|dc:17] or @Embed[Actor.99887766 count="5" classes="pf2e"] to proceed.</p>'
    mech_cn = '<p>用 @Check[athletics|dc:17] 撬门，或 @Embed[Actor.99887766 count="5" classes="pf2e"] 后继续。</p>'
    check(miss(mech_en, mech_cn, True) == [],
          '④ 方括号里的机器参数（dc:17 / 目标 id / count= / classes=）不计入')
    check('17' not in plain(mech_en) and '99887766' not in plain(mech_en)
          and 'pf2e' not in plain(mech_en),
          '⑤ 机器参数连纯文本都进不去（不是靠比对碰巧对上）')

    print('  => ' + ('全绿' if ok else '**有失败**'))
    return ok


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', help='仓库根（`--selftest` 时不需要）')
    ap.add_argument('--pack')
    ap.add_argument('--out')
    ap.add_argument('--min-en', type=int, default=120,
                    help='英文纯文本短于此长度的条目不查（名字、标签噪声大）')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--merge-window', type=int, default=60,
                    help='英文侧同一块内、多少字符内的重复算一份信息（只折英文需求，'
                         '不折中文供给，且跨块不折）。设 0 = 严格多重集，一次不差；调大 = 更宽容')
    ap.add_argument('--no-enricher-text', action='store_true',
                    help='恢复旧口径：把 @X[…] 整段剥掉，连 readaloud=/label= 的正文一起丢。'
                         '**只用来出「纳入前 / 纳入后」的可比数字**，日常不要开')
    ap.add_argument('--selftest', action='store_true',
                    help='双向变异回测：合成样本上证明「删掉中文朗读正文会报、不删不报、'
                         '旧口径下两种都不报」。不读库，秒回')
    ap.add_argument('--with-terms', action='store_true',
                    help='顺带查定译专名。**默认关**：glossary_ec 里混着通用词'
                         '（Shield→盾牌、Counter→反制、blocked→屏蔽），命中全是噪声')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not a.repo:
        ap.error('--repo 是必需的（除非 --selftest）')

    P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gloss = {}
    gp = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json')
    if os.path.exists(gp):
        for k, v in json.load(open(gp, encoding='utf-8')).items():
            zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
            # 只用「多字中文 + 多字英文」的条目当锚点：单字中文误报率太高
            if len(k) >= 5 and len(zh) >= 2 and CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
                gloss[k] = zh

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    packs = [a.pack] if a.pack else sorted(
        f for f in os.listdir(en_dir)
        if f.endswith('.json') and os.path.exists(os.path.join(cn_dir, f)))

    keep_enr = not a.no_enricher_text
    rows = []
    checked = 0
    en_chars = cn_chars = 0
    enr_leaves = 0
    enr_en, enr_cn = {}, {}          # 参数名 -> [个数, 字符数]，只统计**真被查过的**叶
    for pack in packs:
        o = []
        walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
             json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
             [], o)
        for path, e, c in o:
            if not (c and CJK.search(c)):
                continue
            se, sc = {}, {}
            pe = plain(e, keep_enr=keep_enr, stats=se)
            pc = plain(c, cn=True, keep_enr=keep_enr, stats=sc)
            if len(pe) < a.min_en:
                continue
            checked += 1
            en_chars += len(pe)
            cn_chars += len(pc)
            if se or sc:
                enr_leaves += 1
            for src, dst in ((se, enr_en), (sc, enr_cn)):
                for k, v in src.items():
                    d = dst.setdefault(k, [0, 0, 0])
                    for i in range(3):
                        d[i] += v[i]
            # 多重集比对（2026-08-12）。原先是 set，`3 Talent Points`→「2点天赋点」
            # 会被同页别处的 3 掩盖 —— 全书最核心的成长规则错一级而本脚本报 0。
            miss_num = [f'{n}×{k}' if k > 1 else n
                        for n, k in sorted(number_multiset(pe, pc, a.merge_window),
                                           key=lambda t: (-t[1], -len(t[0])))]
            miss_term = ([f'{k}→{v}' for k, v in gloss.items()
                          if re.search(r'\b' + re.escape(k) + r'\b', pe) and v not in pc]
                         if a.with_terms else [])
            if miss_num or miss_term:
                rows.append({'pack': pack, 'path': path,
                             'missing_numbers': miss_num,
                             'missing_terms': miss_term[:6],
                             'en_len': len(pe), 'cn_len': len(pc),
                             'ratio': round(len(pc) / max(len(pe), 1), 2)})

    rows.sort(key=lambda r: -(len(r['missing_numbers']) * 2 + len(r['missing_terms'])))

    # 反空转：先报「我这次捞到多少增强器正文」，再报「查出多少问题」。
    # 捞到 0 个而口径是开着的 —— 那 0 条问题的含义是「无从查起」，不是「查过了没问题」。
    def _fmt(d):
        return ' · '.join(f'{k}= {v[0]} 个 / {v[1]} 字 / 其中带阿拉伯数字 {v[2]} 个'
                          for k, v in sorted(d.items())) or '（无）'
    if keep_enr:
        en_numful = sum(v[2] for v in enr_en.values())
        print(f'增强器可见文本（白名单 {"/".join(ENR_TEXT_PARAMS)}）已计入覆盖率：')
        print(f'  EN: {_fmt(enr_en)}')
        print(f'  CN: {_fmt(enr_cn)}')
        print(f'  落在 {enr_leaves} 条被查的叶上')
        if not enr_en and not enr_cn:
            print('  ⚠ **无从查起**：一个可见文本参数都没捞到。'
                  '这不等于「没问题」——先确认库里确实没有 readaloud=/label=，'
                  '再确认 ENR_PARAM / MARKUP 没被改坏（跑 --selftest）')
        elif en_numful == 0:
            print('  ⚠ **计入了但数字闸咬不到**：捞到的英文可见文本里**一个阿拉伯数字都没有**，'
                  '所以默认判据（数字多重集）对这部分文本的产出必然是 0 命中 ——'
                  '这属于「无从查起」，不是「查过了没问题」。'
                  '判这部分的闸**不在本脚本里**：见 assert_resolutions.py 的 '
                  'R-readaloud-coverage（段级：缺失／中英字符比／段内定译专名／句数）。'
                  '⚠ 不要用 `--with-terms` 代替它 —— 那是叶级的，实测干净时就报 39 条、'
                  '删空后 43 条，没有判别力。')
        if a.with_terms:
            print('  （--with-terms 已开：**它判不了朗读正文**，叶级专名闸干净侧就有几十条噪声；'
                  '朗读正文那部分请看 assert_resolutions.py 的 R-readaloud-coverage）')
    else:
        print('⚠ 旧口径（--no-enricher-text）：@X[…] 整段剥掉，'
              f'readaloud=/label= 的正文**不进覆盖率**。仅供 A/B 对比')
    print(f'口径：{checked} 条叶 / EN 正文 {en_chars} 字 / CN 正文 {cn_chars} 字'
          f'（中英比 {cn_chars / max(en_chars, 1):.3f}）')
    print(f'查了 {checked} 条已译且英文正文 ≥ {a.min_en} 字符的条目')
    print(f'  其中中文丢了英文里的数字或定译专名：**{len(rows)}** 条')
    hard = [r for r in rows if r['missing_numbers']]
    print(f'  丢数字的（信号最强，多半是漏译整句规则）：{len(hard)} 条')
    print(f'\n前 {a.top} 条：')
    for r in rows[:a.top]:
        print(f'  [{r["ratio"]}] {r["path"][-72:]}')
        if r['missing_numbers']:
            print(f'      缺数字: {r["missing_numbers"][:10]}')
        if r['missing_terms']:
            print(f'      缺专名: {r["missing_terms"][:4]}')
    if a.out:
        json.dump({'checked': checked, 'flagged': len(rows),
                   'with_missing_numbers': len(hard),
                   'enricher_text': {'enabled': keep_enr, 'en': enr_en, 'cn': enr_cn,
                                     'leaves': enr_leaves},
                   'plain_chars': {'en': en_chars, 'cn': cn_chars},
                   'items': rows},
                  open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
