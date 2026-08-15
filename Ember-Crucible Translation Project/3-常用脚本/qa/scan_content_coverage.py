#!/usr/bin/env python3
"""不靠长度、靠「跨语言不变量」找出中文没跟上英文的条目。

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
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter

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
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|&(?:amp;)?[Rr]eference\[[^\]]*\]')
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


def plain(s: str, cn: bool = False) -> str:
    """剥掉标记与标签，只留给人读的正文。cn=True 时顺带折算中文数字。

    块级标签换成 `BOUND` 哨兵而不是空格（同样是 1 个字符，长度统计不变），
    这样 `counted()` 能分清「同一句里说了两遍」和「表格两行各有一个」。
    """
    out = ENTITY.sub(' ', MARKUP.sub(' ', s))
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
    ap.add_argument('--repo', required=True)
    ap.add_argument('--pack')
    ap.add_argument('--out')
    ap.add_argument('--min-en', type=int, default=120,
                    help='英文纯文本短于此长度的条目不查（名字、标签噪声大）')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--merge-window', type=int, default=60,
                    help='英文侧同一块内、多少字符内的重复算一份信息（只折英文需求，'
                         '不折中文供给，且跨块不折）。设 0 = 严格多重集，一次不差；调大 = 更宽容')
    ap.add_argument('--with-terms', action='store_true',
                    help='顺带查定译专名。**默认关**：glossary_ec 里混着通用词'
                         '（Shield→盾牌、Counter→反制、blocked→屏蔽），命中全是噪声')
    a = ap.parse_args()

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

    rows = []
    checked = 0
    for pack in packs:
        o = []
        walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
             json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
             [], o)
        for path, e, c in o:
            if not (c and CJK.search(c)):
                continue
            pe, pc = plain(e), plain(c, cn=True)
            if len(pe) < a.min_en:
                continue
            checked += 1
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
                   'with_missing_numbers': len(hard), 'items': rows},
                  open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
