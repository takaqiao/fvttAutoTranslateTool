#!/usr/bin/env python3
"""中文正文里留着的**裸英文专名** —— 有译名却没用上。

判据：某个英文专名在库里已经有确定的中文译名（来自 `name` 字段成对提取），
而中文正文里仍以**裸英文**出现，且不是本项目的双语并列形态（`中文 English`）。

为什么现有闸门全盲
------------------
* 覆盖率只看「这一叶有没有中文」—— 这些叶子中文一大片，算已译；
* 数字覆盖只看数字；标记五项只看标记；`scan_latin_residue` 有意把双语尾巴放行，
  于是裸专名混在里面一起被放行；
* `scan_label_vs_name` 只管 `@UUID{标签}`，管不到散文里的裸名字。

实测起点：`Squish` 的条目 name 早已是「压扁 Squish」，正文里却仍写 `Squish`；
连 `Tauric`（译名「陶里克」）都有裸英文出现。

**性能**：朴素实现是 O(专名 × 叶子)，在本库（3.5 万叶 × 数千专名）跑不完。
这里把全部专名编成**一条**交替正则（按长度降序，长的优先匹配），每叶只扫一遍。

必要的排除（每一条都是实测教训）
--------------------------------
* **双语并列**：`中文 English` 是本项目既定风格（第 3.4 节），English 那一半不算残留。
* **标记内部**：`@UUID[...]` 方括号里、`[[/...]]` 命令体、HTML 属性值里的英文是机器要读的。
* **专名同时也是普通英文词**：`Green` / `Ship` / `Empty` 这类，只在**首字母大写且独立成词**
  时才算；仍会有噪声，所以报告分 `strong`（长度≥2 词或非词典词）与 `weak` 两档。
* **英文原文本身就用英文**：中文照抄一个上游有意保留的英文（系统名 `Crucible`）不算错。
  故只在**该叶英文里这个专名确实存在**、且**中文译名在别处已被采用**时才报。

2026-08-13（第九轮）收口的三类已知假阳性
----------------------------------------
第八轮把这份报告压到 37 处后判定「全是假阳性或已裁 deferred」，并定位出三种固定模式。
它们每轮都要重判一次，所以这一轮改判据而不是再判一次：

1. **双语并列的语序倒装 / 前缀词 / 标签形态**。旧判据只认「译名**紧贴**英文」，
   于是 `沙纳山脉南部 Southern Shana Mountains`（中文加了方位后缀）、
   `劫掠者海洋 The Reaver Ocean`（英文带冠词）、
   `@UUID[…]{虚假生命}False Life`（中文在标签里）都认不出来。
   改成 `bilingual_before()`：先退到这段英文串的起点，再看紧邻的一小段中文里有没有译名。
2. **命中后紧跟一个大写词**。交替正则里没有更长的那个名字时短的会赢：
   `Mage Hand Press`（出版商）被报成法术 `Mage Hand`。凡是命中后紧跟大写词的，
   真正的实体一定比词典里那条长，一律不报。
3. **`&Reference[...]` 方括号内**。`MARKUP` 只认 `@X[…]` 与 `[[…]]`，
   看不见 dnd5e 的 `&Reference[Difficult Terrain]`。它是 enricher 的**查询键**，
   照抄不译才是对的（实测 `1-Ember汉化插件` 英中各 1228 处，逐字节一致）。

用法：
  python scan_bare_english_names.py --repo <repo> [--repo <另一个>] [--out <json>] [--show 40]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import string
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')
# dnd5e 的规则引用 enricher。方括号里是**查询键**，照抄不译（英中逐字节一致），
# 但 `MARKUP` 只认 `@` 开头，看不见它 —— `&Reference[Difficult Terrain]` 因此被
# 报成「裸英文 Difficult Terrain」。与 apply_translations.REFERENCE 同形。
REFERENCE = re.compile(r'&(?:amp;)?[Rr]eference\[[^\]]*\]')
ATTR = re.compile(r'\w+\s*=\s*"[^"]*"')
# 双语并列的英文尾巴
BILINGUAL_TAIL = re.compile(r'\s+[^一-鿿　-〿＀-￯]+$')

# 命中后**紧跟一个大写词** —— 说明真正的实体比词典里那条长（`Mage Hand` vs 出版商
# `Mage Hand Press`）。交替正则按长度降序，长的本来会赢，输只可能是因为词典里没有它。
FOLLOWED_BY_CAP = re.compile(r"[  ][A-Z][A-Za-z'’\-]")

# 从命中处往回退时可以跨过的字符：拉丁字母数字、空格，以及会夹在中文与英文之间的
# ASCII 标点（`{虚假生命}False Life` 的 `}`、`&amp;Reference[` 的那一串）。
BACK_SKIP = frozenset(string.ascii_letters + string.digits + " \t'’-.&;:/|(){}[]")
#: 回退最多跨这么多字符，免得从一整句英文的中间一路退到句首的中文上
BACK_MAX_RUN = 40


def bilingual_before(plain: str, start: int, zh: str) -> bool:
    """命中的英文名是不是「中文 English」双语并列里英文那一半的一部分。

    旧判据 `plain[start-len(zh)-1:start].endswith(zh)` 只认译名**紧贴**英文，
    实测认不出三种常见形态：

    * 中文侧带修饰、语序倒装：`沙纳山脉南部 Southern Shana Mountains`
      （词典是 `Shana Mountains`→沙纳山脉，中文把方位挪到了后面）
    * 英文侧带冠词或前缀词：`劫掠者海洋 The Reaver Ocean`
    * 中文在 `@UUID[…]{标签}` 里、英文紧跟其后：`{虚假生命}False Life 法术`

    做法：先退到这段英文串的起点，再看紧邻的一小段中文里有没有它的译名。
    """
    i = start
    while i > 0 and (start - i) < BACK_MAX_RUN and plain[i - 1] in BACK_SKIP:
        i -= 1
    return zh in plain[max(0, i - len(zh) - 12):i]


# ============================================================================
# 2026-08-15（第十六轮 Z2）：`--min-words 1` 的四类结构性假阳性
# ----------------------------------------------------------------------------
# `--min-words 2` 归零后把闸降到 1，实测 70 处 / 31 个专名。逐处判下来
# **32 处是真缺陷**（已产批次 `round16/batches/z2-bare-english-1w.json`），
# 4 处是待裁的译名问题（`Cyclonic`，见 PROJECT.md Z2），
# 其余 **34 处是下面这四类结构性假阳性** —— 单词级判据独有，两词级碰不到。
#
# ⚠ 写这四条时刻意**没有**加「命中前紧跟大写词就跳过」这一条通用规则
#   （即 FOLLOWED_BY_CAP 的镜像）。它看着能一次盖掉 D&D / Forge of the Artificer /
#   Warhammer Skallith / Keystone Challenge，但实测会**连真缺陷一起消掉**：
#   `Keystone Challenge`（库内 30+ 处作「基石挑战」）与 `Warhammer Skallith`
#   （令牌中文名就叫「战锤斯卡利斯」）都是前面紧跟大写词的**真缺陷**。
#   这正是第九轮 `Sunalins` 那一课 —— 降噪规则必须逐条核对被消掉的是什么。
# ============================================================================

# 1) **整叶就是一个双语并列名**。`folders.<名>` / `scenes.*.notes.<名>` /
#    `Conditions.categories.<名>` 这类「键即英文名、值是中文名 English」的映射，
#    路径不以 `.name` 结尾，逃过了脚本对 `.name` 的整条排除。
#    `bilingual_before()` 也认不出来，因为词典给这个英文投出的中文常常**不是**
#    本处用的那个（`Water`→水域，而这条叶子写的是「水元素 Water」）。
#    判据：整叶去掉首尾空白后**以该叶英文原值整串结尾**，且前缀里**一个拉丁字母都没有**
#    并含中文。这么严是有意的 —— 散文叶的英文原值是一整段，不可能整串结尾。
def whole_leaf_bilingual_tail(plain: str, en_value: str):
    """整叶是「中文 <英文原值>」时，返回英文那一半的起点；否则 None。"""
    s = plain.strip()
    e = en_value.strip()
    if not e or len(e) >= len(s) or not s.endswith(e):
        return None
    prefix = s[:len(s) - len(e)]
    if re.search(r'[A-Za-z]', prefix) or not CJK.search(prefix):
        return None
    return plain.index(s) + len(prefix)


# 2) **姓名表**。`journals.Cultures.pages.*` / `Ancestries.*` 的「姓名：」后面是
#    上游给的取名建议表，几十个英文名用 `、` 或 `,` 串起来，照抄不译才对
#    （译了等于替玩家把取名表改掉）。其中恰好有几个撞上库内实体名：
#    `Tayan` `Orbis` `Gnash` `Rask` `Vial` `Hallow`。
#    阈值取 8 段是量出来的：全库这种大写词逗号串的长度分布是**双峰**——
#    2 段 30 处（普通行文，如「A、B」），然后直接跳到 18 段起（姓名表），
#    **3–17 段一处都没有**。8 落在空档正中，两边各留一倍余量。
NAME_LIST_RUN = re.compile(
    r"[A-Z][A-Za-z'’\-]*(?:\s*[,、;；]\s*[A-Z][A-Za-z'’\-]*){7,}")

# 3) **外部产品 / 模块名**。上游写的是真实世界的书名模块名，全库一致保留英文，
#    交替正则却只认得其中一个词（`Dungeons &amp; Dragons` 里的 `Dragons`→巨龙、
#    `Eberron: Forge of the Artificer` 里的 `Artificer`→工匠）。
#    这是**字面白名单**而不是通用规则，理由见本节开头的 ⚠。
EXTERNAL_TITLES = (
    'Dungeons &amp; Dragons', 'Dungeons & Dragons',   # D&D 5e，全库 8 处
    'Forge of the Artificer',                          # WotC 模块名，全库 4 处
    "Tasha's Cauldron of Everything",                  # 同上，与前者同句出现
)

# 4) **UI 菜单路径**。`Throw &gt; Activation &gt; Consumption` 是叫 GM 照着点的
#    dnd5e 角色卡路径，跟 `@UUID[...]` 一样是「机器/界面要读的」，译了就点不着。
#    要求链上**至少两个 `>`**（三段），普通行文里的单个 `A > B` 不算。
UI_PATH_CHAIN = re.compile(
    r"[A-Z][A-Za-z]*(?:\s*(?:&gt;|>)\s*[A-Z][A-Za-z]*){2,}")


def in_any(spans, start, end):
    return any(s <= start and end <= e for s, e in spans)


def literal_spans(plain: str, literals) -> list:
    out = []
    for lit in literals:
        i = plain.find(lit)
        while i >= 0:
            out.append((i, i + len(lit)))
            i = plain.find(lit, i + 1)
    return out


#: 这些英文词太常见，单独出现时几乎一定不是专名引用
STOP = {
    'Green', 'Ship', 'Empty', 'Ground', 'Surface', 'Entry', 'Basement', 'Library',
    'Gardens', 'Markets', 'Temple', 'Mine', 'Ocean', 'Roofs', 'Fields', 'Overlook',
    'Traps', 'Caste', 'Gala', 'Lookout', 'Promenade', 'Vineyard', 'Pathways',
}


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def head(name):
    s = BILINGUAL_TAIL.sub('', name).strip()
    return s or name.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=40)
    ap.add_argument('--min-words', type=int, default=1,
                    help='英文专名至少几个词才纳入（2 可大幅降噪）')
    a = ap.parse_args()

    # ---- 1. 建「英文专名 -> 中文译名」词典（只取 name 字段，取多数写法）
    votes = collections.defaultdict(collections.Counter)
    for repo in a.repo:
        en_dir = os.path.join(repo, 'compendium', 'en')
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or pack == '_source.json':
                continue
            cn_p = os.path.join(repo, 'compendium', 'cn', pack)
            if not os.path.exists(cn_p):
                continue
            en = dict(walk(load(os.path.join(en_dir, pack)).get('entries', {})))
            cn = dict(walk(load(cn_p).get('entries', {})))
            for path, v in en.items():
                if not path.endswith('.name'):
                    continue
                c = cn.get(path)
                if not c or not CJK.search(c):
                    continue
                if not re.fullmatch(r"[A-Z][A-Za-z'’\-]*(?: [A-Z][A-Za-z'’\-]*)*", v):
                    continue
                if len(v) < 4 or v in STOP:
                    continue
                if len(v.split()) < a.min_words:
                    continue
                votes[v][head(c)] += 1
    DICT = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    print(f'英文专名 -> 中文译名 词典：{len(DICT)} 条')
    if not DICT:
        return

    # ---- 2. 一条交替正则，长的优先
    names = sorted(DICT, key=len, reverse=True)
    RX = re.compile(r'(?<![A-Za-z])(' + '|'.join(re.escape(n) for n in names) + r')(?![A-Za-z])')

    findings, per_name = [], collections.Counter()
    for repo in a.repo:
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        en_dir = os.path.join(repo, 'compendium', 'en')
        for pack in sorted(os.listdir(cn_dir)):
            if not pack.endswith('.json'):
                continue
            en_p = os.path.join(en_dir, pack)
            if not os.path.exists(en_p):
                continue
            cn = dict(walk(load(os.path.join(cn_dir, pack)).get('entries', {})))
            en = dict(walk(load(en_p).get('entries', {})))
            for path, v in cn.items():
                # `.name` 整条排除是**有意的**：本判据查的是「散文里留着裸英文专名」，
                # 而 `.name` 按本项目约定本来就该带英文尾巴（「压扁 Squish」），
                # 在这里查只会把全部双语名报成残留。
                # ⚠ 但因此「应带尾巴的 name 缺了尾巴」不在本脚本辖区 ——
                # 那一维度归 qa/scan_missing_bilingual_tail.py（2026-08-14 新增）。
                # 在它之前 scan_same_en_split.py 的交接声明指向本脚本，是错的。
                if path.endswith('.name') or not CJK.search(v):
                    continue
                ev = en.get(path)
                if not ev:
                    continue
                # 抹掉机器要读的部分与双语并列
                plain = ATTR.sub(' ', REFERENCE.sub(' ', MARKUP.sub(' ', v)))
                plain = TAG.sub(' ', plain)
                # ---- 单词级的四类结构性豁免（成因见文件上半部分的长注释）
                skip_spans = []
                tail = whole_leaf_bilingual_tail(plain, ev)        # 1) 整叶双语名
                if tail is not None:
                    skip_spans.append((tail, len(plain)))
                skip_spans += [m2.span() for m2 in NAME_LIST_RUN.finditer(plain)]   # 2) 姓名表
                skip_spans += literal_spans(plain, EXTERNAL_TITLES)                 # 3) 外部产品名
                skip_spans += [m2.span() for m2 in UI_PATH_CHAIN.finditer(plain)]   # 4) UI 菜单路径
                for m in RX.finditer(plain):
                    if in_any(skip_spans, m.start(), m.end()):
                        continue
                    name = m.group(1)
                    zh = DICT[name]
                    # 双语并列：中文名（可带修饰、可在 {标签} 里）在这段英文之前
                    if bilingual_before(plain, m.start(), zh):
                        continue
                    # 命中后紧跟大写词 —— 真正的实体比词典里那条更长（`Mage Hand Press`）
                    if FOLLOWED_BY_CAP.match(plain, m.end()):
                        continue
                    # 英文原文里必须确实有这个专名，否则中文里的英文另有来源
                    if not re.search(r'(?<![A-Za-z])' + re.escape(name) + r'(?![A-Za-z])', ev):
                        continue
                    per_name[name] += 1
                    findings.append({
                        'repo': repo, 'pack': pack, 'path': path, 'batch_path': path,
                        'english': name, 'should_be': zh,
                        'context': plain[max(0, m.start() - 40):m.end() + 40],
                    })

    print(f'**中文正文里的裸英文专名**：{len(findings)} 处 / {len(per_name)} 个专名')
    for n, c in per_name.most_common(a.show):
        print(f'  {c:4d}×  {n:34s} -> {DICT[n]}')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'dict_size': len(DICT), 'findings': findings,
                       'per_name': dict(per_name)}, f, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
