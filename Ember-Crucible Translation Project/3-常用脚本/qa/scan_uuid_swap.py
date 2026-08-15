#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫 `@UUID[目标]{标签}` 的**标签与目标错位** —— 链接指向了错误的文档。

    python scan_uuid_swap.py --repo <repo> [--repo <另一个>] [--pack <name.json>]
                             [--out <json>] [--min-support N] [--min-ratio R]
                             [--alien-ratio R] [--basis auto|target]
                             [--include-commands] [--show N]

这个闸门补的盲区
----------------
译者把英文语序倒装成中文时（"the Tyraphem on Luxarum" → "在 Luxarum 上的 Tyraphem"），
只搬动了 `{标签}` 文字、没搬 `@UUID[目标]`。结果每条链接的标签都挂到了**邻居的目标**上，
玩家点「提拉斐姆」跳进的是卢克萨鲁姆的页面。

现有的每一道闸对此**完全沉默**，而且不是实现疏漏，是判据本身管不到：

* `apply_translations.py` 的 `markup_signature` 用 `@[A-Za-z]+\\[[^\\]]*\\]` 建签名，
  **按设计剥掉 `{标签}`**（标签是要翻译的可见文字，见 PROJECT.md 第 8 节 2026-08-06 那条决议）。
  错位后目标多重集一字未变 → 0 拒绝。
* `scan_markup_drift.py` 同理，比的是标记多重集。
* `scan_markup_targets.py` 只问「方括号里有没有混进中文」—— 错位的目标全是干净的 id。
* `scan_class_drift.py` / `scan_content_coverage.py` / 覆盖率统计 —— 更加看不见。

一句话：**错位后的叶子，目标多重集与英文相等，标签多重集也与英文相等**（标签只是换了绑定）。
任何「多重集相等即清白」的检查都必然漏掉这一类。要抓它必须查**配对关系**，不是集合。

判据
----
先在全库（跨全部 `--repo`、所有包）建「目标 → 中文标签分布」表。统计单位是**叶子**不是出现
次数，免得同一叶里重复三遍的错标压过别处的正确写法；算某一叶时**先扣掉该叶自己的那一票**，
否则孤例会自我印证。

表默认按 **(目标 id, 该叶英文标签)** 建键（`--basis auto`），不是只按目标 id：同一个目标在
不同语境本来就会带不同标签（页面 `The Party Arrives` 在别处被引作 `Temple Sanctum`），
只按目标 id 统计会把大量**正确**译文判成偏离多数。英文标签是权威的语境锚，钉住它之后，
中文标签的差异才真的是差异。英文侧没有对应链接的叶子（死键等）退回只按目标 id 统计。

在此之上三条证据，互相独立，任一成立即报 BROKEN：

1. **同叶归属**：某目标在本叶的中文标签，其在该目标上的占比低于 `--alien-ratio`（＝外来），
   而它恰好是**同叶另一个目标**的多数标签（占比 ≥ `--min-ratio`、支持 ≥ `--min-support`）。
   报告给出「这个标签本来属于谁」。
2. **轮转闭合**：一句里常常是 3 个以上目标的**轮转**而非两两交换（实测
   `History/The First Dawn` 一句 7 个全错位）。证据 1 命中后再做闭合：凡是「被认定为某个
   错位标签的正主」、自己手上的标签又是外来的目标，一并升级为 BROKEN —— 否则轮转的最后
   一环会掉进 UNCERTAIN。
3. **英文侧配对**（不依赖任何全库统计，只看本叶）：中文标签是纯英文，且逐字等于**同叶另一个
   目标**的英文标签、又不等于自己目标的英文标签。专抓「标签没译 + 又搬错位置」的
   （本库 `{Casia}` 挂在 Primordis 的目标上就是）。

分档
----
* `BROKEN`    —— 上面三条至少一条成立。
* `UNCERTAIN` —— 该目标在该语境下有清晰的多数标签，本叶用了别的写法且占比很低，但**归属不到
  同叶任何目标**。同一目标在库内有多个合法中文标签是常态，所以这一档单独放，不塞进 BROKEN
  污染结论。实际内容多是近音异写（西格纳拉 / 西格纳兰）与术语裁决没执行到位（阿克图里亚 /
  阿克图里安）这类真实不一致 —— 值得看，但不是本检测器负责的缺陷。
* 另有三类**只计数不列出**，它们是别的账或根本不是缺陷：
  - `untranslated` 标签整条留英且等于本目标自己的英文标签 → 「@UUID 标签未译」那条账；
  - `variant` 中文标签包含（或被包含于）该目标的多数标签，如「卢克萨鲁姆 - 无尽战争之境」、
    双语并列「初曙 The First Dawn」 → 合法变体；
  - `no-basis` 本叶的英文标签在全库支持不足（多为一次性的语境化标签），无从判定。

覆盖范围与已知覆盖不到的地方
----------------------------
**覆盖**：`@UUID[…]{标签}`、`@Embed[… flags]{标签}`、`@embed`，以及任何 `@X[<目标>]{标签}`
只要目标是 Foundry 文档 UUID（首段是 `Actor` / `Item` / `JournalEntry` / `Compendium` /
`Scene` / `RollTable` / `Macro` 等文档类型）。`@Embed` 方括号里的 `inline`、`readaloud="…"`
等参数与 `#锚点` 会先剥掉，再取末段 id 作键（同一文档会被写成
`JournalEntry.a.JournalEntryPage.b` 与 `Compendium.pkg.JournalEntry.b` 两种前缀，取末段才合得到一起）。
遍历从**包的根**开始，所以顶层 `label` 与 `folders.*` 也在内 —— 它们不在 `validate_translations.py`
/ `pair_dump.py` 的定义域里。

**默认不进证据 1/2，要 `--include-commands` 才纳入**：
* `[[/skillCheck …]]{标签}`、`[[/hazard …]]{标签}` 等行内命令 —— 「目标」是一条命令而非文档
  身份，同一条命令在全库带几十种合法标签，「多数标签」的前提不成立，开着只会灌满 UNCERTAIN。
* `@ref[actor.name]{生物}` —— 本库 762 处，目标是数据路径不是文档，标签是随句子变的普通名词，同理。
  （证据 3 对这两类**始终生效**，它只用本叶的英文配对、不需要全库统计。）

**本检测器管不到的**：
* 无标签的 `@UUID[目标]`（渲染的是文档名，没有标签可错位）；
* 目标的正确标签在库内**从未**出现过的（没有多数可比）；
* 整叶链接被系统性整体平移、且平移后每个标签在全库都成了「多数」的极端情形；
* 英文侧自己就把标签配错的（本检测器把英文当权威）；
* `lang/cn.json` 与 `scripts/*.mjs` 里的链接 —— 只读 `compendium/`。

CLI 与 `qa/` 其余脚本一致：`--repo` 可重复，`--pack` 限定单包，`--out` 落 JSON。
有 BROKEN 时退出码 1。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:  # Windows 控制台
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CJK = re.compile(r'[一-鿿]')
# `@UUID[target]{label}` / `@Embed[target flags]{label}` / `@Item[…]{…}` …
# 目标里不会出现 `]`，标签里不会出现 `}`；两者都不跨行。
LINK = re.compile(r'@([A-Za-z]+)\[([^\]\n]*)\]\{([^}\n]*)\}')
# `[[/skillCheck awareness 14]]{Label}` —— 带标签的行内命令。
CMD = re.compile(r'\[\[([^\]\n]*)\]\]\{([^}\n]*)\}')
# 目标是不是 Foundry 文档 UUID：看首段是不是文档类型。大小写敏感，`actor.name` 因此排除。
DOC_TYPES = {
    'Actor', 'Item', 'JournalEntry', 'JournalEntryPage', 'Scene', 'RollTable',
    'Macro', 'Cards', 'Card', 'Playlist', 'PlaylistSound', 'Adventure', 'Folder',
    'Compendium', 'ActiveEffect', 'Combat', 'Combatant', 'Note', 'Token',
    'AmbientLight', 'AmbientSound', 'MeasuredTemplate', 'Tile', 'Wall',
    'Drawing', 'Region', 'TableResult', 'User', 'ChatMessage',
}
ARTICLE = re.compile(r'^(?:the|a|an)\s+', re.I)
WS = re.compile(r'\s+')


# --------------------------------------------------------------------------- #
# 抽取
# --------------------------------------------------------------------------- #
def norm_label(s: str) -> str:
    """标签的统计口径：折叠空白、去首尾空格。` 西格纳拉` 与 `西格纳拉` 是同一个写法。"""
    return WS.sub(' ', s).strip()


def norm_en(s) -> str:
    """英文标签的比较口径：大小写、冠词、首尾标点都不算差异。

    少了这一步，`the Cosmos` 与 `Cosmos` 会被证据 3 误判成「挂错了目标」。
    """
    s = WS.sub(' ', s or '').strip().lower()
    s = ARTICLE.sub('', s)
    return s.strip(' .,;:!?"\'’“”')


def target_key(body: str):
    """→ `(键, 完整目标, 是否文档型)`。"""
    tok = body.strip().split(' ')[0] if body.strip() else ''
    tok = tok.split('#')[0]
    if not tok:
        return None, body.strip(), False
    segs = tok.split('.')
    return segs[-1], tok, segs[0] in DOC_TYPES


def links_in(s: str):
    out = []
    for m in LINK.finditer(s):
        key, full, doc = target_key(m.group(2))
        if key is None:
            continue
        out.append({'form': '@' + m.group(1), 'key': key, 'target': full, 'doc': doc,
                    'label': m.group(3), 'norm': norm_label(m.group(3)), 'at': m.start()})
    for m in CMD.finditer(s):
        body = WS.sub(' ', m.group(1)).strip()
        out.append({'form': '[[]]', 'key': 'CMD:' + body, 'target': '[[' + body + ']]',
                    'doc': False, 'label': m.group(2), 'norm': norm_label(m.group(2)),
                    'at': m.start()})
    out.sort(key=lambda d: d['at'])
    return out


def leaf_strings(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaf_strings(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaf_strings(v, path + [str(i)], out)
    elif isinstance(node, str):
        out['.'.join(path)] = node


def batch_path(path: str):
    """批次文件（`apply_translations.py`）用的键：相对 `entries`，folders 加 `(folders)` 前缀。"""
    if path.startswith('entries.'):
        return path[len('entries.'):]
    if path.startswith('folders.'):
        return '(folders).' + path[len('folders.'):]
    return None


def align_en(en_ls, cn_ls):
    """**逐位对齐**：中文侧每个链接 → 它在英文侧对应位置的英文标签（对不上返回 None）。

    2026-08-13（第九轮）替换掉原来的 `en_index`（目标 → 该目标在本叶的**第一个**英文标签，
    多标签时干脆置 None）。那个口径是 UNCERTAIN 档 97% 假阳性的根源：

        `Arcturel Dives / Area Overview` 一页里目标 `9plFRf3Hurd9r7ol` 出现 5 次，
        英文标签分别是 Emelyn Arvoda / Arcturians / Eolas Hathwick / Calandra /
        Hob Korell，中文一一对应且**全对**。取第一个 → 后 4 处全被报成「偏离多数」，
        而且 `en_label` 一律显示成 Emelyn Arvoda，张冠李戴到一眼可见。

    对齐单位是「**同一 target 的出现序号**」而不是全叶序号 —— 与 `scan_label_vs_name`
    2026-08-13e 的修法同源。中文常常重排句序，全叶序号会错位；同一目标自身的引用
    顺序则几乎不变。个数对不上时（英文侧更少）多出来的位置给 None，宁可无锚不乱锚。

    返回一个与 `cn_ls` 等长的列表。空标签也占位，否则序号会串。
    """
    by_key = defaultdict(list)
    for L in en_ls:
        by_key[L['key']].append(L['norm'])
    seen, out = Counter(), []
    for L in cn_ls:
        j = seen[L['key']]
        seen[L['key']] += 1
        labs = by_key.get(L['key']) or []
        out.append(labs[j] if j < len(labs) else None)
    return out


def raw_en_index(links):
    m = defaultdict(list)
    for L in links:
        if L['norm']:
            m[L['key']].append(L['norm'])
    return m


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', action='append', required=True,
                    help='汉化插件仓库根目录，可重复；多数标签表跨全部 --repo 建立')
    ap.add_argument('--pack', help='只查这一个包（文件名），默认全部')
    ap.add_argument('--out', help='报告落盘路径（JSON）')
    ap.add_argument('--min-support', type=int, default=3,
                    help='「多数标签」至少要有多少个**别的**叶子支持（默认 3）')
    ap.add_argument('--min-ratio', type=float, default=0.6,
                    help='「多数标签」在该目标该语境下的最低占比（默认 0.6）')
    ap.add_argument('--alien-ratio', type=float, default=0.3,
                    help='本叶所用标签的占比低于此值才算「外来」（默认 0.3）')
    ap.add_argument('--basis', choices=('auto', 'target'), default='auto',
                    help='auto=按(目标,英文标签)建键、英文缺失时退回目标；target=只按目标（更吵）')
    ap.add_argument('--include-commands', action='store_true',
                    help='把 [[/…]]{标签} 与 @ref[路径]{标签} 也纳入多数标签判据（很吵，见 --help 开头）')
    ap.add_argument('--show', type=int, default=12, help='控制台最多列出几组 BROKEN（默认 12）')
    a = ap.parse_args()

    # ---------------- pass 1：抽链接 ----------------
    cn_links: dict = {}          # leafkey -> [link]
    en_links: dict = {}          # leafkey -> [link]
    per_pack, scanned, labelled, doclike = [], 0, 0, 0

    for repo in a.repo:
        rname = os.path.basename(os.path.normpath(repo))
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        en_dir = os.path.join(repo, 'compendium', 'en')
        if not os.path.isdir(cn_dir):
            print(f'!! 没有 {cn_dir}，跳过')
            continue
        packs = [a.pack] if a.pack else sorted(
            f for f in os.listdir(cn_dir) if f.endswith('.json') and not f.startswith('_'))
        for pack in packs:
            cn_p = os.path.join(cn_dir, pack)
            if not os.path.isfile(cn_p):
                continue
            with open(cn_p, encoding='utf-8') as f:
                cn_doc = json.load(f)
            cn_leaves: dict = {}
            leaf_strings(cn_doc, [], cn_leaves)

            n_link = n_lab = n_doc = 0
            for path, s in cn_leaves.items():
                if '@' not in s and '[[' not in s:
                    continue
                ls = links_in(s)
                if not ls:
                    continue
                cn_links[(rname, pack, path)] = ls
                n_link += len(ls)
                n_lab += sum(1 for L in ls if L['norm'])
                n_doc += sum(1 for L in ls if L['doc'])

            en_p = os.path.join(en_dir, pack)
            if os.path.isfile(en_p):
                with open(en_p, encoding='utf-8') as f:
                    en_doc = json.load(f)
                en_leaves: dict = {}
                leaf_strings(en_doc, [], en_leaves)
                for path, s in en_leaves.items():
                    lk = (rname, pack, path)
                    if lk in cn_links and ('@' in s or '[[' in s):
                        el = links_in(s)
                        if el:
                            en_links[lk] = el

            scanned += n_link
            labelled += n_lab
            doclike += n_doc
            per_pack.append({'repo': rname, 'pack': pack, 'links': n_link,
                             'labelled': n_lab, 'doc_targets': n_doc})
            print(f'[扫] {rname}/{pack:<34} 链接 {n_link:>6}  带标签 {n_lab:>6}  文档型 {n_doc:>6}')

    # ---------------- 建多数标签表 ----------------
    # ctx[(目标, 英文标签)][中文标签] = {叶子}   ← 主表，英文标签当语境锚
    # any_[目标][中文标签]          = {叶子}   ← 英文侧缺链接时的退路
    ctx: dict = defaultdict(lambda: defaultdict(set))
    any_: dict = defaultdict(lambda: defaultdict(set))
    anchor: dict = {}            # leafkey -> [逐位对齐到的英文标签或 None]，与 cn_links[lk] 等长

    for lk, ls in cn_links.items():
        ea = align_en(en_links.get(lk, []), ls)
        anchor[lk] = ea
        for i, L in enumerate(ls):
            if not L['norm'] or not (L['doc'] or a.include_commands):
                continue
            any_[L['key']][L['norm']].add(lk)
            e = norm_en(ea[i]) if ea[i] else None
            if e:
                ctx[(L['key'], e)][L['norm']].add(lk)

    def dist(table, key, leafkey):
        """扣掉本叶自己那一票之后的标签分布。"""
        c = Counter()
        for lab, s in table.get(key, {}).items():
            n = len(s) - (1 if leafkey in s else 0)
            if n:
                c[lab] = n
        return c

    def top(c):
        """分布 → (多数标签, 支持, 总数, 占比, 分布)，达不到阈值返回 None。"""
        if not c:
            return None
        total = sum(c.values())
        lab, n = c.most_common(1)[0]
        if n < a.min_support or n / total < a.min_ratio:
            return None
        return lab, n, total, n / total, c

    def references(link_key, en_lab, leafkey):
        """→ {依据名: (多数标签, 支持, 总数, 占比, 分布)}，两种依据各算一份。

        `target+en` 是主依据（精确，噪声低）；`target` 是兜底（召回高，单用会把大量**正确**
        译文判成偏离多数）。两者的用法不同，见下面 BROKEN / UNCERTAIN 的分工。
        """
        out = {}
        if a.basis == 'auto' and en_lab:
            m = top(dist(ctx, (link_key, en_lab), leafkey))
            if m:
                out['target+en'] = m
        m = top(dist(any_, link_key, leafkey))
        if m:
            out['target'] = m
        return out

    # ---------------- pass 2：逐叶判定 ----------------
    findings, groups = [], []
    bucket = Counter()

    for leafkey in sorted(cn_links):
        rname, pack, path = leafkey
        ls = cn_links[leafkey]
        ea = anchor.get(leafkey) or [None] * len(ls)
        raw_en = raw_en_index(en_links[leafkey]) if leafkey in en_links else {}

        # 该目标在本叶用过的全部英文标签（规范化）—— same_en 守卫用
        en_labs_of = defaultdict(set)
        for i, L in enumerate(ls):
            if ea[i]:
                en_labs_of[L['key']].add(norm_en(ea[i]))

        def en_of(k):
            """该目标在本叶的英文标签集合的代表值。**只用于展示归属**，
            判定一律走逐位对齐的 `ea[i]`。"""
            s = sorted(en_labs_of.get(k) or ())
            if s:
                return ' / '.join(s) if len(s) > 1 else s[0]
            v = raw_en.get(k)
            return v[0] if v else None

        # 逐位建引用表：同一目标的不同出现有不同的英文语境锚，不能再按目标 id 合并一份
        ref_at = {}
        for i, L in enumerate(ls):
            if not (L['doc'] or a.include_commands):
                continue
            ref_at[i] = references(L['key'], norm_en(ea[i]) if ea[i] else None, leafkey)
        ref = {}                       # 目标 → 任一处的引用表（展示归属时用）
        for i, rr in ref_at.items():
            ref.setdefault(ls[i]['key'], rr)

        # 归属索引：某个多数标签在本叶唯一属于哪个目标
        # （同一个多数标签被本叶两个目标共用时属歧义，两边都弃用）
        owner = {}
        for basis in ('target+en', 'target'):
            own = defaultdict(set)
            for i, rr in ref_at.items():
                if basis in rr:
                    own[rr[basis][0]].add(ls[i]['key'])
            owner[basis] = {lab: next(iter(ks)) for lab, ks in own.items() if len(ks) == 1}

        def same_en(owner_k, e_here):
            """本处的英文标签，与「正主」目标在本叶用过的某个英文标签相同
            → 本叶没有任何位置证据，中文名不一致是命名问题不是错位。

            本库真实存在：同一页里两个不同文档的英文标签都叫 `Ordain` / `Kessia`。
            不设这道守卫，它们会被互相判成「标签挂错了对方」。
            """
            return bool(e_here) and norm_en(e_here) in en_labs_of.get(owner_k, set())

        leaf_rows, broken_here = [], []
        for i, L in enumerate(ls):
            if not L['norm']:
                continue
            key, c = L['key'], L['norm']
            e_self = ea[i]                       # ← 逐位对齐取到的**本处**英文标签
            rr = ref_at.get(i, {})
            # 主依据（决定分档与 UNCERTAIN）：英文标签在场就必须用 target+en —— 支持不足即判
            # 「无依据」，**不退回** target，那正是 UNCERTAIN 噪声的来源。英文侧压根没有这条
            # 链接时才用 target。`--basis target` 则一律用 target。
            if a.basis == 'target':
                primary = 'target' if 'target' in rr else None
            elif 'target+en' in rr:
                primary = 'target+en'
            elif e_self:
                primary = None
            else:
                primary = 'target' if 'target' in rr else None
            evidence, owner_key, verdict, used = [], None, None, None
            row = {'i': i, 'form': L['form'], 'target': L['target'], 'key': key,
                   'en_label': e_self, 'cn_label': L['label']}

            def classify(m_lab):
                short, long_ = sorted((c, m_lab), key=len)
                if len(short) >= 2 and short in long_:
                    return 'variant'                                  # 合法变体 / 双语并列
                if not CJK.search(c) and e_self and norm_en(c) == norm_en(e_self):
                    return 'untranslated'                             # 标签没译 —— 另一条账
                return None

            # --- 证据 3：英文侧配对（只用本叶，对所有链接形态生效） ---
            if raw_en and c and not CJK.search(c):
                nc = norm_en(c)
                if nc and (e_self is None or nc != norm_en(e_self)):
                    for k2, labs in raw_en.items():
                        if k2 != key and any(nc == norm_en(x) for x in labs):
                            evidence.append('en-label-misplaced')
                            owner_key, verdict = k2, 'BROKEN'
                            break

            # --- 证据 1：全库多数标签 + 同叶归属。两种依据取并集（召回优先） ---
            # BROKEN 有「标签必须正好是同叶另一目标的多数标签」这条强约束兜着，
            # 所以放宽依据不会像 UNCERTAIN 那样把噪声放进来。
            same_en_hit = None
            for basis in ('target+en', 'target'):
                if basis not in rr:
                    continue
                m_lab, m_n, m_tot, m_ratio, cdist = rr[basis]
                if c == m_lab or classify(m_lab):
                    continue
                if cdist.get(c, 0) / m_tot >= a.alien_ratio:
                    continue
                ow = owner[basis].get(c)
                if not ow or ow == key:
                    continue
                if same_en(ow, e_self):
                    same_en_hit = same_en_hit or (basis, ow)
                    continue
                evidence.append('majority-owner-in-leaf')
                owner_key, verdict, used = owner_key or ow, 'BROKEN', basis
                break

            # --- 分档与计数：UNCERTAIN 只认主依据，才不会被语境变化灌满 ---
            shown_basis = used or primary or (same_en_hit[0] if same_en_hit else None)
            if shown_basis:
                m_lab, m_n, m_tot, m_ratio, cdist = rr[shown_basis]
                share = cdist.get(c, 0) / m_tot if m_tot else 0.0
                row['basis'] = shown_basis
                row['majority'] = {'label': m_lab, 'support': m_n, 'total': m_tot,
                                   'ratio': round(m_ratio, 3)}
                if c != m_lab:
                    row['own_share'] = {'count': cdist.get(c, 0), 'total': m_tot,
                                        'ratio': round(share, 3)}
            if verdict == 'BROKEN':
                pass
            elif not primary:
                # 主依据不成立。但若 BROKEN 扫描撞上了「同叶两个目标英文标签相同」的守卫，
                # 那仍是一条值得人看的中文命名不一致 —— 归 UNCERTAIN，不静默吞掉。
                if same_en_hit:
                    verdict = 'UNCERTAIN'
                    evidence.append('same-en-label')
                    owner_key = owner_key or same_en_hit[1]
                elif L['doc'] or a.include_commands:
                    bucket['no-basis'] += 1
            else:
                m_lab, m_n, m_tot, m_ratio, cdist = rr[primary]
                share = cdist.get(c, 0) / m_tot if m_tot else 0.0
                cls = None if c == m_lab else classify(m_lab)
                if c == m_lab:
                    bucket['ok'] += 1
                elif cls:
                    bucket[cls] += 1
                elif share >= a.alien_ratio:
                    bucket['minority-but-supported'] += 1
                else:
                    verdict = 'UNCERTAIN'
                    if same_en_hit:
                        evidence.append('same-en-label')

            if not verdict:
                continue
            row['evidence'] = evidence
            row['verdict'] = verdict
            if owner_key:
                ow_ref = ref.get(owner_key, {})
                ow_basis = used or ('target+en' if 'target+en' in ow_ref else 'target')
                row['label_belongs_to'] = {
                    'key': owner_key, 'en_label': en_of(owner_key),
                    'majority_cn': ow_ref[ow_basis][0] if ow_basis in ow_ref else None}
            if row.get('majority'):
                row['suggested'] = row['majority']['label']
                row['suggested_is_english'] = not bool(CJK.search(row['suggested']))
            leaf_rows.append(row)
            if verdict == 'BROKEN':
                broken_here.append(row)

        # --- 轮转闭合：错位标签的「正主」自己也拿着外来标签的，一并升级 ---
        if broken_here:
            owners_hit = {r['label_belongs_to']['key'] for r in broken_here
                          if r.get('label_belongs_to')}
            for row in leaf_rows:
                if row['verdict'] == 'UNCERTAIN' and row['key'] in owners_hit:
                    row['verdict'] = 'BROKEN'
                    row['evidence'].append('rotation-closure')

        if not leaf_rows:
            continue
        gid = len(groups)
        groups.append({
            'group': gid, 'repo': rname, 'pack': pack, 'path': path,
            'batch_path': batch_path(path),
            'broken': sum(1 for r in leaf_rows if r['verdict'] == 'BROKEN'),
            'uncertain': sum(1 for r in leaf_rows if r['verdict'] == 'UNCERTAIN'),
            # 整叶的英中配对，逐位对齐 —— 复核时直接看这一列就能判 en_label 对不对
            'leaf_pairs': [{'key': L['key'], 'target': L['target'],
                            'en': ea[i], 'cn': L['label']}
                           for i, L in enumerate(ls) if L['norm']]})
        for r in leaf_rows:
            r.update({'group': gid, 'repo': rname, 'pack': pack, 'path': path,
                      'batch_path': batch_path(path)})
            findings.append(r)

    broken = [f for f in findings if f['verdict'] == 'BROKEN']
    uncertain = [f for f in findings if f['verdict'] == 'UNCERTAIN']
    b_leaves = len({(f['repo'], f['pack'], f['path']) for f in broken})
    u_leaves = len({(f['repo'], f['pack'], f['path']) for f in uncertain})

    # ---------------- 输出 ----------------
    print(f'\n标签-目标错位检测  basis={a.basis}  min-support={a.min_support}'
          f'  min-ratio={a.min_ratio}  alien-ratio={a.alien_ratio}')
    print(f'  扫过链接 {scanned}（带标签 {labelled}，文档型目标 {doclike}）'
          f'，语境键 {len(ctx)} 个 / 目标 id {len(any_)} 个')
    print(f'  BROKEN     {len(broken):>5} 处 / {b_leaves} 叶')
    print(f'  UNCERTAIN  {len(uncertain):>5} 处 / {u_leaves} 叶')
    print(f'  （只计数不列出：与多数一致 {bucket["ok"]}、合法变体 {bucket["variant"]}、'
          f'标签未译 {bucket["untranslated"]}、少数但有支持 {bucket["minority-but-supported"]}、'
          f'无判定依据 {bucket["no-basis"]}）')

    by_pack = Counter((f['repo'], f['pack']) for f in broken)
    if by_pack:
        print('\nBROKEN 按包分布:')
        for (r, p), n in by_pack.most_common():
            print(f'   {n:>5}  {r}/{p}')

    shown = [g for g in groups if g['broken']][:a.show]
    for g in shown:
        print(f'\n组 #{g["group"]}  {g["repo"]}/{g["pack"]}   BROKEN {g["broken"]}')
        print(f'  {g["path"]}')
        for f in findings:
            if f['group'] != g['group'] or f['verdict'] != 'BROKEN':
                continue
            ow = f.get('label_belongs_to') or {}
            print(f'   {f["key"]:<18} 英文 {str(f.get("en_label")):<26} 中文 {f["cn_label"]}')
            if f.get('suggested'):
                fix = f'→ 应为 {f["suggested"]}'
                if f.get('suggested_is_english'):
                    fix += '（⚠ 库内多数写法本身是英文，落笔前先定译名）'
            else:
                fix = f'→ 库内无多数中文写法，按英文标签 {f.get("en_label")!r} 定名'
            print(f'   {"":<18} {fix}'
                  + (f'   (该标签属 {ow.get("key")} / {ow.get("en_label")})' if ow else '')
                  + f'   [{",".join(f["evidence"])}]')
    n_bg = sum(1 for g in groups if g['broken'])
    if n_bg > len(shown):
        print(f'\n  … 另有 {n_bg - len(shown)} 组 BROKEN 未列出（用 --show 调整，或看 --out 的 JSON）')

    if uncertain and a.show:
        print(f'\nUNCERTAIN 抽样（有清晰多数写法、本叶偏离、但归属不到同叶任何目标 —— '
              f'多是命名不一致，不是错位）:')
        for f in uncertain[:a.show]:
            print(f'   {f["repo"]}/{f["pack"]} :: {f["path"][:80]}')
            print(f'     {f["key"]:<18} 英文 {str(f.get("en_label")):<24}'
                  f' 中文 {f["cn_label"]}  ←→ 库内多数 {f.get("suggested")}'
                  + (f'   [{",".join(f["evidence"])}]' if f['evidence'] else ''))
        if len(uncertain) > a.show:
            print(f'   … 另有 {len(uncertain) - a.show} 处，见 --out 的 JSON')

    if a.out:
        payload = {
            'params': {'repos': a.repo, 'pack': a.pack, 'basis': a.basis,
                       'min_support': a.min_support, 'min_ratio': a.min_ratio,
                       'alien_ratio': a.alien_ratio,
                       'include_commands': a.include_commands},
            'summary': {'links_scanned': scanned, 'labelled': labelled,
                        'doc_targets': doclike, 'context_keys': len(ctx),
                        'distinct_targets': len(any_),
                        'BROKEN': len(broken), 'broken_leaves': b_leaves,
                        'UNCERTAIN': len(uncertain), 'uncertain_leaves': u_leaves,
                        'not_listed': dict(bucket),
                        'broken_by_pack': {f'{r}/{p}': n for (r, p), n in by_pack.items()}},
            'per_pack': per_pack, 'groups': groups, 'findings': findings}
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')

    raise SystemExit(1 if broken else 0)


if __name__ == '__main__':
    main()
