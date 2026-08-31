# -*- coding: utf-8 -*-
"""三通道一致性闸：lang / compendium / 运行时硬编码 .mjs 三方对表。

本项目有三条互相独立的翻译通道，玩家在**同一块屏幕上同时**看到它们：

  ① Foundry i18n       `<repo>/lang/cn.json`
  ② Babele             `<repo>/compendium/cn/*.json`
  ③ 运行时硬编码补丁    `1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`

在此脚本之前没有任何检查比对过它们（`qa/` 下也没有任何脚本读 `.js/.mjs`）。
实测漂移见 `5-其他内容/reports/AUDIT-2026-08-12-multiagent.md` 第 2.3 / 2.4 节：

  * `SPELL.GESTURES.Fan`  lang=扇子 / compendium=扇形（英文闸下 扇形 : 扇子 = 44 : 0）。
    Blast / Ward / Ray / Surge / Conjure / Cone 同病。系统用
    `SPELL.NameFormatAdj = "{rune}{gesture}"` 在运行时拼法术名，
    于是玩家学到的是「手势：光线」、法术栏里出现的却是「火焰射线」。
  * `Stride` lang 写「跨步」而 compendium 写「步幅」（第 8 节已裁 Stride→步幅）。
  * `.mjs` 自己的注释写着「译名与 crucible lang 的 KNOWLEDGE.* 逐条对齐，
    改一处要两边一起改」，实测 31 条里已漂 2 条。

三节输出
--------
  A  **lang ↔ compendium**：对每个 lang 键，拿它的**英文值**去 compendium 英文侧
     做词闸，只在英文相同的叶子里比中文写法。判据必须带英文闸 ——
     PROJECT.md 反复强调裸词频既会漏也会误伤（库内「电能」25 / 「阶位」160
     加上英文闸才看出译的是 `electrical energy` / `Rank`，与那两个术语无关）。
     compendium 侧的竞争写法**由叶级对照给出**，不切 n-gram（2026-08-12 audit3 重写，
     见下「A/C 段的词对齐怎么来的」）。

A/C 段的词对齐怎么来的（2026-08-12 重写，替换掉原来的滑动窗口）
--------------------------------------------------------------
旧做法：在英文闸命中的叶子里切 2–4 字 CJK n-gram，按「闸内 df² / 全库 df」打分取头名。
它没有词边界概念，也分不清「译名」与「同句共现词」，于是 PROJECT.md 点名的三个坏例：

    Ward   → 神殿区（是 `Temple Ward` 的译名，被算到 `Ward` 头上）
    Aspect → 会替  （「…手势的法术**会替**换你先前的…」，跨词边界的碎片）
    Shoddy → 品质  （「粗糙**品质**」的后两字，是 `Quality` 的译名）

新做法分两步，**任何候选都必须先是一个词，再谈统计**：

  1. **候选只能来自词表**（有词表就不用猜词边界）。词表 = 叶级对照 + lang + glossary +
     `.mjs` + `@UUID` 标签，都是本项目自己写过的完整词：
       · `leaf`/`name` 整叶等值对照：英文叶去标记后**整条等于**该术语时，中文叶（去双语尾）
         就是它的译法。`name`/`label`/`tokenName`/`adjective` 路径记为 `name` 层 ——
         这正是 PROJECT.md 依据阶梯的最强证据。
       · `uuid` 按目标 id 配对的 `@UUID[同一目标]{标签}` 中英标签。
       · `lang` / `glossary` / `mjs` 三本词典。
       · `biling` 双语并列锚：中文里写成「中文 English」时，English 紧邻的那个中文词。
       · `compose` 组合式：多词术语用各部分的已知译名拼接（`Social`社交 + `Event`事件
         → 社交事件），拼出来的串必须在闸内真实出现才留。
     词表外的串一律不进候选 —— `会替`/`尺锥`/`裂的弧形`/`成该类型` 这类碎片从源头消失。
  2. **三道过滤**，专治「共现词冒充译名」：
       · 互斥度：候选必须主要出现在**没有** lang 写法的叶子里（伴随词 `品质` 与 `粗糙`
         同现 108/129，互斥度 0.16，直接出局）。
       · 特异度：闸内出现率 / 闸外出现率 ≥ `--min-lift`（默认 8）。
       · **归属反查**：候选若是**另一个**英文的已知译名，且那个英文出现在该候选几乎
         全部命中叶的英文侧，就判给那个英文（`神殿区`→`Temple Ward`、`爆炸瓶`→`Blast Flask`、
         `品质`→`Quality`），不再算作本词的竞争写法，只留在 `attributed` 里备查。
     另外每条给出 `gated_free`：术语作为**自由词**（不在更长的大写专名里）出现的叶子数 ——
     `Ward` 111 叶里绝大多数是 `Temple Ward`，这一列一眼看得出来。

  候选带 `tier`：`aligned`（叶级对照/词典，可直接下结论）> `composed` > `distributional`
  （只有词表 + 统计支持，**要看样例再定**）。`--legacy-mine` 可切回旧的 n-gram 挖掘做对照。
  B  **.mjs ↔ lang**：解析 `.mjs` 里的 `const` 替换表（KNOWLEDGE / LANGUAGES /
     ATTUNEMENTS / CALENDAR_MONTHS / EXACT …），按**英文**与两个仓库的 lang 对表。
     顺带给出该英文在 compendium 侧的写法 —— `CALENDAR_MONTHS.Steading=庄园`
     这种「lang 与 .mjs 一致、但两边一起错」的情况才不会漏。
  C  **枚举族内部一致性**：同一组枚举（品质五档 / 温度五档 / 威胁类别 / 防御名 /
     手势 / 屈折 …）在 lang 与 compendium 各处的写法是否成套。族由 lang 的键结构
     自动归纳（点号同父 + camelCase 同前缀），并检测「两档译成同一个词」。

用法
----
    python scan_cross_channel.py \
        --repo "<项目>\\1-Ember汉化插件" --repo "<项目>\\2-Crucible汉化插件" \
        --package "%LOCALAPPDATA%\\FoundryVTT\\Data\\systems\\crucible" \
        --package "%LOCALAPPDATA%\\FoundryVTT\\Data\\modules\\ember" \
        --mjs "<项目>\\1-Ember汉化插件\\scripts\\ember-hardcoded-cn.mjs" \
        --out "<报告目录>\\cross_channel.json"

`--package` 可省略（回落到 `<repo>/lang/en.json` 这份基准），也可以给多个 ——
与 `--repo` 的配对**按 lang 键重合度自动认领**，不依赖参数顺序。

判读要点
--------
  * `LANG_ORPHAN` 是最该先看的一档：lang 的写法在英文闸下**一处支持都没有**，
    而 compendium 有成规模的另一种写法。改 lang 那一边永远是改动面小的那边。
  * `LANG_MINORITY` 次之；`LANG_SPLIT` 是「看一眼」桶，不是判决 —— 两侧都有相当支持时，
    脚本不替人裁，按 PROJECT.md 的依据阶梯（name 字段 > 同卷已译页 > 全库多数）自己定。
  * `LANG_NO_SUPPORT`：英文闸下 lang 写法一处支持都没有，但**也没有**任何够格的对家写法。
    多半是该英文在 compendium 里根本没有对应译名（纯 UI 词、枚举 id），不是缺陷；
    真要下结论就用 `term_gate.py` 单查。
  * **看 `tier`**：`aligned` 的候选有叶级对照/词典直接支持，可以直接下结论；
    `distributional` 只有词表 + 统计支持，**必须先看样例**再定。
  * 计数列的含义：`gated_with_cn` 英文闸命中且有中文的叶子；`gated_free` 其中术语作为
    自由词出现（不在更长大写专名内）的叶子；`lang_support` 这些叶子里含 lang 写法的数量
    （这一列一直是可信的）；候选的 `excl_df` 是「含该候选且**不含** lang 写法」的叶子数。
  * `by_repo` / `--exclude-pack` ：dnd5e 侧（`ember.adventure.json`、`ember.dnd5e-*`）
    按项目作用域**对 crucible 主线没有权威**，术语本来就可以不同。
    只看主线时加 `--exclude-pack 'dnd5e|ember\\.adventure\\.json'`。
  * 计数单位是**叶子数**（同一叶里出现多次只算一次），与 `term_gate.py` 一致。
  * 两个仓库全跑一次约 22 秒。

只读：本脚本不写任何译文，唯一可能写出的文件是 `--out` 指定的报告。
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 正则与常量

CJK_RUN = re.compile(r"[一-鿿]{2,}")
CJK_ANY = re.compile(r"[一-鿿]")
# 一趟剥掉标记：HTML 标签 / enricher 的方括号目标 / [[/…]] 掷骰块 / HTML 实体。
# 只吃方括号里的目标，`{标签}` 是可见文字要留着。
MARKUP = re.compile(r"<[^>]*>|@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|&[A-Za-z#0-9]+;")
WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

# babele/mapping 的保留字，不是译文
SKIP_KEYS = {"_id", "path", "_variants", "_when", "mapping", "_identity"}

# 英文侧可当术语的形状：首字母大写、1–4 个词、不含数字与占位符
TERM_EN = re.compile(r"^[A-Z][A-Za-z'’\-]*(?: [A-Za-z'’\-]+){0,3}$")

# n-gram 预选阶段先扔掉的高频套话（只影响预选名额，最终排序仍靠全库特异度）
STOP_GRAMS = {
    "的一", "一个", "可以", "如果", "目标", "伤害", "攻击", "生物", "效果", "检定",
    "动作", "这个", "他们", "以及", "或者", "因为", "所有", "能够", "进行", "使用",
    "获得", "造成", "一次", "一名", "之一", "时候", "状态", "属性", "角色", "玩家",
    "队伍", "区域", "位置", "地点", "事件", "然后", "可能", "需要", "不会", "已经",
    "没有", "其他", "其中", "一些", "当前", "每个", "任何", "例如", "包括", "它们",
    "自己", "你的", "我们", "这些", "那些", "并且", "但是", "而且", "同时", "之后",
    "之前", "开始", "结束", "出现", "发现", "看到", "知道", "上的", "中的", "不能",
    "取决于", "相当于", "等同于", "通常", "例如", "至少", "至多", "最多", "以外",
    "以内", "范围内", "相同", "不同", "一样", "其余", "另一", "各自", "分别",
}

# 以这些字开头或结尾的 n-gram 基本都是句子碎片（「的品质」「尚未定」），不是译名。
# 只降权不丢弃 —— 万一某个译名真长这样，还看得见。
EDGE_FUNC = set("的了着地得而就也被把从对以之该此并但很更再又所且则与和或及若因当如在是")

SEV = {  # 排序用；数值越大越该先看
    "LANG_ORPHAN": 40, "MJS_LANG_DRIFT": 40, "FAMILY_COLLISION": 40,
    "LANG_MINORITY": 30, "FAMILY_SPLIT": 30, "MJS_ORPHAN_CN": 30,
    "LANG_SPLIT": 20, "LANG_NO_SUPPORT": 15, "LANG_WEAK_SIGNAL": 12, "MJS_NO_LANG": 10,
    "AGREE": 0, "INSUFFICIENT": 0, "TOO_COMMON": 0,
}

# ---- 叶级对照用（2026-08-12 重写 A/C 段词对齐时新增）

# 双语并列的英文尾巴：`申特月神殿 Shent Moon Temple` → 去掉后半截
LATIN_TAIL = re.compile(r"[\s　]*[（(\[]?[A-Za-z][A-Za-z0-9'’\-.,:;!?&/ ]*[)\]）]?[\s　]*$")
# 一个「词」的形状：纯 CJK，允许人名里的间隔号
CN_UNIT = re.compile(r"^[一-鿿]+(?:·[一-鿿]+)*$")
# 拆词表单元用的强分隔符（**不含** `·`，那是名字内部的字符）
UNIT_SPLIT = re.compile(r"[，,。；;：:！？!?…、（）()【】《》\[\]|/\\\n\r\t“”‘’\"'—–~＋+*#§ 　]+")
UUID_LABEL = re.compile(r"@[A-Za-z]+\[([^\]]*)\]\{([^}]*)\}")
# 路径末段属于「名字类」字段 —— 依据阶梯里最强的一层
NAME_TAILS = {"name", "tokenName", "label", "adjective", "subtitle", "prefix", "suffix"}
# 大写但不构成专名的功能词：`The Ward` 里的 Ward 仍是自由词
FUNC_CAPS = {
    "A", "An", "The", "This", "That", "These", "Those", "Its", "It", "Their", "His", "Her",
    "Your", "You", "Our", "My", "In", "On", "At", "To", "Of", "For", "From", "With", "By",
    "And", "Or", "But", "If", "When", "While", "As", "So", "Then", "Than", "Because",
    "Since", "Though", "Although", "Unless", "Until", "After", "Before", "During", "Once",
    "Each", "Every", "Any", "All", "Some", "No", "Not", "One", "Two", "Three", "Both",
    "Either", "Neither", "Such", "More", "Most", "Other", "Another", "Only", "Even",
    "Just", "Also", "However", "Here", "There", "Now", "Yet", "Still", "Very", "Much",
    "Many", "Few", "Several", "Is", "Are", "Was", "Were", "Be", "Been", "Can", "Could",
    "May", "Might", "Must", "Should", "Would", "Will", "Do", "Does", "Did", "Has", "Have",
    "Had", "Make", "Makes", "Use", "Using", "Add", "Gain", "Roll", "See", "Note", "Each",
    "Where", "Which", "Who", "Whose", "What", "How", "Why", "Whenever", "Whether",
}
SRC_WEIGHT = {"name": 6, "leaf": 4, "uuid": 4, "lang": 3, "glossary": 3, "mjs": 3,
              "biling": 2, "compose": 1}


# ---------------------------------------------------------------- 基础工具

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def flatten(obj, prefix=""):
    """lang json → 扁平点分键。与 lang_gap.py 同一套语义。"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, str):
        out[prefix] = obj
    return out


def foundry_lookup(tr, key):
    """复刻 `foundry.utils.getProperty`：先整键命中，否则按点逐级下探。

    与 `qa/lang_gap.py` 里的同名函数保持一致 —— 校验必须复刻被验证系统的查找语义，
    否则「键带点、值却是嵌套对象」这种 Foundry 根本取不到的形态会被当成已翻译
    （2026-08-12 那次 ember 486 键里 372 个静默失效就是这么来的）。
    """
    if key in tr:
        v = tr[key]
        return v if isinstance(v, str) else None
    node = tr
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node if isinstance(node, str) else None


def cjk_ngrams(s, lo=2, hi=4):
    """一条中文里所有 2–4 字连续 CJK 片段（按叶子去重，用于算 document frequency）。"""
    out = set()
    for seg in CJK_RUN.findall(s):
        L = len(seg)
        for k in range(lo, hi + 1):
            for i in range(L - k + 1):
                out.add(seg[i:i + k])
    return out


# ---------------------------------------------------------------- 语料装载

def walk_pack(en, cn, path, out):
    """成对遍历英/中包。**整份文件都走**，不只 `entries` ——

    顶层的 `label` 与 `folders` 从来不在既有普查里（审计报告第 5 节点名的结构性盲点），
    而合集侧边栏与文件夹名玩家天天看得到。
    """
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk_pack(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk_pack(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def load_leaves(repos, exclude_pack):
    """返回 [(repo_name, pack, path, en, cn), …]"""
    leaves = []
    for repo in repos:
        rname = Path(repo).name
        en_dir = Path(repo) / "compendium" / "en"
        cn_dir = Path(repo) / "compendium" / "cn"
        if not en_dir.is_dir():
            print(f"  ! {rname} 没有 compendium/en，跳过", file=sys.stderr)
            continue
        for f in sorted(en_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            if any(rx.search(f.name) for rx in exclude_pack):
                continue
            en = load_json(f)
            cf = cn_dir / f.name
            cn = load_json(cf) if cf.is_file() else {}
            sub = []
            walk_pack(en, cn, [], sub)
            for p, e, c in sub:
                leaves.append((rname, f.name, p, e, c))
    return leaves


def load_lang_pairs(repo, packages):
    """(英文 lang, 中文 lang 原始树, 英文来源, keep_english)。英文优先取 Foundry 包里的当前版本。"""
    repo = Path(repo)
    cn_raw = load_json(repo / "lang" / "cn.json")
    cn_keys = set(flatten(cn_raw))

    best, best_n, best_src = None, -1, None
    for pkg in packages:
        f = Path(pkg) / "lang" / "en.json"
        if not f.is_file():
            continue
        flat = flatten(load_json(f))
        n = len(set(flat) & cn_keys)
        if n > best_n:
            best, best_n, best_src = flat, n, str(f)
    if best is None or best_n == 0:
        f = repo / "lang" / "en.json"
        if f.is_file():
            best, best_src = flatten(load_json(f)), str(f)
        else:
            best, best_src = {}, "(无英文 lang)"

    keep = repo / "lang" / "lang_keep_english.json"
    keep_english = set(load_json(keep)) if keep.is_file() else set()
    return best, cn_raw, best_src, keep_english


# ---------------------------------------------------------------- 英文词闸

def build_gate(leaves, terms, ignore_case):
    """把英文叶子按词切开，用 1–4 词短语查表 —— 比 N 个正则各扫一遍全库快两个数量级。

    切词前先剥标记，所以 `<section class="block …">`、`@UUID[…]` 里的目标 id
    不会被当成正文里的英文。默认**区分大小写**（PROJECT.md：小写 stride 是动词，
    「所跨步踏过的土地」那处与属性 Stride 不是一回事）。

    同一段英文在孪生包里逐字节重复出现，故按英文原文缓存匹配结果。
    """
    if not terms:
        return {}
    tset = {t.lower() for t in terms} if ignore_case else set(terms)
    firsts = {t.split(" ", 1)[0] for t in tset}
    maxn = max(len(t.split()) for t in tset)
    gated = defaultdict(list)
    cache = {}
    for idx, lf in enumerate(leaves):
        en = lf[3]
        if not en:
            continue
        hit = cache.get(en)
        if hit is None:
            toks = WORD.findall(MARKUP.sub(" ", en))
            if ignore_case:
                toks = [t.lower() for t in toks]
            n = len(toks)
            seen = set()
            for i, t0 in enumerate(toks):
                if t0 not in firsts:
                    continue
                for k in range(1, maxn + 1):
                    if i + k > n:
                        break
                    ph = t0 if k == 1 else " ".join(toks[i:i + k])
                    if ph in tset:
                        seen.add(ph)
            hit = tuple(seen)
            cache[en] = hit
        for ph in hit:
            gated[ph].append(idx)
    return gated


def corpus_df_exact(strings, cn_all):
    """给定若干完整中文串，算它们在全库出现在多少条叶子里（C 层子串搜索）。"""
    out = {}
    for s in strings:
        if not s:
            out[s] = 0
            continue
        out[s] = sum(1 for cn in cn_all if s in cn)
    return out


# ================================================================ 叶级对照（A/C 段的新词对齐）
#
# 这一整块是 2026-08-12 audit3 重写的产物，用来替换掉「滑动窗口 n-gram」。
# 原则：**候选先得是个词**（来自本项目自己写过的词表），再谈统计。


def strip_bilingual(s):
    """去标记、去双语并列的英文尾巴、去包裹标点，剩下的才是「这条中文写的词」。"""
    s = MARKUP.sub(" ", s).replace(" ", " ").strip()
    t = LATIN_TAIL.sub("", s).strip()
    t = t.strip(" 　“”‘’\"'（）()[]【】<>《》：:，,。.、;；!！?？-—–·")
    return t if t and CJK_ANY.search(t) else ""


def cn_units(s, max_len=14):
    """一条中文里可以进词表的**完整单元**：整条（去双语尾）＋按强分隔符切出的段。

    这里**不切 n-gram** —— 每个单元都是本项目在某处当作一个词写下来的东西。
    """
    t = strip_bilingual(s)
    if not t:
        return []
    out = []
    if 2 <= len(t) <= max_len and CN_UNIT.match(t):
        out.append(t)
    for part in UNIT_SPLIT.split(t):
        part = part.strip("·")
        if 2 <= len(part) <= max_len and CN_UNIT.match(part):
            out.append(part)
    return out


def norm_en(s):
    """英文键归一：压空白、去尾标点、末词单数化、全小写。用于跨层查同一个词。"""
    s = re.sub(r"\s+", " ", (s or "")).strip().strip(".,:;!?…")
    if not s:
        return ""
    ws = s.split(" ")
    ws[-1] = singular(ws[-1])
    return " ".join(w.lower() for w in ws)


class Evidence:
    """叶级对照 + 三本词典汇总出来的证据层。

    align[en][cn]        某英文 → 中文写法（整叶等值 / UUID 标签 / lang / glossary / mjs / 双语锚）
    src[(en, cn)]        这条对照来自哪些层（name 最强，见 SRC_WEIGHT）
    owner[cn]            这个中文写法是**哪些英文**的已知译名 —— 归属反查用
    parts[单词]          单词级译名，给多词术语拼组合候选
    lex                  词表（所有层的中文单元），候选只能从这里出
    """

    def __init__(self):
        self.align = defaultdict(Counter)
        self.src = defaultdict(set)
        self.owner = defaultdict(Counter)
        self.parts = defaultdict(Counter)
        self.lex = defaultdict(set)

    def add(self, en, cn, source, n=1):
        en = (en or "").strip()
        if not en or not cn:
            return
        key = norm_en(en)
        if not key:
            return
        self.align[key][cn] += n
        self.src[(key, cn)].add(source)
        self.owner[cn][en] += n * SRC_WEIGHT.get(source, 1)
        self.lex[cn].add(source)
        if " " not in key:
            self.parts[key][cn] += n * SRC_WEIGHT.get(source, 1)

    def add_lex(self, cn, source):
        if cn:
            self.lex[cn].add(source)


def build_evidence(leaves, lang_all, glossary, mjs_entries, max_en=48):
    """装配证据层。只读语料，不做任何统计判断。"""
    ev = Evidence()

    # ① 整叶等值对照 —— 英文叶去标记后整条就是那个词时，中文叶就是它的译法。
    #    路径末段是 name/label/tokenName/adjective 的记为 `name` 层（依据阶梯最强的一层）。
    #
    #    ⚠ 词表**只收这种整条即术语的叶子**。早先试过把散文叶按标点切段也塞进词表，
    #    结果「这场社交事件的核心」这类句子片段成了「词」，等于把 n-gram 碎片从后门放回来
    #    （词表一下涨到 13 万条）。散文不提供词边界，就不要从散文里取词。
    for repo, pack, path, en, cn in leaves:
        if not cn:
            continue
        e = MARKUP.sub(" ", en).strip()
        if not e or len(e) > max_en or not re.search(r"[A-Za-z]", e):
            continue
        u = strip_bilingual(cn)
        if not u or len(u) > 14 or not CN_UNIT.match(u):
            continue
        tail = path.rsplit(".", 1)[-1]
        src = "name" if tail in NAME_TAILS else "leaf"
        ev.add(e, u, src)
        for part in cn_units(cn):
            ev.add_lex(part, src)

    # ② @UUID[同一目标]{标签} 的中英标签配对
    for repo, pack, path, en, cn in leaves:
        if not cn or "@" not in en:
            continue
        en_lab = dict(UUID_LABEL.findall(en))
        if not en_lab:
            continue
        cn_lab = dict(UUID_LABEL.findall(cn))
        for tgt, lab in en_lab.items():
            c = cn_lab.get(tgt)
            if not c:
                continue
            u = strip_bilingual(c)
            lab = lab.strip()
            if u and len(u) <= 14 and CN_UNIT.match(u) and len(lab) <= max_en:
                ev.add(lab, u, "uuid")

    # ③ 三本词典
    for rname, pairs in lang_all.items():
        for k, (en, cn) in pairs.items():
            u = strip_bilingual(cn)
            if u and len(u) <= 14 and CN_UNIT.match(u) and len(en.strip()) <= max_en:
                ev.add(en, u, "lang")
    for en, cn in (glossary or {}).items():
        if not isinstance(cn, str):
            continue
        u = strip_bilingual(cn)
        if u and len(u) <= 14 and CN_UNIT.match(u) and len(en.strip()) <= max_en:
            ev.add(en, u, "glossary")
    for _p, _t, en, cn in mjs_entries:
        u = strip_bilingual(cn)
        if u and len(u) <= 14 and CN_UNIT.match(u) and len(en.strip()) <= max_en:
            ev.add(en, u, "mjs")
    return ev


class LexIndex:
    """词表 → 全库命中索引。

    做法是**按首二字分桶**再逐位试匹配，等价于一次 Aho-Corasick，但代码只有十行、
    没有外部依赖。全库唯一中文文本各扫一遍即可，之后所有计数都是集合运算。
    """

    def __init__(self, words, leaves, max_len=14):
        self.words = sorted(w for w in words if 2 <= len(w) <= max_len)
        self.wid = {w: i for i, w in enumerate(self.words)}
        by2 = defaultdict(list)
        for w in self.words:
            by2[w[:2]].append(w)
        for v in by2.values():
            v.sort(key=len, reverse=True)
        self.by2 = dict(by2)

        self.leaf_tid = [-1] * len(leaves)
        tid, texts = {}, []
        for i, lf in enumerate(leaves):
            cn = lf[4]
            if not cn:
                continue
            t = tid.get(cn)
            if t is None:
                t = len(texts)
                tid[cn] = t
                texts.append(cn)
            self.leaf_tid[i] = t
        self.texts = texts
        self.hits = [self._scan(t) for t in texts]

        leaves_per_text = Counter(t for t in self.leaf_tid if t >= 0)
        self.corpus_df = Counter()
        for t, n in leaves_per_text.items():
            for wi in self.hits[t]:
                self.corpus_df[wi] += n

    def _scan(self, txt):
        by2, out, n = self.by2, set(), len(txt)
        for i in range(n - 1):
            lst = by2.get(txt[i:i + 2])
            if lst is None:
                continue
            for w in lst:
                if txt.startswith(w, i):
                    out.add(self.wid[w])
        return tuple(sorted(out))

    def gated_counts(self, idxs):
        """英文闸命中的叶子里，每个词表词各出现在多少叶（叶级 df）。"""
        per_text = Counter()
        for i in idxs:
            t = self.leaf_tid[i]
            if t >= 0:
                per_text[t] += 1
        c = Counter()
        for t, n in per_text.items():
            for wi in self.hits[t]:
                c[wi] += n
        return c

    def df(self, cn):
        wi = self.wid.get(cn)
        if wi is not None:
            return self.corpus_df.get(wi, 0)
        return None


# ---- 自由词 / 受缚词：`Ward` 与 `Temple Ward` 不是一回事

def _free_regex(term):
    return re.compile(r"(?<![A-Za-z'’\-])" + re.escape(term) + r"(?![A-Za-z'’\-])")


def occurs_free(term, en_text, rx=None, stripped=False):
    """术语在这条英文里是否**至少出现一次自由词**（左右都不是构成专名的大写词）。

    `Temple Ward` / `Blast Flask` 里的 Ward / Blast 是受缚的 —— 那两处的中文译的是
    整个专名，不是这个词。句首的大写词（`. Ward is…`）与 The/A/In 这类功能词不算构成专名。
    """
    txt = en_text if stripped else MARKUP.sub(" ", en_text)
    rx = rx or _free_regex(term)
    for m in rx.finditer(txt):
        left, right = txt[:m.start()], txt[m.end():]
        lm = re.search(r"([A-Za-z][A-Za-z'’\-]*)[  ]+$", left)
        bound_l = False
        if lm:
            w = lm.group(1)
            pre = left[:lm.start()].rstrip()
            sentence_start = (pre == "" or pre[-1] in ".!?:;\"”)]|>》」")
            bound_l = w[:1].isupper() and w not in FUNC_CAPS and not sentence_start
        rm = re.match(r"[  ]+([A-Za-z][A-Za-z'’\-]*)", right)
        bound_r = bool(rm) and rm.group(1)[:1].isupper() and rm.group(1) not in FUNC_CAPS
        if not (bound_l or bound_r):
            return True
    return False


# ---- 候选生成

TIER_RANK = {"aligned": 3, "composed": 2, "distributional": 1}
ANCHOR_MAX = 14


def compose_candidates(en, parts, max_out=12):
    """多词术语的组合候选：`Social`社交 + `Event`事件 → 社交事件。

    只产生候选，是否成立由「闸内真的出现过」来验。
    """
    ws = [w for w in norm_en(en).split(" ") if w]
    if not (2 <= len(ws) <= 3):
        return []
    opts = []
    for w in ws:
        r = parts.get(w)
        if not r:
            return []
        opts.append([c for c, _ in r.most_common(2)])
    out = []
    stack = [""]
    for o in opts:
        stack = [pre + c for pre in stack for c in o]
    out.extend(stack)
    if len(ws) == 2:
        out.extend(b + a for a in opts[0] for b in opts[1])
    seen, uniq = set(), []
    for c in out:
        if c not in seen and 2 <= len(c) <= ANCHOR_MAX:
            seen.add(c)
            uniq.append(c)
    return uniq[:max_out]


def anchor_candidates(term, idxs, leaves, lex, limit=200):
    """双语并列锚：中文里写成「中文 English」时，紧贴英文的那个中文词。

    只收**已经在词表里**的锚 —— 否则会把「…只能拥有一个 Ward…」这种残留英文
    前面的整句话收进来。
    """
    # `(?!\s+[A-Z])`：中文写「爆炸瓶 Blast Flask」时，紧贴 Blast 的中文译的是整个
    # `Blast Flask`，不是 `Blast`。后面还跟着大写词就说明术语只是长专名的头，弃。
    rx = re.compile(r"([一-鿿·]{2,%d})[\s　]*[（(]?%s(?![A-Za-z])(?![\s　]+[A-Z])"
                    % (ANCHOR_MAX, re.escape(term)))
    c = Counter()
    for i in idxs[:limit]:
        cn = leaves[i][4]
        if not cn:
            continue
        for g in rx.findall(cn):
            g = g.strip("·")
            if g and g in lex.wid:
                c[g] += 1
    return [g for g, n in c.most_common(6) if n >= 2]


def attribute_owner(cn, en_norm, ev, cand_idxs, leaves, en_cache, min_cov=0.8, max_owners=8):
    """归属反查：这个中文写法是不是**别的**英文的译名？

    `神殿区` 的已知归属是 `Temple Ward`、`品质` 是 `Quality`、`爆炸瓶` 是 `Blast Flask`。
    只要那个英文出现在该候选几乎全部命中叶的英文侧，这个写法就该判给它，
    不是本词的竞争写法 —— 这是旧 n-gram 版最致命的三个坏例的共同病根。
    """
    if not cand_idxs:
        return None
    owners = [(p, w) for p, w in ev.owner.get(cn, {}).items() if norm_en(p) != en_norm]
    if not owners:
        return None
    owners.sort(key=lambda pw: (-pw[1], -len(pw[0])))
    for p, _w in owners[:max_owners]:
        if len(p) < 2 or not re.search(r"[A-Za-z]", p):
            continue
        # 归属反查要**忽略大小写并认复数**：库里既有 `Blast Flask` 也有 `blast flasks`，
        # 逐字匹配会让 `爆炸瓶` 逃过反查、冒充 `Blast` 的译名。
        # 复数与所有格都要认：`<h4>Gamemaster's Summary</h4>` 里的 Gamemaster 若因为后面
        # 那个撇号而匹配不上，`游戏主持人` 就会冒充成本词的译名（实测踩到）。
        rx = re.compile(r"(?<![A-Za-z’])" + re.escape(p) + r"(?:e?s|'s|’s)?(?![A-Za-z])",
                        re.IGNORECASE)
        hit = 0
        for i in cand_idxs:
            e = en_cache.get(i)
            if e is None:
                e = MARKUP.sub(" ", leaves[i][3])
                en_cache[i] = e
            if rx.search(e):
                hit += 1
        if hit / len(cand_idxs) >= min_cov:
            return {"en": p, "coverage": round(hit / len(cand_idxs), 2)}
    return None


def build_candidates(en, idxs, leaves, ev, lex, n_cn_leaves, min_alt, min_lift,
                     free_idxs, en_cache, top=10, probe=60, max_common=800):
    """某个英文术语在 compendium 侧的**候选中文写法**（与 lang 值无关，A/B 两段共用）。

    返回的每条带 tier / sources / gated_df / corpus_df / lift / attributed_to。
    互斥度过滤依赖 lang 值，放在 `pick_alts` 里做。
    """
    gated_n = len(idxs)
    if not gated_n:
        return []
    en_n = norm_en(en)
    gcounts = lex.gated_counts(idxs)
    cands = {}

    def put(cn, tier, sources):
        if not cn or len(cn) < 2:
            return
        d = cands.get(cn)
        if d is None:
            wi = lex.wid.get(cn)
            gd = gcounts.get(wi, 0) if wi is not None else \
                sum(1 for i in idxs if cn in leaves[i][4])
            cdf = lex.corpus_df.get(wi) if wi is not None else None
            if cdf is None:
                cdf = sum(1 for lf in leaves if lf[4] and cn in lf[4])
            cands[cn] = {"cn": cn, "tier": tier, "sources": set(sources),
                         "gated_df": gd, "corpus_df": max(cdf, gd)}
        else:
            d["sources"] |= set(sources)
            if TIER_RANK[tier] > TIER_RANK[d["tier"]]:
                d["tier"] = tier

    # ① 叶级对照 / 词典（最强）
    for cn, _n in ev.align.get(en_n, {}).items():
        put(cn, "aligned", ev.src.get((en_n, cn), {"align"}))
    # ② 双语并列锚
    for cn in anchor_candidates(en, idxs, leaves, lex):
        put(cn, "aligned", {"biling"})
    # ③ 组合式
    for cn in compose_candidates(en, ev.parts):
        put(cn, "composed", {"compose"})
    # ④ 词表内、闸内够格的分布式候选
    for wi, gd in gcounts.most_common(probe):
        if gd < min_alt:
            break
        put(lex.words[wi], "distributional", {"lexicon"})

    outside_n = max(n_cn_leaves - gated_n, 1)
    out = []
    for d in cands.values():
        gd, cdf = d["gated_df"], d["corpus_df"]
        outside = max(cdf - gd, 0)
        rate_in = gd / gated_n
        rate_out = outside / outside_n
        d["lift"] = round(rate_in / rate_out, 1) if rate_out else (999.0 if gd else 0.0)
        d["score"] = round(gd * min(d["lift"], 999) ** 0.5, 2)
        if d["tier"] == "distributional":
            if gd < min_alt or d["lift"] < min_lift:
                continue
            # 全库通用词（游戏主持人 1111 叶 / 区域 2050 / 成功 2188）没有叶级对照撑着，
            # 就不能说它是**这个**术语的译法 —— 它只是碰巧同屏的常用词。
            if cdf > max_common:
                continue
            # 该候选命中的叶子里，术语基本从不作自由词出现 → 它译的是更长的那个专名
            hits = [i for i in idxs if d["cn"] in leaves[i][4]]
            free_ratio = (sum(1 for i in hits if i in free_idxs) / len(hits)) if hits else 0
            if free_ratio < 0.2:
                d["attributed_to"] = {"en": "(更长的大写专名)", "coverage": 1 - free_ratio}
                out.append(d)
                continue
        else:
            hits = [i for i in idxs if d["cn"] in leaves[i][4]]
        # 直接对照到**本词**的（name/leaf/uuid/lang/glossary/mjs 层）不做归属反查 ——
        # 那些层本来就是「这个英文的中文写法」的直接证据。
        direct = bool(d["sources"] & {"name", "leaf", "uuid", "lang", "glossary", "mjs"})
        if not direct and hits:
            att = attribute_owner(d["cn"], en_n, ev, hits, leaves, en_cache)
            if att:
                d["attributed_to"] = att
        out.append(d)

    for d in out:
        d["sources"] = sorted(d["sources"])
    out.sort(key=lambda d: (-TIER_RANK[d["tier"]], -d["gated_df"], -d["score"]))
    return out[:max(top, 10)]


# ================================================================ 旧的 n-gram 挖掘（--legacy-mine）

def mine_preselect(idxs, leaves, cap, preselect=40):
    """从英文闸命中的叶子里挖 compendium 侧中文写法的**候选**。

    两个处理：
      * **按中文原文去重**再计 df。孪生包（`ember.adventure` / `ember.crucible-adventure`）
        与预生角色包里同一段译文逐字节重复出现，不去重的话，重复簇里的句子碎片
        会盖过真正的译名。
      * 叶子多时等距抽样（结果稳定）。真正报出去的计数在 `rank_candidates` 里
        对**全量**闸内叶子重算，所以抽样不影响报出来的数字。
    """
    use = idxs
    if cap and len(idxs) > cap:
        step = len(idxs) / cap
        use = [idxs[int(i * step)] for i in range(cap)]
    texts, seen = [], set()
    for i in use:
        cn = leaves[i][4]
        if not cn:
            continue
        cn = cn if len(cn) <= 4000 else cn[:4000]
        if cn in seen:
            continue
        seen.add(cn)
        texts.append(cn)
    df = Counter()
    for cn in texts:
        df.update(cjk_ngrams(cn))
    cands = [(g, c) for g, c in df.items() if c >= 2 and g not in STOP_GRAMS]
    cands.sort(key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return cands[:preselect], texts


def neighbor_diversity(g, texts, cap=3):
    """候选左右邻字的种数（取小的那个）。

    这一项是分辨「译名」与「重复句子里的碎片」的关键：`未能抵抗` 在全部 38 条里
    左邻永远是「对」、右邻永远是「该」，因为它只是同一句模板的碎片；而真正的译名
    `基础` 会出现在「10 基础伤害」「基础消耗」等各种上下文里，左右邻字都散。
    """
    L, R = set(), set()
    for t in texts:
        i = t.find(g)
        while i >= 0:
            L.add(t[i - 1] if i > 0 else "^")
            j = i + len(g)
            R.add(t[j] if j < len(t) else "$")
            if len(L) >= cap and len(R) >= cap:
                return cap
            i = t.find(g, i + 1)
    return min(len(L), len(R))


def rank_candidates(cands, bg_df, idxs, leaves, texts, en, owners, top=10):
    """排序：score = df_闸内² / df_全库 × 邻字多样性 × 同族兄弟惩罚，入选项再重算精确 df。

    单看 df 会让「伤害」「目标」这类套话霸榜；除以全库 df 之后，只在英文闸命中的
    叶子里才出现的那个词才会浮上来 —— 正是我们要的对家写法。

    `owners` 是「中文写法 → 它是哪些英文的 lang 译名」。同一族枚举的成员总是挤在
    同一句里（「粗糙 / 标准 / 精良 / 卓越 / 大师级」），不压一下的话，兄弟档的译名
    会被当成本档的竞争写法。
    """
    scored = []
    for g, c in cands:
        gdf = max(bg_df.get(g, c), c)
        pen = 1.0 if (c <= 3 or neighbor_diversity(g, texts) >= 2) else 0.2
        own = owners.get(g)
        if own and en not in own:
            pen *= 0.35        # 这写法是**别的**英文的 lang 译名，多半是同族兄弟
        if g[0] in EDGE_FUNC or g[-1] in EDGE_FUNC:
            pen *= 0.15        # 句子碎片
        scored.append((round(c * c / gdf * pen, 2), c, g, gdf, pen))
    scored.sort(key=lambda t: (-t[0], -t[1], -len(t[2])))
    kept = []
    for score, c, g, gdf, pen in scored:
        exact = sum(1 for i in idxs if g in leaves[i][4])
        item = {"cn": g, "gated_df": exact, "corpus_df": gdf,
                "score": round(exact * exact / max(gdf, exact) * pen, 2)}
        # 同一族嵌套 n-gram（扇形 / 扇形手势）只留一个：优先留覆盖面大的那个短串，
        # 但短串特异度掉太多（施法 之于 扇形施法）时仍留长串。
        nested = [k for k in kept if g in k["cn"] or k["cn"] in g]
        if nested:
            k = nested[0]
            if item["gated_df"] > k["gated_df"] and item["score"] >= k["score"] * 0.5:
                kept[kept.index(k)] = item
            continue
        kept.append(item)
        if len(kept) >= top:
            break
    kept.sort(key=lambda d: (-d["score"], -d["gated_df"]))
    return kept


# ---------------------------------------------------------------- A：lang ↔ compendium

def judge(lang_hits, best, gated_cn, min_gated, min_alt):
    """判据只看三个数：英文闸命中叶数、lang 写法的支持数、对家写法的**互斥**支持数。

    对家的计数用 `excl_df`（含该写法且**不含** lang 写法的叶子）而不是裸 df ——
    同一叶里两种写法都出现时，那不是竞争，是共现。
    """
    if gated_cn < min_gated:
        return "INSUFFICIENT"
    top_df = (best.get("excl_df", best["gated_df"]) if best else 0)
    if top_df < min_alt:
        # 没有够格的对家。lang 写法在闸内一处都没有时单独标一档：多半是该英文在
        # compendium 里根本没有对应译名（纯 UI 词 / 枚举 id），但值得人看一眼。
        if lang_hits == 0 and gated_cn >= max(min_gated, 4):
            return "LANG_NO_SUPPORT"
        return "AGREE"
    if lang_hits >= 0.85 * gated_cn:
        return "AGREE"
    ratio = 1.0 if lang_hits == 0 else lang_hits / (lang_hits + top_df)
    if lang_hits and ratio >= 0.66:
        return "AGREE"
    # 只有**叶级对照/词典/组合**撑着的候选才够格下「不一致」的结论。
    # 分布式候选（只有词表 + 统计）另立一档，交给人看样例 —— 这一档的假阳性率高，
    # 混进 ORPHAN/MINORITY 里就会重演「A 段不可信」。
    if best.get("tier") == "distributional":
        return "LANG_WEAK_SIGNAL"
    if lang_hits == 0:
        return "LANG_ORPHAN"
    if ratio <= 0.34:
        return "LANG_MINORITY"
    return "LANG_SPLIT"


def pick_alts(ranked_all, cn_val, idxs_cn, leaves, top, min_alt=2, excl_ratio=0.5):
    """从候选里挑出真正的「对家写法」。A/B 两段共用。

    * 与本方写法相同、或包含本方写法（`扇形施法` ⊃ `扇形`）的，是同一种译法，去掉。
    * 已被归属反查判给别的英文的（`神殿区`→`Temple Ward`），不是本词的对家，去掉。
    * 本方写法的**真子串**（`锥形` ⊂ `锥形区域`）是另一种更短的译法，要留，
      但计数只算「用了短写法、没用本方写法」的叶子，否则短串必然全覆盖长串、
      每条都会被误报成不一致。
    * **互斥度**：分布式候选必须主要出现在没有本方写法的叶子里，否则它只是同句共现的
      伴随词（`粗糙品质` 里的 `品质` 与 `粗糙` 同现 108/129，互斥度 0.16 → 出局）。
    """
    out = []
    for c in ranked_all:
        g = c["cn"]
        if g == cn_val or cn_val in g:
            continue
        if c.get("attributed_to"):
            continue
        # 形容词形不是异译：`SPELL.INFLECTIONS.AbyssAdj`＝深渊**的** 与 compendium 的
        # 名词 `深渊` 是同一个词 —— 那个「的」是法术名拼装用的（迅捷的 燃烧打击）。
        if cn_val == g + "的" or g == cn_val + "的":
            continue
        excl = sum(1 for i in idxs_cn if g in leaves[i][4] and cn_val not in leaves[i][4])
        c = dict(c, excl_df=excl)
        if g in cn_val:
            if excl < 2:
                continue
            c["gated_df"], c["exclusive"] = excl, True
        if c.get("tier") == "distributional":
            if excl < min_alt or excl < excl_ratio * max(c["gated_df"], 1):
                continue
        elif excl < 1:
            continue
        out.append(c)
        if len(out) >= top:
            break
    return out


def analyse_term(en, cn_val, holders, idxs_cn, ranked_all, leaves,
                 corpus_df, min_gated, min_alt, top, too_common=False, free_idxs=None):
    """把某个 lang 术语（英文 en / 中文 cn_val）与 compendium 侧的写法对上。"""
    lang_idx = [i for i in idxs_cn if cn_val in leaves[i][4]]
    alts = pick_alts(ranked_all, cn_val, idxs_cn, leaves, top, min_alt=min_alt)
    best = alts[0] if alts else None
    verdict = ("TOO_COMMON" if too_common else
               judge(len(lang_idx), best, len(idxs_cn), min_gated, min_alt))
    free = [i for i in idxs_cn if free_idxs is None or i in free_idxs]

    sample = []
    pool = ([j for j in idxs_cn if best and best["cn"] in leaves[j][4]
             and cn_val not in leaves[j][4]]
            or [j for j in idxs_cn if best and best["cn"] in leaves[j][4]]
            or idxs_cn)
    for i in pool[:2]:
        r = leaves[i]
        sample.append({"repo": r[0], "pack": r[1], "path": r[2],
                       "en": r[3][:180], "cn": r[4][:180]})

    row = {
        "en": en,
        "lang_cn": cn_val,
        "lang_keys": [f"{r}::{k}" for r, k in holders],
        "verdict": verdict,
        "evidence": best.get("tier") if best else None,
        "gated_with_cn": len(idxs_cn),
        "gated_free": len(free),
        "lang_support": len(lang_idx),
        "lang_support_free": sum(1 for i in free if cn_val in leaves[i][4]),
        "lang_cn_corpus_df": corpus_df.get(cn_val, 0),
        "compendium_top": best,
        "candidates": alts,
        "attributed": [{"cn": c["cn"], "gated_df": c["gated_df"],
                        "attributed_to": c["attributed_to"]}
                       for c in ranked_all if c.get("attributed_to")][:4],
        "by_repo": dict(Counter(leaves[i][0] for i in idxs_cn)),
        "samples": sample,
    }
    row["suggest"] = suggest_a(row)
    return row


def suggest_a(row):
    best = row["compendium_top"]
    lang = row["lang_cn"]
    if row["verdict"] == "TOO_COMMON":
        return "英文闸命中过多（该英文是普通词），本判据不适用"
    if row["verdict"] == "INSUFFICIENT":
        return "英文闸命中太少，证据不足"
    if row["verdict"] == "LANG_NO_SUPPORT":
        att = ("；归属反查显示闸内的写法译的是 "
               + "/".join(f"{c['cn']}←{c['attributed_to']['en']}" for c in row["attributed"])
               if row["attributed"] else "")
        if row.get("gated_free") == 0:
            return (f"该英文在 compendium 里**从不作自由词出现**（{row['gated_with_cn']} 叶"
                    f"全部在更长的大写短语里），lang「{lang}」无从比对，本判据不适用{att}")
        return (f"英文闸命中 {row['gated_with_cn']} 叶（自由词 {row['gated_free']} 叶）里"
                f"一处都没有 lang 写法「{lang}」，但也挖不到够格的对家 —— "
                f"多半是 compendium 根本不用这个词（纯 UI 词/枚举 id），需人工确认{att}")
    if not best:
        return f"compendium 侧未见成规模的竞争写法，lang「{lang}」可留"
    b = best["cn"]
    n = best.get("excl_df", best["gated_df"])
    g = row["gated_with_cn"]
    tier = best.get("tier")
    warn = ("（该候选只有词表+统计支持 tier=distributional，**先看样例再定**）"
            if tier == "distributional" else
            f"（证据层 {tier}: {'/'.join(best.get('sources', []))}）")
    if row["verdict"] == "LANG_ORPHAN":
        tail = ("，且该写法在全库一处都没有（孤例）" if row["lang_cn_corpus_df"] == 0
                else f"，全库另有 {row['lang_cn_corpus_df']} 叶含此写法但英文都不是本词")
        return (f"取 compendium 多数值「{b}」（{n}/{g}）；lang「{lang}」在英文闸下 0 处{tail}。"
                f"改 lang 一处即可，改动面最小{warn}")
    if row["verdict"] == "LANG_WEAK_SIGNAL":
        return (f"只有分布式信号：闸内「{b}」{n}/{g} 且与 lang「{lang}」互斥，"
                f"但它没有任何叶级对照支持，**很可能只是同屏常用词**。"
                f"要下结论先看样例，或 `term_gate.py --en \"\\b{row['en']}\\b\" --cn \"{lang},{b}\"`")
    if row["verdict"] == "LANG_MINORITY":
        return (f"取 compendium 多数值「{b}」（{n}/{g}），lang「{lang}」只有 "
                f"{row['lang_support']} 处支持 —— 改 lang 那边改动面小{warn}")
    if row["verdict"] == "LANG_SPLIT":
        return (f"两侧都有支持：lang「{lang}」{row['lang_support']} : compendium「{b}」{n}，"
                f"需按依据阶梯（name 字段 > 同卷已译页 > 全库多数）裁一个{warn}")
    return f"lang「{lang}」是多数派（{row['lang_support']}/{g}），无需改"


# ---------------------------------------------------------------- B：.mjs 解析

CONST_OBJ = re.compile(r"\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*\{")
PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def slice_object(src, open_idx):
    """从 `{` 开始按花括号配对切出对象字面量，跳过字符串与注释。"""
    depth, i, n = 0, open_idx, len(src)
    while i < n:
        c = src[i]
        if c in "\"'`":
            q, i = c, i + 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == q:
                    break
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "/":
            i = src.find("\n", i)
            if i < 0:
                return None
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            if j < 0:
                return None
            i = j + 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx:i + 1]
        i += 1
    return None


def parse_mjs_tables(path):
    """抽出 `.mjs` 里所有 `const NAME = { "en": "cn", … }` 形态的替换表。

    数组形态（PATTERNS / PREFIXED）是逻辑不是数据，故意不取。
    """
    src = Path(path).read_text(encoding="utf-8-sig")
    tables = {}
    for m in CONST_OBJ.finditer(src):
        body = slice_object(src, m.end() - 1)
        if body is None:
            continue
        pairs = PAIR.findall(body)
        if pairs:
            tables[m.group(1)] = [(k, v) for k, v in pairs]
    return tables


def suggest_b(row):
    if row["verdict"] == "MJS_LANG_DRIFT":
        if row.get("lang_drift"):
            d = row["lang_drift"][0]
            return (f".mjs 写「{row['mjs_cn']}」而 lang {d['key']} 写「{d['cn']}」——"
                    f"两条通道同屏可见，必须统一"
                    f"（.mjs 自己的注释就写着「改一处要两边一起改」）")
        return f".mjs 内部同英文异译：{' / '.join(row.get('mjs_internal_variants', []))}"
    if row["verdict"] == "MJS_ORPHAN_CN":
        b = row["compendium_top"]
        return (f".mjs 与 lang 一致，但英文闸下 compendium 写的是「{b['cn']}」"
                f"（{b['gated_df']}/{row['compendium_gated']}），.mjs 值 0 处支持 ——"
                f"两边一起错的典型，改 compendium 之前先决定取哪个")
    if row["verdict"] == "MJS_NO_LANG":
        return "lang 里没有同英文的键（多半是 Ember 新增、只能靠 .mjs），仅记录"
    return "与 lang 一致"


# ---------------------------------------------------------------- C：枚举族

CAMEL = re.compile(r"^([A-Z][a-z]{2,})([A-Z][A-Za-z]*)$")
ENUMISH = re.compile(r"^[A-Z][A-Z0-9_]*$")   # 枚举组的父段一般是全大写


def singular(e):
    """粗糙的单数化，只用来判断两个枚举成员是不是同一个词的单复数。

    `Milestone`/`Milestones`、`Somatic Gesture`/`Somatic Gestures`、`Prefix`/`Prefixes`
    本来就该译成同一个词，不能算「两档撞名」。
    """
    e = e.strip().lower()
    if e.endswith("ies") and len(e) > 4:
        return e[:-3] + "y"
    if e.endswith(("ses", "xes", "zes", "ches", "shes")):
        return e[:-2]
    if e.endswith("s") and not e.endswith("ss"):
        return e[:-1]
    return e


def derive_families(term_rows, min_size=3):
    """从 lang 键结构归纳枚举族。

    两种形态都要认：
      * 点号同父   `SPELL.GESTURES.*`、`TEMPERATURE_TIERS.*`、`…THREAT_RANKS.*`
      * camel 同前缀 `ITEM.QualityShoddy/Standard/Fine/Superior/Masterwork`
        —— 品质五档就是这个形状，只按点号分组会散掉。
    """
    dot, camel = defaultdict(list), defaultdict(list)
    for key, en, cn, repo in term_rows:
        parent, last = key.rsplit(".", 1) if "." in key else ("", key)
        if ENUMISH.match(parent.rsplit(".", 1)[-1] if parent else ""):
            dot[(repo, parent)].append((key, en, cn))
        m = CAMEL.match(last)
        if m:
            camel[(repo, f"{parent}.{m.group(1)}*" if parent else f"{m.group(1)}*")]\
                .append((key, en, cn))

    out = {}
    for k, v in dot.items():
        if len(v) >= min_size:
            out[k] = sorted(v)
    for k, v in camel.items():
        if len(v) < min_size:
            continue
        keys = {x[0] for x in v}
        if any(keys == {x[0] for x in ex} for ex in out.values()):
            continue                      # 与某个点号族成员完全相同，别报两遍
        out[k] = sorted(v)
    return out


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(
        description="lang / compendium / ember-hardcoded-cn.mjs 三通道一致性闸")
    ap.add_argument("--repo", action="append", required=True, help="汉化插件仓库目录，可重复")
    ap.add_argument("--package", action="append", default=[],
                    help="Foundry 里的系统/模块目录（英文 lang 来源），可重复；按键重合度自动配对")
    ap.add_argument("--mjs", action="append", default=[],
                    help="运行时硬编码补丁 .mjs，可重复")
    ap.add_argument("--out", help="报告 JSON 输出路径")
    ap.add_argument("--section", default="ABC", help="只跑其中几节，如 --section AB")
    ap.add_argument("--exclude-pack", action="append", default=[],
                    help="排除包文件名的正则（例：--exclude-pack 'dnd5e|ember\\.adventure'）")
    ap.add_argument("--min-gated", type=int, default=3, help="英文闸命中叶子数下限（默认 3）")
    ap.add_argument("--min-alt", type=int, default=3, help="竞争写法的 df 下限（默认 3）")
    ap.add_argument("--max-gated", type=int, default=3000,
                    help="命中超过这个数就判 TOO_COMMON（该英文是普通词，判据不适用）")
    ap.add_argument("--mine-cap", type=int, default=500,
                    help="（仅 --legacy-mine）挖候选写法时最多看多少条闸内叶子")
    ap.add_argument("--min-lift", type=float, default=8.0,
                    help="分布式候选的特异度下限：闸内出现率 / 闸外出现率（默认 8）")
    ap.add_argument("--max-common", type=int, default=800,
                    help="分布式候选的全库 df 上限（默认 800≈2%%）：通用词没有叶级对照撑着"
                         "就不能算某个术语的译法")
    ap.add_argument("--glossary",
                    help="glossary_ec.json 路径（作为词表与归属反查的一层；缺省自动找）")
    ap.add_argument("--legacy-mine", action="store_true",
                    help="切回 2026-08-12 之前的 n-gram 滑动窗口挖掘（只用于对照，判据是坏的）")
    ap.add_argument("--top", type=int, default=4, help="每条最多列几个候选写法")
    ap.add_argument("--ignore-case", action="store_true",
                    help="英文闸忽略大小写（默认区分：小写 stride 是动词，与属性 Stride 不同）")
    ap.add_argument("--include-agree", action="store_true", help="连一致的条目也打印")
    ap.add_argument("--limit", type=int, default=60, help="每节最多打印多少条")
    ap.add_argument("--strict", action="store_true",
                    help="有 LANG_ORPHAN / MJS_LANG_DRIFT / FAMILY_COLLISION 时退出码 1")
    a = ap.parse_args()

    t0 = time.time()
    excl = [re.compile(x) for x in a.exclude_pack]
    sections = a.section.upper()

    # ---- 1. 装语料
    leaves = load_leaves(a.repo, excl)
    cn_all = [lf[4] for lf in leaves if lf[4]]
    print(f"compendium 叶子 {len(leaves)}（有中文 {len(cn_all)}），{len(a.repo)} 个仓库")

    # ---- 2. 装 lang，挑出「术语形状」的键
    lang_all, term_rows = {}, []
    for repo in a.repo:
        en_flat, cn_raw, src, keep = load_lang_pairs(repo, a.package)
        rname = Path(repo).name
        pairs = {}
        for k, en in en_flat.items():
            cn = foundry_lookup(cn_raw, k)
            if cn is None:
                continue          # 查不到 = 键形态错，交给 lang_gap.py 的 UNREACHABLE
            pairs[k] = (en, cn)
            if k in keep:
                continue
            e = en.strip()
            if not TERM_EN.match(e) or len(e) > 32:
                continue
            c = cn.strip()
            if not CJK_ANY.search(c) or "{" in c or len(c) > 12 or " " in c:
                continue
            term_rows.append((k, e, c, rname))
        lang_all[rname] = pairs
        print(f"  lang {rname}: {len(pairs)} 键（英文基准 {src}）")
    print(f"  术语形状的 lang 键：{len(term_rows)}")

    # ---- 3. .mjs 表
    mjs_tables, mjs_entries = {}, []
    for p in a.mjs:
        t = parse_mjs_tables(p)
        mjs_tables[Path(p).name] = {k: len(v) for k, v in t.items()}
        for tname, pairs in t.items():
            for en, cn in pairs:
                mjs_entries.append((str(p), tname, en, cn))
    if a.mjs:
        print(f"  .mjs：{len(mjs_entries)} 条替换，"
              f"{sum(len(v) for v in mjs_tables.values())} 张表 / {len(a.mjs)} 个文件")

    # ---- 4. 英文词闸（lang 术语 + .mjs 里 ≤4 词的英文一起过一遍）
    by_en = defaultdict(lambda: defaultdict(list))     # en -> cn -> [(repo,key)]
    for k, en, cn, repo in term_rows:
        by_en[en][cn].append((repo, k))
    mjs_terms = {en.strip() for _, _, en, _ in mjs_entries
                 if TERM_EN.match(en.strip()) and len(en.strip()) <= 32}
    gate_terms = set(by_en) | mjs_terms
    gated = build_gate(leaves, gate_terms, a.ignore_case)
    print(f"  英文闸词条 {len(gate_terms)}，命中 {len(gated)} 个"
          f"（{round(time.time() - t0, 1)}s）")

    def gidx(en):
        return gated.get(en.lower() if a.ignore_case else en, [])

    # ---- 5. 候选写法（每个英文只做一次）
    gated_cn = {}
    for en in sorted(gate_terms):
        gated_cn[en] = [i for i in gidx(en) if leaves[i][4]]

    exact_targets = ({cn for d in by_en.values() for cn in d}
                     | {c.strip() for _, _, _, c in mjs_entries})

    if a.legacy_mine:
        # —— 旧的滑动窗口，只为对照保留。它产出的是 n-gram 碎片，不要拿来下结论。
        pre, cand_union, mine_texts = {}, set(), {}
        for en, idxs in gated_cn.items():
            if not idxs or len(idxs) > a.max_gated:
                continue
            c, texts = mine_preselect(idxs, leaves, a.mine_cap)
            pre[en], mine_texts[en] = c, texts
            cand_union.update(g for g, _ in c)
        short = {s for s in exact_targets if 2 <= len(s) <= 4 and CJK_RUN.fullmatch(s)}
        bg_df = Counter()
        probe = cand_union | short
        if probe:
            for cn in cn_all:
                hit = probe & cjk_ngrams(cn)
                if hit:
                    bg_df.update(hit)
        corpus_df = {s: bg_df.get(s, 0) for s in short}
        corpus_df.update(corpus_df_exact(exact_targets - short, cn_all))
        owners = defaultdict(set)
        for _k, en, cn, _r in term_rows:
            owners[cn].add(en)
        ranked = {}
        for en, cands in pre.items():
            ranked[en] = rank_candidates(cands, bg_df, gated_cn[en], leaves,
                                         mine_texts[en], en, owners, top=10)
        free_idx = {en: set(gated_cn[en]) for en in gated_cn}
        print(f"  [legacy] 候选 n-gram {len(cand_union)}（{round(time.time() - t0, 1)}s）")
    else:
        # —— 叶级对照：证据层 → 词表 → 全库命中索引 → 每词的候选
        gpath = a.glossary or (Path(__file__).resolve().parents[2]
                               / "5-其他内容" / "glossary" / "glossary_ec.json")
        glossary = {}
        if Path(gpath).is_file():
            try:
                glossary = load_json(gpath)
            except Exception as e:                        # noqa: BLE001
                print(f"  ! glossary 读取失败：{e}", file=sys.stderr)
        ev = build_evidence(leaves, lang_all, glossary, mjs_entries)
        lex = LexIndex(set(ev.lex) | {s for s in exact_targets if 2 <= len(s) <= 14}, leaves)
        print(f"  证据层：对照 {len(ev.align)} 个英文 / 词表 {len(lex.words)} 词"
              f"（glossary {len(glossary)} 条）（{round(time.time() - t0, 1)}s）")

        corpus_df = {}
        for s in exact_targets:
            d = lex.df(s)
            corpus_df[s] = d if d is not None else sum(1 for cn in cn_all if s in cn)

        # 自由词命中（术语不在更长的大写专名里）——`Ward` 与 `Temple Ward` 之别
        free_idx, en_cache = {}, {}

        def stripped_en(i):
            e = en_cache.get(i)
            if e is None:
                e = MARKUP.sub(" ", leaves[i][3])
                en_cache[i] = e
            return e

        for en, idxs in gated_cn.items():
            if not idxs or len(idxs) > a.max_gated:
                free_idx[en] = set(idxs)
                continue
            rx = _free_regex(en)
            free_idx[en] = {i for i in idxs
                            if occurs_free(en, stripped_en(i), rx, stripped=True)}

        n_cn_leaves = len(cn_all)
        ranked = {}
        for en, idxs in gated_cn.items():
            if not idxs or len(idxs) > a.max_gated:
                continue
            ranked[en] = build_candidates(en, idxs, leaves, ev, lex, n_cn_leaves,
                                          a.min_alt, a.min_lift, free_idx[en], en_cache,
                                          top=10, max_common=a.max_common)
        print(f"  候选生成完成（{round(time.time() - t0, 1)}s）")

    report = {
        "repos": [str(Path(r)) for r in a.repo],
        "packages": [str(Path(p)) for p in a.package],
        "mjs_tables": mjs_tables,
        "options": {"min_gated": a.min_gated, "min_alt": a.min_alt,
                    "max_gated": a.max_gated, "mine_cap": a.mine_cap,
                    "min_lift": a.min_lift, "legacy_mine": a.legacy_mine,
                    "ignore_case": a.ignore_case, "exclude_pack": a.exclude_pack},
        "corpus": {"leaves": len(leaves), "leaves_with_cn": len(cn_all),
                   "lang_keys": {r: len(v) for r, v in lang_all.items()},
                   "lang_term_keys": len(term_rows),
                   "mjs_entries": len(mjs_entries)},
    }

    # ================================================== A
    rows_a, analysis = [], {}
    if "A" in sections:
        for en, variants in by_en.items():
            idxs = gated_cn.get(en, [])
            rk = ranked.get(en, [])
            too_common = len(idxs) > a.max_gated
            for cn_val, holders in variants.items():
                row = analyse_term(en, cn_val, holders, idxs, rk, leaves,
                                   corpus_df, a.min_gated, a.min_alt, a.top, too_common,
                                   free_idxs=free_idx.get(en))
                if len(variants) > 1:
                    row["lang_internal_variants"] = sorted(variants)
                    row["suggest"] += ("；另注意 lang 内部同英文异译："
                                       + " / ".join(sorted(variants)))
                analysis[(en, cn_val)] = row
                rows_a.append(row)
        rows_a.sort(key=lambda r: (-SEV.get(r["verdict"], 0),
                                   -(r["compendium_top"]["gated_df"]
                                     if r["compendium_top"] else 0)))
        # 一致 / 证据不足的行留在报告里（用来回答「为什么 X 没被报出来」），
        # 但不留样例与候选，否则报告会大得没法看
        for r in rows_a:
            if r["verdict"] in ("AGREE", "INSUFFICIENT", "TOO_COMMON"):
                r["samples"], r["candidates"] = [], r["candidates"][:1]
        report["A_lang_vs_compendium"] = rows_a

    # ================================================== B
    rows_b = []
    if "B" in sections and mjs_entries:
        lang_by_en = defaultdict(list)
        for rname, pairs in lang_all.items():
            for k, (en, cn) in pairs.items():
                lang_by_en[en.strip()].append((rname, k, cn))
        seen_mjs = defaultdict(set)
        for _, _, en, cn in mjs_entries:
            seen_mjs[en.strip()].add(cn.strip())
        for path, tname, en, cn in mjs_entries:
            en_s, cn_s = en.strip(), cn.strip()
            counter = lang_by_en.get(en_s, [])
            drift = [c for c in counter if c[2].strip() != cn_s]
            idxs = gated_cn.get(en_s, [])
            comp_top, comp_support = None, None
            if idxs and len(idxs) <= a.max_gated:
                comp_support = sum(1 for i in idxs if cn_s in leaves[i][4])
                alts = pick_alts(ranked.get(en_s, []), cn_s, idxs, leaves, a.top)
                comp_top = alts[0] if alts else None
            if drift:
                verdict = "MJS_LANG_DRIFT"
            elif (comp_top and comp_support == 0
                  and comp_top["gated_df"] >= a.min_alt
                  and len(idxs) >= a.min_gated):
                verdict = "MJS_ORPHAN_CN"
            elif counter:
                verdict = "AGREE"
            else:
                verdict = "MJS_NO_LANG"
            row = {
                "table": tname, "file": Path(path).name, "en": en_s, "mjs_cn": cn_s,
                "verdict": verdict,
                "lang_matches": [{"repo": r, "key": k, "cn": c} for r, k, c in counter],
                "lang_drift": [{"repo": r, "key": k, "cn": c} for r, k, c in drift],
                "mjs_cn_corpus_df": corpus_df.get(cn_s, 0),
                "compendium_gated": len(idxs),
                "compendium_support": comp_support,
                "compendium_top": comp_top,
            }
            if len(seen_mjs[en_s]) > 1:
                row["mjs_internal_variants"] = sorted(seen_mjs[en_s])
                if verdict in ("AGREE", "MJS_NO_LANG"):
                    row["verdict"] = verdict = "MJS_LANG_DRIFT"
            row["suggest"] = suggest_b(row)
            rows_b.append(row)
        rows_b.sort(key=lambda r: (-SEV.get(r["verdict"], 0), r["table"], r["en"]))
        report["B_mjs_vs_lang"] = rows_b

    # ================================================== C
    rows_c = []
    if "C" in sections:
        for (repo, fam), members in sorted(derive_families(term_rows).items()):
            mrows, renderings = [], set()
            for key, en, cn in members:
                r = analysis.get((en, cn))
                if r is None:
                    r = analyse_term(en, cn, [(repo, key)], gated_cn.get(en, []),
                                     ranked.get(en, []), leaves, corpus_df,
                                     a.min_gated, a.min_alt, a.top,
                                     len(gated_cn.get(en, [])) > a.max_gated,
                                     free_idxs=free_idx.get(en))
                renderings.add(cn)
                bad_v = r["verdict"] in ("LANG_ORPHAN", "LANG_MINORITY", "LANG_SPLIT")
                if bad_v and r["compendium_top"]:
                    renderings.add(r["compendium_top"]["cn"])
                mrows.append({
                    "key": key, "en": en, "lang_cn": cn, "verdict": r["verdict"],
                    "gated_with_cn": r["gated_with_cn"],
                    "lang_support": r["lang_support"],
                    "lang_cn_corpus_df": r["lang_cn_corpus_df"],
                    "compendium_top": r["compendium_top"],
                })
            # 撞名：两档译成同一个词。单复数（Milestone/Milestones）本来就该同译，不算。
            byrender = defaultdict(set)
            for m in mrows:
                byrender[m["lang_cn"]].add(singular(m["en"]))
            dupes = sorted(cn for cn, ens in byrender.items() if len(ens) > 1)
            bad = [m for m in mrows
                   if m["verdict"] in ("LANG_ORPHAN", "LANG_MINORITY", "LANG_SPLIT")]
            orphan = [m for m in mrows if m["verdict"] == "LANG_ORPHAN"]
            verdict = ("FAMILY_COLLISION" if dupes else
                       "FAMILY_SPLIT" if (orphan or len(bad) >= 2) else "AGREE")
            rows_c.append({
                "family": fam, "repo": repo, "size": len(mrows), "verdict": verdict,
                "lang_collisions": dupes,
                "members_disagreeing": len(bad),
                "members_orphan": len(orphan),
                "distinct_renderings": len(renderings),
                "renderings": sorted(renderings),
                "members": mrows,
            })
        rows_c.sort(key=lambda r: (-SEV.get(r["verdict"], 0), -r["members_orphan"],
                                   -r["members_disagreeing"]))
        report["C_enum_families"] = rows_c

    # ---- 打印与落盘
    print_a(rows_a, a)
    print_b(rows_b, a)
    print_c(rows_c, a)

    counts = {"A": dict(Counter(r["verdict"] for r in rows_a)),
              "B": dict(Counter(r["verdict"] for r in rows_b)),
              "C": dict(Counter(r["verdict"] for r in rows_c))}
    report["summary"] = counts
    print("\n" + "=" * 78)
    print(f"A lang↔compendium : {counts['A']}")
    print(f"B .mjs↔lang       : {counts['B']}")
    print(f"C 枚举族           : {counts['C']}")
    print(f"用时 {round(time.time() - t0, 1)}s")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"→ {a.out}")

    if a.strict:
        # A 段只把**有叶级对照支持**的孤例算作硬失败：distributional 档只有词表+统计支持，
        # 按判读要点要人先看样例，不该卡住流水线。
        hard_a = sum(1 for r in rows_a if r["verdict"] == "LANG_ORPHAN"
                     and r.get("evidence") != "distributional")
        hard = (hard_a + counts["B"].get("MJS_LANG_DRIFT", 0)
                + counts["C"].get("FAMILY_COLLISION", 0))
        if hard:
            sys.exit(1)


# ---------------------------------------------------------------- 打印

def print_a(rows, a):
    if not rows:
        return
    print("\n" + "=" * 78)
    print("A. lang ↔ compendium（英文闸：只在英文相同的叶子里比中文）")
    print("=" * 78)
    shown = 0
    for r in rows:
        if r["verdict"] in ("AGREE", "INSUFFICIENT", "TOO_COMMON") and not a.include_agree:
            continue
        if shown >= a.limit:
            print("  …（还有更多，见 --out 的 JSON）")
            break
        shown += 1
        keys = r["lang_keys"]
        print(f"\n[{r['verdict']}] {r['en']}   lang=「{r['lang_cn']}」  "
              f"（{', '.join(keys[:3])}{'…' if len(keys) > 3 else ''}）")
        print(f"    英文闸命中 {r['gated_with_cn']} 叶（自由词 {r.get('gated_free', '-')} 叶）："
              f"lang 写法 {r['lang_support']} 处；该写法全库 {r['lang_cn_corpus_df']} 叶  "
              f"{r['by_repo']}")
        if r["candidates"]:
            print("    compendium 候选：" + " · ".join(
                f"{c['cn']} {c.get('excl_df', c['gated_df'])}/{c['gated_df']}"
                f"（全库{c['corpus_df']}·{c.get('tier', '?')[:5]}）" for c in r["candidates"]))
        if r.get("attributed"):
            print("    归属反查（不算对家）：" + " · ".join(
                f"{c['cn']}→{c['attributed_to']['en']}" for c in r["attributed"]))
        print(f"    → {r['suggest']}")
        for s in r["samples"][:1]:
            print(f"      例 {s['repo']}/{s['pack']}::{s['path']}")
            print(f"         EN: {s['en']}")
            print(f"         CN: {s['cn']}")
    if not shown:
        print("  （无）")


def print_b(rows, a):
    if not rows:
        return
    print("\n" + "=" * 78)
    print("B. ember-hardcoded-cn.mjs ↔ lang（运行时替换表 vs Foundry i18n）")
    print("=" * 78)
    shown = 0
    for r in rows:
        if r["verdict"] in ("AGREE", "MJS_NO_LANG") and not a.include_agree:
            continue
        if shown >= a.limit:
            print("  …（还有更多，见 --out 的 JSON）")
            break
        shown += 1
        print(f"\n[{r['verdict']}] {r['table']}.{r['en']}   .mjs=「{r['mjs_cn']}」")
        for d in r["lang_drift"]:
            print(f"    lang {d['repo']}::{d['key']} =「{d['cn']}」")
        if r["compendium_top"]:
            print(f"    compendium 英文闸 {r['compendium_gated']} 叶：.mjs 值 "
                  f"{r['compendium_support']} 处，多数写法「{r['compendium_top']['cn']}」"
                  f" {r['compendium_top']['gated_df']} 处")
        print(f"    → {r['suggest']}")
    if not shown:
        print("  （无漂移）")
    c = Counter(r["verdict"] for r in rows)
    print(f"\n  合计 {len(rows)} 条：与 lang 一致 {c.get('AGREE', 0)}、"
          f"漂移 {c.get('MJS_LANG_DRIFT', 0)}、"
          f"与 compendium 不符 {c.get('MJS_ORPHAN_CN', 0)}、"
          f"lang 无对应键 {c.get('MJS_NO_LANG', 0)}（Ember 新增，正常）")


def print_c(rows, a):
    if not rows:
        return
    print("\n" + "=" * 78)
    print("C. 枚举族内部一致性（同一组枚举在 lang 与 compendium 是否成套）")
    print("=" * 78)
    shown = 0
    for r in rows:
        if r["verdict"] == "AGREE" and not a.include_agree:
            continue
        if shown >= a.limit:
            print("  …（还有更多，见 --out 的 JSON）")
            break
        shown += 1
        print(f"\n[{r['verdict']}] {r['family']}  @{r['repo']}  {r['size']} 档；"
              f"{r['members_disagreeing']} 档与 compendium 不一致"
              f"（其中孤例 {r['members_orphan']}）；全族 {r['distinct_renderings']} 种写法")
        if r["lang_collisions"]:
            print(f"    ⚠ lang 内部撞名（两档同一译名）：{r['lang_collisions']}")
        quiet = 0
        for m in r["members"]:
            if m["verdict"] not in ("LANG_ORPHAN", "LANG_MINORITY", "LANG_SPLIT") \
                    and m["lang_cn"] not in r["lang_collisions"]:
                quiet += 1
                continue
            b = m["compendium_top"]
            tail = (f"  compendium 多数「{b['cn']}」{b['gated_df']}" if b
                    else "  compendium 无竞争写法")
            print(f"    {m['en'][:22]:<24} lang=「{m['lang_cn']}」 支持 "
                  f"{m['lang_support']}/{m['gated_with_cn']}{tail}   [{m['verdict']}]")
        if quiet:
            print(f"    （另有 {quiet} 档与 compendium 一致或证据不足，见 JSON）")
    if not shown:
        print("  （无）")


if __name__ == "__main__":
    main()
