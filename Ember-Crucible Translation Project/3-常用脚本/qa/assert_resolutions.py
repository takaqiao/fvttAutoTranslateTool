#!/usr/bin/env python3
"""把 PROJECT.md 第 8 节的裁决**编译成可执行断言**，跑在 5.4 全套后面。

  python assert_resolutions.py [--rules <json>] [--repo <repo> ...] [--verbose]

为什么需要这个
--------------
第 8 节现在有上百条裁决，**全是散文**。没有任何机制阻止下一轮悄悄推翻某一条 ——
十四轮里已经出过好几次险：`scan_uuid_swap` 改判据降噪时消掉过一条真缺陷；
`Scout` 的处理差点与已归档豁免冲突；给 crucible 的 CI 断言照抄进 ember 差点让下一次发版失败；
`Trident's Point` 的「本地区+地图」差点被术语替换改坏。**这些都是靠人当场想起来才没出事。**

散文裁决 → 机器断言，一致性就从「靠记忆维持」变成「靠闸维持」。

断言类型
--------
`term_gated`      英文侧命中 `en` 的叶，中文必须含 `cn_required`（且不得含 `cn_forbidden`）
`cn_absent`       某个中文串在 compendium **与 lang** 全库不得出现（已清零的错译不许回潮）
`sense_gated`     同一个英文词有**机制义／普通名词义**两支，按上下文窗口分类后只闸得住的那两个方向
`distinct_terms`  一组术语的中文必须两两不同，**且每个术语都要过英文闸读库核对**（防撞名 + 防空转）
`term_domains`    同一个英文词按域分裂成多个中文，逐域钉死（防下一轮「顺手统一」）
`lang_parity`     两仓 `lang/cn.json` 的键数必须等于英文侧键数
`anchor_ids`      标题上的显式 `id=` 数量不得低于阈值（撑着锚点链接）
`no_bilingual_tail` 指定字段的中文不得带「中文 English」双语尾巴
`exclusions_closed` `same_en_split` 的分组必须全部在已归档豁免表内
`leaf_literal`    指定叶必须/不得包含某个字面串（用于登记「绝不能动」的假阳性）
`block_aligned_gate` **叶内**判据：按块级标签把中英切块后逐块比术语类别（第十八轮 Y2 新增）
`block_sense_gate`   同上，但块内做机制义／普通名词义分类，块内单一义项时上正向闸
`enricher_slot_gate` **槽位级**判据：按 (动词,目标) 把两侧 `@X[…]{标签}` 配对，逐个比标签里的术语
                     （第十九轮 Y6 新增，补 `split_blocks` 把标签连花括号一起涂空留下的洞）
`glossary_value`  词表的 base 层与产物层都必须是某个值（防「词表把错误洗成权威」）
`version_matrix`  PROJECT.md 抬头与版本矩阵必须与两仓 module.json 一致（这一行漂过两次）

每条断言都带 `decision`（对应第 8 节哪一天的裁决）与 `why`，失败时一并打印 ——
让下一个人看到的不是「断言 R-xx 挂了」，而是「你违反了 2026-08-13j 定的那条，理由是……」。

第十六轮补的通则：**任何断言都必须能说出「我扫了多少个叶 / 多少个键」**
------------------------------------------------------------------
说不出来的那种，是**自检**，不是断言。这条通则是被咬出来的，本项目已经实测到
**三种不同的空转形态**——写新断言前请拿这三条当自检清单逐条过一遍：

（a）**判据写坏了** —— `R-catwalk`：`en` 在 JSON 里写成 `"\bcatwalk"`（单反斜杠），
     被 JSON 当成退格符吃掉，正则变成 `\x08catwalk`，一个都匹配不到，断言一路报绿，
     而库里实际有 10 叶违规。
     ▸ 防法：`min_hits`。**这一类 `min_hits` 能防。**

（b）**判据压根没读库** —— `R-region-area-map`：旧版 `distinct_terms` 的注释白纸黑字
     自陈「纯配置自检，不读库」，只比较一组术语的中文两两不同。于是它一直全绿，而
     ember lang 的 `EMBER.CALENDAR.REGION` 反着写了「区域地图」**四个发布版没人发现**。
     ▸ 防法：强制声明 `scan`（lang / compendium）。**`min_hits` 对这一类无效** ——
       它连「命中数」这个概念都没有，没读库就没有命中数可言。

（c）**读的是别人早先写下的报告快照** —— `R-exclusions-closed`：判据读磁盘上的
     `4-临时脚本/2026-08-15-round15/qa2/same_en_final.json`（第十五轮产物，15 组），
     而第十六轮库里实跑 `scan_same_en_split` 是 **14 组 / 123 叶**。这条断言因此
     **只与「上次谁跑过扫描并写了那个文件」一样新**：库里新分叉出来，它不会变红。
     ▸ 防法：**现场调用扫描器**（本轮改法），或至少校验报告 mtime 晚于两仓
       `compendium/cn` 的最新 mtime、不满足就判失败。**`min_hits` 与 `scan` 都防不住这一类** ——
       它「读了库」，只不过读的是库的一张过期照片。

（d）**闸在跑，但豁免表已经空了** —— 第十八轮收尾实测：两条块级断言的 `except_blocks` 里
     合计躺着 **7 条**再也匹配不到的条目（登记的内容欠账都已修完），而全套依旧 52/0 全绿。
     豁免不命中只让 detail 里的 `n_exempt` 变小，**没有任何断言会因此变红，只能靠人记**。
     后果与 `R-dives-mine` 一样：欠账还清了、豁免却留着，遮住了同一处未来的回潮。
     ▸ 防法：`max_unused_exempt`（默认 0，见 `_unused_exempt`）。
       **前三种防法对这一类全部无效** —— 它判据没写坏、读了本次运行时的库、命中数也正常。

通则一句话：**任何断言都必须能说出「我这次扫了多少叶 / 多少个键」，说不出来的是自检不是断言。**
推论：判据的数据来源必须是**本次运行时现读的库**；凡是从磁盘上另一个文件里拿结论的，
都要能证明那个文件比库新，否则就是形态 (c)。

所以本轮起：`distinct_terms` 的每个术语都必须声明能读库的英文闸（lang / compendium），
没有 `scan` 的 `distinct_terms` 规则**直接判失败**（见 `a_distinct_terms` 顶部）；
`exclusions_closed` 现场重跑 `scan_same_en_split.py`（见 `a_exclusions_closed` 顶部）。

已知局限（**别把它当成全覆盖**，这是本项目方法教训 1 的又一处应用）
-----------------------------------------------------------------
1. `term_gated` 是**叶级**的：只要求该叶中文里**出现过**定译。一叶里提到该术语 5 次、
   只错 1 次的情况**它抓不到**。回测实测：把一叶里 3 处「邪术师」全改成「术士」才会响，
   只改 1 处不响。要抓叶内部分错译，得做逐位对齐（成本高得多），本轮没做。
   ▸ **第十八轮 Y2 做了**：`block_aligned_gate` / `block_sense_gate` 两个新类型把「按块级
     标签切块、再逐块对齐」固化下来，专门补三条断言各自 why 里写死的叶内盲区
     （R-shard-god 的 70 叶 · R-arcturel-vs-arcturian 的叶内串行 · R-rank-sense-compendium
     的混合叶／无法分类叶与缺失的正向闸）。见 `a_block_aligned_gate` 的 docstring。
     ⚠ 它**不是万能的替代品**：叶级判据仍然管着「整叶一次都没提」这种情况，两者互补。
   ⚠ 反过来，叶级也会**假阳性**：一叶里同时出现 `Shard God` 与 `Shard Gods` 时，
   中文只要重写了其中一种说法，另一条闸就会误报。
   ▸ 第十六轮终段的**绕法**（`R-shard-god`）：不去做逐位对齐，而是把闸只下在
     **英文侧不含歧义**的那些叶上 —— `(?s)^(?!.*<另一形态>).*<本形态>` 这种
     「只含单数 / 只含复数」的负向先行断言。实测单数专属 264 叶 264/264 干净、
     复数专属 117 叶 114/117 干净。同时含两种形态的 70 叶仍然不查，那是本判据的边界。
     这个形态可以复用到任何「同一词根多形态、叶内混排」的裁决上。
2. `cn_absent` 反过来会**误伤合法用法**：某个废弃写法如果在别处是正当中文，会假阳性。
   目前每条都验过全库为 0，加新条目前务必先数一遍。必要时用 `except_paths` 登记豁免
   （可以写成 `{"代币": [...]}` 形态，只豁免某一个词，别把整叶从所有词的检查里摘出去）。
3. 断言只覆盖第 8 节里**能机械表达**的那些裁决。像「改动面小的那边优先」「name 与正文
   冲突时多数该改 name」这类**方法性**裁决，本质上表达不成断言，仍然只能靠人读第 8 节。
4. lang 侧英文闸**先剥占位符**（`{rank}` `{level}` 之类）再匹配。不剥的话
   `HAZARD.TooltipDamage` 那种「英文只在 `{rank}` 里出现、中文当然没有对应汉字」的键
   会被判违规。compendium 侧**不剥** —— 那里的 `@UUID[...]{Shard Gods}` 花括号内是正文标签。

自检与回测
----------
`--selftest` 跑判据自身的正反例（双语尾巴那条最容易写错，第一版就把「圣堂区路人 A」
这种编号后缀全报成了尾巴）。
`--root <另一棵树>` 用于灵敏度回测：往副本里注入违规，确认断言真的会响 ——
**只测特异度（全绿）是不够的**，那样「所有断言都返回空」也能过。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 项目根
DEFAULT_RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")
REPOS = {"ember": "1-Ember汉化插件", "crucible": "2-Crucible汉化插件"}

# 双语尾巴：中文 …… 空白 + 拉丁串结尾。
#
# ⚠ **必须排除单字母编号。** 第一版写成 `[A-Z][A-Za-z'\-]*`，把「圣堂区路人 A」
# 「菌丝旷野底图 B」这类**编号后缀**全报成双语尾巴（14 处全是假阳性）——
# 而那正是既定约定要求的写法（英文侧就是 `Hallows Passerby A`，字母是编号本体，
# 前面留半角空格是库内 name 的惯例）。
#
# 判据取与 `scan_same_en_split` 的归一规则同源的定义：尾巴是**英文名本身**被附在中文后面，
# 所以要求结尾那串拉丁**至少有 2 个字母**。单个字母（或数字）是编号，不是名字。
_TAIL = re.compile(r"[一-鿿].*?\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)$")


def has_bilingual_tail(text):
    m = _TAIL.match(text.strip())
    if not m:
        return False
    letters = re.sub(r"[^A-Za-z]", "", m.group(1))
    return len(letters) >= 2


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def load_pack_pairs(repo_dir):
    """产出 (pack, path, en, cn)；只收两侧都有的叶。"""
    en_dir = os.path.join(repo_dir, "compendium", "en")
    cn_dir = os.path.join(repo_dir, "compendium", "cn")
    if not os.path.isdir(en_dir):
        return
    for fname in sorted(os.listdir(en_dir)):
        if not fname.endswith(".json") or fname == "_source.json":
            continue
        cn_path = os.path.join(cn_dir, fname)
        if not os.path.exists(cn_path):
            continue
        en = dict(walk(json.load(open(os.path.join(en_dir, fname), encoding="utf-8-sig"))))
        cn = dict(walk(json.load(open(cn_path, encoding="utf-8-sig"))))
        for p, ev in en.items():
            cv = cn.get(p)
            if cv is not None:
                yield fname, p, ev, cv


def load_lang_pairs(repo_dir, upstream_dir):
    """产出 (key, en, cn)；只收两侧都有的键。

    英文侧取的是**上游安装目录**里的 `lang/en.json`（模块/系统本体），
    中文侧是本仓的 `lang/cn.json`。两边都要 `walk` 展平：上游 en.json 是嵌套的，
    而本仓 cn.json 是扁平的（`flatten_lang.py` 的产物），不展平会得到 92 : 486 这种假差异。
    """
    en_p = os.path.join(upstream_dir, "lang", "en.json")
    cn_p = os.path.join(repo_dir, "lang", "cn.json")
    if not (os.path.exists(en_p) and os.path.exists(cn_p)):
        return
    en = dict(walk(json.load(open(en_p, encoding="utf-8-sig"))))
    cn = dict(walk(json.load(open(cn_p, encoding="utf-8-sig"))))
    for k, ev in en.items():
        cv = cn.get(k)
        if cv is not None:
            yield k, ev, cv


class Ctx:
    """一次性把两个仓库读进来，所有断言共用（否则每条断言重读一遍太慢）。"""

    def __init__(self, repos, meta=None):
        self.repos = repos
        self.meta = meta or {}
        self.pairs = {}
        for name, d in repos.items():
            self.pairs[name] = list(load_pack_pairs(d))
        # lang 通道。上游路径写在 rules 的 meta.lang_sources 里（与 R-lang-parity 同源）。
        # ⚠ 这里**故意不受 `--root` 影响**：英文基准永远取真实安装目录，
        # 灵敏度回测只需要往副本树的 `lang/cn.json` 里注入违规就能生效。
        self.lang = {}
        for name, src in (self.meta.get("lang_sources") or {}).items():
            if name in repos:
                self.lang[name] = list(load_lang_pairs(repos[name], os.path.expandvars(src)))

    def all_pairs(self, scope):
        for name in (scope or self.repos.keys()):
            for row in self.pairs.get(name, []):
                yield (name,) + row

    def all_lang(self, scope):
        for name in (scope or self.lang.keys()):
            for k, ev, cv in self.lang.get(name, []):
                yield name, k, ev, cv


# lang 侧英文闸要先剥掉占位符再匹配 —— 见模块 docstring 的「已知局限 4」。
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _paths_matcher(spec):
    """把 except_paths 编成一个「路径是否豁免」的函数。空 spec 返回 None。"""
    if not spec:
        return None
    pats = [re.escape(s) if not s.startswith("re:") else s[3:] for s in spec]
    rx = re.compile("|".join(pats))
    return lambda path: bool(rx.search(path))


# ----------------------------------------------------------------- 断言实现

def a_term_gated(rule, ctx):
    # ⚠ 大小写不敏感是**默认**。本项目已经被同一个坑咬过两次：
    # `split_region_area_map.py` 的 `\b(Region|Area) Maps?\b` 漏掉小写形态，
    # 导致 72 叶被误判成「需人判」、主控据此判定「全库拆分做不了」——判据错了结论就跟着错。
    # 专名的大小写在本库里从来不是判据的一部分，所以默认忽略，要区分请显式写 "case_sensitive": true。
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    en_re = re.compile(rule["en"], flags)
    # `cn_required` 是可选的：有些裁决只有「禁用写法」而没有「每叶都必须出现的定译」——
    # 例如 `Token`，345 个命中叶里有 60 叶的英文是普通名词义（a token of good luck），
    # 中文当然不含「指示物」。那种规则写成「闸内零容忍某几个词」才对，
    # 硬要求 cn_required 只会造出 60 条假阳性。
    req = rule.get("cn_required")
    forbid = rule.get("cn_forbidden", [])
    # 有意的例外（例：`Drakeling Scales`=幼龙鳞片 —— 材料名不是生物指称，
    # 「龙兽鳞」不是中文词而「龙鳞」是。登记在这里，免得下一轮「顺手统一」）
    except_re = _paths_matcher(rule.get("except_paths"))
    bad = []
    hits = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if not en_re.search(ev):
            continue
        if except_re and except_re(path):
            continue
        hits += 1
        if req and req not in cv:
            bad.append((repo, pack, path, f"英文命中但中文无「{req}」"))
            continue
        for f in forbid:
            if f in cv:
                bad.append((repo, pack, path, f"中文同时出现禁用写法「{f}」"))
                break

    # ⚑ **命中 0 叶不是「通过」，是「这条断言根本没在跑」。**
    # 实测代价：`R-catwalk` 的 `en` 在 JSON 里写成了 `"\bcatwalk"`（单反斜杠），
    # 被 JSON 当成**退格符**吃掉，正则变成 `\x08catwalk`，匹配不到任何东西 ——
    # 断言一路报绿，而库里实际有 10 叶违规。这正是本项目「静默全绿」那一类失败。
    min_hits = rule.get("min_hits", 1)
    if hits < min_hits:
        bad.append(("-", "-", rule["en"],
                    f"英文闸只命中 {hits} 叶（要求 ≥{min_hits}）—— 这条断言在空转，"
                    f"多半是正则被 JSON 转义吃掉了（`\\b` 要写成 `\\\\b`），或上游改了措辞"))
    return bad, f"英文闸命中 {hits} 叶"


def a_cn_absent(rule, ctx):
    """某些中文写法在全库不得出现。

    第十六轮改了三处：
    1. `cn` 可以是**一组**词（`Token` 那条要同时禁「令牌」和「代币」）。
    2. 扫描面从 compendium 扩到 **compendium + lang**。原来只扫 compendium ——
       与 `distinct_terms` 不读库是同一类盲区，只是没那么显眼。实测本轮 18 个禁用词
       在 lang 侧全为 0，所以这次扩面不产生任何新失败，纯属补洞。
    3. 可选的 `en` 英文闸**只用来算命中数**（`min_hits` 反空转），不改变「全库零容忍」的语义。
       意义是：万一上游把这个词整个删了，`cn_absent` 会永远报绿而没人知道它已经不设防。
    """
    needles = rule["cn"]
    if isinstance(needles, str):
        needles = [needles]
    raw_exc = rule.get("except_paths") or {}
    if isinstance(raw_exc, list):                     # 一份豁免表管所有词
        raw_exc = {n: raw_exc for n in needles}
    exc = {n: _paths_matcher(v) for n, v in raw_exc.items()}

    bad = []
    n_leaf = n_key = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        n_leaf += 1
        for needle in needles:
            if needle in cv and not (exc.get(needle) and exc[needle](path)):
                bad.append((repo, pack, path, f"出现了已废弃的写法「{needle}」"))
    for repo, key, ev, cv in ctx.all_lang(rule.get("scope")):
        n_key += 1
        for needle in needles:
            if needle in cv and not (exc.get(needle) and exc[needle](key)):
                bad.append((repo, "lang/cn.json", key, f"出现了已废弃的写法「{needle}」"))

    detail = f"扫 {n_leaf} 叶 + {n_key} 个 lang 键，禁用词 {len(needles)} 个"
    if rule.get("en"):
        flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
        en_re = re.compile(rule["en"], flags)
        hits = sum(1 for _, _, _, ev, _ in ctx.all_pairs(rule.get("scope")) if en_re.search(ev))
        hits += sum(1 for _, _, ev, _ in ctx.all_lang(rule.get("scope"))
                    if en_re.search(_PLACEHOLDER.sub(" ", ev)))
        detail += f"；英文闸 `{rule['en']}` 命中 {hits}"
        min_hits = rule.get("min_hits", 1)
        if hits < min_hits:
            bad.append(("-", "-", rule["en"],
                        f"英文闸只命中 {hits}（要求 ≥{min_hits}）—— 这条 cn_absent 已经不设防了："
                        f"要么上游把这个概念删了/改了措辞，要么正则被 JSON 转义吃掉"))
    return bad, detail


_TAGS = re.compile(r"<[^>]+>")


def a_sense_gated(rule, ctx):
    """同一个英文词有**机制义**和**普通名词义**两支，只有机制义该用那个定译。

    为什么需要一个新类型（第十六轮终段补）
    --------------------------------------
    `R-tier-rank-level` 的 `scan` 只写了 `lang`，实测只看住 62 个 lang 键；
    而第十六轮在 **compendium 侧改了 176 叶**（英文闸 IGNORECASE 下 阶位 221 : 等级 51），
    那一整片**没有任何闸看守**。可是 compendium 侧又不能照搬 `distinct_terms`：
    英文 `rank` 在本库里是两个词 ——

      · 机制义：`You gain the Novice rank in Arcana` / `Attunement Rank 1` / `Soulbound Rank`
      · 普通名词义：`denote their civic rank`（公民地位）/ `within the ranks of Shard Gods`（行列）
        / `rank as a Commander`（军衔）/ `rank-and-file`（普通一兵）

    —— 后者**不归那条裁决管**，硬要求它们含「阶位」会造出成片假阳性。

    判据（GAME/COMMON 两张正则表与 `4-临时脚本/2026-08-15-round16/probes/split_rank.py` 同源，
    COMMON 优先级高于 GAME）把每一处出现按**剥标签后的上下文窗口**分类，
    然后**只闸得住的那两个方向**：

      ① 反向闸 `require_en_support`：中文用了定译，英文侧却一个该词都没有 → 违规。
         实测全库 40674 叶里「阶位」出现在 221 叶，**221 叶全部**英文含 `rank`，零违规。
         这一条抓的是「整词替换扫到了别的义项」（把 Tier / level 的叶一起刷成阶位）。
      ② 普通名词义专属叶 `forbid_when_all_common`：该叶所有出现都判 COMMON → 中文不得含定译。
         实测 96 叶，**无一含「阶位」**，零违规。这一条抓的是「顺手全库统一」。

    ⚠ **判据边界，别把绿读成全覆盖**：**故意不做**「GAME → 中文必须含阶位」那个正向闸。
    实测纯 GAME 的 230 叶里有 13 叶中文正当地没有「阶位」——
    `1 rank of exhaustion`（＝1 层力竭）· `close ranks with enemy characters`（＝并肩结阵）·
    `join their ranks`（＝加入他们）· 更新日志里被整段改写的 `Soulbound (rank 1 only)`。
    分类器把它们判成 GAME 是因为窗口里有 `exhaustion` / `skill` 这些词，这是分类器的粗糙处，
    不是译文的错。上正向闸就是 13 条假阳性，所以这条断言**证明不了**「每一处机制义都译对了」。
    """
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    occ = re.compile(rule["en"], flags)
    game = re.compile(rule["sense"]["game"], flags)
    common = re.compile(rule["sense"]["common"], flags)
    win = rule.get("window", 90)
    cn = rule["cn"]
    exc = _paths_matcher(rule.get("except_paths"))
    bad = []
    n_en = n_cn = 0
    kinds_n = {"GAME": 0, "COMMON": 0, "UNKNOWN": 0, "MIX": 0}

    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        has_cn = cn in cv
        if has_cn:
            n_cn += 1
        ms = list(occ.finditer(ev))
        if not ms:
            if has_cn and rule.get("require_en_support", True) and not (exc and exc(path)):
                bad.append((repo, pack, path,
                            f"中文用了机制义定译「{cn}」，而英文侧一个 `{rule['en']}` 都没有 ——"
                            f"多半是整词替换扫到了别的义项（Tier / level）"))
            continue
        n_en += 1
        seen = set()
        for m in ms:
            window = _TAGS.sub(" ", ev[max(0, m.start() - win): m.end() + win])
            if common.search(window):
                seen.add("COMMON")
            elif game.search(window):
                seen.add("GAME")
            else:
                seen.add("UNKNOWN")
        if "UNKNOWN" in seen:
            bucket = "UNKNOWN"
        elif seen == {"COMMON"}:
            bucket = "COMMON"
        elif seen == {"GAME"}:
            bucket = "GAME"
        else:
            bucket = "MIX"
        kinds_n[bucket] += 1
        if bucket == "COMMON" and has_cn and rule.get("forbid_when_all_common", True):
            if not (exc and exc(path)):
                bad.append((repo, pack, path,
                            f"这一叶的 `{rule['en']}` 全部是普通名词义（公民地位／行列／军衔），"
                            f"中文却用了机制义定译「{cn}」"))

    detail = (f"英文闸命中 {n_en} 叶（机制义 {kinds_n['GAME']} / 普通名词义 {kinds_n['COMMON']} / "
              f"混合 {kinds_n['MIX']} / 无法分类 {kinds_n['UNKNOWN']}）· 中文含「{cn}」{n_cn} 叶")
    for field, got, why in (("min_en_leaves", n_en, "英文闸"),
                            ("min_cn_leaves", n_cn, f"中文「{cn}」"),
                            ("min_common_leaves", kinds_n["COMMON"], "普通名词义分桶")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{why}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"要么正则被 JSON 转义吃掉，要么上游改了措辞，要么分类表失效"))
    return bad, detail


def _gate_one(term, ctx, scan, scope):
    """一个术语的读库核对。返回 (命中数, 违规列表)。

    `en_gate` 缺省是 `\\b<en>\\b`；`cn_gate` 缺省是定译本身 —— 但允许写一个**更宽的
    可接受形态**：`level` 的定译是「等级」，而 lang 里 `Level {x}` 正当地写成「{x} 级」，
    所以 `cn_gate` 取「级」。放宽的每一处都要在规则的 `why` 里写清为什么。
    """
    en_gate = term.get("en_gate") or rf"\b{re.escape(term['en'])}\b"
    flags = 0 if term.get("case_sensitive") else re.IGNORECASE
    rx = re.compile(en_gate, flags)
    need = term.get("cn_gate", term["cn"])
    forbid = term.get("cn_forbidden", [])
    exc = _paths_matcher(term.get("except_paths"))
    hits = 0
    bad = []
    if "lang" in scan:
        for repo, key, ev, cv in ctx.all_lang(scope):
            if not rx.search(_PLACEHOLDER.sub(" ", ev)):
                continue
            if exc and exc(key):
                continue
            hits += 1
            if need not in cv:
                bad.append((repo, "lang/cn.json", key,
                            f"英文是 {ev[:60]!r}，中文里没有「{need}」：{cv[:50]!r}"))
            for f in forbid:
                if f in cv:
                    bad.append((repo, "lang/cn.json", key, f"中文出现禁用写法「{f}」：{cv[:50]!r}"))
    if "compendium" in scan:
        for repo, pack, path, ev, cv in ctx.all_pairs(scope):
            if not rx.search(ev):
                continue
            if exc and exc(path):
                continue
            hits += 1
            if need not in cv:
                bad.append((repo, pack, path, f"英文命中 `{en_gate}` 但中文无「{need}」"))
            for f in forbid:
                if f in cv:
                    bad.append((repo, pack, path, f"中文出现禁用写法「{f}」"))
    return hits, bad


def a_distinct_terms(rule, ctx):
    """一组 {en, cn} 的中文必须两两不同，**并且逐个过英文闸读库核对**。

    ⚠ 旧版只做前半句（注释原文：「纯配置自检，不读库」）。后果实测：`R-region-area-map`
    一直全绿，而 ember lang 的 `EMBER.CALENDAR.REGION` 反着写成「区域地图」，
    四个发布版没人发现 —— 断言不读库就等于没有断言。
    所以现在**没有 `scan` 的规则直接判失败**，不给「只做配置自检」留后门。
    """
    seen = {}
    bad = []
    for item in rule["terms"]:
        cn = item["cn"]
        if cn in seen:
            bad.append(("-", "配置", item["en"], f"与「{seen[cn]}」共用中文「{cn}」"))
        seen[cn] = item["en"]

    scan = rule.get("scan")
    if not scan:
        bad.append(("-", "配置", rule["id"],
                    "这条 distinct_terms 没有声明 `scan` —— 它只比了配置、一个字节的库都没读，"
                    "正是 R-region-area-map 空转四个版本的那种形态。请补 lang 或 compendium 闸"))
        return bad, f"{len(rule['terms'])} 个术语（**未读库**）"

    total = 0
    per = []
    for item in rule["terms"]:
        if not item.get("gate", True):               # 显式声明「这个词没有可用闸」的，要写 why
            per.append(f"{item['en']}:—")
            continue
        h, b = _gate_one(item, ctx, scan, rule.get("scope"))
        total += h
        per.append(f"{item['en']}:{h}")
        bad.extend(b)

    min_hits = rule.get("min_hits", 1)
    if total < min_hits:
        bad.append(("-", "配置", rule["id"],
                    f"英文闸合计只命中 {total}（要求 ≥{min_hits}）—— 这条断言在空转"))
    return bad, f"{'+'.join(scan)} 闸命中 {total}（{' '.join(per)}）"


def a_term_domains(rule, ctx):
    """同一个英文词按**域**分裂成多个中文，逐域钉死。

    这是 `distinct_terms` 的孪生形态，区别在于：`distinct_terms` 管的是**不同英文**不许
    共用中文；`term_domains` 管的是**同一个英文**必须按域保持不同中文 ——
    后者更容易被下一轮「按多数派统一」一刀切掉，因为从中文侧看它就像一处分裂。

    每个域可以由 lang 键（`lang`）或 compendium 英文闸（`gates`）界定，
    并各自带 `cn_forbidden`（＝别的域的中文不许渗进来）。
    """
    bad = []
    total = 0
    per = []
    for dom in rule["domains"]:
        name = dom.get("name", "?")
        n = 0
        forbid = dom.get("cn_forbidden", [])
        for repo, keys in (dom.get("lang") or {}).items():
            langmap = {k: (ev, cv) for _, k, ev, cv in ctx.all_lang([repo])}
            for key, want in keys.items():
                if key not in langmap:
                    bad.append((repo, "lang/cn.json", key,
                                f"这个 lang 键不见了（上游改键名？）—— 域「{name}」失去看守"))
                    continue
                n += 1
                ev, cv = langmap[key]
                if want not in cv:
                    bad.append((repo, "lang/cn.json", key,
                                f"域「{name}」要求中文含「{want}」，实为 {cv[:40]!r}（英文 {ev[:40]!r}）"))
                for f in forbid:
                    if f in cv:
                        bad.append((repo, "lang/cn.json", key,
                                    f"域「{name}」里混进了别域的写法「{f}」：{cv[:40]!r}"))
        for g in (dom.get("gates") or []):
            # ⚠ 字段名要翻译一次：`gates[].en` 与 term_gated 一样是**原始正则**，
            # 而 `_gate_one` 里的 `en` 是**字面词**（会被 re.escape 后套上 \b）。
            # 第一版直接 `{**g}` 传过去，`\b(Fear|Command) Aura\b` 被整个 escape 成字面串，
            # 闸命中 0 —— 幸亏 min_hits 把它抓出来了，这正是护栏该干的事。
            h, b = _gate_one({"en_gate": g["en"], "cn": g.get("cn_required", ""),
                              "cn_forbidden": g.get("cn_forbidden", []),
                              "except_paths": g.get("except_paths"),
                              "case_sensitive": g.get("case_sensitive")},
                             ctx, ["compendium"], rule.get("scope"))
            n += h
            bad.extend(b)
        total += n
        per.append(f"{name}:{n}")

    min_hits = rule.get("min_hits", 1)
    if total < min_hits:
        bad.append(("-", "配置", rule["id"],
                    f"三域合计只命中 {total}（要求 ≥{min_hits}）—— 这条断言在空转"))
    return bad, f"合计 {total}（{' / '.join(per)}）"


def a_lang_parity(rule, ctx):
    bad = []
    detail = []
    for name, pkg in rule["packages"].items():
        repo = ctx.repos.get(name)
        if not repo:
            continue
        cn_p = os.path.join(repo, "lang", "cn.json")
        en_p = os.path.join(os.path.expandvars(pkg), "lang", "en.json")
        if not (os.path.exists(cn_p) and os.path.exists(en_p)):
            bad.append((name, "-", cn_p, "lang 文件找不到，无法核对"))
            continue
        cn = dict(walk(json.load(open(cn_p, encoding="utf-8-sig"))))
        en = dict(walk(json.load(open(en_p, encoding="utf-8-sig"))))
        detail.append(f"{name}: cn {len(cn)} / en {len(en)}")
        if len(cn) != len(en):
            bad.append((name, "-", "lang/cn.json", f"键数 {len(cn)} != 英文侧 {len(en)}"))
    return bad, " | ".join(detail)


def a_anchor_ids(rule, ctx):
    pat = re.compile(r"<h[1-6][^>]*\sid=", re.IGNORECASE)
    n = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        n += len(pat.findall(cv))
    if n < rule["min"]:
        return [("-", "-", "-", f"标题显式 id 只有 {n} 个，低于阈值 {rule['min']}")], f"实测 {n}"
    return [], f"实测 {n} 个（阈值 {rule['min']}）"


def a_no_bilingual_tail(rule, ctx):
    fields = tuple(rule["fields"])
    bad = []
    n = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        seg = path.replace("[", ".").split(".")
        # 字段名可能是最后一段（tokenName），也可能是倒数第二段（encounterTokens.<名>）
        if not (seg[-1] in fields or (len(seg) >= 2 and seg[-2] in fields)):
            continue
        n += 1
        if has_bilingual_tail(cv):
            bad.append((repo, pack, path, f"带双语尾巴：{cv[:40]!r}"))
    return bad, f"检查 {n} 个该约定下的叶"


def _exc_key(en, rule):
    """把一个分叉组的英文串压成能在豁免表里查的键。

    短串（专名 / 短标签）直接用原串；长正文用配置里给的**特征词**（`long_keys`），
    因为豁免表记的是「`Adelyne` 那条」而不是整段英文。找不到特征词就退回前 30 字符。
    """
    if len(en) < rule.get("short_len", 40):
        return en
    m = re.search("|".join(rule.get("long_keys", ["Adelyne", r"lookup @name"])), en)
    return m.group(0) if m else en[:30]


# ============================================================ 块对齐（第十八轮 Y2）
#
# 为什么要有这一层：`term_gated` / `distinct_terms` / `sense_gated` 全是**叶级**的，
# 而本项目最大的三块无闸区恰恰都在叶**内部**（三条的 why 里各自写着）：
#   ① R-rank-sense-compendium：混合叶 8 / 无法分类叶 57 整叶不查，纯 GAME 叶不上正向闸
#   ② R-shard-god：同叶单复数并存的 70 叶，负向先行断言结构上就不进闸
#   ③ R-arcturel-vs-arcturian：96% 的叶两个词都有，叶级判据抓不到叶内单处串行
#
# 判据形态取自 `4-临时脚本/2026-08-15-round16/probes/split_dives.py`：标签是机械的、
# 两侧逐字节相同，所以切出来的块数两侧应当相等；不等的**报出来**而不是静默跳过。
#
# ⚠ 与 split_dives 的两处**有意不同**，都是本轮实测逼出来的：
#
# 1. **只按块级标签切，行内标签剥成空格。** split_dives 按 `<[^>]+>` 全切，本轮实测那样太细：
#    中文「定语在前」会把词搬过 `<strong>` 边界 —— `Cora Attunement.description` 的英文块
#    「damage equal to 2 times your attunement rank」，中文的「同调阶位」搬到了前一块
#    「你获得等同于同调阶位 2 倍的」。全切的话正向闸报出 42 处，其中成片是这一类；
#    改成只按 p/li/td/h*/br… 切之后掉到 12 处。段落／列表项／表格格仍然比整叶细一个量级。
#
# 2. **富文本增强器连标签一起涂掉**（`@UUID[…]{标签}` 的花括号也涂）。
#    实测 `A Brush With Death` 的英文是**裸** `@UUID[…]`（Foundry 渲染目标名），中文补了
#    `{阿克图里安}`；不涂标签就是「EN 空 / CN 有」的假阳性。标签另有闸看着
#    （R-arcturian-split 的 `\{Arcturians\}` 域 · R-arcturian-actor-card · scan_uuid_swap）。
_BLOCK_TAG = re.compile(
    r"</?(?:p|div|li|ul|ol|tr|td|th|table|thead|tbody|tfoot|caption|h[1-6]|br|hr|"
    r"section|article|aside|header|footer|blockquote|figure|figcaption|dl|dt|dd|pre)\b[^>]*>",
    re.IGNORECASE)
_INLINE_TAG = re.compile(r"<[^>]+>")
_ENRICHER = [re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?"), re.compile(r"\[\[[^\]]*\]\]")]


def split_blocks(s):
    """按块级标签切块；每块内的行内标签与增强器涂成等长空格（保持位置，便于人对照）。"""
    for p in _ENRICHER:
        s = p.sub(lambda m: " " * len(m.group()), s)
    return [_INLINE_TAG.sub(" ", b) for b in _BLOCK_TAG.split(s)]


def _class_re(spec, flags):
    """`[{"re": …, "cls": "X"}, …]` → (合并后的正则, 组名→类名)。

    合并成一条带命名组的交替式，**顺序即优先级**（正则交替是最左优先），
    所以 `Shard Goddess` 必须写在 `Shard Gods` 前面、`阿克图里安人` 写在 `阿克图里安` 前面。
    ⚠ 不用 `m.lastgroup` 取类：条目自身可以带内层括号，内层组一旦参与匹配
    `lastgroup` 就会变成 None。改为按插入顺序找第一个非 None 的命名组。
    """
    parts, names = [], {}
    for i, item in enumerate(spec):
        g = f"c{i}"
        names[g] = item["cls"]
        parts.append(f"(?P<{g}>{item['re']})")
    return re.compile("|".join(parts), flags), names


def _classes(rx, names, text):
    out = []
    for m in rx.finditer(text):
        for g, cls in names.items():
            if m.group(g) is not None:
                out.append(cls)
                break
    return out


def _block_exempt(rule, path, i, en_sig="", cn_sig=""):
    """`except_blocks` 的每一条必须**四项全中**（路径后缀 + 块号 + 两侧类串）。

    故意写这么死：块号会随译文改动漂移，漂了就重新报出来让人看 —— 这个方向是对的。
    宁可将来红一次让人重判，也不要一条谁也不记得为什么存在的死豁免（R-dives-mine 的教训）。

    命中返回该条在 `except_blocks` 里的**下标**，没命中返回 `None`。
    ⚠ 返回下标而不是 True，是因为调用方要数**每条豁免各被用了几次** —— 见 `_unused_exempt`。
    下标 0 是假值，所以调用方必须写 `is not None`，不能写 `if j:`。
    """
    for j, e in enumerate(rule.get("except_blocks", [])):
        if (path.endswith(e["path"]) and e["block"] == i
                and e.get("en", en_sig) == en_sig and e.get("cn", cn_sig) == cn_sig):
            return j
    return None


def _unused_exempt(rule, used, field="except_blocks", label="块"):
    """**死豁免闸**：`except_blocks` 里一次都没命中的条目要当场吵出来。返回 (违规行, 死豁免条数)。

    `field` / `label` 只是为了让第十九轮新增的 `enricher_slot_gate` 复用同一套逻辑
    （它的豁免表叫 `except_slots`、单位是「槽」）—— **不许把这段判据复制第二份**，
    两份判据迟早分叉，那正是本项目反复吃亏的形态（见 `_run_same_en_split` 的注释）。

    这是本文件通则的**第四种空转形态**（前三种写在模块 docstring 里：判据写坏 / 没读库 /
    读的是库的过期快照）。这一种最隐蔽，因为**闸本身照常在跑、照常全绿**：
    豁免不命中只让 detail 里的 `n_exempt` 变小一点，**没有任何断言会因此变红**。

    实测代价（第十八轮收尾，复核单元实跑）：两条块级断言的豁免表里合计躺着 **7 条**
    再也匹配不到的条目 —— `R-rank-sense-blocks` 5 条（Shine On 块 4/14/31 与
    The Old Flame 块 264/268，都是升报后译文已经改对了）、`R-arcturel-arcturian-blocks`
    2 条（Sadri Zhalimorne 块21 与 Constructed Companion 块7，同样已修复）——
    而全套依旧 52 通过 / 0 失败，**只能靠人记**。

    这正是 `R-dives-mine` 的形态：欠账还清了、豁免却留着，于是**遮住了未来的回潮** ——
    下一轮同一个块再错回去，会被一条谁也不记得为什么存在的豁免直接吞掉。

    默认上限 **0**：豁免一旦不再命中就必须由人决定「删掉」还是「块号漂了要重判」，
    不许静静地留着。`max_unused_exempt` 可以调高，但调高就要在 `why` 里说明为什么。
    """
    dead = [(j, e) for j, e in enumerate(rule.get(field, [])) if not used[j]]
    cap = rule.get("max_unused_exempt", 0)
    if len(dead) <= cap:
        return [], len(dead)
    return ([("-", "配置", f"{rule['id']} {field}[{j}]",
              f"这条豁免一次都没命中（{e['path']} {label}{e.get('block', e.get('en', ''))}）—— **死豁免**："
              f"要么它登记的内容欠账已经修完了、该删掉，要么块号／类串漂了、该重新判一次。"
              f"留着它只会遮住同一处将来的回潮（R-dives-mine 形态）"
              + (f"；本条允许 {cap} 条未命中，实有 {len(dead)} 条" if cap else ""))
             for j, e in dead], len(dead))


def _iter_blocks(rule, ctx, leaf_re):
    """产出 (repo, pack, path, 英文块, 中文块)；块数不等的直接算 shape 异常。"""
    exc = _paths_matcher(rule.get("except_paths"))
    n_leaf = n_shape = 0
    shape_bad = []
    rows = []
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if not leaf_re.search(ev):
            continue
        if exc and exc(path):
            continue
        n_leaf += 1
        eb, cb = split_blocks(ev), split_blocks(cv)
        if len(eb) != len(cb):
            n_shape += 1
            shape_bad.append((repo, pack, path,
                              f"块级标签结构两侧不同（{len(eb)} vs {len(cb)}）——本条无法逐块判，"
                              f"该由 scan_markup_drift 先处理"))
            continue
        rows.append((repo, pack, path, eb, cb))
    return rows, n_leaf, n_shape, shape_bad


def a_block_aligned_gate(rule, ctx):
    """按块把英文与中文的**术语类别**对齐。两种 mode，各有各的适用面：

    `sequence` —— 两侧类别**序列必须逐位相等**。最强的一档，能抓「叶内单处串行」
        （城名写成族名）与「漏译一处」。代价是对语序调换敏感，所以只用在中文承载得住
        逐位对应的那些二分上。实测 `Arcturel`/`Arcturian`：1714 块里 1705 块逐位相等。

    `count_ge` —— 中文各类计数**不得少于**英文（可多不可少），另可对指定类加反向存在闸。
        用在中文语法**扛不住**逐位对应的地方。实测 `Shard God`：按单／复数逐位对齐，
        779 块里 46 块不齐，逐条看过**全部是合法中文** —— 中文不标复数，且惯于把
        `the Shard God X` 译成「碎片诸神之一的 X」、把 `three Shard Gods of Fire and four of
        Battle` 拆成「三位火焰之碎片之神和四位战斗之碎片之神」。**是判据不成立，不是译文错**，
        所以单复数不进类。中文真正承载得住的是「女神 vs 神」，那一支用反向闸一处不许错。
        「可多不可少」放过的是**代词还原**（英文 they → 中文点名），实测 13 块残差
        无一例外是中文多；抓得住的是整块漏译一处、把女神并进神、把某一类整支改名。

    反空转：`min_leaves`（闸下多少叶）与 `min_blocks`（有词块数）双护栏，
    detail 里必报「扫了多少叶 / 多少块」——满足本文件通则。
    `max_shape_mismatch` 默认 0：块级标签结构不齐的叶算失败，因为那等于**本条判不了它**，
    而判不了必须吵出来，不能静静地算通过。
    `max_unused_exempt` 默认 0：`except_blocks` 里一次都没命中的条目算失败 —— 理由与实测
    代价见 `_unused_exempt` 的 docstring（那是**闸照常全绿**却已经不设防的第四种形态）。
    """
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    leaf_re = re.compile(rule["leaf_gate"], flags)
    en_rx, en_names = _class_re(rule["en_tokens"], flags)
    cn_rx, cn_names = _class_re(rule["cn_tokens"], 0)
    mode = rule.get("mode", "sequence")

    rows, n_leaf, n_shape, bad = _iter_blocks(rule, ctx, leaf_re)
    n_block = n_ok = n_exempt = 0
    used = [0] * len(rule.get("except_blocks", []))
    for repo, pack, path, eb, cb in rows:
        for i, (e, c) in enumerate(zip(eb, cb)):
            ee = _classes(en_rx, en_names, e)
            cc = _classes(cn_rx, cn_names, c)
            if not ee and not cc:
                continue
            n_block += 1
            en_sig, cn_sig = "".join(ee), "".join(cc)
            why = None
            if mode == "sequence":
                if ee != cc:
                    why = f"块内类别序列不对齐：英文 {en_sig or '∅'} / 中文 {cn_sig or '∅'}"
            elif mode == "count_ge":
                ec, cnt = {}, {}
                for k in ee:
                    ec[k] = ec.get(k, 0) + 1
                for k in cc:
                    cnt[k] = cnt.get(k, 0) + 1
                short = [f"「{k}」类英文 {n} 处、中文只有 {cnt.get(k, 0)} 处" for k, n in ec.items()
                         if cnt.get(k, 0) < n]
                back = [f"中文出现「{k}」类而英文块内没有" for k in rule.get("backward_classes", [])
                        if cnt.get(k, 0) and not ec.get(k, 0)]
                if short or back:
                    why = "；".join(short + back) + f"（英文 {en_sig or '∅'} / 中文 {cn_sig or '∅'}）"
            else:
                why = f"未知 mode {mode!r}"
            if why is None:
                n_ok += 1
                continue
            j = _block_exempt(rule, path, i, en_sig, cn_sig)
            if j is not None:
                used[j] += 1
                n_exempt += 1
            else:
                bad.append((repo, pack, f"{path} 块{i}", why))

    dead_rows, n_dead = _unused_exempt(rule, used)
    bad.extend(dead_rows)
    detail = (f"{mode} 闸：闸下 {n_leaf} 叶（结构不齐 {n_shape}）· 有词块 {n_block} 块"
              f"（对齐 {n_ok} · 已登记豁免 {len(used)} 条 / 命中 {n_exempt} 块 / 死豁免 {n_dead} 条）")
    if n_shape > rule.get("max_shape_mismatch", 0):
        bad.append(("-", "配置", rule["id"],
                    f"块级标签结构不齐的叶有 {n_shape}（上限 {rule.get('max_shape_mismatch', 0)}）"))
    for field, got, why in (("min_leaves", n_leaf, "闸下叶数"), ("min_blocks", n_block, "有词块数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{why}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"正则被 JSON 转义吃掉了？上游改了措辞？切块规则被改坏了？"))
    return bad, detail


def a_block_sense_gate(rule, ctx):
    """`sense_gated` 的块级版：把义项分类的窗口从**整叶**收到**块内**，于是块内单一义项
    的地方终于能上**正向闸**（叶级版明确写着「故意不做正向闸」，因为纯 GAME 的 230 叶里
    有 13 叶中文正当地没有定译）。

    块级带来两处判据修正，都是实测逼出来的，都是**收紧分类、不是放宽闸**：
      · `strong_game`：块小了之后 COMMON 的 `ranks of` 会咬到 `Ranks of attunement
        progression`（叶级时那一叶别处还有 GAME、落进 MIX 桶所以从没暴露）。同调／魂印／
        `Rank N` 是本系统的机制专名，优先级必须高于 COMMON 的泛化措辞。
      · `exempt`：`rank of exhaustion`（＝层）· `close ranks`（＝并肩结阵）·
        `join their ranks`（＝加入他们）是**第三个义项**，本来就不归「阶位」那条裁决管。
        块内出现即整块不判 —— 这是把它们从分类里摘出去，不是给它们放行。

    ⚠ **判据边界**：块内混合义项（MIX）与无法分类（UNKNOWN）仍然不判 —— 但注意这里的
    「不判」比叶级小得多：叶级是整叶 57 片不判，块级只是那一段落不判，同叶其它段落照判。
    """
    occ = re.compile(rule["occ"], re.IGNORECASE)
    leaf_re = re.compile(rule.get("leaf_gate", rule["occ"]), re.IGNORECASE)
    game = re.compile(rule["sense"]["game"], re.IGNORECASE)
    common = re.compile(rule["sense"]["common"], re.IGNORECASE)
    strong = re.compile(rule["sense"]["strong_game"], re.IGNORECASE)
    exempt = re.compile(rule["sense"]["exempt"], re.IGNORECASE)
    win = rule.get("window", 90)
    need = rule["cn"]

    rows, n_leaf, n_shape, bad = _iter_blocks(rule, ctx, leaf_re)
    k = {"GAME": 0, "COMMON": 0, "MIX": 0, "UNKNOWN": 0, "EXEMPT": 0}
    n_block = n_exempt = 0
    used = [0] * len(rule.get("except_blocks", []))
    for repo, pack, path, eb, cb in rows:
        for i, (e, c) in enumerate(zip(eb, cb)):
            ms = list(occ.finditer(e))
            if not ms:
                continue
            n_block += 1
            seen = set()
            for m in ms:
                w = e[max(0, m.start() - win): m.end() + win]
                if exempt.search(w):
                    seen.add("EXEMPT")
                elif strong.search(w):
                    seen.add("GAME")
                elif common.search(w):
                    seen.add("COMMON")
                elif game.search(w):
                    seen.add("GAME")
                else:
                    seen.add("UNKNOWN")
            bucket = ("EXEMPT" if "EXEMPT" in seen else "UNKNOWN" if "UNKNOWN" in seen
                      else seen.pop() if len(seen) == 1 else "MIX")
            k[bucket] += 1
            if bucket == "GAME" and need not in c and c.strip():
                why = (f"块内 `{rule['occ']}` 全部是机制义，中文这一块却没有「{need}」："
                       f"EN {e.strip()[:60]!r} / CN {c.strip()[:40]!r}")
            elif bucket == "COMMON" and need in c:
                why = (f"块内 `{rule['occ']}` 全部是普通名词义（组织层级／行列／军衔），"
                       f"中文却用了机制义定译「{need}」：EN {e.strip()[:60]!r}")
            else:
                continue
            j = _block_exempt(rule, path, i)
            if j is not None:
                used[j] += 1
                n_exempt += 1
            else:
                bad.append((repo, pack, f"{path} 块{i}", why))

    dead_rows, n_dead = _unused_exempt(rule, used)
    bad.extend(dead_rows)
    detail = (f"闸下 {n_leaf} 叶（结构不齐 {n_shape}）· 含 `{rule['occ']}` 的块 {n_block}"
              f"（机制义 {k['GAME']} / 普通名词义 {k['COMMON']} / 混合 {k['MIX']} / "
              f"无法分类 {k['UNKNOWN']} / 第三义项 {k['EXEMPT']}）· 已登记豁免 {len(used)} 条 / "
              f"命中 {n_exempt} 块 / 死豁免 {n_dead} 条")
    if n_shape > rule.get("max_shape_mismatch", 0):
        bad.append(("-", "配置", rule["id"], f"块级标签结构不齐的叶有 {n_shape}"))
    for field, got, why in (("min_leaves", n_leaf, "闸下叶数"),
                            ("min_blocks", n_block, "含该词的块数"),
                            ("min_game_blocks", k["GAME"], "机制义块数"),
                            ("min_common_blocks", k["COMMON"], "普通名词义块数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{why}只数到 {got}（要求 ≥{want}）—— 这条断言在空转"))
    return bad, detail


def _run_same_en_split(ctx, rule):
    """**现场**跑一遍 `scan_same_en_split.py`，返回 (分叉组, 扫到的英文唯一串数, 出错说明)。

    走子进程而不是 import，是因为该扫描器的分组逻辑整个写在 `main()` 里；
    复制一份到这里就等于开了第二个判据，两边迟早分叉 —— 那正是本项目反复吃亏的形态。
    """
    script = os.path.join(HERE, rule.get("scanner", "scan_same_en_split.py"))
    if not os.path.exists(script):
        return None, 0, f"找不到扫描器 {script}"
    fd, tmp = tempfile.mkstemp(suffix=".same_en.json")
    os.close(fd)
    cmd = [sys.executable, script]
    for d in ctx.repos.values():
        cmd += ["--repo", d]
    cmd += ["--show", "0", "--out", tmp]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return None, 0, f"扫描器退出码 {proc.returncode}：{(proc.stderr or '')[-300:]}"
        groups = json.load(open(tmp, encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001 —— 跑不起来必须判失败，不能判通过
        return None, 0, f"扫描器跑不起来：{exc!r}"
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    m = re.search(r"英文唯一串（有中文的）\s*(\d+)", proc.stdout or "")
    return groups, int(m.group(1)) if m else 0, None


def a_exclusions_closed(rule, ctx):
    """`same_en_split` 的分叉组必须全部已在归档豁免表里 —— **现场重跑扫描器**。

    ⚠ 第十六轮改法，原因是这条断言被实测抓到在**空转**（模块 docstring 的形态 (c)）：
    旧版读的是磁盘上的 `4-临时脚本/2026-08-15-round15/qa2/same_en_final.json`
    ——第十五轮的产物、15 组——而第十六轮库里实跑是 **14 组 / 123 叶**。
    也就是说它**只与「上次谁跑过扫描并写了那个文件」一样新**，库里新分叉出来它不会变红，
    而这条断言存在的唯一理由恰恰是「冒出新组就要有人看」。

    现在的形态：每次都子进程跑一遍 `scan_same_en_split.py`（跑在 `--root` 指定的那棵树上），
    并把「扫到多少条英文唯一串 / 报出多少组多少叶」打进 detail —— 满足本文件的通则
    「任何断言都必须能说出我这次扫了多少」。扫描器跑不起来一律**判失败**，不判通过。

    ⚠ 第十七轮（2026-08-15，A7）**只动了豁免表的位置**，判据形态没变：
    那份 125 条的表原本在 `4-临时脚本/2026-08-13-round12/findings/EXCLUSIONS.json`，
    被 `.gitignore` 的 `4-临时脚本/**/*.json` 挡在仓库外 —— 换台机器 clone 下来它就不存在，
    本函数会走下面「找不到归档豁免表」那条分支报失败。**那是判据环境坏了，不是库坏了**，
    而报出来的样子和真缺陷一模一样。现已挪到 `5-其他内容/EXCLUSIONS.same_en_split.json`
    （路径写在规则的 `exclusions` 字段里，不在本文件里硬编码）。
    ⚠ 它与 `5-其他内容/EXCLUSIONS.json` **是两张不同的表，别合并**：本函数吃的是**裸 list**，
    靠 `` `英文串` `` 这种反引号写法在每条的 what/why 里做子串匹配；那一张是
    `{meta, exclusions:[…]}`，给人每轮读的项目级登记表。合并会当场把本判据打瘸。
    """
    exc_p = os.path.join(ROOT, rule["exclusions"])
    if not os.path.exists(exc_p):
        # 非空 bad ⇒ 本条判**失败**（不是 skipped）。有意如此：表没了就等于没设防，
        # 而「没设防」必须吵出来。detail 里点明这多半是判据环境问题而非库的问题。
        return ([("-", "-", exc_p,
                  "找不到归档豁免表，无法核对 —— 先确认它是不是又被挪回 4-临时脚本/ 被 .gitignore 挡掉了")],
                "归档豁免表缺失（判据环境问题，不是库的问题）")
    exc = json.load(open(exc_p, encoding="utf-8"))
    blob = " ".join(f"{e.get('what','')} {e.get('why','')}" for e in exc)

    groups, n_en, err = _run_same_en_split(ctx, rule)
    if err:
        return [("-", "-", "scan_same_en_split", err + " —— 判失败而不是判通过：跑不了就等于没设防")], "扫描失败"

    bad = []
    loose = 0
    for g in groups:
        en = g.get("en", "")
        key = _exc_key(en, rule)
        if f"`{key}`" in blob:                     # 豁免表的写法是 `Shield` 护盾术(23) / 盾牌(11)
            continue
        if key in blob:                            # 长正文那两条只能松匹配，记下来让人看得见
            loose += 1
            continue
        bad.append(("-", "-", en[:60],
                    "该分叉组不在已归档豁免表里 —— 要么是新缺陷，要么要补一条豁免"))

    n_leaf = sum(g.get("n_leaf", 0) for g in groups)
    min_en = rule.get("min_en_strings", 1000)
    if n_en < min_en:
        bad.append(("-", "-", rule["id"],
                    f"扫描器只报了 {n_en} 条英文唯一串（要求 ≥{min_en}）—— 它多半没读到库，"
                    f"这条断言在空转"))
    return bad, (f"现场扫描：英文唯一串 {n_en} 条 → 分叉 {len(groups)} 组 / {n_leaf} 叶；"
                 f"归档 {len(exc)} 条（其中 {loose} 组靠松匹配过闸）")


def glossary_value_matches(want, got):
    """词表值判据：产物层允许带双语尾巴（既定书写约定），所以只比**中文头部**。

    ⚠ 头部是**第一个空格前的整词**，不是前缀。这一条看着琐碎，实际是个会立刻把断言打红的坑：
    第十六轮收尾要给 `Cosmology` 登记词表值时，按 §8 的散文裁决「Cosmology＝宇宙」写 want='宇宙'，
    而产物是「宇宙观 Cosmology」、头部取到「宇宙观」—— **'宇宙' ≠ '宇宙观'，当场判失败**。
    正解是给这个**词表键**单列 want='宇宙观'（锚点页名本身就是「宇宙观 Cosmology」），
    与「正文里泛指用法的 cosmology/cosmological＝宇宙」区分开：
    一个是**键的值**，一个是**词根的译法**，两者可以不同。

    ⚠ 另一半：base 里的**多义 list**（`Ordain`＝["奥尔丹","授命"]、`Shield`＝[…]）在这里恒判不过。
    那是有意的 —— 多义词不该用「词表值必须等于某一个中文」来看守，
    该用读库英文闸（见 R-ordain-vs-ordani）。所以它们**不该出现在 entries 里**。
    """
    head = got.split(" ")[0] if isinstance(got, str) else got
    return want in (got, head)


def a_glossary_value(rule, ctx):
    """词表的 base 层与产物层**都**必须是这个值。

    只查产物是不够的：词表是构建产物，`build_glossary.py` 会用 base + harvest 重建，
    只改产物下一次构建就退回去（2026-08-13j 的裁决）。所以两层一起查。
    这一条守的是本项目最隐蔽的机制 —— **词表把错误洗成权威**：
    包里改对了，词表停在旧值，下一轮有人拿词表当依据反向回灌。
    """
    bad = []
    checked = 0
    for layer, rel in rule["files"].items():
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.exists(path):
            # base 层在项目根之外（fvtt/），跑在别的树上时可能不存在，跳过而不是假失败
            continue
        d = json.load(open(path, encoding="utf-8"))
        for key, want in rule["entries"].items():
            if key not in d:
                if rule.get("require_present", True):
                    bad.append((layer, os.path.basename(path), key, f"词表里缺这个锚点键（应为「{want}」）"))
                continue
            checked += 1
            got = d[key]
            if not glossary_value_matches(want, got):
                bad.append((layer, os.path.basename(path), key, f"是「{got}」，应为「{want}」"))
    return bad, f"两层合计核对 {checked} 条"


def a_version_matrix(rule, ctx):
    """PROJECT.md 抬头与版本矩阵里的版本号，必须与两仓 `module.json` 一致。

    为什么值得一条断言：这一行**历史上漂过两次**（停在 0.9.6/1.1.7 与 0.9.7/1.1.10 各一次）。
    每次都是「发了版、追加了本轮小节、但没回头改抬头」——纯粹靠人记得，就一定会漏。
    新会话第一件事就是读抬头判断现状，读到过期版本会直接判断错「现在到哪一步了」。
    """
    import re as _re
    doc_p = os.path.join(ROOT, rule.get("doc", "PROJECT.md"))
    if not os.path.exists(doc_p):
        return [("-", "-", doc_p, "找不到 PROJECT.md")], "跳过"
    doc = open(doc_p, encoding="utf-8").read()
    bad = []
    detail = []
    for name, repo in ctx.repos.items():
        mj = os.path.join(repo, "module.json")
        if not os.path.exists(mj):
            continue
        manifest = json.load(open(mj, encoding="utf-8"))
        ver = manifest.get("version")
        pkg = manifest.get("id", name)
        prefix = rule["tag_prefix"].get(name, "")
        want = f"{prefix}{ver}"
        detail.append(f"{name}={want}")
        # 抬头段（前 40 行）里必须出现当前版本
        head = "\n".join(doc.splitlines()[:40])
        if want not in head:
            bad.append((name, "PROJECT.md", "抬头段",
                        f"抬头没有写当前版本 {want}（module.json 是 {ver}）—— 抬头又漂了"))

        # ⚑ 只查抬头是不够的：正文里还有「发版状态：…」那一行，它停在 0.9.4 / v1.1.4
        # 漂了四个版本仍然全绿 —— 断言自己的覆盖面就是个盲区。
        #
        # ⚠ 但**不能全文乱扫**：§1 有大量**有意保留**的历史引用（「上一版抬头：… 0.9.5 / v1.1.5」）、
        # §6 年表更是逐版记录。第一版按「§6 之前都算正文」扫，立刻在那些历史行上报了 8 处假阳性。
        # 正确的判据是**只查声称「现在」的那些行** —— 由 current_markers 明确列出。
        markers = rule.get("current_markers", ["当前已发布", "发版状态", "当前版本"])
        for i, line in enumerate(doc.splitlines(), 1):
            if not any(mk in line for mk in markers):
                continue
            for m in _re.finditer(rf"{_re.escape(pkg)}[` ]+v?(\d+\.\d+\.\d+)", line):
                if m.group(1) != ver:
                    bad.append((name, "PROJECT.md", f"第 {i} 行",
                                f"这一行声称的是**当前状态**，却写着 {pkg} {m.group(1)}，"
                                f"而 module.json 是 {ver}"))
        # 版本矩阵那一行
        row = _re.search(rf"^\|\s*{_re.escape(pkg)}\s*\|.*$", doc, _re.MULTILINE)
        if not row:
            bad.append((name, "PROJECT.md", "版本矩阵", f"矩阵里没有 {pkg} 这一行"))
        elif want not in row.group(0):
            bad.append((name, "PROJECT.md", "版本矩阵",
                        f"矩阵写的不是 {want}：{row.group(0)[:90]}"))
    return bad, " | ".join(detail)


def a_leaf_literal(rule, ctx):
    """指定叶必须／不得包含某个字面串。

    `pack` 与 `path` 都可以写成**列表** —— 本库大量内容是孪生两包各一份
    （`ember.adventure.json` / `ember.crucible-adventure.json`），只钉一包等于放另一包不管。
    `min_leaves` 是这一类的反空转护栏：路径写错、上游改名、只钉到孪生的一半，
    都会让命中数掉下来，而不是静静地按「0 叶待查 = 通过」放行。
    """
    packs = rule["pack"] if isinstance(rule["pack"], list) else [rule["pack"]]
    paths = rule["path"] if isinstance(rule["path"], list) else [rule["path"]]
    bad = []
    checked = 0
    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if pack not in packs or path.removeprefix("entries.") not in paths:
            continue
        checked += 1
        for must in rule.get("must_contain", []):
            if must not in cv:
                bad.append((repo, pack, path, f"必须包含「{must}」但没有：{cv[:40]!r}"))
        for never in rule.get("must_not_contain", []):
            if never in cv:
                bad.append((repo, pack, path, f"不得包含「{never}」但出现了：{cv[:40]!r}"))
    want = rule.get("min_leaves", 1)
    if checked < want:
        bad.append(("-", packs[0], paths[0],
                    f"只命中 {checked} 叶（要求 ≥{want}）—— 这几叶找不到了（上游改名？路径写错？"
                    f"孪生包只钉了一半？），断言失效，需要人看"))
    return bad, f"命中 {checked} 叶（要求 ≥{want}）"


# ====================================================== 增强器槽位（第十九轮 Y6）
#
# 为什么要有这一层：`split_blocks`（见上面它的注释）把 `@X[…]{标签}` **连花括号一起涂空**，
# 于是标签里的译名整条不进任何块级闸。第十八轮把这个洞量化过，第十九轮又逐处复核了一遍：
#
#   全库中文「阿克图瑞尔|阿克图里安」共 **2218 处，其中 278 处（12.5%）落在增强器内**。
#   把那 278 处逐个改错、每处重跑全部读库断言 —— **只有 89 处**会让某条断言变红，
#   且全是**叶级间接覆盖**（只有该叶里这一处是唯一带该词的地方时才响）；**其余 189 处无闸**。
#
# ⚠ 第十八轮那句「标签另有 R-arcturian-split 的 `{Arcturians}` 域 / R-arcturian-actor-card /
# scan_uuid_swap 看着」**逐条落实全不成立**，已在第十八轮末尾按实测改写：
# `R-arcturian-actor-card` 只钉 `actors.Arcturian` 的 name/tokenName 四叶，那四叶里根本没有增强器；
# `scan_uuid_swap` 根本不在断言套里（是独立扫描器），它自己的 docstring 也写明这类不归它管。
# **教训：`why` 里写「另有 X 闸看着」是可证伪断言，写之前必须逐处变异回测。**
#
# 本闸与 `scan_label_vs_name` 的分工（**不许做成它的重复**）
# ------------------------------------------------------
# `scan_label_vs_name` 比的是「`@UUID{标签}` 的中文 ↔ **目标文档的中文 name**」，
# 且只报「英文标签本来就等于目标英文 name」的那些 —— 它管的是**标签与目标是否同名**。
# 本闸管的是另一件事：标签里的**术语**（地名／族名／专名）与既定译名是否一致，
# 而标签本身**可以**与目标 name 不同（作者有意换称呼时 scan_label_vs_name 直接不看）。
# 举例：`@UUID[…]{Arcturel Dives}` 的中文写成「阿克图里安矿渊」——
# 标签与目标 name 的关系没变（都不是目标名），scan_label_vs_name 一声不吭，本闸报红。
#
# 做成断言（新 kind）而不是独立扫描器的理由
# ----------------------------------------
# 判据要用的两侧叶文本 `ctx.pairs` 已经在内存里；独立扫描器要**重读一遍 4 万叶**
# 再走一次子进程。`a_exclusions_closed` 走子进程是因为分组逻辑本来就写在
# `scan_same_en_split.main()` 里、复制一份就等于开第二个判据 —— 本条没有这个前提。
#
# ⚠ 配对不按「出现序号」，按 **(动词, 目标)** 分组后组内取序 —— 这是实测改的
# ------------------------------------------------------------------------
# 第一版按整叶出现序号逐个配对（Y6 任务书里写的做法），实测 **30650 对里有 1388 对
# （4.5%）配歪**：中文「定语在前」经常把整个增强器搬到另一位置
# （`Lake Jinro Lunar Shrine` / `Mythspire Observatory` 成片如此）。配歪的后果不是漏报而是
# **假阳性**：`EN=Arcturel / CN=杰夫赫尔家族`、`EN=Arcturian / CN=奥尔丹`、
# `EN=Arcturian Liquor / CN=烧瓶` 这些「一眼看去像串行」的条目，全部是配对错位造出来的幻影。
# 改成按 (动词, 目标) 分组后，全库 **30650 个增强器 0 个配不上**，那 4 类幻影同时消失。
# 目标相同的多个增强器（同一目标在一叶里出现多次）在组内仍按出现序取，位置精度不丢。
_AT_ENR = re.compile(r"@([A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?")
_ENR_PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*"([^"]*)"')
# 增强器里**玩家能看见的文本**：花括号标签，以及 `@Embed[… label="…" readaloud="…"]`
# 的这两个参数 —— 实测 `readaloud=` 里塞的是整段朗读正文（**48 处**，
# EN 侧 16966 字符 / 最长 **951** 字，CN 侧 5392 字符 / 最长 311 字），
# ⚠ 上一版写「46 处，最长 300+ 字」两处都不准：漏的 2 处在**小写** `@embed[` 里
# （本文件的 `_AT_ENR` 是 `@([A-Za-z]+)\[`，吃小写，所以那 2 处一样在定义域内）；
# 「最长 300+」只对 CN 侧成立，EN 侧最长是 951。
# 那整段同样被 split_blocks 涂空，是本洞里最大的单块。其余参数（`count=` `classes=`）
# 是机械值，不进槽位。
_ENR_TEXT_PARAMS = ("label", "readaloud")


def _enr_key(m):
    """(动词小写, 目标) —— 目标取方括号内第一个空格前的串（`@Embed` 后面还跟着参数）。"""
    return m.group(1).lower(), m.group(2).split(" ", 1)[0]


def _enr_slots(m, wanted):
    out = {}
    if m.group(3) is not None:
        out["label"] = m.group(3)
    for pm in _ENR_PARAM.finditer(m.group(2)):
        n = pm.group(1).lower()
        if n in wanted:
            out["param:" + n] = pm.group(2)
    return out


def _enr_pairs(ev, cv, wanted):
    """按 (动词,目标) 分组、组内取序配对。返回 (配对列表, 配不上的个数)。"""
    eg, cg = {}, {}
    for m in _AT_ENR.finditer(ev):
        eg.setdefault(_enr_key(m), []).append(m)
    for m in _AT_ENR.finditer(cv):
        cg.setdefault(_enr_key(m), []).append(m)
    pairs, unpaired = [], 0
    for k in set(eg) | set(cg):
        a, b = eg.get(k, []), cg.get(k, [])
        n = min(len(a), len(b))
        for x, y in zip(a[:n], b[:n]):
            pairs.append((k[1], _enr_slots(x, wanted), _enr_slots(y, wanted)))
        unpaired += abs(len(a) - len(b))
    return pairs, unpaired


def a_enricher_slot_gate(rule, ctx):
    """增强器**槽位级**的术语闸：逐个「英文标签 ↔ 中文标签」比既定译名。

    判据（正向为主，反向可选）
    --------------------------
    * `en_tokens` / `cn_tokens` 与 `block_aligned_gate` 同一套有序交替式（`_class_re`），
      **顺序即优先级**，所以派生词要写在词根前面。
    * 正向：英文槽里出现某一类 → 中文槽必须出现同一类。**这一条就能抓住全部串行**
      （把 `Arcturel` 的中文写成族名，正向闸立刻少一个 E 类）。
    * 反向 `forbid_absent`：中文槽出现了英文槽里没有的类 → 违规。抓的是「凭空多出一个专名」。
      默认**关**：中文标签合法地补上下文的情形不少（`{Trinkets}`→「阿克图里安小饰品」），
      开之前必须先在当前库上实跑一遍看假阳性。

    覆盖不到的地方（照本项目规矩写死，不许含糊）
    -------------------------------------------
    * **中文有标签而英文那一侧是裸增强器**的槽（实测 582 个）没有英文槽可比
      （Foundry 对裸 `@UUID` 渲染目标文档名，中文侧补个标签是**正当做法**）。
      `cn_only_leaf_fallback` 打开后退回**整叶英文**，且**只做反向闸**：
      中文槽里出现的类，整叶英文里必须出现过；反过来**不要求**中文槽含整叶英文的类。
      ⚠ 这个方向是实测逼出来的：第一版写成正向（整叶英文只有一类 → 中文槽必须含该类），
      当场造出 **70 条假阳性** —— 一叶英文里提到 Arcturian，不代表这叶里每个
      中文标签（`{月华花}` `{拉斯克}` `{迷惘}`）都得带族名。反向则站得住：
      中文标签凭空冒出一个整叶英文里根本没有的专名，那要么是串行要么是加戏。
      ⚠ 反向闸的合法英文依据有**两个来源**，缺一个就还会假阳性（也是实测出来的）：
      ① 本叶英文；② **同一目标在全库别处的英文标签**。裸 `@UUID` 由 Foundry 渲染
      目标文档名，中文补个标签正是在写那个名字 —— `@UUID[RollTable.BUurPyycyDIuox5L]`
      在本叶英文里一个 Arctur 字样都没有，而全库别处 28 次写着 `{Arcturian Trinkets}`，
      中文写「阿克图里安小饰品」当然是对的。只认来源 ① 会把这 4 槽误报。
    * 方括号**内部**不判：那是机器参数，既定约定要求照抄英文（`[[/culture Ordani]]`）。
    * **只认 `@Verb[…]` 形态，不认 `[[/verb …]]{标签}`。** 后者实测中文侧 20318 个、
      带花括号标签的 349 个（含中文 344 个），拿七条断言的全部已裁术语去扫这 349 个标签
      **一处都不命中**（伤害类型／灾害名／法术名居多）。所以这不是「漏了」而是**量过、当前为空**；
      哪天已裁术语进了那一格，这里要跟着扩。
    * 增强器配不上对的（实测 0 个）**报出来**，不静默跳过。

    反空转：`min_leaves` / `min_slots`（英文侧有类命中的槽数）双护栏，
    detail 必报「扫了多少叶 / 配了多少对 / 判了多少槽」。
    `max_unused_exempt` 默认 0，与两个块级闸共用 `_unused_exempt`。
    """
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    en_rx, en_names = _class_re(rule["en_tokens"], flags)
    cn_rx, cn_names = _class_re(rule["cn_tokens"], 0)
    wanted = tuple(rule.get("params", _ENR_TEXT_PARAMS))
    slot_filter = set(rule["slots"]) if rule.get("slots") else None
    exc = _paths_matcher(rule.get("except_paths"))
    fallback = rule.get("cn_only_leaf_fallback", False)

    bad = []
    n_leaf = n_pair = n_unpaired = n_slot = n_gated = n_ok = 0
    n_cn_only = n_fb_gated = n_exempt = 0
    used = [0] * len(rule.get("except_slots", []))

    # 预扫：目标 -> 全库英文标签里出现过的类。只有 `cn_only_leaf_fallback` 用得上，
    # 所以不开就不扫（省一遍 4 万叶）。
    tgt_cls = {}
    if fallback:
        for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
            if "@" not in ev:
                continue
            for m in _AT_ENR.finditer(ev):
                for txt in _enr_slots(m, wanted).values():
                    c = _classes(en_rx, en_names, txt)
                    if c:
                        tgt_cls.setdefault(_enr_key(m)[1], set()).update(c)

    for repo, pack, path, ev, cv in ctx.all_pairs(rule.get("scope")):
        if "@" not in ev and "@" not in cv:
            continue
        if exc and exc(path):
            continue
        n_leaf += 1
        pairs, unpaired = _enr_pairs(ev, cv, wanted)
        n_pair += len(pairs)
        if unpaired:
            n_unpaired += unpaired
            bad.append((repo, pack, path,
                        f"有 {unpaired} 个增强器两侧按 (动词,目标) 配不上对 —— 本条判不了它们，"
                        f"该先由 scan_markup_drift / scan_markup_targets 处理"))
        leaf_en_cls = None
        for tgt, es, cs in pairs:
            for slot in set(es) | set(cs):
                if slot_filter and slot not in slot_filter:
                    continue
                n_slot += 1
                et, ct = es.get(slot), cs.get(slot)
                if ct is None:
                    continue                       # 英文有标签、中文裸 —— 中文侧没字可判
                cc = _classes(cn_rx, cn_names, ct)
                if et is None:
                    n_cn_only += 1
                    if not fallback or not cc:
                        continue
                    if leaf_en_cls is None:
                        leaf_en_cls = set(_classes(en_rx, en_names, ev))
                    allow = leaf_en_cls | tgt_cls.get(tgt, set())
                    ee, src = sorted(allow), "整叶英文+同目标别处英文标签"
                    n_fb_gated += 1
                    miss, extra = [], [c for c in dict.fromkeys(cc) if c not in allow]
                else:
                    ee, src = _classes(en_rx, en_names, et), "英文槽"
                    # ⚠ 英文槽一个类都没命中时**不能直接跳过** —— 反向闸要抓的正是
                    #   「英文槽里没有、中文槽里冒出来」这一形态（`{the Tradeway}`→
                    #   「阿克图里安贸易道」）。第一版写成 `if not ee: continue`，
                    #   于是 forbid_absent 在最该响的那一格永远响不了，自检当场抓出来。
                    if not ee and not (rule.get("forbid_absent") and cc):
                        continue
                    n_gated += 1
                    miss = [c for c in dict.fromkeys(ee) if c not in cc]
                    extra = ([c for c in dict.fromkeys(cc) if c not in ee]
                             if rule.get("forbid_absent") else [])
                if not miss and not extra:
                    n_ok += 1
                    continue
                why = (f"{src}是 {'/'.join(dict.fromkeys(ee))} 类，中文槽是 "
                       f"{'/'.join(dict.fromkeys(cc)) or '∅'} 类"
                       + (f"（缺 {'/'.join(miss)}）" if miss else "")
                       + (f"（多出 {'/'.join(extra)}）" if extra else "")
                       + f"：EN {(et or '（英文侧裸增强器）')[:60]!r} / CN {ct[:40]!r}")
                j = None
                for k2, e2 in enumerate(rule.get("except_slots", [])):
                    if (path.endswith(e2["path"]) and e2.get("slot", slot) == slot
                            and e2.get("en", et) == et and e2.get("cn", ct) == ct):
                        j = k2
                        break
                if j is not None:
                    used[j] += 1
                    n_exempt += 1
                else:
                    bad.append((repo, pack, f"{path} → {tgt[:34]} [{slot}]", why))

    dead_rows, n_dead = _unused_exempt(rule, used, "except_slots", "槽 ")
    bad.extend(dead_rows)
    detail = (f"闸下 {n_leaf} 叶 · 配对增强器 {n_pair} 个（配不上 {n_unpaired}）· "
              f"可见文本槽 {n_slot}（英文槽有类命中 {n_gated}"
              + (f" + 整叶回退 {n_fb_gated}" if fallback else "")
              + f" · 中文独有槽 {n_cn_only} · 判过 {n_gated + n_fb_gated} 一致 {n_ok}）· "
              f"已登记豁免 {len(used)} 条 / 命中 {n_exempt} 槽 / 死豁免 {n_dead} 条")
    for field, got, txt in (("min_leaves", n_leaf, "闸下叶数"),
                            ("min_slots", n_slot, "可见文本槽数"),
                            ("min_gated", n_gated + n_fb_gated, "真正判过的槽数")):
        want = rule.get(field)
        if want is not None and got < want:
            bad.append(("-", "配置", rule["id"],
                        f"{txt}只数到 {got}（要求 ≥{want}）—— 这条断言在空转："
                        f"正则被 JSON 转义吃掉了？上游改了标签措辞？增强器切法被改坏了？"))
    if n_unpaired > rule.get("max_unpaired", 0):
        bad.append(("-", "配置", rule["id"],
                    f"配不上对的增强器有 {n_unpaired}（上限 {rule.get('max_unpaired', 0)}）"))
    return bad, detail


KINDS = {
    "term_gated": a_term_gated,
    "cn_absent": a_cn_absent,
    "sense_gated": a_sense_gated,
    "distinct_terms": a_distinct_terms,
    "term_domains": a_term_domains,
    "lang_parity": a_lang_parity,
    "anchor_ids": a_anchor_ids,
    "no_bilingual_tail": a_no_bilingual_tail,
    "exclusions_closed": a_exclusions_closed,
    "leaf_literal": a_leaf_literal,
    "block_aligned_gate": a_block_aligned_gate,
    "block_sense_gate": a_block_sense_gate,
    "enricher_slot_gate": a_enricher_slot_gate,
    "glossary_value": a_glossary_value,
    "version_matrix": a_version_matrix,
}


# ----------------------------------------------------------------- 自检

SELFTEST = [
    # (值, 是否应判为双语尾巴)  —— 正例来自库内真实写法，反例来自本项目既定约定
    ("螯蛛艾斯 Cheliceraeth", True),
    ("血鸟 Gore Bird", True),
    ("赛洛克弓手 Thayloc Courser", True),
    ("圣堂区路人 A", False),        # 单字母编号，英文侧就是 Hallows Passerby A
    ("菌丝旷野底图 B", False),      # 同上
    ("软泥池 2", False),            # 数字编号
    ("环境音效（2）", False),        # 全角括注
    ("斥候", False),                # 纯裸中文
    ("莫伊雷语", False),
    ("瀑布外部东北", False),
]


# 第十六轮补：读库闸本身的正反例。
#
# 光测「双语尾巴」是不够的 —— 上一次事故（`distinct_terms` 空转）恰恰出在**闸的机制**上，
# 而不是出在某个字符串判据上。这里用一棵合成的假库把 `_gate_one` 的四种行为钉住：
#   命中/不命中 · 占位符剥离 · cn_gate 放宽形态 · cn_forbidden。
class _FakeCtx:
    def __init__(self, lang=(), pairs=()):
        self._lang = list(lang)
        self._pairs = list(pairs)

    def all_lang(self, scope=None):
        return iter(self._lang)

    def all_pairs(self, scope=None):
        return iter(self._pairs)


GATE_SELFTEST = [
    # (说明, term, ctx, scan, 期望命中, 期望违规数)
    ("lang 英文闸命中且中文正确",
     {"en": "Region Map", "cn": "地区地图"},
     _FakeCtx(lang=[("ember", "K1", "Region Map", "地区地图")]), ["lang"], 1, 0),
    ("lang 英文闸命中但中文写反 —— 这正是四个版本没人发现的那种",
     {"en": "Region Map", "cn": "地区地图"},
     _FakeCtx(lang=[("ember", "K1", "Region Map", "区域地图")]), ["lang"], 1, 1),
    ("占位符 {rank} 要先剥掉，否则这条会被误判成违规",
     {"en": "Rank", "cn": "阶位"},
     _FakeCtx(lang=[("cruc", "K2", "{rank} vs. {defense}", "{rank} 对抗 {defense}")]),
     ["lang"], 0, 0),
    ("cn_gate 放宽：Level 的定译是「等级」，但「{x} 级」也合法",
     {"en": "level", "cn": "等级", "cn_gate": "级"},
     _FakeCtx(lang=[("ember", "K3", "Level {x}", "{x} 级")]), ["lang"], 1, 0),
    ("cn_forbidden：闸内出现禁用写法要响",
     {"en": "Cyclonic", "cn": "气旋", "cn_forbidden": ["旋风的"]},
     _FakeCtx(pairs=[("ember", "p.json", "a.text", "Cyclonic blast", "气旋冲击（旋风的）")]),
     ["compendium"], 1, 1),
    ("except_paths：登记过的叶不计命中也不报违规",
     {"en": "Cyclonic", "cn": "气旋", "except_paths": ["items.Multiattack"]},
     _FakeCtx(pairs=[("ember", "p.json", "actors.X.items.Multiattack.description",
                      "[[/item Cyclonic]]", "用 [[/item Cyclonic]] 攻击")]),
     ["compendium"], 0, 0),
    ("英文不命中就不该产生任何检查",
     {"en": "Region Map", "cn": "地区地图"},
     _FakeCtx(lang=[("ember", "K4", "Area Map", "区域地图")]), ["lang"], 0, 0),
]


# 第十六轮终段补：`sense_gated` 的正反例。
#
# 这个类型比 `_gate_one` 更容易写坏，因为它有**三条**可以各自失效的判据
# （反向闸 / COMMON 专属闸 / 上下文分类），而且分类是靠一个窗口正则做的 ——
# 窗口取错、忘了剥标签、COMMON 与 GAME 的优先级写反，都会静静地把闸变成摆设。
# 下面六条把这三件事各钉一次，其中「剥标签」那条来自实测：`Bonus | Rank | Scale`
# 在原文里被 `<td>` 拆成三格，不剥标签窗口里根本凑不出 `Bonus`（split_rank.py 第一版栽过）。
_SENSE_RULE = {
    "id": "SELFTEST", "kind": "sense_gated", "en": r"\branks?\b", "cn": "阶位",
    "sense": {"game": r"(Novice|training|skill|Attunement|Soulbound|Rank\s*\d|\bBonus\b|Scale)",
              "common": r"(civic|social|ranks of|rank[- ]and[- ]file|rank as an? )"},
    "window": 90,
}

SENSE_SELFTEST = [
    ("普通名词义专属叶里出现「阶位」→ 要响",
     [("e", "p.json", "a.text", "It denotes their civic rank in the city.", "标示其在城中的阶位")], 1),
    ("同一叶普通名词义、中文写「地位」→ 不响",
     [("e", "p.json", "a.text", "It denotes their civic rank in the city.", "标示其在城中的公民地位")], 0),
    ("机制义叶中文没有「阶位」→ **不响**（正向闸是故意不做的，见 docstring 判据边界）",
     [("e", "p.json", "a.text", "You gain the Novice rank in Arcana.", "你在奥秘上获得新手层级")], 0),
    ("反向闸：中文有「阶位」而英文一个 rank 都没有 → 要响",
     [("e", "p.json", "a.text", "Tier 3 creatures are dangerous.", "3 阶位的生物很危险")], 1),
    ("COMMON 优先级高于 GAME：窗口里同时有 skill 和 ranks of 时判 COMMON",
     [("e", "p.json", "a.text", "grit is required to endure the training to join the ranks of the order",
       "需要毅力才能熬过训练加入该教团的阶位")], 1),
    ("必须先剥标签：`Bonus|Rank|Scale` 被 <td> 拆开，剥了才判得出 GAME（判 GAME 就不报）",
     [("e", "p.json", "a.text", "<tr><td>Bonus</td><td>Rank</td><td>Scale</td></tr>",
       "<tr><td>加值</td><td>阶位</td><td>尺度</td></tr>")], 0),
]


# 第十六轮收尾补：`glossary_value` 的头部判据正反例。
#
# 加这一组的直接理由是它**真的咬过一次**：按 §8 的散文裁决把 `Cosmology` 登成 want='宇宙'，
# 产物「宇宙观 Cosmology」的头部是「宇宙观」，断言当场变红。那不是库错了，是登记值选错了 ——
# 而这种「判据形状决定了该登什么」的知识，只写在 why 里会随人走，钉成正反例才留得下来。
GLOSSARY_SELFTEST = [
    # (说明, want, got, 期望)
    ("base 层裸中文、与登记值相同 → 过", "宇宙观", "宇宙观", True),
    ("产物层双语尾巴，只比中文头部 → 过", "宇宙观", "宇宙观 Cosmology", True),
    ("⚠ 头部是**整词**不是前缀：want='宇宙' 对 got='宇宙观 Cosmology' 必须**不过**",
     "宇宙", "宇宙观 Cosmology", False),
    ("同一个坑的另一半：want='宇宙' 对裸中文「宇宙观」也必须不过", "宇宙", "宇宙观", False),
    ("英文尾巴有多个词时靠 got 全等那一支过（头部只切到第一个空格）",
     "凯西安沙刀", "凯西安沙刀 Kessian Sand Knife", True),
    ("base 里的多义 list（Ordain / Shield）恒判不过 —— 它们不该出现在 entries 里",
     "奥尔丹", ["奥尔丹", "授命"], False),
]


# 第十八轮 Y2 补：两个块对齐类型的正反例。
#
# 这两个类型比前面所有类型都更容易「看起来在跑、其实判不着」，因为它们有**四件**
# 可以各自失效的东西：切块规则 · 类别正则的顺序 · 对齐判据本身 · 豁免匹配。
# 下面把四件各钉一次，其中三条直接来自本轮实测踩到的坑：
#   · 行内标签**不**切块（中文定语在前会把词搬过 `<strong>`，全切会造出成片假阳性）
#   · `count_ge` 而非逐位对齐（中文不标复数，S/P 逐位对齐这个判据本身不成立）
#   · 混合义项叶里的**单一义项块**必须判得动（这正是叶级 sense_gated 整叶放弃的那片）
_ALIGN_RULE = {
    "id": "SELFTEST-align", "kind": "block_aligned_gate", "mode": "sequence",
    "leaf_gate": r"\bArctur", "min_leaves": 1, "min_blocks": 1,
    "en_tokens": [{"re": r"\bArcturians?\b", "cls": "I"}, {"re": r"\bArcturel\b", "cls": "E"}],
    "cn_tokens": [{"re": "阿克图里安人", "cls": "I"}, {"re": "阿克图里安", "cls": "I"},
                  {"re": "阿克图瑞尔", "cls": "E"}],
}
_COUNT_RULE = {
    "id": "SELFTEST-count", "kind": "block_aligned_gate", "mode": "count_ge",
    "leaf_gate": r"\bShards? God", "min_leaves": 1, "min_blocks": 1,
    "backward_classes": ["F"],
    "en_tokens": [{"re": r"\bShards? Goddess(?:es)?\b", "cls": "F"},
                  {"re": r"\bShards? Gods?\b", "cls": "G"}],
    "cn_tokens": [{"re": "碎片女神", "cls": "F"}, {"re": "碎片诸神", "cls": "G"},
                  {"re": "碎片之神", "cls": "G"}],
}


def _P(en, cn):
    return [("e", "p.json", "j.Some Page.text", en, cn)]


BLOCK_ALIGN_SELFTEST = [
    ("逐块对齐、两侧一致 → 不响", _ALIGN_RULE,
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图里安住所。</p>"), 0),
    ("⚑ **叶内单处串行** → 要响（这正是叶级判据看不见的：整叶两个中文都在，闸会放行）",
     _ALIGN_RULE,
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图瑞尔住所。</p>"), 1),
    ("语序调换会响 —— 这是逐位对齐**已知的代价**，实测 1714 块里只有 1 块，登记在 except_blocks",
     _ALIGN_RULE, _P("<p>Arcturian shops of Arcturel</p>", "<p>阿克图瑞尔的阿克图里安商铺</p>"), 1),
    ("行内标签**不**切块：`<strong>` 两侧的词算同一块（全切的话这条会假阳性）",
     _ALIGN_RULE, _P("<p>the <strong>Arcturel</strong> Dives</p>", "<p>阿克图瑞尔矿渊</p>"), 0),
    # ⚠ 这一条的 leaf_gate 必须真的在**英文原文**里命中，否则整叶根本不进闸、
    #   测到的就只是 min_leaves 在响。实测第一版写成纯 @UUID 的英文，闸下 0 叶、假绿。
    ("增强器标签两侧都涂掉：英文裸 @UUID、中文补了 {标签} → 不响",
     _ALIGN_RULE, _P("<p>the @UUID[Actor.x] bard of Arcturel</p>",
                     "<p>这位@UUID[Actor.x]{阿克图里安}吟游诗人来自阿克图瑞尔</p>"), 0),
    # 结构不齐会**同时**从三个方向吵：该叶一条 + max_shape_mismatch 超限一条 +
    # 该叶被跳过导致 min_blocks 掉到 0 一条。三条都该在 —— 判不了就是不能算通过。
    ("块级标签结构两侧不同 → 要响（判不了必须吵出来，不能当通过）",
     _ALIGN_RULE, _P("<p>Arcturel</p><p>Arcturian</p>", "<p>阿克图瑞尔 阿克图里安</p>"), 3),
    ("英文闸一块都没命中 → min_blocks 把空转抓出来",
     _ALIGN_RULE, _P("<p>Nothing here.</p>", "<p>这里什么都没有。</p>"), 2),
    ("count_ge：单／复数**不进类**，`Shard Gods` 对「碎片之神」不响（中文不标复数）",
     _COUNT_RULE, _P("<p>Shard Gods are mortal ascendants.</p>", "<p>碎片之神是飞升的凡人。</p>"), 0),
    ("count_ge：代词还原让中文多出一处 → 不响（可多不可少）",
     _COUNT_RULE, _P("<p>A Shard God arrived. They blessed it.</p>",
                     "<p>一位碎片之神到来。碎片诸神赐下祝福。</p>"), 0),
    ("count_ge：块内两处英文、中文只译了一处 → 要响",
     _COUNT_RULE, _P("<p>Shard God A fought Shard God B.</p>", "<p>碎片之神 A 与 B 交战。</p>"), 1),
    ("count_ge：把「碎片女神」并进「碎片之神」→ 要响（女神那一支中文扛得住，一处不许错）",
     _COUNT_RULE, _P("<p>the shard goddess Scoris</p>", "<p>碎片之神斯科里斯</p>"), 1),
    ("count_ge 反向闸：英文没有 Goddess 而中文写了「碎片女神」→ 要响",
     _COUNT_RULE, _P("<p>the Shard Gods gathered</p>", "<p>碎片女神们聚集</p>"), 1),
    # ⚑ 死豁免闸（`max_unused_exempt`，默认 0）。这两条钉的是**第四种空转形态**：
    #    闸本身照常跑、照常全绿，只是豁免表里躺着再也匹配不到的条目。实测代价见
    #    `_unused_exempt` 的 docstring（收尾时两条块级断言合计躺着 7 条，全套仍 52/0）。
    ("⚑ 豁免一条都没命中 → 要响（死豁免自己吵出来，不再只能靠人记）",
     dict(_ALIGN_RULE, except_blocks=[
         {"path": "j.Nowhere At All.text", "block": 99, "why": "自检：故意留一条永远匹配不到的"}]),
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图里安住所。</p>"), 1),
    ("同一条豁免真的命中 → 不响（证明上一条响的是「没命中」而不是「有豁免」）",
     dict(_ALIGN_RULE, except_blocks=[
         {"path": "j.Some Page.text", "block": 3, "en": "I", "cn": "E", "why": "自检：这条真命中"}]),
     _P("<p>Arcturel is a city.</p><p>Arcturian dwellings.</p>",
        "<p>阿克图瑞尔是一座城。</p><p>阿克图瑞尔住所。</p>"), 0),
]

_SENSE_BLOCK_RULE = {
    "id": "SELFTEST-bsense", "kind": "block_sense_gate", "occ": r"\branks?\b",
    "leaf_gate": r"\branks?\b", "cn": "阶位", "window": 90,
    "min_leaves": 1, "min_blocks": 1,
    "sense": {
        "strong_game": r"(attunement|attuned|soulbound|soulmark|Rank\s*\d)",
        "game": r"(Novice|training|skill|Attunement|Soulbound|exhaustion|Scale|Rank\s*\d|\bBonus\b)",
        "common": r"(ranks? (depending|based) on|civic|social|ranks of|rank[- ]and[- ]file|rank as an? )",
        "exempt": r"(ranks? of[^.]{0,40}?exhaust|close ranks|join(ing)? their ranks)",
    },
}

# ⚠ 第二项是这一条自检往 `_SENSE_BLOCK_RULE` 上打的**规则覆盖**（`None` ＝不覆盖）。
# 原来的写法是「note 里含 'except_blocks' 就塞一张表」，靠**注释文字**驱动判据 ——
# 加第二条豁免用例时它当场就不够用了（几条都含那个词、却要各自不同的表），所以改成显式一列。
BLOCK_SENSE_SELFTEST = [
    ("⚑ **正向闸**：块内全机制义、中文没有「阶位」→ 要响（叶级 sense_gated 故意不做这个方向）",
     None, _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手层级。</p>"), 1),
    ("块内全机制义、中文有「阶位」→ 不响",
     None, _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手阶位。</p>"), 0),
    ("反向闸：块内全普通名词义、中文却用「阶位」→ 要响",
     None, _P("<p>It denotes their civic rank.</p>", "<p>标示其公民阶位。</p>"), 1),
    ("⚑ **混合义项叶里的单一义项块判得动** —— 叶级版对这一叶是 MIX、整叶放弃",
     None, _P("<p>It denotes their civic rank.</p><p>You gain the Novice rank in Arcana.</p>",
              "<p>标示其公民地位。</p><p>你在奥秘上获得新手层级。</p>"), 1),
    ("第三义项 `rank of exhaustion`（＝层）不归这条裁决管 → 不响，哪怕中间夹着增强器",
     None, _P("<p>Each character gains one rank of &amp;Reference[exhaustion] and must save.</p>",
              "<p>每名角色获得一级力竭，并且必须豁免。</p>"), 0),
    ("strong_game 压过 common：`Ranks of attunement progression` 是机制义，不是「行列」",
     None, _P("<p>There are now five full Ranks of attunement progression.</p>",
              "<p>现在同调进阶共有完整的五个阶位。</p>"), 0),
    ("行内标签**不**切块：中文把「同调阶位」搬到了 `<strong>` 前面 → 不响（全切会假阳性）",
     None, _P("<p>You gain resistance to <strong>Acid</strong> damage equal to 2 times "
              "your attunement rank.</p>",
              "<p>你获得等同于同调阶位 2 倍的<strong>强酸</strong>伤害抗性。</p>"), 0),
    ("组织内部层级 `ranks depending on experience and skill` 判 COMMON，中文写「等级」不响",
     None, _P("<p>Within the Guard there are a number of ranks depending on experience and skill.</p>",
              "<p>卫队内部依照经验与技能设有多个等级。</p>"), 0),
    ("登记过的块不再报（登记的是内容欠账，必须同时升报）",
     {"except_blocks": [{"path": "j.Some Page.text", "block": 1, "why": "自检：这条真命中"}]},
     _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手层级。</p>"), 0),
    # ⚑ 死豁免闸在 block_sense_gate 这一侧的正反例。与 block_aligned_gate 那边成对，
    #    因为两个类型各有一份自己的 `used` 计数，只钉一边等于另一边没测。
    ("⚑ 豁免一条都没命中 → 要响（`max_unused_exempt` 默认 0，见 _unused_exempt）",
     {"except_blocks": [{"path": "j.Nowhere.text", "block": 99, "why": "自检：永远匹配不到"}]},
     _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手阶位。</p>"), 1),
    ("把上限显式放宽到 1 → 同一条死豁免不再响（证明响的是「未命中」本身，不是「有豁免」）",
     {"except_blocks": [{"path": "j.Nowhere.text", "block": 99, "why": "自检：永远匹配不到"}],
      "max_unused_exempt": 1},
     _P("<p>You gain the Novice rank in Arcana.</p>", "<p>你在奥秘上获得新手阶位。</p>"), 0),
]


_SLOT_RULE = {
    "id": "SELFTEST-slot", "kind": "enricher_slot_gate", "forbid_absent": True,
    "min_leaves": 1, "min_slots": 1, "min_gated": 1,
    "en_tokens": [{"re": r"\bArcturel\w*", "cls": "E"}, {"re": r"\bArcturians?\b", "cls": "I"}],
    "cn_tokens": [{"re": "阿克图瑞尔", "cls": "E"}, {"re": "阿克图里安", "cls": "I"}],
}

# ⚑ 增强器槽位闸（第十九轮 Y6）的正反例。
#
# 这一套的重点不是「术语比得对不对」（那和 block_aligned_gate 同一套 `_class_re`，
# 已经在那边钉过），而是**这个类型特有的三件事**，每一件都是实测踩出来的：
#   ① 配对按 (动词, 目标) 而不是出现序号 —— 全库 4.5% 的增强器被中文语序搬过位；
#   ② 中文独有槽的回退方向是**反向**，且合法英文依据有两个来源（本叶英文 + 同目标别处的英文标签）；
#   ③ 方括号内部不判、`@Embed` 的 label/readaloud 参数要判。
SLOT_SELFTEST = [
    ("标签两侧一致 → 不响", None,
     _P("<p>@UUID[JournalEntry.a]{Arcturel} and @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.a]{阿克图瑞尔}与@UUID[JournalEntry.b]{阿克图里安人}</p>"), 0),
    ("⚑ **标签里单处串行** → 要响（这正是块级闸看不见的：split_blocks 把标签整条涂空）", None,
     _P("<p>@UUID[JournalEntry.a]{Arcturel} and @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.a]{阿克图里安}与@UUID[JournalEntry.b]{阿克图里安人}</p>"), 1),
    ("⚑ **中文把两个增强器搬了位** → 不响（按 (动词,目标) 配对；按出现序号配会造出 2 条幻影）", None,
     _P("<p>@UUID[JournalEntry.a]{Arcturel} shops of @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.b]{阿克图里安}的@UUID[JournalEntry.a]{阿克图瑞尔}商铺</p>"), 0),
    # ⚠ 这两条把 `min_gated` 显式放到 0：它们本来就该「一个类都判不到」，
    #   不放宽的话响的是反空转护栏而不是被测的那件事，测了个寂寞。
    ("方括号**内部**不判：目标串里出现术语也不看（既定约定要求照抄英文）", {"min_gated": 0},
     _P("<p>@UUID[Compendium.ember.x.Arcturel]{the city}</p>",
        "<p>@UUID[Compendium.ember.x.Arcturel]{这座城}</p>"), 0),
    ("`@Embed` 的 readaloud 参数是可见正文，要判 → 中文串行时要响", None,
     _P('<p>@Embed[Actor.z readaloud="These Arcturians are wary."]</p>',
        '<p>@Embed[Actor.z readaloud="这些阿克图瑞尔人心存戒备。"]</p>'), 1),
    ("反向闸 forbid_absent：英文槽没有的类中文槽冒出来 → 要响", None,
     _P("<p>@UUID[JournalEntry.a]{the Tradeway}</p>", "<p>@UUID[JournalEntry.a]{阿克图里安贸易道}</p>"), 1),
    ("英文有标签、中文裸增强器 → 不响（中文侧没字可判，不是缺陷）", {"min_gated": 0},
     _P("<p>@UUID[JournalEntry.a]{Arcturel}</p>", "<p>@UUID[JournalEntry.a]</p>"), 0),
    # ↓ 中文独有槽（英文裸 @UUID、Foundry 渲染目标名）的三条。回退**只做反向**。
    ("⚑ 中文独有槽：回退关着 → 不响", None,
     _P("<p>Arcturel: @UUID[JournalEntry.a] @UUID[JournalEntry.c]{Arcturel}</p>",
        "<p>阿克图瑞尔：@UUID[JournalEntry.a]{阿克图里安} @UUID[JournalEntry.c]{阿克图瑞尔}</p>"), 0),
    ("⚑ 中文独有槽：回退开着、类不在本叶英文里 → 要响", {"cn_only_leaf_fallback": True},
     _P("<p>Arcturel: @UUID[JournalEntry.a] @UUID[JournalEntry.c]{Arcturel}</p>",
        "<p>阿克图瑞尔：@UUID[JournalEntry.a]{阿克图里安} @UUID[JournalEntry.c]{阿克图瑞尔}</p>"), 1),
    ("⚑ 中文独有槽：类不在本叶英文里，但**同一目标在别处的英文标签**是这一类 → 不响",
     {"cn_only_leaf_fallback": True},
     [("e", "p.json", "j.A.text", "<p>Arcturel: @UUID[JournalEntry.a]</p>",
       "<p>阿克图瑞尔：@UUID[JournalEntry.a]{阿克图里安小饰品}</p>"),
      ("e", "p.json", "j.B.text", "<p>@UUID[JournalEntry.a]{Arcturian Trinkets}</p>",
       "<p>@UUID[JournalEntry.a]{阿克图里安小饰品}</p>")], 0),
    ("⚑ 正向回退是错的：本叶英文有 I 类，不代表每个中文标签都得带族名 → 不响",
     {"cn_only_leaf_fallback": True},
     _P("<p>Arcturians live here. @UUID[JournalEntry.a] @UUID[JournalEntry.c]{Arcturel}</p>",
        "<p>阿克图里安人住在这里。@UUID[JournalEntry.a]{月华花} @UUID[JournalEntry.c]{阿克图瑞尔}</p>"), 0),
    ("英文槽一个类都没命中 → min_gated 把空转抓出来", None,
     _P("<p>@UUID[JournalEntry.a]{Nothing}</p>", "<p>@UUID[JournalEntry.a]{什么都没有}</p>"), 1),
    ("登记过的槽不再报（登记的是已裁的合法例外）",
     {"except_slots": [{"path": "j.Some Page.text", "en": "Arcturel", "cn": "阿克图里安",
                        "why": "自检：这条真命中"}]},
     _P("<p>@UUID[JournalEntry.a]{Arcturel} @UUID[JournalEntry.b]{Arcturians}</p>",
        "<p>@UUID[JournalEntry.a]{阿克图里安} @UUID[JournalEntry.b]{阿克图里安人}</p>"), 0),
    ("⚑ 豁免一条都没命中 → 要响（死豁免闸在本类型这一侧也要有自己的用例）",
     {"except_slots": [{"path": "j.Nowhere.text", "en": "x", "cn": "y", "why": "自检：永远匹配不到"}]},
     _P("<p>@UUID[JournalEntry.a]{Arcturel}</p>", "<p>@UUID[JournalEntry.a]{阿克图瑞尔}</p>"), 1),
    ("把上限显式放宽到 1 → 同一条死豁免不再响（证明响的是「未命中」本身）",
     {"except_slots": [{"path": "j.Nowhere.text", "en": "x", "cn": "y", "why": "自检：永远匹配不到"}],
      "max_unused_exempt": 1},
     _P("<p>@UUID[JournalEntry.a]{Arcturel}</p>", "<p>@UUID[JournalEntry.a]{阿克图瑞尔}</p>"), 0),
]


def run_selftest():
    bad = 0
    for value, want in SELFTEST:
        got = has_bilingual_tail(value)
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  {flag} {value!r:32s} 期望 {want} 实得 {got}")
    print(f"\n双语尾巴判据：{len(SELFTEST) - bad} / {len(SELFTEST)} 通过")

    print("\n读库闸（_gate_one）正反例：")
    gbad = 0
    for note, term, ctx, scan, want_h, want_b in GATE_SELFTEST:
        h, b = _gate_one(term, ctx, scan, None)
        ok = (h == want_h and len(b) == want_b)
        if not ok:
            gbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望 命中{want_h}/违规{want_b}，实得 命中{h}/违规{len(b)}")
    print(f"\n读库闸：{len(GATE_SELFTEST) - gbad} / {len(GATE_SELFTEST)} 通过")

    print("\n机制义／普通名词义分类闸（sense_gated）正反例：")
    sbad = 0
    for note, pairs, want_b in SENSE_SELFTEST:
        b, detail = a_sense_gated(_SENSE_RULE, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            sbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nsense_gated：{len(SENSE_SELFTEST) - sbad} / {len(SENSE_SELFTEST)} 通过")

    print("\n词表值头部判据（glossary_value）正反例：")
    vbad = 0
    for note, want, got, expect in GLOSSARY_SELFTEST:
        ok = glossary_value_matches(want, got) == expect
        if not ok:
            vbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        want={want!r} got={got!r} 期望 {expect}")
    print(f"\nglossary_value：{len(GLOSSARY_SELFTEST) - vbad} / {len(GLOSSARY_SELFTEST)} 通过")

    print("\n块对齐闸（block_aligned_gate）正反例：")
    abad = 0
    for note, rule, pairs, want_b in BLOCK_ALIGN_SELFTEST:
        b, detail = a_block_aligned_gate(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            abad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nblock_aligned_gate：{len(BLOCK_ALIGN_SELFTEST) - abad} / {len(BLOCK_ALIGN_SELFTEST)} 通过")

    print("\n块级义项闸（block_sense_gate）正反例：")
    bbad = 0
    for note, override, pairs, want_b in BLOCK_SENSE_SELFTEST:
        rule = dict(_SENSE_BLOCK_RULE, **(override or {}))
        b, detail = a_block_sense_gate(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            bbad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nblock_sense_gate：{len(BLOCK_SENSE_SELFTEST) - bbad} / {len(BLOCK_SENSE_SELFTEST)} 通过")

    print("\n增强器槽位闸（enricher_slot_gate）正反例：")
    ebad = 0
    for note, override, pairs, want_b in SLOT_SELFTEST:
        rule = dict(_SLOT_RULE, **(override or {}))
        b, detail = a_enricher_slot_gate(rule, _FakeCtx(pairs=pairs))
        ok = len(b) == want_b
        if not ok:
            ebad += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {note}")
        print(f"        期望违规 {want_b}，实得 {len(b)}　（{detail}）")
    print(f"\nenricher_slot_gate：{len(SLOT_SELFTEST) - ebad} / {len(SLOT_SELFTEST)} 通过")
    return 1 if (bad or gbad or sbad or vbad or abad or bbad or ebad) else 0


def main():
    # Windows 控制台默认 gbk，规则里的 ⚠ / ▸ 会直接把脚本炸成 UnicodeEncodeError，
    # 而那看起来像「断言崩了」。与本目录其它扫描器统一：输出一律走 utf-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                              # noqa: BLE001 —— 老 Python / 被重定向时无所谓
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--repo", action="append", help="限定仓库（默认两个都跑）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-show", type=int, default=6)
    ap.add_argument("--selftest", action="store_true", help="只跑判据自身的正反例回测")
    ap.add_argument("--root", help="改用另一棵项目树（灵敏度回测用：往副本里注入违规，确认断言真的会响）")
    a = ap.parse_args()

    if a.selftest:
        return run_selftest()

    rules = json.load(open(a.rules, encoding="utf-8"))
    global ROOT
    if a.root:
        ROOT = os.path.abspath(a.root)
    repos = {}
    for name, rel in REPOS.items():
        d = os.path.join(ROOT, rel)
        if a.repo and rel not in a.repo and name not in a.repo:
            continue
        if os.path.isdir(d):
            repos[name] = d
    ctx = Ctx(repos, rules.get("meta"))
    total_leaves = sum(len(v) for v in ctx.pairs.values())
    total_keys = sum(len(v) for v in ctx.lang.values())
    print(f"读入 {len(repos)} 个仓库 / {total_leaves} 对中英叶 / {total_keys} 对中英 lang 键\n")
    if not total_keys:
        print("⚠ lang 通道一个键都没读到 —— meta.lang_sources 指向的上游安装目录不在？\n"
              "  依赖 lang 闸的断言会以「空转」形态报失败，那是**判据环境问题**，不是库的问题。\n")

    failed = passed = skipped = 0
    for rule in rules["assertions"]:
        fn = KINDS.get(rule["kind"])
        if not fn:
            print(f"  ?? {rule['id']}: 未知断言类型 {rule['kind']}")
            skipped += 1
            continue
        try:
            bad, detail = fn(rule, ctx)
        except Exception as exc:                       # 断言自己炸了也要说清楚，不能静默
            print(f"  !! {rule['id']}: 断言执行出错 {exc!r}")
            failed += 1
            continue
        if bad:
            failed += 1
            print(f"  FAIL  {rule['id']}  —— {rule['title']}")
            print(f"        裁决 {rule['decision']}：{rule['why']}")
            print(f"        {detail}，违反 {len(bad)} 处：")
            for repo, pack, path, why in bad[:a.max_show]:
                print(f"          [{repo}/{pack}] {str(path)[:78]}")
                print(f"            {why}")
            if len(bad) > a.max_show:
                print(f"          …另 {len(bad) - a.max_show} 处")
        else:
            passed += 1
            if a.verbose:
                print(f"  ok    {rule['id']}  {rule['title']}  （{detail}）")

    print(f"\n{'=' * 62}")
    print(f"通过 {passed} / 失败 {failed} / 跳过 {skipped}")
    if failed:
        print("\n⚠ 失败的每一条都对应第 8 节的一条既定裁决。")
        print("  正确的处理是：要么改回来，要么**显式推翻那条裁决并同时改断言**——")
        print("  不要只改断言让它变绿，那正是这套东西要防的事。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
