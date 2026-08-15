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
    return 1 if (bad or gbad or sbad or vbad) else 0


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
