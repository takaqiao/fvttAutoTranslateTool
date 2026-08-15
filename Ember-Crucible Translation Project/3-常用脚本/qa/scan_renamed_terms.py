# -*- coding: utf-8 -*-
"""Upstream-renamed / deleted proper nouns whose Chinese rendering is still in cn.

WHY THIS EXISTS (PROJECT.md sec.5.4 + sec.8 `Arcturel Upper`):
  `port_orphans.py` only moves paths, it never touches the translated text.  When
  upstream renames a thing, the *old* name survives inside the Chinese ——
  `Rune: Lightning`->`Rune: Storm` left 28 stale 「闪电」; `Arcturel Upper/Lower`
  left an English tail that does not exist anywhere in today's English baseline.
  Every existing gate is blind to this: markup signature matches, number coverage
  matches, length ratio is normal, the Chinese prose reads fine.

TWO DETECTORS (same criterion, two viewing angles):

  A. `en-tail`  —— the English half of a 「中文 English」 bilingual name (or any
     Title-Case English run sitting inside the Chinese) does not exist in the
     CURRENT English baseline.  This is the purest form: the stale name is
     literally written out.  `scan_latin_residue.py` deliberately whitelists
     bilingual tails, so nothing else looks at them.

  B. `cn-term`  —— an English proper noun that existed in the OLD baseline and is
     gone from the CURRENT one, whose Chinese rendering is still present in
     compendium/cn.  Needs an English gate, because a Chinese word can have more
     than one English source (「闪电」is both `Lightning` and a metaphor).

MANDATORY EXCLUSIONS (each one is a real lesson, do not remove):
  * variant spelling —— `Reaper Ocean` vs `Reaver Ocean` (sec.8, 2026-08-11).  A
    name is only "gone" if no case/plural/hyphen/apostrophe/edit-distance-1
    variant survives in the current baseline.
  * English gate —— a finding only stands when the leaf's own CURRENT English
    contains nothing that could explain the Chinese.  Same three-bucket logic as
    `4-临时脚本/2026-08-12-fix/term_gate.py`.
  * common Chinese words are held to a stricter bar (see --max-cn-leaves).

Precision beats recall here: 5 true findings are worth more than 200 with 190
false ones.

⚠ 基准缺包 = 探测器 B 静默不扫（第十九轮 Y5 的同型盲区，第二十轮补上）
----------------------------------------------------------------------
探测器 B 的候选**全部**来自 `--old` 基准目录里的包（`load_pack_leaves(a.old)`）。
基准里没有的包，其旧英文专名从来没进过候选表 —— 而报告长得和「扫过、干净」一样。
实测 2026-08-15：`--repo 1-Ember汉化插件 --old ember-cn-v1.0.15-shipped-en`
→ `old-en 14878` 叶，只来自 **3** 个包（`_repaired.json` + adversary + character），
仓里却有 10 个 en 包 / 9 个有中文，A/B 两段各报 0。
这条尤其要紧：本闸是 §5.4 第 11 项，**多条断言在 `why` 里点名依赖它兜底**
（如 `R-moon-hollow`：「`cn_absent` 抓不到上游换了尊号而中文没跟，那要靠
`scan_renamed_terms`」）—— 那句依赖此前对 ember 的另外 6 个包**不成立**。
⚠⚠ **第二十轮已补上出口，别再照旧结论说「补不回来」**：那 6 个包的正确基准不是
`ember-cn-v1.0.15-shipped-en/`（v1.0.15 那会儿模块只有 3 个 crucible 侧包），而是
**`ember-cn-v1.1.0-shipped-en/`** —— 它们是 v1.1.0 才加进来的，而 v1.1.0 的 `compendium/en/`
一直躺在 `1-Ember汉化插件` 自己的 git 历史里。拿 v1.1.0 那份跑：**扫 10 包 / 缺 0 / 零告警**。
⚠ **但只换基准还不够**（第二十轮 A/B 实证）：本探测器还要求该术语**挖得成候选、且配得出中文写法**。
月亮尊号的散文形态 `{Aura}, The Hollow Moon,` 根本进不了候选；标签形态
`{Aura - The Hollow Moon}` 进得了，但词表里只有裸键 `The Hollow Moon`、配不出整串中文，
于是 `renderings` 为空、**静默 continue**。第二十轮已往 base 词表补了三条整串键
（`Aura - The Hollow Moon` = 奥拉 - 空洞之月，等三条），兜底这才真正成立。
**通则：说「某某闸兜着」之前，要把「基准覆盖得到」和「术语配得出中文」两个前提都验一遍。**

现在每次运行都印「本次扫 N 个包 / 仓里共 M 个包 / 缺 K 个」以及缺掉的中文叶数；
`--strict-coverage` 让缺包直接以退出码 3 结束。语义与
`scan_number_drift.py` / `scan_dropped_terms.py` / `scan_marker_followup.py` /
`scan_en_drift.py` 完全一致。

范围注记：**探测器 A 不受基准缺包影响** —— 它扫的是当前 `compendium/cn` 全部包、
对照当前 `compendium/en` 全部包，`--old` 只用来给结论标注 `in_old_baseline`。
覆盖行报的是**探测器 B 的定义域**（也就是 `--old` 的包覆盖）。

Usage:
  python scan_renamed_terms.py --repo "<repoDir>" --old "<baselineDir>" \
         [--mode both|en-tail|cn-term] [--out x.json] [--show 40] \
         [--strict-coverage]
"""
import argparse, json, os, re, sys, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# leaves whose content is legitimately Latin / not prose
SKIP_PATH_RX = re.compile(
    r"(?:^|\.)(?:pronunciation|img|src|texture|type|folder|sort|uuid|id|_id"
    r"|scale|width|height|elevation|key|mode|priority|value|formula)$", re.I)

CJK_RX = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# ---------------------------------------------------------------- tree walking


def leaves(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            leaves(v, path + [str(k)], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, path + [str(i)], out)
    elif isinstance(obj, str):
        out.append((".".join(path), obj))


def load_pack_leaves(d):
    """dir -> {relname: [(path, str)]}, skipping _source.json and broken JSON.

    旧基准目录的文件名与当前 `compendium/en` 不同名，必须归一化，否则整包漏读：
      * v1.0.15 那份用了 `-en.json` 后缀；
      * 战役正文的 `ember.crucible-adventure-en.json` **本身就是损坏 JSON**
        （PROJECT.md 3.5：第 44 行多一个 `}`），`_repaired.json` 才是修好的那份。
    `sorted()` 让 `_repaired.json` 排在 `ember.*-en.json` 前面，配合 setdefault ＝先到先得，
    与 `scan_en_drift.baseline_packs()` 保持同一套语义。用赋值会让损坏的那个把修好的覆盖掉。
    """
    res = {}
    if not os.path.isdir(d):
        return res
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn == "_source.json":
            continue
        key = ("ember.crucible-adventure.json" if fn == "_repaired.json"
               else fn.replace("-en.json", ".json"))
        if key in res:                      # 先到先得：修好的那份已经收下了
            continue
        try:
            j = json.load(open(os.path.join(d, fn), encoding="utf-8-sig"))
        except Exception as e:
            print(f"  ! skip unreadable {fn}: {e}", file=sys.stderr)
            continue
        out = []
        leaves(j.get("entries", {}), ["entries"], out)
        res[key] = out
    return res


# ----------------------------------------------------------- 基准覆盖了几个包
def cjk_leaf_count(rows):
    """一个 cn 包里**含中文**的叶数 —— 探测器 B 能追的残留就住在这些叶里。"""
    return sum(1 for _p, s in (rows or []) if CJK_RX.search(s))


def coverage(repo, baseline, old_packs, cn_packs):
    """扫了几个包 / 仓里共几个包 / 基准缺几个包（连带缺掉多少条中文叶）。

    ⚠ **反空转第 (d) 型的探针**。前三种空转形态（判据写坏 / 压根不读库 / 读旧报告
    快照）的防法对它全部无效：判据没问题、库也读了、报告是当场生成的 —— 只是基准
    目录里**压根没有那个包**，它的旧英文专名一个也没进过探测器 B 的候选表，报告上
    一片绿，实情是没扫。

    与 `scan_number_drift.coverage()` / `scan_dropped_terms.coverage()` /
    `scan_marker_followup.coverage()` / `scan_en_drift.coverage()` 同一套语义。

    与那四条的**唯一实现差异**（有意为之）：`scanned` 不是从目录列表推的，而是取
    `load_pack_leaves(--old)` **真正读成功**的包名。理由是本脚本的 loader 比它们多
    一条 fallback —— `_repaired.json` 若读不出来会回落到 `ember.*-en.json`，
    再照目录推一遍就会与实情脱节。取「真读进来的那些」才是不可空转的口径。
    """
    en_dir = os.path.join(repo, "compendium", "en")
    cn_dir = os.path.join(repo, "compendium", "cn")
    repo_packs = sorted(f for f in os.listdir(en_dir)
                        if f.endswith(".json") and f != "_source.json")
    scanned, missing = [], []
    for f in repo_packs:
        if f in old_packs:
            scanned.append(f)
        else:
            missing.append({"pack": f,
                            "cn_present": os.path.exists(os.path.join(cn_dir, f)),
                            "cn_cjk_leaves": cjk_leaf_count(cn_packs.get(f))})
    return {
        "baseline": os.path.abspath(baseline),
        "repo_en_packs": len(repo_packs),
        "scanned_packs": scanned,
        "missing_packs": missing,
        "cn_cjk_leaves_uncovered": sum(m["cn_cjk_leaves"] for m in missing),
        # 基准里有、仓里已经没有的包（上游删包 / 改名）。不影响本次结果，但要看得见。
        "baseline_only_packs": [f for f in sorted(old_packs) if f not in repo_packs],
    }


def print_coverage(cov):
    print(f"  包覆盖（探测器 B 的定义域）：本次扫 {len(cov['scanned_packs'])} 个包 / "
          f"仓里 en 共 {cov['repo_en_packs']} 个包 / 基准缺 {len(cov['missing_packs'])} 个")
    if cov["missing_packs"]:
        print("  ⚠ 下列包**基准里没有，探测器 B 一条也没看** —— 它们的 0 告警不是「干净」，是「没扫」：")
        for m in cov["missing_packs"]:
            tail = "" if m["cn_present"] else "（无 cn 文件）"
            print(f"       {m['pack']}  含中文的叶 {m['cn_cjk_leaves']}{tail}")
        print(f"     合计未进闸的中文叶 {cov['cn_cjk_leaves_uncovered']}")
        print("     ⚠ 有断言在 why 里写「靠 scan_renamed_terms 兜底」—— 那句话对上列包不成立。")
        print("     历史快照补不回来（那些包发版当时就没捕获）；将来靠 "
              "3-常用脚本/qa/capture_baseline.py 在**升级前**截全包基准。")
    if cov["baseline_only_packs"]:
        print(f"  ⚠ 基准里有、仓里 en 已没有的包 {len(cov['baseline_only_packs'])} 个："
              f"{'、'.join(cov['baseline_only_packs'])}")


# ------------------------------------------------------------ text normalising

RX_REF_SPAN = re.compile(r'<span class="reference">.*?</span>', re.S)
RX_INLINE = re.compile(r"\[\[.*?\]\]", re.S)
RX_ENRICH_LBL = re.compile(r"(?:@|&amp;|&)[A-Za-z]+\[[^\]]*\](?:\{([^}]*)\})?")
RX_TAG = re.compile(r"<[^>]*>")
RX_ENTITY = re.compile(r"&[A-Za-z#0-9]+;")

APOS = {"’": "'", "‘": "'", "ʼ": "'", "´": "'"}
DASH = {"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-"}


def unify(s):
    for a, b in APOS.items():
        s = s.replace(a, b)
    for a, b in DASH.items():
        s = s.replace(a, b)
    return s


SEG = " ¦ "   # markup boundary marker: not whitespace, so a Title-Case run
                   # cannot silently span it.  Without it `<h3>Discord</h3><p>
                   # <a>Discord</a>` collapses to the "phrase" `Discord Discord`
                   # and every heading+first-word pair becomes a fake finding.
                   # norm_spaced() maps it back to a space, so the corpus that
                   # answers "does this name still exist" is unaffected.


def prose(s):
    """Strip functional markup, keep enricher labels (they are visible text)."""
    s = RX_REF_SPAN.sub(SEG, s)
    s = RX_INLINE.sub(SEG, s)
    s = RX_ENRICH_LBL.sub(lambda m: SEG + (m.group(1) or "") + SEG, s)
    s = RX_TAG.sub(SEG, s)
    s = RX_ENTITY.sub(" ", s)
    return unify(s)


RX_NONWORD = re.compile(r"[^a-z0-9]+")


def flat(s):
    return RX_NONWORD.sub("", unify(s).lower())


RX_WORD = re.compile(r"[A-Za-z]+")


def words_lower(s):
    return [w.lower() for w in RX_WORD.findall(s)]


def norm_spaced(s):
    """lowercase, punctuation -> single space; for word-boundary phrase search."""
    return " " + re.sub(r"[^a-z0-9]+", " ", unify(s).lower()).strip() + " "


# ------------------------------------------------------------------- SymSpell-1


def deletes1(w):
    return {w[:i] + w[i + 1:] for i in range(len(w))}


class WordIndex:
    """distance-<=1 neighbour lookup over a word set (SymSpell deletes)."""

    def __init__(self, words):
        self.words = set(words)
        self.idx = collections.defaultdict(set)
        for w in self.words:
            if len(w) < 4:
                continue
            self.idx[w].add(w)
            for d in deletes1(w):
                self.idx[d].add(w)

    def has(self, w):
        return w in self.words

    def near(self, w):
        """words within edit distance 1 (excluding exact)."""
        if len(w) < 4:
            return set()
        cand = set(self.idx.get(w, ()))
        for d in deletes1(w):
            cand |= self.idx.get(d, set())
        cand.discard(w)
        return cand


# ------------------------------------------------------------------- corpora


class Corpus:
    """Two layers on purpose.

    `flat` / `norm` are built from PROSE ONLY (html tags, enricher targets and
    ids stripped).  Ids must NOT count as "the name still exists": upstream keeps
    `Scene.emberArcturelUpper` forever after renaming the scene to `Tradeway`,
    so an id-inclusive corpus reports the renamed name as still present and the
    whole detector goes silent.  That is exactly the bug this scan is about.

    `idflat` keeps the raw text (ids included) and is only used to annotate a
    finding with `only_in_ids` — a strong *positive* signal of a rename.
    """

    def __init__(self, packs):
        norm_chunks, flat_chunks, id_chunks, wset = [], [], [], set()
        for pack, rows in packs.items():
            for p, s in rows:
                pr = prose(s)
                norm_chunks.append(norm_spaced(pr))
                flat_chunks.append(flat(pr))
                id_chunks.append(flat(s))
                wset.update(words_lower(pr))
        self.norm = "|".join(norm_chunks)
        self.flat = "|".join(flat_chunks)
        self.idflat = "|".join(id_chunks)
        self.words = WordIndex(wset)

    @staticmethod
    def _pattern(phrase):
        """words joined by an optional space, both ends anchored on a non-word.

        Anchoring is not optional: a bare flat-substring test made
        `Ring of Warmth` "still present" because the corpus contains
        「a fee-LING OF WARMTH」.  Over-exclusion is silent, so it is the
        dangerous direction.
        """
        # digits are part of a name (`Chapter 1 Events`, `Silk Rope (50 ft.)`);
        # dropping them made every such name look deleted.
        ws = [re.escape(w) for w in re.findall(r"[a-z0-9]+", unify(phrase).lower())]
        if not ws:
            return None
        # last word: tolerate singular <-> plural in either direction
        last = ws[-1]
        alts = {last, last + "s", last + "es"}
        if last.endswith("s"):
            alts.add(last[:-1])
        if last.endswith("es"):
            alts.add(last[:-2])
        ws[-1] = "(?:" + "|".join(sorted(alts, key=len, reverse=True)) + ")"
        return re.compile(r"(?<![a-z0-9])" + r" ?".join(ws) + r"(?![a-z0-9])")

    @staticmethod
    def _flat_forms(phrase):
        f = flat(phrase)
        if not f:
            return []
        forms = {f, f + "s", f + "es"}
        if f.endswith("es"):
            forms.add(f[:-2])
        if f.endswith("s"):
            forms.add(f[:-1])
        return sorted(forms)

    def has_phrase(self, phrase):
        """case / space / hyphen / apostrophe / plural insensitive presence.

        Two stages only for speed: the space-stripped substring test is a
        NECESSARY condition for the anchored match (dropping spaces can only add
        matches), so it is a sound C-speed pre-filter for the 20 MB corpus.
        """
        forms = self._flat_forms(phrase)
        if not forms:
            return True
        if not any(f in self.flat for f in forms):
            return False
        rx = self._pattern(phrase)
        return bool(rx) and bool(rx.search(self.norm))

    def in_ids_only(self, phrase):
        return (not self.has_phrase(phrase)) and flat(phrase) in self.idflat


# --------------------------------------------------------- candidate mining

CONNECTORS = {"of", "the", "and", "in", "at", "on", "to", "de", "del", "van",
              "von", "der", "la", "le", "for", "a", "an"}

# A Title-Case run.  Two things the first draft got wrong:
#   * `[A-Z][a-z]+` stops at the second capital, so `McDonald` came out as `Mc`
#     and every such name was reported as missing upstream.
#   * a run must not END on a connector, or `Leather Working to` /
#     `Recap of de` become findings.
_TW = r"[A-Z][a-z'’][A-Za-z'’]*"
_CONN = r"of|the|and|in|at|on|to|de|del|van|von|der|la|le|for"
RX_TITLE_RUN = re.compile(
    r"\b(" + _TW + r"(?:(?:[-\s]+(?:" + _CONN + r"))*[-\s]+" + _TW + r")+)")

RX_UUID_LABEL = re.compile(r"@[A-Za-z]+\[[^\]]*\]\{([^}]*)\}")

# by-design English, module/system names, mechanics
WHITELIST_WORDS = {
    "crucible", "ember", "dnd", "dnd5e", "foundry", "vtt", "gm", "gms", "pc",
    "pcs", "npc", "npcs", "dc", "hp", "ac", "xp", "the", "and", "of", "a", "an",
    "level", "tier", "damage", "check", "test", "save", "action", "round",
    "early", "access", "chapter", "part", "appendix", "table", "sidebar",
    "adventure", "campaign", "player", "players", "character", "characters",
}


def is_propername(word, old_lower_words):
    """A proper noun never shows up lowercase in the same corpus."""
    w = word.lower()
    if len(w) < 3 or w in WHITELIST_WORDS or w in CONNECTORS:
        return False
    return w not in old_lower_words


def phrase_words(phrase):
    return [w for w in words_lower(phrase) if w]


RX_WORD_SPAN = re.compile(r"[A-Za-z]+")


def variant_alive(cur, phrase, max_subs=200):
    """Is the phrase alive upstream under a different spelling?

    sec.8 (2026-08-11) `Reaper Ocean` / `Reaver Ocean`: sampling one spelling
    gives a biased sample.  A merely *similar* word existing somewhere is not
    enough evidence — we rebuild the whole phrase with the neighbour substituted
    and require THAT to exist.  Otherwise `altar`~`alter` would silently kill
    every finding that happens to contain a common word.
    """
    spans = [(m.start(), m.end(), m.group(0).lower())
             for m in RX_WORD_SPAN.finditer(phrase)]
    tried = 0
    for st, en, w in spans:
        if len(w) < 4 or w in CONNECTORS:
            continue
        for alt in sorted(cur.words.near(w)):
            tried += 1
            if tried > max_subs:
                return None
            cand = phrase[:st] + alt + phrase[en:]
            if cur.has_phrase(cand):
                return cand
    return None


# --------------------------------------------------------- CN part extraction


def cn_part(value):
    """'申特月神殿 Shent Moon Temple' -> '申特月神殿'  (up to the last CJK char)."""
    if not value:
        return ""
    m = None
    for m in CJK_RX.finditer(value):
        pass
    if m is None:
        return ""
    s = value[: m.end()].strip()
    return s.strip(" 　\t·-—:：,，.。")


def cn_core(s):
    """drop wrapping quotes / brackets so search is robust."""
    return s.strip("“”\"'‘’《》「」()（）[]【】 ")


# ==============================================================  detector A

# Single Title-Case token, used only when --min-run-words 1 is asked for.
# A one-word run cannot lean on "two capitalised words in a row" as evidence of
# being a name, so detector A holds it to a much harder bar (see below).
RX_TITLE_TOKEN = re.compile(r"\b([A-Z][a-z'’][A-Za-z'’]{2,})")


def detect_en_tail(repo, cur, old, cur_cn_packs, old_lower_words, args):
    """English runs inside CN text that no longer exist in the current baseline."""
    findings, excluded = [], []
    seen = {}
    minw = max(1, getattr(args, "min_run_words", 2))
    for pack, rows in cur_cn_packs.items():
        for p, s in rows:
            if SKIP_PATH_RX.search(p):
                continue
            if not CJK_RX.search(s):
                continue
            pr = prose(s)
            spans = []
            for m in RX_TITLE_RUN.finditer(pr):
                spans.append((m.start(1), m.end(1)))
                run = re.sub(r"\s+", " ", m.group(1)).strip()
                pw = phrase_words(run)
                if len(pw) < 2:
                    continue
                if all(w in WHITELIST_WORDS or w in CONNECTORS for w in pw):
                    continue
                bucket = seen.setdefault(run.lower(), {"run": run, "hits": []})
                bucket["hits"].append((pack, p))
            if minw > 1:
                continue
            # --- single-token pass (opt-in).  `Petalzon` / `Opalix` / `Lamberite`
            # are one word each: detector A structurally could not see them and
            # they only surfaced through detector B, which needs an old baseline.
            # Bar: the token must not exist ANYWHERE in the current English
            # (not just "not as a phrase"), must never appear lowercase in either
            # baseline (i.e. it is a name, not an ordinary word), and must not be
            # a one-edit variant of a living word.
            for m in RX_TITLE_TOKEN.finditer(pr):
                if any(a <= m.start(1) < b for a, b in spans):
                    continue                      # already covered by a multi-word run
                tok = m.group(1)
                lw = tok.lower()
                if lw in WHITELIST_WORDS or lw in CONNECTORS:
                    continue
                if lw in old_lower_words:         # ordinary word, not a name
                    continue
                # A digit glued to the token means it is part of a machine
                # string, not a name.  Upstream ships `@CriticalSuccess12]`
                # (a typo for `@CriticalSuccess[12]`); the CN copies it
                # byte-for-byte, and the phrase matcher, which anchors on a
                # non-alphanumeric, then declares `CriticalSuccess` missing
                # from the English.  Faithful copy, not residue.
                if m.end(1) < len(pr) and pr[m.end(1)].isdigit():
                    continue
                bucket = seen.setdefault(lw, {"run": tok, "hits": []})
                bucket["hits"].append((pack, p))

    for key, b in sorted(seen.items()):
        run = b["run"]
        if cur.has_phrase(run):
            continue
        pw = phrase_words(run)
        alt = variant_alive(cur, run)
        if alt:
            excluded.append({"mode": "en-tail", "phrase": run,
                             "why": "alive upstream under a variant spelling",
                             "variant": alt, "hits": len(b["hits"])})
            continue
        findings.append({
            "mode": "en-tail",
            "repo": os.path.basename(repo),
            "phrase": run,
            "in_old_baseline": old.has_phrase(run) if old else None,
            "only_in_ids": cur.in_ids_only(run),
            "cur_has_word": {w: cur.words.has(w) for w in pw},
            "hits": [{"pack": pk, "path": pp} for pk, pp in b["hits"][:40]],
            "hit_count": len(b["hits"]),
        })
    return findings, excluded


# ==============================================================  detector B


def content_subphrases(name, max_out=24):
    """Contiguous word windows of `name` holding >= 2 CONTENT words.

    Used as a prose escape hatch for the English gate.  The content-word floor
    is the whole point: a window like `Bag of` carries one content word and must
    NOT be able to explain anything, or a page saying `Bag of Coins` would
    silence the genuine `Bag of Bronze Coins` residue.
    """
    ws = words_lower(name)
    out, seen_w = [], set()
    for i in range(len(ws)):
        for j in range(i + 2, len(ws) + 1):
            win = ws[i:j]
            # len>=3 drops the `s` that `Terracini's` splits into — otherwise
            # the window "terracini s" is 2 "content" words and a page saying
            # `Terracini's Tavern` would excuse the stale 特拉奇尼的餐馆.
            if sum(1 for w in win if len(w) >= 3
                   and w not in CONNECTORS and w not in WHITELIST_WORDS) < 2:
                continue
            s = " ".join(win)
            if s in seen_w or s == " ".join(ws):
                continue
            seen_w.add(s)
            out.append(s)
            if len(out) >= max_out:
                return out
    return out


def build_name_map(en_packs, cn_packs):
    """(en name) -> set(cn rendering), paired by identical leaf path."""
    cn_idx = {}
    for pack, rows in cn_packs.items():
        for p, s in rows:
            cn_idx[(pack, p)] = s
    m = collections.defaultdict(collections.Counter)
    for pack, rows in en_packs.items():
        for p, s in rows:
            if not (p.endswith(".name") or p.endswith(".tokenName")
                    or p == "name" or p.endswith(".label")):
                continue
            c = cn_idx.get((pack, p))
            if not c:
                continue
            cp = cn_core(cn_part(c))
            if len(CJK_RX.findall(cp)) < 2:
                continue
            m[s.strip()][cp] += 1
    return m


def detect_cn_term(repo, cur, old, cur_en_packs, cur_cn_packs, old_en_packs,
                   old_lower_words, glossary, ids, args, old_cn_packs=None):
    """OLD English proper nouns gone from CURRENT English, whose CN is still used."""
    # --- 1. candidate old proper nouns: name fields + UUID labels
    cands = {}

    def add(name, src):
        n = re.sub(r"\s+", " ", name).strip()
        if not n or len(n) < 4 or CJK_RX.search(n):
            return
        pw = phrase_words(n)
        if not pw or all(w in WHITELIST_WORDS or w in CONNECTORS for w in pw):
            return
        cands.setdefault(n, set()).add(src)

    for pack, rows in old_en_packs.items():
        for p, s in rows:
            if p.endswith(".name") or p.endswith(".tokenName") or p == "name":
                add(s, "name")
            elif not SKIP_PATH_RX.search(p):
                for m in RX_UUID_LABEL.finditer(s):
                    add(m.group(1), "uuid-label")

    # --- 2. gone from current baseline?
    gone, excluded = {}, []
    for n, src in cands.items():
        if cur.has_phrase(n):
            continue
        alt = variant_alive(cur, n)
        if alt:
            excluded.append({"mode": "cn-term", "phrase": n,
                             "why": "alive upstream under a variant spelling",
                             "variant": alt})
            continue
        gone[n] = {"src": sorted(src), "only_in_ids": cur.in_ids_only(n)}

    # --- 3. english -> chinese renderings
    #
    # Pairing the OLD english against the CURRENT cn only works while the leaf
    # path survives the rename.  It usually does not: the path key contains the
    # name, so a rename moves the key and the pair evaporates.  In the first run
    # 18 of 47 deleted names (38%) had no CN rendering to hunt for at all.
    # `--old-cn` fixes that at the root by pairing the old english with the cn of
    # THAT SAME RELEASE (`git show <old tag>:compendium/cn/...`), where the paths
    # do match by construction.
    old_map = build_name_map(old_en_packs, cur_cn_packs)
    if old_cn_packs:
        for en, cns in build_name_map(old_en_packs, old_cn_packs).items():
            old_map[en].update(cns)
    cur_map = build_name_map(cur_en_packs, cur_cn_packs)
    # english name -> its current CN rendering, for annotating uuid targets
    cn_name_of = {en: c.most_common(1)[0][0] for en, c in cur_map.items() if c}

    # reverse map: chinese rendering -> english sources still present upstream
    rev = collections.defaultdict(set)
    for en, cns in cur_map.items():
        for c in cns:
            rev[c].add(en)
    for en, cn in glossary.items():
        c = cn_core(cn_part(cn))
        if len(CJK_RX.findall(c)) >= 2:
            rev[c].add(en)

    # Chinese renderings of names that are STILL ALIVE upstream.  A deleted
    # name whose Chinese is a proper substring of one of these is a common word,
    # not residue: `Closet 1` -> 「壁橱」 fires on all 45 leaves of Marlstone
    # Manor because 壁橱 is simply "closet" and today's names are
    # 赫菲丝的楼下壁橱 / 客用壁橱 / 备用壁橱.  The per-leaf English gate cannot
    # catch this — it only knows whole names, and 壁橱 is never a whole name.
    living_cn = set()
    for en, cns in cur_map.items():
        if not cur.has_phrase(en):
            continue
        living_cn.update(cns)

    # --- 4. hunt the chinese renderings in cn
    cn_rows = [(pk, p, s) for pk, rows in cur_cn_packs.items() for p, s in rows]
    en_by_path = {(pk, p): s for pk, rows in cur_en_packs.items() for p, s in rows}

    findings = []
    for n, info in sorted(gone.items()):
        renderings = collections.Counter()
        for c, k in old_map.get(n, {}).items():
            renderings[c] += k
        g = glossary.get(n)
        if g:
            c = cn_core(cn_part(g))
            if len(CJK_RX.findall(c)) >= 2:
                renderings[c] += 1
        if not renderings:
            continue
        subphrases = content_subphrases(n)
        for cn_term in renderings:
            if len(CJK_RX.findall(cn_term)) < 2:
                continue
            host = next((c for c in living_cn
                         if len(c) > len(cn_term) and cn_term in c), None)
            if host:
                excluded.append({"mode": "cn-term", "phrase": n, "cn": cn_term,
                                 "why": "cn rendering is a component of a name "
                                        "that is still alive upstream",
                                 "living_cn_name": host})
                continue
            hits, gated_out = [], 0
            for pk, p, s in cn_rows:
                if cn_term not in s:
                    continue
                if SKIP_PATH_RX.search(p):
                    continue
                en = en_by_path.get((pk, p), "")
                # english gate: does this leaf's english explain the chinese?
                explained = None
                if en:
                    lo = norm_spaced(prose(en))
                    fl = flat(en)
                    for alt in rev.get(cn_term, ()):  # a still-living english source
                        if alt.strip() == n:
                            continue
                        af = flat(alt)
                        if af and (af in fl or (" " + norm_spaced(alt).strip() + " ") in lo):
                            explained = alt
                            break
                    if not explained:
                        # ...or the leaf's English keeps part of the deleted name
                        # as ordinary prose.  Upstream dropped the ITEM
                        # `Bronze Plate Mail`, but the same page still says
                        # "wear the bronze plate", which is what 青铜板甲
                        # translates.  Two CONTENT words are required, so
                        # `Bag of` cannot excuse the real `Bag of Bronze Coins`
                        # residue on a page that says `Bag of Coins`.
                        for sub in subphrases:
                            if (" " + sub + " ") in lo:
                                explained = sub + " (prose fragment)"
                                break
                if explained:
                    gated_out += 1
                    continue
                h = {"pack": pk, "path": p,
                     "en_present": bool(en),
                     "en_excerpt": (prose(en)[:160] if en else "")}
                # If the stale rendering sits in a @UUID label and we have the
                # id table, the target's CURRENT name settles the case outright
                # -- no majority vote, no guessing.
                if ids:
                    for m in re.finditer(r"@UUID\[([^\]]*)\]\{([^}]*)\}", s):
                        if cn_term not in m.group(2):
                            continue
                        tid = m.group(1).split(".")[-1]
                        tinfo = ids.get(tid) or {}
                        if tinfo.get("name"):
                            h["uuid_target_now"] = tinfo["name"]
                            h["uuid_target_cn"] = cn_name_of.get(tinfo["name"])
                        break
                hits.append(h)
            if not hits:
                continue
            if len(hits) > args.max_cn_leaves:
                excluded.append({"mode": "cn-term", "phrase": n, "cn": cn_term,
                                 "why": f"cn term too common ({len(hits)} leaves) "
                                        f"- needs manual English gate",
                                 "gated_out": gated_out})
                continue
            # `Torch x8` -> 「火把」: the gate explained 152 leaves and left 1.
            # A rendering that is overwhelmingly produced by some *other*, still
            # living English name is a common word, and the leftovers are almost
            # certainly ordinary uses, not residue of the deleted name.
            if gated_out >= args.noise_ratio * len(hits):
                excluded.append({"mode": "cn-term", "phrase": n, "cn": cn_term,
                                 "why": f"cn rendering mostly explained by other "
                                        f"english ({gated_out} gated vs "
                                        f"{len(hits)} left)"})
                continue
            findings.append({
                "mode": "cn-term",
                "repo": os.path.basename(repo),
                "phrase": n,
                "cn": cn_term,
                "src": info["src"],
                "only_in_ids": info["only_in_ids"],
                
                "gated_out_by_english": gated_out,
                "hit_count": len(hits),
                "hits": hits[:40],
            })
    return findings, excluded


# ==================================================================== main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--old", required=True, help="old english-baseline dir")
    ap.add_argument("--old-cn", default="",
                    help="cn/ dir of the SAME old release (e.g. exported with "
                         "`git show v1.0.15:compendium/cn/x.json`).  Gives a real "
                         "old-EN -> old-CN dictionary; without it a rename that "
                         "moved the leaf path leaves nothing to search for.")
    ap.add_argument("--min-run-words", type=int, default=2,
                    help="detector A: minimum words in a Title-Case run.  1 also "
                         "scans single tokens (Petalzon / Opalix), held to a "
                         "harder bar; 2 is the conservative default.")
    ap.add_argument("--mode", default="both",
                    choices=["both", "en-tail", "cn-term"])
    ap.add_argument("--glossary", default="")
    ap.add_argument("--ids", default="",
                    help="id -> {name} table from dump_ids.mjs.  With it a stale "
                         "@UUID label is settled by the target's CURRENT name "
                         "instead of a majority vote.")
    ap.add_argument("--max-cn-leaves", type=int, default=60,
                    help="a CN rendering hit in more leaves than this is treated "
                         "as a common word and pushed to the excluded list")
    ap.add_argument("--noise-ratio", type=float, default=3.0,
                    help="drop a CN rendering when the English gate already "
                         "explained >= ratio x the remaining hits")
    ap.add_argument("--out", default="")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--strict-coverage", action="store_true",
                    help="基准缺包时以退出码 3 结束（默认只告警：回测脚本会拿单包小仓跑，"
                         "默认非零会把它们全打红）")
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, "compendium", "en")
    cn_dir = os.path.join(a.repo, "compendium", "cn")
    print(f"repo   : {a.repo}")
    print(f"old    : {a.old}")
    cur_en = load_pack_leaves(en_dir)
    cur_cn = load_pack_leaves(cn_dir)
    old_en = load_pack_leaves(a.old)
    old_cn = load_pack_leaves(a.old_cn) if a.old_cn else None
    print(f"leaves : cur-en {sum(len(v) for v in cur_en.values())} / "
          f"cur-cn {sum(len(v) for v in cur_cn.values())} / "
          f"old-en {sum(len(v) for v in old_en.values())}"
          + (f" / old-cn {sum(len(v) for v in old_cn.values())}" if old_cn else ""))
    cov = coverage(a.repo, a.old, old_en, cur_cn)
    print_coverage(cov)
    for pk in cov["scanned_packs"]:
        print(f"     · {pk}  基准叶 {len(old_en.get(pk, ()))}"
              f"  当前英文叶 {len(cur_en.get(pk, ()))}"
              f"  含中文的叶 {cjk_leaf_count(cur_cn.get(pk))}")

    cur = Corpus(cur_en)
    old = Corpus(old_en)
    # a proper noun is a word that never appears lowercase in EITHER baseline
    old_lower = set()
    for packs in (old_en, cur_en):
        for rows in packs.values():
            for p, s in rows:
                for w in RX_WORD.findall(prose(s)):
                    if w[:1].islower():
                        old_lower.add(w.lower())

    ids = {}
    if a.ids and os.path.isfile(a.ids):
        ids = json.load(open(a.ids, encoding="utf-8-sig"))
        print(f"ids     : {len(ids)} documents")

    glossary = {}
    if a.glossary and os.path.isfile(a.glossary):
        glossary = json.load(open(a.glossary, encoding="utf-8-sig"))
        print(f"glossary: {len(glossary)} entries")

    findings, excluded = [], []
    if a.mode in ("both", "en-tail"):
        f, x = detect_en_tail(a.repo, cur, old, cur_cn, old_lower, a)
        findings += f
        excluded += x
        print(f"[A en-tail] findings {len(f)}  excluded {len(x)}")
    if a.mode in ("both", "cn-term"):
        f, x = detect_cn_term(a.repo, cur, old, cur_en, cur_cn, old_en,
                              old_lower, glossary, ids, a, old_cn)
        findings += f
        excluded += x
        print(f"[B cn-term] findings {len(f)}  excluded {len(x)}")

    findings.sort(key=lambda d: (-d.get("hit_count", 0), d.get("phrase", "")))
    for d in findings[: a.show]:
        tag = d["mode"]
        extra = f" cn={d['cn']!r}" if d.get("cn") else ""
        print(f"  [{tag}] {d['phrase']!r}{extra}  hits={d['hit_count']} "
              f"in_old={d.get('in_old_baseline')} "
              f"gated_out={d.get('gated_out_by_english')}")
        for h in d["hits"][:3]:
            print(f"        {h['pack']} :: {h['path']}")

    # 报告里必须带覆盖口径：只写「findings 0」的报告，事后分不清是「干净」还是「没扫」。
    meta = {"tool": os.path.basename(__file__), "repo": os.path.abspath(a.repo),
            "coverage": cov, "argv": sys.argv}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"meta": meta, "repo": a.repo, "old": a.old, "mode": a.mode,
                   "findings": findings, "excluded": excluded},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> {a.out}")
    if a.strict_coverage and cov["missing_packs"]:
        print(f"✗ --strict-coverage：基准缺 {len(cov['missing_packs'])} 个包，"
              f"本次结果不构成全库结论。")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
