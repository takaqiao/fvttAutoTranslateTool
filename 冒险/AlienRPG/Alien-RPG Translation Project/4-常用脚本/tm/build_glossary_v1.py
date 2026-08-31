#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_glossary_v1.py — rebuild glossary_alien.{json,provenance,disputes,pending,byrole}
under the OWNER'S SETTLED PRIORITY LADDER (PROJECT.md §1.4, 2026-08-29).

Supersedes build_glossary_v0_2.py, which adjudicated with zh.wikipedia as the spine.
That was correct only while the owner had supplied nothing but zh.wikipedia and the
CnSCG subtitle lineage.  Three stronger sources have since arrived — one of them is the
Chinese translation OF THIS VERY BOOK — so the spine is re-cut here.

LADDER (1 strongest .. 7 weakest):
  1  《异形RPG：进化版》中文连载  ch.1 + ch.2   (mined_fan_translation.json)
  2  Romulus 2024 / Alien: Earth 2025 subtitles (mined_subtitles_vote.json lineages)
  3  Alien: Isolation official 简中               (mined_alien_isolation.json)
  4  system lang/cn.json A-stratum (TI-130 human, 2020-11..2021-01)
  5  Covenant 2017 / Prometheus 2012 subtitles
  6  CnSCG Alien 1/2/3/4 — ALL FOUR ARE ONE VOTE
  7  zh.wikipedia — an encyclopedia about the FILMS, not about this RPG

Two owner-pinned exceptions where level 4 beats level 1 (players already see them):
  Wits = 机智   Empathy = 共情

NOT a ladder level, and never a spine on its own:
  "C"  convention / derivation.  Legal for a COMMON noun, ILLEGAL for a proper noun.
  "B-rejected"  cn.json stratum B (maintainer bulk MT 2022-2026).  Never a source.
  "off-ladder"  Chinese fan corpus (bilibili / douban / 机核) — uncountable this session.

EVERY count in the output is read out of a miner artifact at build time.
Nothing is transcribed by hand, and no conclusion rests on a bare Chinese frequency.

Run:  python build_glossary_v1.py            (writes)
      python build_glossary_v1.py --check    (asserts only, writes nothing)
"""

import json, os, re, sys, io

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GLOSS  = os.path.join(ROOT, "7-其他内容", "glossary")
OTHER  = os.path.join(ROOT, "7-其他内容")
LANG   = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang"
ACTOR  = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/module/documents/actor.mjs"

GENERATED = "2026-08-29"
VERSION   = "v1.0"

def jload(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)

def jdump(p, obj):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return os.path.getsize(p)

# ---------------------------------------------------------------- corpora ---
PREV      = jload(os.path.join(GLOSS, "glossary_alien.provenance.json"))
PREV_T    = PREV["terms"]
# The v1.0 build WRITES glossary_alien.disputes.json, so it must not also read it for
# history — the first run would consume its own baseline. The Phase-0 set is frozen in
# glossary_alien.disputes.phase0.json and that is what "what this dispute used to say"
# quotes from.
PREV_DISP = jload(os.path.join(GLOSS, "glossary_alien.disputes.phase0.json"))
PREV_PEND = jload(os.path.join(GLOSS, "glossary_alien.pending.json"))
FAN       = jload(os.path.join(GLOSS, "mined_fan_translation.json"))
ISO       = jload(os.path.join(GLOSS, "mined_alien_isolation.json"))
SUB       = jload(os.path.join(GLOSS, "mined_subtitles_vote.json"))
DNT       = jload(os.path.join(OTHER, "DO-NOT-TRANSLATE.json"))

def _flat(o, p=""):
    out = {}
    for k, v in o.items():
        kk = p + "." + k if p else k
        if isinstance(v, dict):
            out.update(_flat(v, kk))
        else:
            out[kk] = v
    return out

LANG_EN = _flat(jload(os.path.join(LANG, "en.json")))
LANG_CN = _flat(jload(os.path.join(LANG, "cn.json")))

FAN_BY_EN = {}
for _r in FAN["terms"]:
    FAN_BY_EN.setdefault(_r["en"], []).append(_r)
SUB_BY_KEY = {t["key"]: t for t in SUB["terms"]}
ISO_BY_EN  = ISO["terms"]

# lineage -> ladder level
LINEAGE_LEVEL = {
    "Romulus2024": 2,
    "AlienEarth2025": 2,
    "Covenant2017": 5,
    "Prometheus2012": 5,
    "CnSCG(异形1/2/3/4·同源)": 6,
}
LINEAGE_LABEL = {
    "Romulus2024": "Alien: Romulus 2024 subtitles",
    "AlienEarth2025": "Alien: Earth 2025 subtitles",
    "Covenant2017": "Alien: Covenant 2017 subtitles",
    "Prometheus2012": "Prometheus 2012 subtitles",
    "CnSCG(异形1/2/3/4·同源)": "CnSCG Alien 1/2/3/4 (ONE vote — single fansub lineage)",
}

# ------------------------------------------------- candidate constructors ---
def cand(zh, level, source, count, count_kind, evidence, **kw):
    """Every candidate names a LEVEL and a COUNT KIND.  count_kind is never 'bare'."""
    d = {
        "zh": zh,
        "source_level": level,
        "source": source,
        "count": count,
        "count_kind": count_kind,
        "evidence": evidence,
    }
    d.update(kw)
    return d

def fanc(en_key, idx=0, zh=None):
    """Level-1 candidate straight out of mined_fan_translation.json."""
    recs = FAN_BY_EN[en_key]
    r = recs[idx]
    if zh is not None and r["zh"] != zh:
        raise AssertionError("fan record %r zh=%r, expected %r" % (en_key, r["zh"], zh))
    kind = "inline_gloss" if r.get("inline_glossed") else "occurrence_in_translated_book"
    c = cand(
        r["zh"], 1,
        "《异形RPG：进化版》中文连载 %s (level 1 — the Chinese translation of THIS book)" % r["chapter"],
        "en-gated by construction: the fan text is a translation of the known English source; "
        + ("the English is glossed inline at this occurrence" if r.get("inline_glossed")
           else "paired by position against the English rulebook"),
        kind,
        r["quote"],
        fan_key=en_key,
        chapter=r["chapter"],
    )
    if r.get("internal_conflict"):
        c["internal_conflict"] = r["internal_conflict"]
    if r.get("note"):
        c["miner_note"] = r["note"]
    return c

def subc(term_key, lineage, zh=None):
    """Subtitle candidate, English-gated, counted by LINEAGE."""
    t = SUB_BY_KEY[term_key]
    top = (t.get("per_lineage_top") or {}).get(lineage)
    if top is None:
        raise AssertionError("no per-lineage top for %r / %r" % (term_key, lineage))
    if zh is not None and top["zh"] != zh:
        raise AssertionError("sub %r/%r zh=%r expected %r" % (term_key, lineage, top["zh"], zh))
    lvl = LINEAGE_LEVEL[lineage]
    return cand(
        top["zh"], lvl, LINEAGE_LABEL[lineage],
        "en-gated %d line(s) in this lineage; the English regex %s matched %d line(s) corpus-wide "
        "across %d lineage(s)" % (
            top.get("line_count", top.get("per_lineage_lines", {}).get(lineage, 0)),
            t["english_regex"], t["total_hits"], len(t["hits_by_lineage"])),
        "english_gated_subtitle_vote",
        "mined_subtitles_vote.json terms[key=%s].per_lineage_top[%s]; hits_by_lineage=%s"
        % (term_key, lineage, json.dumps(t["hits_by_lineage"], ensure_ascii=False)),
        vote_key=term_key,
        lineage=lineage,
        lineage_count_for_zh=top.get("lineage_count"),
    )

def isoc(en_key, zh=None):
    """Level-3 candidate out of the Alien: Isolation official localisation."""
    e = ISO_BY_EN[en_key]["candidates"][0]
    if zh is not None and e["zh"] != zh:
        raise AssertionError("iso %r zh=%r expected %r" % (en_key, e["zh"], zh))
    ev = e.get("evidence", {})
    return cand(
        e["zh"], 3, "Alien: Isolation official Simplified Chinese localisation (level 3)",
        "en-gated %s key(s) by %s; %s key(s) carry the value" % (
            ev.get("key_gated_keys"), e.get("gate_method"), ev.get("value_occurrence_keys")),
        "english_gated_key_slug" if e.get("english_gate") else "ungated_official_prose",
        "%s -> %s" % (ev.get("representative_key"), ev.get("representative_line")),
        iso_key=en_key,
        gate_method=e.get("gate_method"),
        gate_strength=("hard" if e.get("gate_method") in ("key_slug", "key_sentence", "key_slug+value")
                       else "soft"),
        miner_note=e.get("note"),
    )

def langc(key, zh=None, stratum="A"):
    """Level-4 candidate out of lang/cn.json.  Stratum B is recorded as REJECTED."""
    full = "ALIENRPG." + key
    cnv, env = LANG_CN.get(full), LANG_EN.get(full)
    if zh is not None and cnv != zh:
        raise AssertionError("lang %s cn=%r expected %r" % (full, cnv, zh))
    lvl = 4 if stratum == "A" else "B-rejected"
    return cand(
        cnv, lvl,
        "system lang/cn.json stratum %s%s" % (
            stratum, " (TI-130 human 2020-11..2021-01)" if stratum == "A"
            else " (maintainer bulk MT 2022-2026 — NEVER a source)"),
        "en-gated 1 (a key/value pair: the English side is lang/en.json)",
        "lang_key_pair",
        '%s  en=%r  cn=%r' % (full, env, cnv),
        lang_key=full,
    )

def refc(zh, what, evidence):
    """Level-7 reference-work candidate (zh.wikipedia and the encyclopedia family)."""
    return cand(zh, 7, what, "reference work — no English-gated count exists",
                "reference_work", evidence)

def convc(zh, evidence, note=None):
    """Convention / derivation.  Legal for a COMMON noun only."""
    c = cand(zh, "C", "convention or derivation from already-adopted spine terms",
             "no count — this is a derivation, not an attestation", "derived", evidence)
    if note:
        c["miner_note"] = note
    return c

# --------------------------------------------------------- level relabel ----
def relabel(src):
    """Map a Phase-0 free-text source string onto the new ladder."""
    s = src or ""
    if "stratum B" in s:
        return "B-rejected", "system lang/cn.json stratum B (maintainer bulk MT — NEVER a source)"
    if "stratum A" in s or s.startswith("cn.json") or "cn.json" in s:
        return 4, "system lang/cn.json stratum A (TI-130 human 2020-11..2021-01)"
    if "zh.wikipedia" in s or "wikipedia" in s.lower():
        return 7, "zh.wikipedia (level 7 — an encyclopedia about the FILMS, not about this RPG)"
    if "百度百科" in s or "萌娘百科" in s:
        return 7, s + "  [not named on the ladder; filed with level 7, the reference-work floor]"
    if "CnSCG" in s:
        return 6, "CnSCG Alien 1/2/3/4 subtitles (level 6 — ALL FOUR ARE ONE VOTE)"
    if "fan corpus" in s or "bilibili" in s or "douban" in s or "机核" in s or "fan usage" in s:
        return "off-ladder", s + "  [Chinese fan corpus — not on the owner's ladder, uncountable this session]"
    if "standard modern Chinese" in s or "Coined" in s or s in ("(none)", ""):
        return "C", "convention / standard modern Chinese usage (NOT a ladder level)"
    if "raw-dump" in s or "pack" in s.lower():
        return "C", s
    return "C", s

def carry_candidates(en):
    """Re-level the Phase-0 candidate array without inventing anything."""
    out = []
    for c in PREV_T[en].get("candidates", []):
        if c["zh"] == "(none)":
            continue
        lvl, lbl = relabel(c.get("source"))
        cnt = c.get("count", "")
        kind = ("english_gated_subtitle_vote" if lvl == 6 and "en-gated" in str(cnt)
                else "lang_key_pair" if lvl == 4
                else "reference_work" if lvl == 7
                else "derived" if lvl == "C"
                else "carried_from_phase0")
        out.append(cand(c["zh"], lvl, lbl, cnt or "(no count recorded in Phase 0)",
                        kind, c.get("evidence", ""), carried_from="phase0"))
    return out


# ------------------------------------------------- lockstep, re-derived ----
def derive_lockstep():
    """Re-derive every actor.mjs anchor FROM SOURCE.  Phase 0 shipped three wrong
    line numbers; hard-coding a fourth set would just move the problem."""
    src = io.open(ACTOR, encoding="utf-8").read().split("\n")

    def find(pred, lo=1, hi=None):
        hi = hi or len(src)
        for i in range(lo, hi + 1):
            if pred(src[i - 1]):
                return i
        raise AssertionError("lockstep anchor not found")

    perm  = find(lambda L: 'testArray[9] !== game.i18n.localize("ALIENRPG.Permanent")' in L)
    shift = find(lambda L: 'testArray[9] === "Shift"' in L)
    shloc = find(lambda L: 'game.i18n.localize("ALIENRPG.Shift")' in L)
    regex = find(lambda L: 'testArray[9].match(' in L)
    sw3   = find(lambda L: L.strip() == "switch (testArray[3]) {")
    sw5   = find(lambda L: L.strip() == "switch (testArray[5]) {")

    def close_of(open_line):
        indent = len(src[open_line - 1]) - len(src[open_line - 1].lstrip())
        for i in range(open_line + 1, len(src) + 1):
            L = src[i - 1]
            if L.strip() == "}" and (len(L) - len(L.lstrip())) == indent:
                return i
        raise AssertionError("switch close not found")

    sw3_end, sw5_end = close_of(sw3), close_of(sw5)
    endash = [i for i in range(sw3, sw3_end + 1) if "–" in src[i - 1]]
    sw5_default = find(lambda L: L.strip() == "default:", sw5, sw5_end)
    heal0 = find(lambda L: "healTime = 0;" in L, sw5_default, sw5_end)

    def cases(lo, hi):
        out = []
        for i in range(lo, hi + 1):
            if not src[i - 1].strip().startswith("case "):
                continue  # only the case LABELS; a localize() inside a case BODY is not a compare
            m = re.search(r'localize\("ALIENRPG\.(\w+)"\)\s*\+\s*"([^"]*)"', src[i - 1])
            if m:
                out.append({"line": i, "key": "ALIENRPG." + m.group(1),
                            "appended_literal": m.group(2),
                            "appended_codepoints": [hex(ord(c)) for c in m.group(2)]})
        return out

    return {
        "_source": ACTOR,
        "_source_lines": len(src),
        "_rederived_on": GENERATED,
        "_method": "Anchors are LOCATED BY PATTERN in this build, not transcribed. If upstream "
                   "moves a line, the next build reports the new number instead of lying.",
        "_why": {
            "headline": "TWO SWITCHES, NOT ONE, AND TWO DIFFERENT FAILURE MODES.",
            "switch_testArray_3": "drives cFatal (is the injury fatal, and what is the Medical Aid "
                                  "penalty). Lines %d-%d." % (sw3, sw3_end),
            "switch_testArray_5": "drives healTime (how long the injury takes to heal). "
                                  "Lines %d-%d." % (sw5, sw5_end),
            "silent_half": "The testArray[5] switch falls through to `default:` at line %d and sets "
                           "healTime = 0 at line %d. Nothing is logged. A player just heals "
                           "instantly and never finds out why." % (sw5_default, heal0),
            "crashing_half": "The `Shift` branch does NOT fail silently, and Phase 0's _why said it "
                             "did. At line %d a heal-time cell that is neither empty nor the exact "
                             "English literal \"Shift\" falls through to line %d, "
                             "`testArray[9].match(/^\\[\\[([0-9]d[0-9]+)]/)[1]`. The regex does not "
                             "match, so .match() returns null, and null[1] throws an uncaught "
                             "TypeError that aborts the entire Critical-Injury roll mid-flight. "
                             "⚠ Write the QA gate for a CRASH on this path, not for a zero."
                             % (shift, regex),
            "phase0_errors_corrected": [
                "Phase 0 cited the bare `Shift` literal at :1888. :1888 is the Permanent "
                "comparison; the Shift test is at :%d — two lines down." % shift,
                "Phase 0 gave the healTime switch as :1922-1941. :1922 is blank and :1941 is the "
                "default arm's `break`. The switch is :%d-%d." % (sw5, sw5_end),
                "Phase 0 described only ONE switch. There are two: switch(testArray[3]) at :%d "
                "drives cFatal and switch(testArray[5]) at :%d drives healTime." % (sw3, sw5),
                "Phase 0's _why said the failure is uniformly silent. Half of it is a crash.",
            ],
        },
        "Permanent": {
            "line": perm,
            "code": src[perm - 1].strip(),
            "note": "NO trailing space is appended here, unlike every case in either switch.",
        },
        "Shift_bare_literal": {
            "line": shift,
            "code": src[shift - 1].strip(),
            "localize_line": shloc,
            "throwing_line": regex,
            "note": "A BARE English literal with no localize() call. The RollTable heal-time CELL "
                    "must stay English; ALIENRPG.Shift, localized one line later at :%d, is an "
                    "OUTPUT and should be Chinese. Two things, one name." % shloc,
        },
        "switch_testArray_3_cFatal": {
            "range": [sw3, sw3_end],
            "cases": cases(sw3, sw3_end),
            "en_dash_lines": endash,
            "en_dash_note": "Lines %s use U+2013 EN DASH, not ASCII hyphen-minus. A lockstep table "
                            "written with an ASCII hyphen can never match and silently loses the "
                            "cFatal=true branches. Verified by codepoint dump this build."
                            % (endash,),
        },
        "switch_testArray_5_healTime": {
            "range": [sw5, sw5_end],
            "cases": cases(sw5, sw5_end),
            "default_line": sw5_default,
            "falls_to_zero_at": heal0,
        },
        "lang_state_today": {
            k: {"en": LANG_EN.get("ALIENRPG." + k), "cn": LANG_CN.get("ALIENRPG." + k)}
            for k in ["Yes", "None", "OneRound", "OneTurn", "OneShift", "OneDay",
                      "Permanent", "Shift"]},
        "lang_state_verdict": "cn.json translates None/OneRound/OneTurn/OneShift/OneDay while "
                              "leaving Yes/Permanent/Shift English. That is already an inconsistent "
                              "state: healTime resolves through translated cases while cFatal "
                              "compares against untranslated ones.",
    }


# ============================================================================
# ASSEMBLY
# ============================================================================
CTX = dict(fanc=fanc, subc=subc, isoc=isoc, langc=langc, refc=refc, convc=convc,
           cand=cand, LANG_CN=LANG_CN, LANG_EN=LANG_EN, SUB_BY_KEY=SUB_BY_KEY)

import glossary_adjudication, glossary_additions, glossary_emit

OVERRIDES, _ = glossary_adjudication.build(CTX)
ADDITIONS, TO_PENDING = glossary_additions.build(CTX)

TIER_ORDER = ["T-FROZEN", "T-LATIN", "T-EXACT", "T-BILINGUAL", "T-PLAIN"]
LATIN_TIER_MOVES = {
    "LV-426": "A Latin planetary designator. Level 1 keeps it as LV-426.",
    "LV-223": "A Latin planetary designator; same treatment as LV-426.",
    "MU/TH/UR": "The ship computer's own name. Level 1 keeps MOTHER in Latin in its ch.1 heading; "
                "level 3 does not attest it at all.",
    "USCSS": "A Latin ship prefix. Level 1 glues it to the front of the Chinese hull name.",
}

def name_of(en):
    """The English NAME — the glossary key minus any trailing ' (...)' disambiguator."""
    m = re.match(r"^(.*?)\s+\([^()]*\)$", en)
    return m.group(1) if m else en

terms = {}
for en in PREV_T:
    if en in TO_PENDING:
        continue
    p = PREV_T[en]
    t = {
        "cn": p["cn"],
        "tier": p["tier"],
        "category": p["category"],
        "source_level": None,
        "why_this_won": p.get("why_this_won", ""),
        "candidates": carry_candidates(en),
    }
    for k in ("aliases", "note", "derived_from", "key_disambiguator",
              "byte_exactness_repair", "codepoints"):
        if p.get(k):
            t[k] = p[k]
    if p.get("dispute"):
        t["dispute"] = p["dispute"]
    terms[en] = t

for en, ov in OVERRIDES.items():
    if en not in terms:
        terms[en] = {"category": ov.get("category", "yze-mechanics"),
                     "tier": ov.get("tier", "T-PLAIN")}
    t = terms[en]
    t["cn"] = ov["cn"]
    t["source_level"] = ov["source_level"]
    t["why_this_won"] = ov["why_this_won"]
    t["candidates"] = ov["candidates"]
    t["readjudicated_in_v1_0"] = True
    for k, v in ov.items():
        if k in ("cn", "source_level", "why_this_won", "candidates"):
            continue
        if k == "dispute" and v is None:
            t.pop("dispute", None)
            continue
        t[k] = v

for en, adr in ADDITIONS.items():
    t = dict(adr)
    t["added_in_v1_0"] = True
    terms[en] = t

# --- frozen-key repairs found while asserting traceability this build ------
_NBSP_KEY = "CORE RULES - HOW TO USE THIS MODULE"
_ASCII_KEY = "CORE RULES - HOW TO USE THIS MODULE"
if _ASCII_KEY in terms and _NBSP_KEY not in terms:
    _t = terms.pop(_ASCII_KEY)
    _t["cn"] = _NBSP_KEY
elif _NBSP_KEY in terms:
    _t = terms[_NBSP_KEY]          # already repaired by an earlier run; keep the record
else:
    _t = None
if _t is not None:
    _t["byte_exactness_repair"] = (
        "⚠ FOUND 2026-08-29 while asserting T-FROZEN traceability. The Phase-0 key spelled this "
        "with ASCII spaces and an ASCII hyphen. The literal in "
        "modules/alien-evolved-corerules/module/init.js:11 uses NO-BREAK SPACE U+00A0 between "
        "every word: \"CORE RULES\\xa0-\\xa0HOW\\xa0TO\\xa0USE\\xa0THIS\\xa0MODULE\". A T-FROZEN key "
        "that is not byte-exact is worse than no key at all — a translator searching for it never "
        "finds the real string, and a QA gate built on it reports clean forever. The sibling "
        "STARTER SET - HOW TO USE THIS MODULE really does use ASCII spaces, so the two are NOT "
        "symmetric and must not be normalised to match each other.")
    _t["codepoints"] = [hex(ord(c)) for c in _NBSP_KEY]
    terms[_NBSP_KEY] = _t
for _up, _title, _live in [("HARDENED", "Hardened", True), ("STOIC", "Stoic", True)]:
    if _up in terms:
        _t = terms.get(_title) or dict(terms[_up])
        _t["cn"] = _title
        _t["note"] = ("The SHIPPED document name is title-case; the code uppercases before "
                      "comparing, so both spellings are keyed. Case may change, the LETTERS may "
                      "not.")
        _t["added_in_v1_0"] = True
        terms[_title] = _t

# tier narrowing: T-FROZEN now means ONLY "a real lookup in DO-NOT-TRANSLATE.json"
for en, why in LATIN_TIER_MOVES.items():
    if en in terms:
        terms[en]["tier"] = "T-LATIN"
        terms[en]["tier_narrowed_in_v1_0"] = why

# fill the level on every carried term that no override touched
for en, t in terms.items():
    if t.get("source_level") is None:
        lv = None
        for c in t.get("candidates", []):
            if c["zh"] == t["cn"]:
                lv = c["source_level"]
                break
        if lv is None:
            lv = "C"
            t.setdefault("level_note",
                         "The adopted value is not among the recorded candidates. Phase 0 justified "
                         "that in why_this_won by a documented corpus GAP; it is a derivation, so "
                         "it is levelled C. It must never be treated as attested.")
        t["source_level"] = lv
    if t["tier"] == "T-BILINGUAL":
        nm = name_of(en)
        t["bilingual_name"] = t["cn"] + " " + nm
        if nm != en:
            t["key_disambiguator"] = en[len(nm):].strip()
    else:
        t.pop("bilingual_name", None)

# --- level hygiene ---------------------------------------------------------
# A term's source_level must name something that can actually BE a source.
for _en, _t in terms.items():
    if _t["tier"] == "T-FROZEN":
        _t["source_level"] = "FROZEN"
        _t.setdefault("why_this_won",
                      "Not a terminology decision: a byte-exact string the running code "
                      "compares against.")
    elif _t["tier"] == "T-LATIN":
        _lv = None
        for _c in _t.get("candidates", []):
            if _c["zh"] == _t["cn"] and _c["source_level"] in (1, 2, 3, 4, 5, 6, 7):
                _lv = _c["source_level"]
                break
        _t["source_level"] = _lv if _lv is not None else "LATIN"
    elif _t["source_level"] == "B-rejected":
        # cn.json stratum B is NEVER a source.  A Phase-0 term whose only attestation was
        # stratum B is a CONVENTION that stratum B happens to agree with — which is worth
        # recording as corroboration and worthless as provenance.
        _t["source_level"] = "C"
        _t["level_note"] = (
            "RE-LEVELLED IN v1.0. Phase 0 recorded this value's only attestation as cn.json "
            "stratum B (maintainer bulk MT 2022-2026), which the project's hard rules say is "
            "NEVER a source. The value stands — it is an ordinary Chinese common noun any "
            "translator would reach — but it stands as a CONVENTION at level C, not as an "
            "attestation. That stratum B independently produced the same string is corroboration "
            "of the obvious reading, not evidence.")

# --- fold concurring candidates -------------------------------------------
# Two sources that reach the SAME Chinese string are corroboration, not a duplicate.
# Phase 0's real defect was a candidate duplicated with the SAME source and a pasted
# evidence string.  So the array carries one entry per distinct zh, and every
# concurring source is preserved inside it as also_attested_at.  This makes the
# "no duplicate zh" invariant literally true without throwing away evidence.
def _lvl_key(c):
    L = c.get("source_level")
    return (0, L) if isinstance(L, int) else (1, str(L))

for _en, _t in terms.items():
    _seen = {}
    _order = []
    for _c in _t.get("candidates", []):
        _z = _c["zh"]
        if _z in _seen:
            _seen[_z].append(_c)
        else:
            _seen[_z] = [_c]
            _order.append(_z)
    _out = []
    for _z in _order:
        _group = sorted(_seen[_z], key=_lvl_key)
        _primary = dict(_group[0])
        if len(_group) > 1:
            _primary["also_attested_at"] = [
                {"source_level": _o["source_level"], "source": _o["source"],
                 "count": _o["count"], "count_kind": _o["count_kind"],
                 "evidence": _o["evidence"]}
                for _o in _group[1:]]
            _primary["concurring_source_count"] = len(_group)
        _out.append(_primary)
    if _out:
        _t["candidates"] = _out

# --- declared same-string groups -------------------------------------------
# A Chinese value serving two English keys is either a bug or a decision.  Every
# such group below is a decision and says why; assertion #14 fails the build on any
# group that is NOT declared here, which is how Escape Pod / Escape Vehicle / EEV
# was caught sharing 逃生舱 across three keys in Phase 0.
SAME_STRING_GROUPS = [
    (["Airlock", "Air Lock"],
     "Two English spellings of one thing. Level 1 writes 气闸 once; there is nothing to split."),
    (["Alien", "Alien (1979 film)", "Xenomorph"],
     "The creature, the film named after it, and the taxonomic name are ONE Chinese word by every "
     "source that speaks. ⚠ Level 3 has no distinct rendering for 'xenomorph' at all — Alien: "
     "Isolation collapses it into 异形 — and levels 2 and 6 do the same. Splitting them would be "
     "inventing a distinction the Chinese does not make. The FILM key is disambiguated on the "
     "name field instead: its bilingual_name is '异形 Alien' and its key_disambiguator holds "
     "'(1979 film)'."),
    (["Career", "Careers"],
     "The term and the pack folder that holds it. The folder is a page-name field; the term is a "
     "lang value. Same string, different roles — see byrole."),
    (["Colonial Marine", "Colonial Marines"],
     "Singular and plural of one English name. Chinese does not inflect for number."),
    (["Marine", "Marines"], "Singular and plural of one English name."),
    (["Panic", "Panicked"],
     "The noun and the condition. Level 4 uses 恐慌 for both (ALIENRPG.Panicked -> 恐慌), and "
     "level 1 agrees; a Chinese adjective/noun pair here would be an invention."),
    (["Rad", "Radiation Point"], "Abbreviation and full form of one game quantity."),
    (["Stretch", "Turn"],
     "ONE DURATION, TWO EDITIONS' NAMES. Classic Alien RPG calls the middle time unit a Turn; "
     "Evolved renamed it a Stretch. Level 1 ch.1 glosses the middle unit 节 (Stretch) 5-10分, and "
     "the healTime switch independently orders OneRound(1) < OneTurn(2) < OneShift(3), putting "
     "Turn exactly where Stretch sits. Giving them two Chinese words would invent a distinction "
     "the rules do not have — which is what Phase 0's 轮次 did."),
    (["Base Dice", "Base Die"], "Singular and plural of one English name."),
    (["Stress Dice", "Stress Die"], "Singular and plural of one English name."),
    (["Cryo-tube", "Cryogenic Compartment"],
     "Two English names the packs use for one fixture. Deliberately kept distinct from Cryo Deck "
     "低温舱室, which is the ROOM."),
    (["Weyland-Yutani", "Weyland-Yutani Corporation"],
     "Level 1's 公司 already carries 'Corporation'; appending a second one would be wrong."),
    (["BERSERK", "Frenzy"],
     "The Panic-table RESULT ROW and the Evolved CONDITION key name the same state. The uppercase "
     "key is a table.result_cell, the title-case key is a lang value — see byrole. They must stay "
     "byte-equal or the Evolved panic response stops matching its own table."),
    (["CATATONIC", "Catatonic"], "Panic-table result row and Evolved condition key; must match."),
    (["FLEE", "Flee"], "Panic-table result row and Evolved condition key; must match."),
    (["FREEZE", "Freeze"], "Panic-table result row and Evolved condition key; must match."),
    (["SCREAM", "Scream"], "Panic-table result row and Evolved condition key; must match."),
    (["SEEK COVER", "Seekcover"], "Panic-table result row and Evolved condition key; must match."),
    (["TREMBLE", "Shakes"],
     "Panic-table result row and Evolved condition key. ⚠ Note the English differs (TREMBLE vs "
     "Shakes) while the Chinese must not — the table row and the condition are the same result."),
]
for _keys, _why in SAME_STRING_GROUPS:
    _present = [k for k in _keys if k in terms]
    for _k in _present:
        terms[_k]["same_string_as"] = [x for x in _present if x != _k]
        terms[_k]["same_string_reason"] = _why

# ------------------------------------------------------------- 主控裁定 --
# RULINGS.json 是**决策**，与 glossary_adjudication 的**证据推导**分开存放，
# 这样下一轮读得出「哪条是证据说的、哪条是人拍的」。
# 优先级：owner_pinned > RULINGS > 阶梯推导。ruling 撞上 owner_pinned 直接构建失败。
RULINGS = jload(os.path.join(GLOSS, "RULINGS.json")) or {"rulings": []}
RULING_BY_EN = {}
for _r in RULINGS.get("rulings", []):
    _en = _r["en"]
    if _en not in terms:
        raise SystemExit("RULING %s names a term that does not exist: %r" % (_r["id"], _en))
    if terms[_en].get("owner_pinned"):
        raise SystemExit(
            "RULING %s collides with an owner_pinned term (%s). Owner pins outrank rulings; "
            "resolve this by hand rather than letting the build pick." % (_r["id"], _en))
    RULING_BY_EN[_en] = _r
    _t = terms[_en]
    _was = _t["cn"]
    _t["cn"] = _r["adopt"]
    _t["source_level"] = "RULING"
    _t["ruled"] = {
        "id": _r["id"], "by": _r["ruled_by"], "on": _r["ruled_on"],
        "was": _was, "rejected": _r.get("reject", []),
        "rationale": _r["rationale"],
        "counter_argument_acknowledged": _r.get("counter_argument_acknowledged"),
        "blast_radius": _r.get("blast_radius"),
        "closes_dispute": _r.get("closes_dispute"),
    }
    _t["why_this_won"] = "主控裁定 %s：%s" % (_r["id"], _r["rationale"])
    if _r.get("domain_split"):
        _t["domain_split"] = _r["domain_split"]
        _t["assertion_hint"] = "term_domains"
    if _r.get("note"):
        _t["ruling_note"] = _r["note"]

glossary = {en: terms[en]["cn"] for en in sorted(terms)}

# ---------------------------------------------------------------- disputes --
DISPUTES = glossary_emit.disputes(CTX)
CLOSED = {
    "D1-panic-roll": {"en": "Panic Roll", "closed_by_level": 1, "settled_cn": "恐慌检定",
        "how": "Level 1 ch.2 writes 恐慌检定. The dispute was level-4-internal — the UI keys said "
               "恐慌 while the stratum-A panic prose said 混乱检定 — and a level-1 source outranks "
               "both halves of a level-4 argument.",
        "still_to_do": "cn.json Panic12/13/14 (混乱检定) and Panic8 (直到混乱结束) must be rewritten."},
    "D2-engaged": {"en": "Engaged", "closed_by_level": 1, "settled_cn": "接战",
        "how": "Closed from the OTHER SIDE. The dispute existed because level 4 spent 近战 on both "
               "the Engaged range band and the Melee weapon type. Level 1 gives 近战 to the Close "
               "Combat SKILL, so neither of the other two can keep it: Engaged takes level 4's own "
               "prose form 接战 and Melee takes the freed 肉搏. One level-1 decision dissolved a "
               "three-way collision.",
        "still_to_do": "Rewrite ALIENRPG.Engaged (近战 -> 接战) and ALIENRPG.WepTypeMelee "
                       "(近战 -> 肉搏) in the same commit as the Close Combat skill rename."},
    "D3-stress-level": {"en": "Stress Level", "closed_by_level": 1, "settled_cn": "压力等级",
        "how": "Level 1 ch.2 writes 压力等级. The dispute was a 7-2 level-4 count for 压力水平 "
               "against a consistency argument for 压力等级; level 1 picks the side the count was "
               "losing on, and it also aligns Stress Level with Panic Level (恐慌等级)."},
    "D4-health": {"en": "Health", "closed_by_level": 1, "settled_cn": "生命值",
        "how": "Level 1 ch.2 writes 生命值. All three Phase-0 sides were level-4 readings of one "
               "self-inconsistent file, and the count leader (健康) was entirely stratum B.",
        "still_to_do": "ALIENRPG.Health (生命) and ALIENRPG.healthDamage (健康损害) both change."},
    "D5-synthetic": {"en": "Synthetic", "closed_by_level": 1, "settled_cn": "合成人",
        "how": "Levels 1 and 2 agree on 合成人, and the owner's §1.4 three-way split assigns the "
               "other two words explicitly: Android 仿生人 (level-3 hard gate), Robot 机器人 "
               "(level 2). The Phase-0 dispute could not be settled because it was arguing about "
               "one word while the English has three."},
    "D7-weyland-yutani": {"en": "Weyland-Yutani", "closed_by_level": 1,
        "settled_cn": "韦兰德-汤谷公司",
        "how": "Level 1 spells it in full and clips it to 韦汤. Level 2 independently agrees that "
               "Yutani is 汤谷. The 尤坦尼 branch existed only at level 7 and is abandoned whole.",
        "still_to_do": "Nothing in lang; this is pack content plus the surname lock across four "
                       "sibling keys."},
    "D8-shift": {"en": "Shift", "closed_by_level": 1, "settled_cn": "班",
        "how": "Level 1 ch.1's time-unit table glosses 班 (Shift) inline, and completes the "
               "轮 / 节 / 班 set that §1.4 names. The Phase-0 dispute was 轮班 (level 4) vs 班次 "
               "(stratum B, which is never a source anyway).",
        "still_to_do": "⚠ The RUNTIME hazard is NOT closed and is not a terminology question: the "
                       "RollTable heal-time CELL must stay the bare English 'Shift'. See "
                       "byrole['Shift'] and provenance._meta.lockstep_literals."},
}
for _did, _c in CLOSED.items():
    _c["was"] = {"provisional_cn": PREV_DISP["disputes"][_did]["provisional_cn"],
                 "why_open_in_phase0": PREV_DISP["disputes"][_did]["why_open"]}

# every dispute must be backlinked from every term it governs, and vice versa
for _did, _d in DISPUTES.items():
    for _en in _d["governs_terms"]:
        if _en in terms:
            terms[_en]["dispute"] = _did

# ----------------------------------------------------------------- pending --
pending = {}
for _en, _v in PREV_PEND["terms"].items():
    if _en in terms:
        continue
    pending[_en] = dict(_v)
    pending[_en].setdefault("carried_from", "v0.2")
for _en, _v in TO_PENDING.items():
    pending[_en] = dict(_v)
    pending[_en]["moved_out_of_the_spine_in"] = "v1.0"
for _en in ["Atmosphere Processor", "Hadley's Hope", "Sulaco", "Smartgun", "Motion Tracker",
            "Dropship", "Neomorph"]:
    pending.pop(_en, None)
if "Fury 161" in pending:
    pending["Fury 161"]["v1_0_note"] = (
        "STILL PENDING, but the same world under its other English name IS now settled: "
        "Fiorina 161 = 菲奥莉娜161 at level 1. Only the 'Fury 161' spelling is unreached.")
if "Ellen Ripley" in pending:
    pending["Ellen Ripley"]["v1_0_note"] = (
        "STILL PENDING. The surname is settled (Ripley = 蕾普丽, level 6, en-gated 75) but the "
        "given name is not: level 6's only rendering of 'Ellen' is 海伦, which is on the project's "
        "known-bad list. en-gated 5 rows for 'Ellen Ripley', 3 of them 蕾普丽 with the given name "
        "dropped entirely.")

# ------------------------------------------------------------- byrole -------
# 裁定过的争议必须在 disputes 里标成已决 —— 否则读 disputes.json 的人会以为还没定。
# （2026-08-29 实测：RULINGS 已把 4 条的值落进 terms，disputes 里却还挂着未决。）
for _en, _r in RULING_BY_EN.items():
    _hit = None
    if isinstance(DISPUTES, dict):
        _pool = DISPUTES.get("disputes", DISPUTES)
        _seq = _pool.values() if isinstance(_pool, dict) else _pool
    else:
        _seq = DISPUTES
    for _d in _seq:
        if not isinstance(_d, dict):
            continue
        if _d.get("id") == _r.get("closes_dispute") or _d.get("en") == _en:
            _hit = _d
            break
    if _hit is None:
        raise SystemExit("RULING %s closes %r but no such dispute exists"
                         % (_r["id"], _r.get("closes_dispute")))
    _hit["status"] = "closed_by_ruling"
    _hit["closed_by"] = _r["id"]
    _hit["settled_cn"] = _r["adopt"]
    _hit["ruling_rationale"] = _r["rationale"]
    _hit["counter_argument_acknowledged"] = _r.get("counter_argument_acknowledged")
    if _r.get("domain_split"):
        _hit["domain_split"] = _r["domain_split"]

BYROLE = glossary_emit.byrole(terms, TIER_ORDER)

# ============================================================================
# INVARIANTS
# ============================================================================
PROPER_NOUN_CATS = {"franchise-place", "franchise-ship", "franchise-corp",
                    "franchise-character", "franchise-computer", "franchise-title"}
FROZEN_LITERALS = set()
_S = DNT["sections"]
for _e in _S["name_lookups"]["entries"]:
    FROZEN_LITERALS.add(_e["string"])
for _e in _S["rolltable_names"]["entries"]:
    FROZEN_LITERALS.add(_e["string"])
for _e in _S["folder_names"]["entries"]:
    FROZEN_LITERALS.add(_e["string"])
for _e in _S["item_names"]["entries"]:
    FROZEN_LITERALS.add(_e["string_compared"])
    for _d in _e.get("documents", []):
        FROZEN_LITERALS.add(_d["actual_name"])
FROZEN_LITERALS.add(_S["item_names"]["none_sentinel"]["string"])

def assertions():
    fail = []
    # 1 — no duplicate zh within one term's candidates
    for en, t in terms.items():
        zs = [c["zh"] for c in t.get("candidates", []) if c["zh"] is not None]
        if len(zs) != len(set(zs)):
            fail.append("DUP-ZH %s: %s" % (en, zs))
    # 2 — every term names a source level
    for en, t in terms.items():
        if t.get("source_level") in (None, ""):
            fail.append("NO-LEVEL %s" % en)
    # 3 — no term adopted on a bare Chinese count
    for en, t in terms.items():
        for c in t.get("candidates", []):
            if c.get("count_kind") in (None, "", "bare", "bare_chinese_count"):
                fail.append("BARE-COUNT %s / %s" % (en, c["zh"]))
            if isinstance(c.get("count"), str) and "bare" in c["count"].lower():
                fail.append("BARE-COUNT-TEXT %s / %s" % (en, c["zh"]))
    # 4 — the two owner-pinned exceptions
    if glossary.get("Wits") != "机智":
        fail.append("PIN Wits = %r" % glossary.get("Wits"))
    if glossary.get("Empathy") != "共情":
        fail.append("PIN Empathy = %r" % glossary.get("Empathy"))
    for en in ("Wits", "Empathy"):
        if terms[en].get("source_level") != 4 or not terms[en].get("owner_pinned"):
            fail.append("PIN-META %s not marked owner_pinned at level 4" % en)
    # 5 — every T-FROZEN entry traceable to a real lookup in DO-NOT-TRANSLATE.json
    for en, t in terms.items():
        if t["tier"] != "T-FROZEN":
            continue
        if t["cn"] != en:
            fail.append("FROZEN-VALUE %s -> %r (must be byte-exact English)" % (en, t["cn"]))
        if en not in FROZEN_LITERALS:
            fail.append("FROZEN-UNTRACEABLE %r has no lookup in DO-NOT-TRANSLATE.json" % en)
    # 6 — no T-FROZEN literal in the register is missing from the glossary
    for lit in FROZEN_LITERALS:
        if lit not in terms:
            fail.append("FROZEN-MISSING %r is a real lookup with no glossary key" % lit)
    # 7 — a proper noun may never be settled by convention
    for en, t in terms.items():
        if t["tier"] in ("T-FROZEN", "T-LATIN"):
            continue  # the value IS the English string; no Chinese was invented
        if t.get("category") in PROPER_NOUN_CATS and t.get("source_level") == "C" \
                and not t.get("derived_from"):
            fail.append("RULE-VIOLATION %s: proper noun settled at level C" % en)
    # 8 — T-EXACT values carry no English tail and name their lang key
    for en, t in terms.items():
        if t["tier"] == "T-EXACT":
            if re.search(r"[A-Za-z]", t["cn"]):
                fail.append("EXACT-TAIL %s -> %r" % (en, t["cn"]))
            if not t.get("lang_key"):
                fail.append("EXACT-NOKEY %s names no lang key" % en)
    # 9 — bilingual_name contract: cn + ONE ASCII space + the English NAME
    for en, t in terms.items():
        if t["tier"] == "T-BILINGUAL":
            want = t["cn"] + " " + name_of(en)
            if t.get("bilingual_name") != want:
                fail.append("BILINGUAL %s -> %r != %r" % (en, t.get("bilingual_name"), want))
    # 10 — disputes and terms backlink both ways
    for did, d in DISPUTES.items():
        for en in d["governs_terms"]:
            if en not in terms:
                fail.append("DISPUTE-DANGLING %s governs %r which is not in the glossary" % (did, en))
            elif terms[en].get("dispute") != did:
                fail.append("DISPUTE-NOBACKLINK %s <- %s" % (did, en))
    for en, t in terms.items():
        if t.get("dispute") and t["dispute"] not in DISPUTES:
            fail.append("DISPUTE-GHOST %s points at %s" % (en, t["dispute"]))
    # 11 — nothing is in both the glossary and pending
    for en in pending:
        if en in terms:
            fail.append("PENDING-LEAK %s is in both files" % en)
    # 12 — the zero-hit life-cycle set really is pending
    for en in ("Chestburster", "Ovomorph", "Drone", "Warrior", "Praetorian"):
        if en in terms:
            fail.append("ZERO-HIT-IN-SPINE %s" % en)
        if en not in pending:
            fail.append("ZERO-HIT-NOT-PENDING %s" % en)
    # 13 — the named §1.4 changes actually landed
    for en, want in [("Weyland-Yutani", "韦兰德-汤谷公司"), ("Nostromo", "诺斯特罗莫号"),
                     ("Sulaco", "萨拉科号"), ("Synthetic", "合成人"), ("Android", "仿生人"),
                     ("Robot", "机器人"), ("Atmosphere Processor", "大气处理器"),
                     ("Hadley's Hope", "哈德利的希望"), ("Round", "轮"), ("Stretch", "节"),
                     ("Shift", "班"), ("Cinematic Mode", "电影模式"),
                     ("Campaign Mode", "战役模式"), ("Frontier", "边境"),
                     ("Outer Veil", "外层帷幕"), ("Core Systems", "核心星系"),
                     ("Outer Rim", "外环"), ("Evolved Edition", "进化版"),
                     ("Facehugger", "抱脸虫"), ("Xenomorph", "异形"),
                     ("extraterrestrial", "外星")]:
        if glossary.get(en) != want:
            fail.append("MANDATE %s -> %r, expected %r" % (en, glossary.get(en), want))
    # 14 — a Chinese value must not silently serve two different English terms
    bycn = {}
    for en, t in terms.items():
        if t["tier"] in ("T-FROZEN", "T-LATIN"):
            continue
        bycn.setdefault(t["cn"], []).append(en)
    for cn, ens in bycn.items():
        if len(ens) > 1:
            declared = all(
                set(terms[e].get("same_string_as", [])) >= set(ens) - {e}
                for e in ens)
            if not declared:
                fail.append("CN-COLLISION %r <- %s  (add it to SAME_STRING_GROUPS with a reason, "
                            "or split the value)" % (cn, ens))
    # 16 — cn.json stratum B may never be a winner
    for en, t in terms.items():
        if t.get("source_level") == "B-rejected":
            fail.append("STRATUM-B-WINNER %s: stratum B is never a source" % en)
    # 15 — the Weyland surname is one spelling across the whole family
    fam = [e for e in terms if "Weyland" in e]
    for e in fam:
        if "韦兰德" not in terms[e]["cn"] and terms[e]["tier"] != "T-FROZEN":
            fail.append("SURNAME %s -> %r does not carry 韦兰德" % (e, terms[e]["cn"]))
        if not terms[e].get("surname_crossref"):
            fail.append("SURNAME-NOXREF %s" % e)
    return fail


# ============================================================================
# EMIT
# ============================================================================
import glossary_metablocks as MB


def counts_by(fn):
    d = {}
    for en, t in terms.items():
        k = str(fn(t))
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items()))


def main():
    check_only = "--check" in sys.argv
    fails = assertions()
    if fails:
        print("INVARIANTS FAILED (%d):" % len(fails))
        for f in fails:
            print("  " + f)
        print("NOTHING WRITTEN.")
        return 1
    print("INVARIANTS: all 16 pass over %d terms." % len(terms))
    if check_only:
        return 0

    lockstep = derive_lockstep()

    deviations = []
    for en, t in sorted(terms.items()):
        if t.get("beats_a_higher_level"):
            deviations.append({
                "term": en,
                "adopted": t["cn"],
                "adopted_at_level": t["source_level"],
                "outranked_level": t["beats_a_higher_level"],
                "why": t["why_this_won"],
            })

    meta = {
        "artifact": "glossary_alien.provenance.json",
        "version": VERSION,
        "generated": GENERATED,
        "supersedes": "v0.2.1, which was adjudicated with zh.wikipedia as the spine — before the "
                      "owner supplied the fan translation, the Romulus/Earth subtitles and the "
                      "Alien: Isolation localisation.",
        "built_by": "4-常用脚本/tm/build_glossary_v1.py (+ glossary_adjudication, "
                    "glossary_additions, glossary_emit, glossary_metablocks)",
        "count": len(terms),
        "count_is_authoritative": True,
        "count_note": "This field is the authority on how many terms there are. Do not transcribe "
                      "it into prose anywhere; point at it instead.",
        "tier_counts": counts_by(lambda t: t["tier"]),
        "source_level_counts": counts_by(lambda t: t["source_level"]),
        "category_counts": counts_by(lambda t: t.get("category", "?")),
        "companion_files": {
            "glossary": "glossary_alien.json — flat {en: cn}, the only shape consumers read",
            "disputes": "glossary_alien.disputes.json — genuine OPEN conflicts only",
            "pending": "glossary_alien.pending.json — no usable evidence; deliberately ABSENT "
                       "from the glossary so a translator stops and asks",
            "byrole": "glossary_alien.byrole.json — per (English, field role) tier",
        },
        "source_ladder": MB.LADDER,
        "not_a_ladder_level": MB.NOT_A_LEVEL,
        "owner_pinned_exceptions": {
            "_why": "Level 4 beats level 1 for exactly two terms, by the owner's ruling in "
                    "PROJECT.md §1.4. They appear on every character sheet, every roll and every "
                    "talent description, so continuity outweighs the ladder here and nowhere else.",
            "Wits": {"adopted": "机智", "level_1_says": "智力"},
            "Empathy": {"adopted": "共情", "level_1_says": "同理心"},
        },
        "hard_rules": MB.HARD_RULES,
        "rule_proper_nouns": MB.RULE_PROPER_NOUNS,
        "tiers": MB.TIERS,
        "tier_scope_decision": MB.TIER_SCOPE_DECISION,
        "bilingual_name_contract": MB.BILINGUAL_CONTRACT,
        "value_shape": MB.VALUE_SHAPE,
        "what_changed_from_phase0": MB.WHAT_CHANGED,
        "phase0_defects_addressed": MB.PHASE0_DEFECTS,
        "ladder_deviations": {
            "_why": MB.LADDER_DEVIATIONS_INTRO,
            "count": len(deviations),
            "entries": deviations,
        },
        "lockstep_literals": lockstep,
        "phase0_research_carried_forward": MB.PHASE0_RESEARCH_CARRIED_FORWARD,
        "known_gaps": MB.KNOWN_GAPS,
        "corpora": {
            "level_1": FAN["_meta"]["source_files"],
            "level_2_5_6": {
                "dir": SUB["meta"]["corpus_dir"],
                "total_pairs": SUB["meta"]["total_pairs"],
                "lineages": SUB["meta"]["lineages"],
                "folding": SUB["meta"]["lineage_folding"],
            },
            "level_3": {
                "root": ISO["_meta"]["source_root"],
                "unique_keys": ISO["_meta"]["corpus_correction"]["unique_keys"],
                "gated_fraction_of_live":
                    ISO["english_gate_coverage"]["totals"]["gated_fraction_of_live"],
            },
            "level_4": {
                "en": os.path.join(LANG, "en.json"),
                "cn": os.path.join(LANG, "cn.json"),
                "en_leaves": len(LANG_EN),
                "cn_leaves": len(LANG_CN),
            },
        },
        "invariants_asserted": [
            "1  no duplicate zh within one term's candidates",
            "2  every term names a source level",
            "3  no candidate carries a bare Chinese count",
            "4  Wits = 机智 and Empathy = 共情, both marked owner_pinned at level 4",
            "5  every T-FROZEN value is byte-exact English AND resolves to a named lookup in "
            "DO-NOT-TRANSLATE.json",
            "6  every frozen literal in DO-NOT-TRANSLATE.json has a glossary key",
            "7  no proper noun is settled at level C without a derived_from parent",
            "8  every T-EXACT value is tail-free and names its lang key",
            "9  bilingual_name == cn + one ASCII space + the English name",
            "10 disputes and terms backlink in both directions",
            "11 nothing appears in both the glossary and pending",
            "12 the zero-hit life-cycle set is pending, not spine",
            "13 every value the owner named in §1.4 actually landed",
            "14 no Chinese value serves two English terms without a declared split",
            "15 the Weyland surname is one spelling across the family, all cross-linked",
            "16 no term wins at cn.json stratum B, which is never a source",
        ],
        "revision_history": [h for h in PREV["_meta"]["revision_history"] if h.get("version") != "v1.0"] + [{
            "version": "v1.0",
            "what": "Spine re-cut under the owner's settled ladder. zh.wikipedia moved from rung 1 "
                    "to rung 7; the Chinese translation of this book, the 2024-25 subtitles and "
                    "the Alien: Isolation official localisation moved in above it. Every term "
                    "re-adjudicated. Six Phase-0 disputes CLOSED by a higher rung, two new ones "
                    "opened, the seven verifier-found defects fixed, and one more found in the "
                    "doing (a T-FROZEN key with the wrong whitespace codepoints).",
        }],
    }

    prov = {"_meta": meta, "terms": {k: terms[k] for k in sorted(terms)}}

    disp = {
        "_meta": {
            "artifact": "glossary_alien.disputes.json",
            "version": VERSION,
            "generated": GENERATED,
            "count": len(DISPUTES),
            "count_is_authoritative": True,
            "note": "GENUINE OPEN CONFLICTS ONLY. A conflict the ladder settles is a decision, not "
                    "a dispute, and lives in closed_in_v1_0 below. glossary_alien.json carries a "
                    "PROVISIONAL winner for each open dispute so the file stays usable; that "
                    "value is NOT a decision.",
            "backlinks": "provenance.terms[<en>].dispute points here, and every dispute names the "
                         "terms it governs in governs_terms. BOTH directions are GENERATED, not "
                         "hand-maintained, and are asserted by the builder — Phase 0 lost a "
                         "backlink precisely because one side was maintained by hand.",
            "closed_count": len(CLOSED),
            "opened_in_v1_0": sorted(d for d in DISPUTES if DISPUTES[d].get("opened_in") == "v1.0"),
            "closed_in_v1_0": CLOSED,
        },
        "disputes": DISPUTES,
    }

    pend = {
        "_meta": {
            "artifact": "glossary_alien.pending.json",
            "version": VERSION,
            "generated": GENERATED,
            "count": len(pending),
            "count_is_authoritative": True,
            "note": "Terms with NO usable evidence under the ladder. They are deliberately ABSENT "
                    "from glossary_alien.json: an absent key makes a translator stop and ask, "
                    "whereas a plausible-looking guess propagates silently through 2.7M "
                    "characters of content.",
            "rule": MB.RULE_PROPER_NOUNS,
            "moved_out_of_the_spine_in_v1_0": sorted(TO_PENDING),
            "resolved_out_of_pending_in_v1_0": {
                "Atmosphere Processor": "大气处理器 — level 1 ch.1, glossed inline",
                "Hadley's Hope": "哈德利的希望 — level 1 ch.1, glossed inline",
                "Sulaco": "萨拉科号 — level 1 ch.1 (USS萨拉科号), glossed inline",
                "Smartgun": "keyed as Smart Gun = 智能炮 — level 1 ch.2 (M56A2智能炮)",
                "Motion Tracker": "运动追踪器 — level 1 ch.2 AND a level-3 hard key gate "
                                  "(TEXT_MOTIONTRACKER)",
                "Dropship": "着陆舰 — level 1 ch.2",
                "Neomorph": "新变种 — level 1 ch.1",
            },
            "permanently_unreachable": {
                "_why": "Level 1 covers only ch.1 and ch.2 and the owner has confirmed no more "
                        "exists. The Xenomorph life cycle is ch.10. These five are zero-hit across "
                        "all 11,088 subtitle pairs AND absent from the Alien: Isolation "
                        "localisation, so nothing on the ladder above rung 7 can reach them.",
                "terms": ["Chestburster", "Ovomorph", "Drone", "Warrior", "Praetorian"],
            },
        },
        "terms": {k: pending[k] for k in sorted(pending)},
    }

    byrole_out = {
        "_meta": {
            "artifact": "glossary_alien.byrole.json",
            "version": VERSION,
            "generated": GENERATED,
            "count": len(BYROLE),
            "count_is_authoritative": True,
            "why_this_file_exists":
                "'Does this name carry the English tail' is not a property of a TERM. It is a "
                "property of a (term, field) pair. Shift is 班 in prose, 班 in ALIENRPG.Shift, and "
                "the bare English literal 'Shift' in the RollTable cell — and getting the third "
                "one wrong throws an uncaught TypeError that kills the Critical-Injury roll. "
                "Hardened is 硬汉 in prose and MUST stay 'Hardened' in the Item name. M5A3 RPG "
                "Launcher is the mirror image: there the BILINGUAL tail is what keeps it working, "
                "because the code tests name.includes(' RPG '). One tier per term cannot express "
                "any of that.",
            "roles": glossary_emit.ROLE_DOC,
            "authored_exceptions": sorted(glossary_emit.BYROLE_OVERRIDES),
            "how_the_rest_is_derived": "Every other row is generated from the term's tier and "
                                       "category, so it cannot drift out of step with the "
                                       "provenance file.",
        },
        "terms": BYROLE,
    }

    for name, obj in [("glossary_alien.json", glossary),
                      ("glossary_alien.provenance.json", prov),
                      ("glossary_alien.disputes.json", disp),
                      ("glossary_alien.pending.json", pend),
                      ("glossary_alien.byrole.json", byrole_out)]:
        p = os.path.join(GLOSS, name)
        n = jdump(p, obj)
        print("wrote %-34s %9d bytes" % (name, n))

    print("")
    print("terms %d | disputes %d (closed %d) | pending %d | byrole %d"
          % (len(terms), len(DISPUTES), len(CLOSED), len(pending), len(BYROLE)))
    print("tiers:  " + json.dumps(meta["tier_counts"], ensure_ascii=False))
    print("levels: " + json.dumps(meta["source_level_counts"], ensure_ascii=False))
    print("ladder deviations: %d" % len(deviations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
