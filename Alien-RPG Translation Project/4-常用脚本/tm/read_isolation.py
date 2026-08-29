# -*- coding: utf-8 -*-
"""
read_isolation.py -- reusable reader for the Alien: Isolation official Simplified
Chinese localisation shipped at

    C:/Users/Taka/Desktop/fvtt/AlienRPG/简体汉化/

FACTS ABOUT THE TREE (measured 2026-08-29, do not re-discover):
  * 768 .TXT files on disk, but only **64 are unique**. The 704 files under
    DATA/ENV/PRODUCTION/DLC/<MAP>/TEXT/ENGLISH/ are 11 byte-identical copies of
    the 64 files in DATA/TEXT/ENGLISH/ (verified by md5: 768 paths -> 64 hashes).
    Counting them multiplies every statistic by 12. This reader reads the base
    tree ONLY, unless include_dlc_copies=True.
  * Encoding is **UTF-16LE with BOM** (ff fe), NOT utf-8-sig.
  * Block format is:   [KEY]\n{value}\n\n     (value may contain newlines)
  * Unique key count: 13565.  Unique Chinese chars in values: 201328.
    (x12 = 162780 keys / 2415936 chars, which is where the "162,804 keys /
    6.75M chars" figure in circulation comes from.)
  * The directory is named ENGLISH but contains **only Chinese**. There is no
    English mirror in this tree and none elsewhere on this machine: DATA/UI/
    FONTS_CN.GFX is a compiled Scaleform font library with zero extractable
    ASCII strings, and DATA/FONT_CONFIG.XML is font mapping only.

THE ENGLISH-GATE PROBLEM
  This project forbids adopting a term from a bare Chinese frequency count.
  Because the English side is absent, only two mechanical gates exist here:

    GATE "key"     -- the KEY identifier is itself English and content-bearing,
                      e.g. [TEXT_USE]{使用}, [AI_EXTERN_360_PRESENCE_KILLED_BY_
                      ANDROID]{被仿生人所杀}. This is a genuine English gate: the
                      slug comes from the English source string.
    GATE "inline"  -- the localisers LEFT the English proper noun inside the
                      Chinese value, e.g. {Torrens号}, {Weyland-Yutani}. The
                      English literal is physically present, so the pairing is
                      not inferred. This gate also proves a T-FROZEN decision.

  Everything else (voice lines keyed A1_CUTSCENES_SAM_C0100_0004, terminal text
  keyed A1_T0001_TEX_4066) is **ungated**: the key is an opaque line ID. Terms
  seen only there must be marked english_gate=false, capped at confidence
  'low', and may only CORROBORATE a rendering attested elsewhere.

  AUDITED COVERAGE (2026-08-29, reproduce with `python read_isolation.py`):
      13565 total keys -> 1018 placeholder stubs -> 12547 live
      gated  2447 (19.50% of live)   ungated 10100 (80.50% of live)
  The gated fifth is concentrated in FOUR files -- GLOBAL.TXT (166-entry item
  dictionary, effectively a bilingual en/zh word list), UI.TXT, EXTERNAL.TXT
  (achievements) and TUTORIALS.TXT. Every dialogue file, every terminal-log
  file and the whole of LOCATIONS.TXT are 100% ungated, which means every
  station location name and every spoken line is corroboration-only.

USAGE
    from read_isolation import load, find, key_gate_english
    recs = load()                      # list[Rec]
    for r in find(recs, '抱脸虫'): ...  # substring search over values
    find(recs, re.compile(r'Sevastopol'))
"""
from __future__ import annotations

import os
import re
import glob
import collections
from dataclasses import dataclass

ROOT = r"C:\Users\Taka\Desktop\fvtt\AlienRPG\简体汉化"
BASE_TEXT = os.path.join(ROOT, "DATA", "TEXT", "ENGLISH")
DLC_GLOB = os.path.join(ROOT, "DATA", "ENV", "PRODUCTION", "DLC", "*", "TEXT", "ENGLISH", "*.TXT")

BLOCK_RE = re.compile(r'^\[([^\]\r\n]+)\]\r?\n\{(.*?)\}\s*$', re.S | re.M)

# control tokens the game substitutes at runtime, e.g. #crouch#, and markup
CTRL_RE = re.compile(r'#[A-Za-z0-9_]+#')
MARKUP_RE = re.compile(r'<[^>]{1,80}>')
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/'\-\.]{1,}")

# Structural / speaker / emotion / scene codes. These appear in keys but say
# NOTHING about what the string means, so they must never count as a gate.
# Curated by inspecting every alpha key segment with frequency >= 3.
STOPSEG = set("""A1 AI UI DLG EXTERN TUT POP TEXT LOC TEX CON OBJ LOG ALT ALT2 ALT01 XBOX PS3
PS4 WIN VITA CHAR BESPOKE WAV1 WAV2 PRI SEC DESC LOCK TITLE TITL HEADER BODY PROMPT MES
FRONTEND PACK TROPHY ACHIEVEMENT PRESENCE CLIP KB MBUTTON DEALIEN SGF SGM RIC RIP SAM TAY
WAI LIN MAR HEY SIN SPE ELL LAM ASH DAL ADV VER MOT PAR AN1 AN2 AN3 KUH SIC SII SICE SICDB
SICAF SIIE SIIDB SIIAF HBWSB HBWR HBWIRE HBWE HBWFB HBWI HBWB DSVF TRAN TERM GEMSECSYS
LORENZ ALI AND RIO PLA PAN ALA ATT INT WAR SFC SWF XTRA1 XTRA2 IC LC PO AD SE RE CA CT DE
AL NPC STR UP CUTSCENES OBJECTIVE NAME SECONDARY""".split())

# Speaker-bark files: keys are purely speaker+emotion codes (A1_CV1_G_USE_SIIE_CA_B),
# never content slugs. Excluded from the key gate wholesale.
BARK_FILES = {'CV1.TXT', 'CV2.TXT', 'CV3.TXT', 'CV4.TXT', 'CV5.TXT', 'CV6.TXT',
              'SS1.TXT', 'SS2.TXT', 'SS3.TXT', 'G0000.TXT', 'G0001.TXT', 'G0002.TXT',
              'G0003.TXT', 'P0001.TXT', 'P0002.TXT'}

VOWEL_RE = re.compile(r'[AEIOUaeiou]')
# a mission/line/speaker code: M0401, C0100, CV1, G0001, or a bare short acronym
CODE_RE = re.compile(r'^([A-Z]{1,4}\d{1,5}[A-Z]?|\d+|[A-Z]{1,3})$')


@dataclass(frozen=True)
class Rec:
    file: str          # basename, e.g. 'GLOBAL.TXT'
    key: str
    value: str

    @property
    def clean(self) -> str:
        """value with runtime control tokens and markup stripped"""
        return MARKUP_RE.sub('', CTRL_RE.sub('', self.value))

    @property
    def is_placeholder(self) -> bool:
        """untranslated stub: value is the key repeated verbatim, or empty (1018 of 13565)"""
        return self.value.strip() == self.key or not self.value.strip()

    @property
    def key_words(self) -> list[str]:
        """content-bearing English words recoverable from the key, [] if opaque"""
        if self.is_placeholder or self.file in BARK_FILES:
            return []
        out = []
        for s in self.key.split('_'):
            u = s.upper()
            if not s or u in STOPSEG or CODE_RE.match(u) or len(s) < 3:
                continue
            if not VOWEL_RE.search(s):
                continue
            out.append(s)
        return out

    @property
    def inline_english(self) -> list[str]:
        """English literals the localisers left inside the Chinese value"""
        if self.is_placeholder:
            return []
        v = re.sub(r'@[a-z_]+@', '', self.clean)
        return [t for t in LATIN_RE.findall(v)
                if len(t) >= 3 and VOWEL_RE.search(t) and not CODE_RE.match(t.upper())]

    @property
    def gate(self) -> str:
        """'key' | 'inline' | 'both' | 'none' | 'placeholder'"""
        if self.is_placeholder:
            return 'placeholder'
        k = bool(self.key_words)
        i = bool(self.inline_english)
        return 'both' if (k and i) else 'key' if k else 'inline' if i else 'none'


def _parse(path: str) -> list[Rec]:
    txt = open(path, 'rb').read().decode('utf-16')
    b = os.path.basename(path)
    return [Rec(b, k, v) for k, v in BLOCK_RE.findall(txt)]


def load(include_dlc_copies: bool = False) -> list[Rec]:
    """Load the 64 unique files. The DLC tree is 11 exact copies; skip by default."""
    recs: list[Rec] = []
    for p in sorted(glob.glob(os.path.join(BASE_TEXT, "*.TXT"))):
        recs.extend(_parse(p))
    if include_dlc_copies:
        for p in sorted(glob.glob(DLC_GLOB)):
            recs.extend(_parse(p))
    return recs


def find(recs, needle, field: str = 'value') -> list[Rec]:
    """substring (str) or regex (compiled pattern) search over 'value' or 'key'."""
    get = (lambda r: r.value) if field == 'value' else (lambda r: r.key)
    if hasattr(needle, 'search'):
        return [r for r in recs if needle.search(get(r))]
    return [r for r in recs if needle in get(r)]


def key_gate_english(recs, word: str) -> list[Rec]:
    """Records whose KEY contains the English word -- a real English gate."""
    w = word.upper()
    return [r for r in recs if w in [s.upper() for s in r.key_words]]


def gate_coverage(recs) -> dict:
    """Audited 2026-08-29: 19.50% of live keys gated, 80.50% ungated.
    If these numbers move, the stoplist above was edited -- re-audit before trusting."""
    c = collections.Counter(r.gate for r in recs)
    n = len(recs)
    live = n - c['placeholder']
    gated = c['key'] + c['inline'] + c['both']
    return {
        'total_keys': n,
        'placeholder': c['placeholder'],
        'live_keys': live,
        'gate_key_only': c['key'],
        'gate_inline_only': c['inline'],
        'gate_both': c['both'],
        'gated_total': gated,
        'ungated': c['none'],
        'gated_fraction_of_live': round(gated / live, 4),
        'ungated_fraction_of_live': round(c['none'] / live, 4),
    }


if __name__ == '__main__':
    import sys
    rs = load()
    print(f"loaded {len(rs)} unique keys from {len(set(r.file for r in rs))} files")
    print(gate_coverage(rs))
    if len(sys.argv) > 1:
        pat = re.compile(sys.argv[1])
        hits = find(rs, pat)
        print(f"\n{len(hits)} hits for {sys.argv[1]!r}")
        for r in hits[:30]:
            print(f"  {r.file}:[{r.key}] gate={r.gate} {r.value[:110]!r}")
