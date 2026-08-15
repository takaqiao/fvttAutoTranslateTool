# -*- coding: utf-8 -*-
"""Probe: will each enricher occurrence actually FIRE under crucible 0.10.1 / ember 0.6.0?

Method
------
1. Transcribe the exact upstream enricher regexes (crucible-compiled.mjs registerEnrichers,
   ember.mjs registerEnrichers, ember dnd5e registerEnrichers$1) into Python with re.ASCII
   so that JS `\\w` semantics (ASCII-only, CJK excluded) are preserved.
2. Find every *candidate* enricher-looking token; check whether any upstream pattern matches
   at that exact offset. Non-matching -> the token renders as literal text in the journal.
3. Validate captured argument values against upstream value tables (skills, statuses,
   knowledge, languages, rules, action tags, currency).
4. Run identically over EN and CN so "upstream is also broken" can be separated from
   "we broke it in translation".

False positives (documented):
  - dnd5e-only enrichers ([[/check]] [[/save]] [[/damage]] [[lookup]] ...) are listed in
    DND5E_HEADS and reported separately, not as crucible failures.
  - Foundry core enrichers (@UUID @Embed inline rolls) are in CORE_HEADS.
  - A candidate inside a `<sub data-system="dnd5e">` block is still scanned; the head
    classification handles it.
"""
import json, os, re, sys, collections

A = re.ASCII
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPOS = {
    "ember": os.path.join(ROOT, "1-Ember汉化插件"),
    "crucible": os.path.join(ROOT, "2-Crucible汉化插件"),
}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_inventory import walk_json

# ---------------------------------------------------------------- upstream patterns
# crucible 0.10.1 -- crucible-compiled.mjs :46106
CRUCIBLE = [
    ("award",         r"\[\[/award ([-\w\s]+)\]\]"),
    ("counterspell",  r"\[\[/counterspell ([\w\s=]+)\]\]"),
    ("hazard",        r"\[\[/hazard ([\w\s]+)\]\](?:\{([^}]+)\})?"),
    ("skillCheck",    r"\[\[/skillCheck ([\w\s]+)\]\]"),
    ("knowledge",     r"\[\[/knowledge (\w+)\]\]"),
    ("language",      r"\[\[/language (\w+)\]\]"),
    ("dnd5eSkill",    r"\[\[/skill ([\w\s]+)\]\]"),
    ("talent",        r"\[\[/talent ([\w\-.]+)\]\]"),
    ("condition",     r"@Condition\[(\w+)\]"),
    ("action",        r"@Action\[([\w\-.]+) (\w+)\]"),
    ("spell",         r"@Spell\[([\w.]+)\]"),
    ("rule",          r"@Rule\[([\w.]+)\](?:\{([^}]+)\})?"),
    ("milestone",     r"\[\[/milestone( \d+)?\]\]"),
    ("ref",           r"@ref\[([\w.]+)\](?:\{([^}]+)\})?"),
    ("loot",          r"@Loot\[([\w\-.]+)((?:\s+[\w]+=?[\w]*)*)\](?:\{([^}]+)\})?"),
    ("scroll",        r"@Scroll\[([\w\s]+)\](?:\{([^}]+)\})?"),
]
# ember 0.6.0 -- ember.mjs :129403 (+ dnd5e-specific :123656)
EMBER = [
    ("date",          r"\[\[/date ([A-Z]{2})([+-]?)([0-9]+)\]\]"),
    ("ancestry",      r"\[\[/ancestry (\w+)\]\]"),
    ("culture",       r"\[\[/culture (\w+)\]\]"),
    ("path",          r"\[\[/path (\w+)\]\]"),
    ("attunement",    r"\[\[/attunement (\w+)(?: ([+-]?\d+) (\w+))?\]\]"),
    ("emberLanguage", r"\[\[/language (\w+)\]\]"),
    ("soundscape",    r"\[\[/soundscape ([\w\s=]+)\]\]"),
    ("eventState",    r"\[\[/eventState (\w+)( \w+)?\]\]"),
    ("outcome",       r"\[\[/outcome (\w+)\]\]"),
    ("advantage",     r"@Advantage\[(-?\d)\]"),
    ("critical",      r"@Critical(Success|Failure)\[(\d{1,2})\]"),
    ("emberKnowledge", r"\[\[/knowledge (\w+)\]\]"),
]
PATTERNS = [(n, re.compile(p, A)) for n, p in CRUCIBLE + EMBER]

CORE_HEADS = {"UUID", "Embed", "embed", "Compendium", "Macro", "Draw", "Localize",
              "Lookup", "Actor", "Item", "JournalEntry", "RollTable", "Scene"}
DND5E_BB = {"/check", "/save", "/damage", "/attack", "/item", "/tool", "/heal",
            "/healing", "/ability", "/concentration", "lookup", "/lookup",
            "/skill"}
CORE_BB = {"/roll", "/r", "/gmroll", "/gmr", "/blindroll", "/br", "/selfroll",
           "/sr", "/publicroll", "/pr"}

# ---------------------------------------------------------------- upstream value tables
SKILLS = {"athletics", "awareness", "stealth", "wilderness", "arcana", "medicine",
          "science", "society", "deception", "diplomacy", "intimidation", "performance"}
DND5E_SKILL_MAPPING = {
    "acr", "acrobatics", "ani", "animalHandling", "arc", "arcana", "ath", "athletics",
    "dec", "deception", "his", "history", "ins", "insight", "itm", "intimidation",
    "inv", "investigation", "med", "medicine", "nat", "nature", "prc", "perception",
    "prf", "performance", "per", "persuasion", "rel", "religion", "slt", "sleightOfHand",
    "ste", "stealth", "sur", "survival"}
STATUSES = {"weakened","dead","broken","insane","staggered","stunned","prone","restrained",
    "slowed","hastened","disoriented","exhausted","blinded","burrowing","flying","deafened",
    "silenced","enraged","frightened","invisible","invulnerable","limitless","resolute",
    "guarded","exposed","flanked","diseased","paralyzed","asleep","suffocating",
    "incapacitated","unaware","falling","bleeding","burning","freezing","confused",
    "corroding","decaying","dominated","entropy","irradiated","mending","inspired",
    "poisoned","shocked"}
ACTION_TAGS = {"dualwield","onehand","finesse","brute","projectile","mechanical","shield",
    "talisman","unarmed","unarmored","afterStrike","rest","vocal","auditory","reaction",
    "noncombat","flanking","consume","spell","composed","iconicSpell","summon","strike",
    "melee","ranged","mainhand","twohand","offhand","thrown","natural","hazard","generic",
    "reload","disarm","deadly","difficult","empowered","keen","accurate","harmless",
    "undetectable","weakened","severe","fortitude","reflex","willpower","healing",
    "rallying","maintained","movement"}
KNOWLEDGE = {"alchemy","ancients","artifacts","arts","beasts","celestials","cosmology",
    "crafts","crime","dragons","elementals","fey","fiends","forensics","gods","intrigue",
    "legends","machines","monsters","plants","politics","rituals","seafaring","souls",
    "subterranea","tracking","trade","undeath","warfare","weather"}          # 'outsiders' deleted by ember
KNOWLEDGE_EMBER = KNOWLEDGE | {"abyssals", "aedir", "leviathans", "shent", "outsiders"}  # outsiders is an alias
LANGUAGES = {"common", "sign"}
LANGUAGES_EMBER = LANGUAGES | {"arcden","cascal","forest","hardac","imperial","solical",
    "mithia","luma","kaziric","scripta","wyrdic","pathward","scor","towyr","windclaw",
    "abyssal","draconic","druidic","lunix","caligon","eonic","harmos","cant"}
CURRENCY = {"cp", "sp", "gp", "pp"}

# candidate finders (deliberately loose)
CAND_AT = re.compile(r"@[A-Za-z][A-Za-z0-9_]*\[", A)
CAND_BB = re.compile(r"\[\[/?[A-Za-z][A-Za-z0-9_]*", A)


def flat(path):
    d = json.load(open(path, encoding="utf-8"))
    sink = []
    walk_json(d, [], sink)
    return dict(sink)


def analyse(s):
    """Return list of (start, head, matched_pattern_name_or_None, snippet)."""
    starts = {}
    for m in PATTERNS:
        pass
    matched = {}   # start -> (name, end)
    for name, rx in PATTERNS:
        for mm in rx.finditer(s):
            st = mm.start()
            if st not in matched or mm.end() > matched[st][1]:
                matched[st] = (name, mm.end(), mm)
    out = []
    for cm in CAND_AT.finditer(s):
        head = s[cm.start() + 1:cm.end() - 1]
        out.append((cm.start(), "@" + head, matched.get(cm.start()), s[cm.start():cm.start() + 90]))
    for cm in CAND_BB.finditer(s):
        head = s[cm.start() + 2:cm.end()]
        out.append((cm.start(), head, matched.get(cm.start()), s[cm.start():cm.start() + 90]))
    return out


def classify(head):
    if head.startswith("@"):
        return "core" if head[1:] in CORE_HEADS else "sys"
    if head in CORE_BB:
        return "core"
    if head in DND5E_BB:
        return "dnd5e"
    return "sys"


def main():
    unmatched = []      # candidate that no upstream pattern accepts
    badvalue = []       # matched but argument value not in upstream table
    for repo, base in REPOS.items():
        for side in ("en", "cn"):
            d = os.path.join(base, "compendium", side)
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json") or fn == "_source.json":
                    continue
                for jp, s in flat(os.path.join(d, fn)).items():
                    if "@" not in s and "[[" not in s:
                        continue
                    for st, head, mt, snip in analyse(s):
                        cls = classify(head)
                        if mt is None:
                            if cls == "sys":
                                unmatched.append({"repo": repo, "side": side, "file": fn,
                                                  "jpath": jp, "head": head, "snip": snip})
                            continue
                        name, end, mm = mt
                        g = mm.groups()
                        bad = None
                        if name == "skillCheck":
                            sk = g[0].split(" ")[0]
                            if sk not in SKILLS and sk not in DND5E_SKILL_MAPPING:
                                bad = "skill id"
                        elif name == "dnd5eSkill":
                            if g[0].split(" ")[0] not in DND5E_SKILL_MAPPING:
                                bad = "dnd5e skill id"
                        elif name == "condition":
                            if g[0] not in STATUSES:
                                bad = "status id"
                        elif name in ("knowledge", "emberKnowledge"):
                            if g[0] not in KNOWLEDGE_EMBER:
                                bad = "knowledge id"
                        elif name in ("language", "emberLanguage"):
                            if g[0] not in LANGUAGES_EMBER:
                                bad = "language id"
                        elif name == "rule":
                            rid = g[0]
                            head2, _, rest = rid.partition(".")
                            if head2 == "condition":
                                if rest not in STATUSES:
                                    bad = "rule condition id"
                            elif head2 == "action":
                                if rest not in ACTION_TAGS:
                                    bad = "rule action tag"
                            else:
                                bad = "rule namespace"
                        elif name == "ref":
                            rid = g[0]
                            head2, _, rest = rid.partition(".")
                            if head2 == "condition" and rest not in STATUSES:
                                bad = "ref condition id"
                            elif head2 == "action" and rest not in ACTION_TAGS:
                                bad = "ref action tag"
                        elif name == "hazard":
                            parts = g[0].split()
                            if not parts or not parts[0].isdigit():
                                bad = "hazard danger not numeric"
                            else:
                                for t in parts[1:]:
                                    if t not in ACTION_TAGS:
                                        bad = "hazard tag %r" % t
                                        break
                        elif name == "award":
                            for part in g[0].split():
                                if part == "each":
                                    continue
                                mo = re.match(r"^(.+?)(\D+)$", part)
                                if not mo or mo.group(2).lower() not in CURRENCY:
                                    bad = "award term %r" % part
                                    break
                        if bad:
                            badvalue.append({"repo": repo, "side": side, "file": fn,
                                             "jpath": jp, "enricher": name, "why": bad,
                                             "snip": snip})
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(unmatched, open(os.path.join(here, "unmatched.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(badvalue, open(os.path.join(here, "badvalue.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("UNMATCHED (system enricher that will not fire):")
    c = collections.Counter((u["repo"], u["side"], u["head"]) for u in unmatched)
    for k, v in c.most_common(60):
        print("   ", k, v)
    print("BADVALUE:")
    c2 = collections.Counter((b["repo"], b["side"], b["enricher"], b["why"]) for b in badvalue)
    for k, v in c2.most_common(60):
        print("   ", k, v)


if __name__ == "__main__":
    main()
