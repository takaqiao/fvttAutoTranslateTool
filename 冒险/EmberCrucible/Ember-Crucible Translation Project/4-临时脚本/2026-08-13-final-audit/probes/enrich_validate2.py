# -*- coding: utf-8 -*-
"""Corrected validator.

Fixes over v1:
  * SYSTEM.ACTION.TAGS is built dynamically in crucible-compiled.mjs (lines 4639-4734):
    movement action ids + DAMAGE_TYPES + ABILITIES + ["health","morale"] + "skill" + SKILL ids
    are all appended. v1's static list produced ~250 false positives.
  * Adds swap-block awareness: ember's finalizeEnrichedHTML (ember.mjs:23219) deletes the
    <sub data-system="X"> / <div data-system="X"> branch that does not match game.system.id,
    so a token inside the dnd5e branch is never displayed under crucible and must not be
    reported as a crucible defect.
"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_validate as V
from enrich_inventory import walk_json

MOVEMENT = {"walk", "step", "crawl", "jump", "climb", "swim", "fly", "blink", "burrow"}
DAMAGE_TYPES = {"bludgeoning", "corruption", "piercing", "slashing", "poison", "acid",
                "fire", "cold", "electricity", "psychic", "radiant", "void"}
ABILITIES = {"wisdom", "presence", "intellect", "strength", "toughness", "dexterity"}
RESOURCES = {"health", "morale"}
TAGS = (V.ACTION_TAGS | MOVEMENT | DAMAGE_TYPES | ABILITIES | RESOURCES
        | {"skill"} | V.SKILLS)
V.ACTION_TAGS = TAGS

# marker = every <sub|div data-system="..."> open, and every </sub>|</div> close
MARK = re.compile(r'<(sub|div)\b[^>]*\bdata-system="(\w+)"[^>]*>|</(sub|div)>|<(sub|div)\b[^>]*>')


_CACHE = {}


def _marks(s):
    out = _CACHE.get(id(s))
    if out is None:
        out = [(m.start(), m.group(1), m.group(2), m.group(3), m.group(4))
               for m in MARK.finditer(s)]
        _CACHE.clear()
        _CACHE[id(s)] = out
    return out


def swap_system(s, pos):
    """Innermost data-system attribute governing offset pos, or None."""
    stack = []
    for start, otag, sysid, ctag, ptag in _marks(s):
        if start >= pos:
            break
        if sysid:
            stack.append((otag, sysid))
        elif ctag:
            if stack and stack[-1][0] == ctag:
                stack.pop()
        elif ptag:
            stack.append((ptag, None))
    for tag, sysid in reversed(stack):
        if sysid:
            return sysid
    return None


def flat(path):
    d = json.load(open(path, encoding="utf-8"))
    sink = []
    walk_json(d, [], sink)
    return dict(sink)


def main():
    rows = []
    for repo, base in V.REPOS.items():
        for side in ("en", "cn"):
            d = os.path.join(base, "compendium", side)
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json") or fn == "_source.json":
                    continue
                pack_system = "dnd5e" if fn == "ember.adventure.json" or "dnd5e" in fn else "crucible"
                for jp, s in flat(os.path.join(d, fn)).items():
                    if "@" not in s and "[[" not in s:
                        continue
                    for st, head, mt, snip in V.analyse(s):
                        cls = V.classify(head)
                        vis = swap_system(s, st)
                        if mt is None:
                            if cls == "sys":
                                rows.append({"kind": "UNMATCHED", "repo": repo, "side": side,
                                             "file": fn, "jpath": jp, "head": head,
                                             "swap": vis, "pack": pack_system, "snip": snip})
                            continue
                        name, end, mm = mt
                        g = mm.groups()
                        bad = None
                        if name == "skillCheck":
                            sk = g[0].split(" ")[0]
                            if sk not in V.SKILLS and sk not in V.DND5E_SKILL_MAPPING:
                                bad = "skill id %r" % sk
                        elif name == "dnd5eSkill":
                            sk = g[0].split(" ")[0]
                            if sk not in V.DND5E_SKILL_MAPPING:
                                bad = "dnd5e skill id %r" % sk
                        elif name == "condition":
                            if g[0] not in V.STATUSES:
                                bad = "status id %r" % g[0]
                        elif name in ("knowledge", "emberKnowledge"):
                            if g[0] not in V.KNOWLEDGE_EMBER:
                                bad = "knowledge id %r" % g[0]
                        elif name in ("language", "emberLanguage"):
                            if g[0] not in V.LANGUAGES_EMBER:
                                bad = "language id %r" % g[0]
                        elif name == "rule":
                            h2, _, rest = g[0].partition(".")
                            if h2 == "condition" and rest not in V.STATUSES:
                                bad = "rule condition %r" % rest
                            elif h2 == "action" and rest not in TAGS:
                                bad = "rule action tag %r" % rest
                            elif h2 not in ("condition", "action"):
                                bad = "rule namespace %r" % h2
                        elif name == "ref":
                            h2, _, rest = g[0].partition(".")
                            if h2 == "condition" and rest not in V.STATUSES:
                                bad = "ref condition %r" % rest
                            elif h2 == "action" and rest not in TAGS:
                                bad = "ref action tag %r" % rest
                            elif h2 not in ("condition", "action"):
                                bad = "ref namespace %r" % h2
                        elif name == "hazard":
                            parts = g[0].split()
                            if not parts or not parts[0].isdigit():
                                bad = "hazard danger %r" % (parts[0] if parts else "")
                            else:
                                for t in parts[1:]:
                                    if t not in TAGS:
                                        bad = "hazard tag %r" % t
                                        break
                        elif name == "award":
                            for part in g[0].split():
                                if part == "each":
                                    continue
                                mo = re.match(r"^(.+?)(\D+)$", part)
                                if not mo or mo.group(2).lower() not in V.CURRENCY:
                                    bad = "award term %r" % part
                                    break
                        if bad:
                            rows.append({"kind": "BADVALUE", "repo": repo, "side": side,
                                         "file": fn, "jpath": jp, "enricher": name,
                                         "why": bad, "swap": vis, "pack": pack_system,
                                         "snip": snip})
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(rows, open(os.path.join(here, "validate2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def key(r):
        return (r["kind"], r["repo"], r["file"], str(r["swap"]),
                r.get("why") or r.get("head"))
    print("### VISIBLE-UNDER-CRUCIBLE ONLY (swap != dnd5e), side=cn")
    c = collections.Counter(key(r) for r in rows
                            if r["side"] == "cn" and r["swap"] != "dnd5e")
    for k, v in sorted(c.items()):
        print("  %-10s %-32s swap=%-8s %-34s %d" % (k[0], k[2], str(k[3]), k[4], v))
    print()
    print("### side=en, same filter (for upstream/ours separation)")
    c2 = collections.Counter(key(r) for r in rows
                             if r["side"] == "en" and r["swap"] != "dnd5e")
    for k, v in sorted(c2.items()):
        print("  %-10s %-32s swap=%-8s %-34s %d" % (k[0], k[2], str(k[3]), k[4], v))


if __name__ == "__main__":
    main()
