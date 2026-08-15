# -*- coding: utf-8 -*-
"""
tb_blast_radius.py -- Step 4: size the two type-blind-write findings using the
project's OWN extracted EN corpus (no LevelDB, no Foundry runtime needed).

Two counts, both reproducible from files in this repo:

  (1) description-shape census.
      `crucibleDescription.extract` (babele-mappings.js:63-70) preserves the
      SOURCE shape: a plain string for HTMLField item types, {public,private}
      for the one SchemaField type (CruciblePhysicalItem). So counting the
      shape of every `description` leaf in compendium/en/*.json tells us
      exactly how many documents `migrateLegacyDescriptionShape` /
      `sanitizeItemDataShape` would coerce string -> object.

  (2) actor-embedded item census inside the two Adventure packs, i.e. how much
      translated text lives under `entries[*].actors[*].items[*]` -- the
      subtree that `degradeActorUpdatePayload` deletes from every actor update
      during an adventure re-import.

False positives / limits:
  * The EN extract only contains TRANSLATABLE leaves, so an item with no
    description at all is invisible here. Counts are therefore a LOWER bound
    on the number of affected documents.
  * A `description` leaf nested under `actions.<id>.description` is a plain
    HTMLField on the action sub-model, not the item's own field; those are
    counted separately and are NOT part of the coercion set.
"""
import json
import os
import sys
from collections import Counter

BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + ["[%d]" % i], out)
    else:
        out.append((path, node))


def main():
    shape = Counter()
    per_pack = {}
    action_desc = 0
    adventure_items = Counter()

    for repo in REPOS:
        d = os.path.join(BASE, repo, "compendium", "en")
        if not os.path.isdir(d):
            print("missing", d, file=sys.stderr)
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            data = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            entries = data.get("entries", {})
            s_str = s_obj = 0

            def scan(node, in_actions=False, in_actor_items=False):
                nonlocal s_str, s_obj, action_desc
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "description":
                            if in_actions:
                                action_desc += 1
                            elif isinstance(v, str):
                                s_str += 1
                                if in_actor_items:
                                    adventure_items["desc_str_in_actor_items"] += 1
                            elif isinstance(v, dict):
                                s_obj += 1
                                if in_actor_items:
                                    adventure_items["desc_obj_in_actor_items"] += 1
                            continue
                        scan(v, in_actions=(k == "actions") or in_actions,
                             in_actor_items=in_actor_items or (k == "items" and in_actor_items is not None and _actor_ctx))
                elif isinstance(node, list):
                    for v in node:
                        scan(v, in_actions, in_actor_items)

            # simple two-pass: whole pack, then just the actors subtree
            _actor_ctx = False
            scan(entries)
            per_pack[fn] = {"desc_string": s_str, "desc_object": s_obj}
            shape["string"] += s_str
            shape["object"] += s_obj

            # adventure actor-embedded item census
            for ekey, ev in entries.items():
                if not isinstance(ev, dict):
                    continue
                actors = ev.get("actors")
                if not isinstance(actors, dict):
                    continue
                adventure_items["packs_with_actors"] += 0
                n_actors = 0
                n_items = 0
                n_leaves = 0
                for aname, av in actors.items():
                    n_actors += 1
                    items = (av or {}).get("items")
                    if isinstance(items, dict):
                        n_items += len(items)
                        leaves = []
                        walk(items, [], leaves)
                        n_leaves += sum(1 for _p, v in leaves if isinstance(v, str) and v.strip())
                adventure_items["%s::actors" % fn] = n_actors
                adventure_items["%s::actor_items" % fn] = n_items
                adventure_items["%s::actor_item_text_leaves" % fn] = n_leaves

    print("=== (1) description shape census over compendium/en (both repos) ===")
    print("    string-shaped (HTMLField types -> WOULD BE COERCED): %d" % shape["string"])
    print("    object-shaped ({public,private}, physical items)   : %d" % shape["object"])
    print("    action-level description leaves (not in scope)     : %d" % action_desc)
    print()
    for fn, v in sorted(per_pack.items(), key=lambda kv: -kv[1]["desc_string"]):
        if v["desc_string"] or v["desc_object"]:
            print("      %-46s str=%-6d obj=%d" % (fn, v["desc_string"], v["desc_object"]))
    print()
    print("=== (2) adventure actor-embedded item census ===")
    for k, v in sorted(adventure_items.items()):
        print("    %-60s %s" % (k, v))

    here = os.path.dirname(os.path.abspath(__file__))
    json.dump({"shape": dict(shape), "per_pack": per_pack,
               "action_description_leaves": action_desc,
               "adventure": dict(adventure_items)},
              open(os.path.join(here, "tb_blast_radius.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
