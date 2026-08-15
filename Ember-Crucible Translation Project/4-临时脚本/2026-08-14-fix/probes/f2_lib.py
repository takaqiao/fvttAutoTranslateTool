# -*- coding: utf-8 -*-
"""F2 shard helper: load packs, walk leaves, emit batch-compatible paths."""
from __future__ import annotations
import json, os, sys, io, re

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER = os.path.join(ROOT, "1-Ember汉化插件")
CRUC = os.path.join(ROOT, "2-Crucible汉化插件")

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load(repo, side, pack):
    p = os.path.join(repo, 'compendium', side, pack)
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(node, prefix=""):
    """Yield (dotted_path, value) for every string leaf."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(node, str):
        yield prefix, node


def leaves(doc):
    """Yield batch-compatible paths: entries.* plain, folders.* prefixed (folders)."""
    for k, v in (doc.get('entries') or {}).items():
        for p, s in walk(v, k):
            yield f"{p}", s
    for k, v in (doc.get('folders') or {}).items():
        for p, s in walk(v, k):
            yield f"(folders).{p}", s


def pairs(repo, pack):
    """Yield (path, en, cn) for leaves present in both."""
    en = load(repo, 'en', pack)
    cn = load(repo, 'cn', pack)
    cnmap = dict(leaves(cn))
    for p, e in leaves(en):
        if p in cnmap:
            yield p, e, cnmap[p]


def cnmap(repo, pack):
    return dict(leaves(load(repo, 'cn', pack)))


def enmap(repo, pack):
    return dict(leaves(load(repo, 'en', pack)))


EM_PACKS = ["ember.adventure.json", "ember.character.json", "ember.crucible-adventure.json",
            "ember.crucible-adversary.json", "ember.crucible-affixes.json",
            "ember.crucible-character.json", "ember.crucible-effects.json",
            "ember.crucible-items.json", "ember.dnd5e-effects.json"]
CR_PACKS = ["crucible._packs-folders.json", "crucible.adversary-equipment.json",
            "crucible.adversary-talents.json", "crucible.affixes.json", "crucible.ancestry.json",
            "crucible.archetype.json", "crucible.background.json", "crucible.equipment.json",
            "crucible.macros.json", "crucible.playtest.json", "crucible.pregens.json",
            "crucible.rules.json", "crucible.spell.json", "crucible.summons.json",
            "crucible.talent.json", "crucible.taxonomy.json"]
ALL = [(EMBER, p) for p in EM_PACKS] + [(CRUC, p) for p in CR_PACKS]
