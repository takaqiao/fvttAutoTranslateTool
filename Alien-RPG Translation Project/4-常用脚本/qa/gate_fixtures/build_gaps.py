# -*- coding: utf-8 -*-
"""GAP fixtures: real, game-breaking corruptions that the two gates DO NOT catch today."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import build_fixture as B

GAPS = {}
def gap(name):
    def deco(fn):
        GAPS[name] = fn
        return fn
    return deco

@gap("G1_danger_event_table_translated")
def _(docs, lang):
    """corerules Macro 'Roll on Danger Event Detail Table' does
       game.tables.getName("EV - 48a. LS - DANGER EVENT DETAIL") UNGUARDED."""
    g, n = B.find(docs["corerules"], "tables", "EV - 48a. LS - DANGER EVENT DETAIL")
    n["name"] = "EV - 48a. 生命支持 - 危险事件细节"

@gap("G2_hardened_stoic_talents")
def _(docs, lang):
    """actor-character.mjs:449/452 compare name.toUpperCase() to HARDENED / STOIC."""
    for en in ("Hardened", "Stoic"):
        g, n = B.find(docs["corerules"], "items", en)
        n["name"] = {"Hardened": "硬汉 Hardened", "Stoic": "坚忍 Stoic"}[en]

@gap("G3_spacecraft_crit_split")
def _(docs, lang):
    """actor.mjs:2134/2146 read testArray[5] out of the spaceship crit row."""
    g, n = B.find(docs["corerules"], "tables", "Spaceship Major Component Damage")
    d = n["results"]["2-2"]["description"]
    n["results"]["2-2"]["description"] = d.replace("<br /><br />", "<br />")

@gap("G4_synthetic_crit_split")
def _(docs, lang):
    """actor.mjs:2101/2111 read testArray[1] out of the synthetic crit row."""
    g, n = B.find(docs["corerules"], "tables", "Critical Injuries on Synthetics")
    d = n["results"]["1-1"]["description"]
    n["results"]["1-1"]["description"] = d.replace(": </b>", "：</b>")   # full-width colon

@gap("G5_rpg_launcher_name")
def _(docs, lang):
    """character-sheet.mjs:428 i.name.includes(' RPG ') sets ammo weight 0.5 vs 0.25."""
    g, n = B.find(docs["corerules"], "items", "M5A3 RPG Launcher")
    n["name"] = "M5A3 火箭发射器"

@gap("G6_macro_command_translated")
def _(docs, lang):
    """Babele default-mappings.js:163 maps Macro.command; the system macro compares
       t.folder.name === 'Alien Mother Tables'."""
    g, n = B.find(docs["system"], "macros", "Alien - Roll on selected Mother table V10")
    n["command"] = "(async () => { game.tables.contents.forEach((t) => { if (t.folder && t.folder.name === '异形母体表') {} }); })();"

@gap("G7_xenomorph_crit_split")
def _(docs, lang):
    """creature branch: Critical Injuries on Xenomorphs feeds testArray[0]/[1]."""
    g, n = B.find(docs["corerules"], "tables", "Critical Injuries on Xenomorphs")
    d = n["results"]["2-2"]["description"]
    n["results"]["2-2"]["description"] = d.replace(":</b>", "：</b>")

if __name__ == "__main__":
    for name, fn in GAPS.items():
        r = B.build(os.path.join(B.TMP, name), fn)
        print(name, "->", r)
