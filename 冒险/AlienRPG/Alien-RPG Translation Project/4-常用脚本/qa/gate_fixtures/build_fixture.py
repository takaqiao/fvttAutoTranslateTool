# -*- coding: utf-8 -*-
"""Build the gate fixtures: one GOOD tree (both gates must pass) + N MUTANT trees
(each must make exactly one gate fail).

The trees are built from `compendium/en/*` and the register itself, so they follow the
register automatically: freeze a new string in DO-NOT-TRANSLATE.json and GOOD keeps it
English on the next build.

    python build_fixture.py      # GOOD + 23 mutants
    python build_gaps.py         # 7 gap fixtures (the 2026-08-29 misses)
    python run_gates.py          # matrix of both gates x every fixture
    python run_gates.py GOOD     # one fixture, with full gate stdout

Fixtures land in %TEMP%/alien-gate-fixtures (override with $ALIEN_GATE_FIXTURES).
"""
import json, os, re, sys, shutil, copy
sys.stdout.reconfigure(encoding='utf-8')

PROJ = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project"
TMP  = os.environ.get("ALIEN_GATE_FIXTURES") or os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp", "alien-gate-fixtures")
REG  = json.load(open(os.path.join(PROJ, "7-其他内容", "DO-NOT-TRANSLATE.json"), encoding="utf-8"))
PID  = REG["measured_against"]["pack_identity"]

EN_SRC = {
    "system":     (os.path.join(PROJ, "1-系统汉化插件",   "compendium", "en", PID["system"]["babele_file"]),     "r1"),
    "starterset": (os.path.join(PROJ, "2-新手包汉化插件", "compendium", "en", PID["starterset"]["babele_file"]), "r2"),
    "corerules":  (os.path.join(PROJ, "3-核心书汉化插件", "compendium", "en", PID["corerules"]["babele_file"]),  "r3"),
}

# ---- the lang/cn.json the fixture ships -------------------------------------
LANG_CN = {
    "ALIENRPG": {
        "Yes": "是", "None": "无", "OneRound": "一轮", "OneTurn": "一回合",
        "OneShift": "一班", "OneDay": "一天", "Permanent": "永久", "Shift": "Shift",
        "SkillheavyMach": "重型机械", "SkillcloseCbt": "近战", "Skillstamina": "耐力",
        "SkillrangedCbt": "射击", "Skillmobility": "机动", "Skillpiloting": "驾驶",
        "Skillcommand": "指挥", "Skillmanipulation": "操纵", "SkillmedicalAid": "医疗",
        "Skillobservation": "观察", "Skillsurvival": "求生", "Skillcomtech": "科技",
    }
}
SKILL_ITEM_CN = {e["item_name_en"]: LANG_CN["ALIENRPG"][e["lang_key"].split(".", 1)[1]]
                 for e in REG["sections"]["exact_match_to_lang"]["entries"]}

FROZEN_NAMES = set()
for e in REG["sections"]["name_lookups"]["entries"]:
    FROZEN_NAMES.add(e["string"])
for e in REG["sections"]["rolltable_names"]["entries"]:
    FROZEN_NAMES.add(e["string"])
for e in REG["sections"]["folder_names"]["entries"]:
    FROZEN_NAMES.add(e["string"])
for e in REG["sections"]["item_names"]["entries"]:
    for d in e["documents"]:
        FROZEN_NAMES.add(d["actual_name"])
PREFIX = REG["sections"]["rolltable_names"]["prefix_filters"][0]

# ---- crit-row translation ----------------------------------------------------
LABELS = [("INJURY", "伤情"), ("FATAL", "致命"), ("TIME LIMIT", "时限"),
          ("EFFECTS", "影响"), ("HEALING TIME", "痊愈时间")]
CELL3 = {"Yes": "是", "No": "否"}
CELL5 = {"One Round": "一轮", "One Turn": "一回合", "One Shift": "一班", "One Day": "一天",
         "None": "无"}

def cn_crit(desc):
    out = desc
    for en, cn in LABELS:                      # labels are free, ': ' must survive
        out = out.replace(">%s: <" % en, ">%s: <" % cn)
        out = out.replace(">%s:<" % en, ">%s:<" % cn)   # the corerules no-space defect: keep it
    # FATAL cell: lang value + literal suffix
    out = re.sub(r"(</?b>|</?strong>)Yes(, \u20131 |, \u20132 | )", lambda m: m.group(1) + "是" + m.group(2), out)
    out = out.replace(">Yes ", ">是 ").replace(">Yes, \u20131 ", ">是, \u20131 ").replace(">Yes, \u20132 ", ">是, \u20132 ")
    out = out.replace(">No ", ">否 ")
    for en, cn in sorted(CELL5.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(">%s <" % en, ">%s <" % cn)
        out = out.replace(">%s <br" % en, ">%s <br" % cn)
    out = out.replace("Permanent", "永久")
    out = out.replace("]] days", "]] 天")
    return out

def translate(doc, pack):
    """Return a plausible, RULE-ABIDING cn translation of one Babele en file."""
    d = copy.deepcopy(doc)
    def do_folders(fold):
        for en in list(fold):
            fold[en] = en if en in FROZEN_NAMES else ("【译】" + en)
    def do_group(group, kind):
        for en, node in group.items():
            if not isinstance(node, dict):
                continue
            if kind == "item" and en in SKILL_ITEM_CN:
                node["name"] = SKILL_ITEM_CN[en]              # T-EXACT
            elif en in FROZEN_NAMES:
                node["name"] = en                            # T-FROZEN
            elif kind == "table" and en.startswith(PREFIX["prefix"]):
                node["name"] = PREFIX["prefix"] + " 【译】"    # keep the prefix
            elif "name" in node:
                node["name"] = "【译】" + en
            if kind == "table" and isinstance(node.get("results"), dict):
                for rng, row in node["results"].items():
                    if isinstance(row, dict) and isinstance(row.get("description"), str):
                        row["description"] = cn_crit(row["description"])
            for sub, subkind in (("items", "item"), ("tables", "table"), ("actors", "actor"),
                                 ("journals", "journal"), ("pages", "page"), ("scenes", "scene"),
                                 ("macros", "macro"), ("results", None)):
                if subkind and isinstance(node.get(sub), dict):
                    do_group(node[sub], subkind)
            if isinstance(node.get("folders"), dict):
                do_folders(node["folders"])
    if isinstance(d.get("folders"), dict):
        do_folders(d["folders"])
    for en, node in d.get("entries", {}).items():
        if en in FROZEN_NAMES:
            node["name"] = en
        for sub, kind in (("items", "item"), ("tables", "table"), ("actors", "actor"),
                          ("journals", "journal"), ("scenes", "scene"), ("macros", "macro")):
            if isinstance(node.get(sub), dict):
                do_group(node[sub], kind)
        if isinstance(node.get("folders"), dict):
            do_folders(node["folders"])
    return d


def add_table_refs(d, pack):
    """Write the actor rTables/cTables refs the register measured, so the gate has
    something to check (a real translator's actor node would carry them)."""
    n = 0
    refs = [r for r in REG["sections"]["actor_table_refs"]["refs"] if r["pack"] == pack]
    for ent in d.get("entries", {}).values():
        actors = ent.get("actors")
        if not isinstance(actors, dict):
            continue
        for r in refs:
            node = actors.get(r["actor"])
            if isinstance(node, dict):
                node[r["field"].split(".")[-1]] = r["value"]
                n += 1
    return n


def build(root, mutate=None):
    if os.path.isdir(root):
        shutil.rmtree(root)
    langdir = os.path.join(root, "r1", "lang")
    os.makedirs(langdir)
    lang = copy.deepcopy(LANG_CN)
    docs = {}
    for pack, (src, repo) in EN_SRC.items():
        d = translate(json.load(open(src, encoding="utf-8")), pack)
        add_table_refs(d, pack)
        docs[pack] = d
    if mutate:
        mutate(docs, lang)
    for pack, (src, repo) in EN_SRC.items():
        cn_dir = os.path.join(root, repo, "compendium", "cn")
        os.makedirs(cn_dir, exist_ok=True)
        json.dump(docs[pack], open(os.path.join(cn_dir, PID[pack]["babele_file"]), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    json.dump(lang, open(os.path.join(langdir, "cn.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    shutil.copy(os.path.join(PROJ, "1-系统汉化插件", "lang", "en.json"), os.path.join(langdir, "en.json"))
    return root

# ------------------------------------------------------------------ mutations
def find(d, group, en):
    for ent in d.get("entries", {}).values():
        g = ent.get(group)
        if isinstance(g, dict) and en in g:
            return g, g[en]
        for sub in ("actors", "items", "tables", "journals", "scenes", "macros"):
            s = ent.get(sub)
            if isinstance(s, dict):
                for node in s.values():
                    if isinstance(node, dict) and isinstance(node.get(group), dict) and en in node[group]:
                        return node[group], node[group][en]
    return None, None

MUT = {}
def mut(name):
    def deco(fn):
        MUT[name] = fn
        return fn
    return deco

@mut("M1_adventure_name")
def _(docs, lang):
    docs["system"]["entries"]["Alien RPG System"]["name"] = "异形 RPG 系统"

@mut("M2_welcome_journal")
def _(docs, lang):
    g, n = find(docs["system"], "journals", "MU/TH/ER Instructions.")
    n["name"] = "母亲指令。"

@mut("M3_scene_name")
def _(docs, lang):
    g, n = find(docs["corerules"], "scenes", "Alien Evolved Core Rules")
    n["name"] = "异形进化版核心规则"

@mut("M4_rolltable_name")
def _(docs, lang):
    g, n = find(docs["system"], "tables", "Panic Table")
    n["name"] = "恐慌表"

@mut("M5_folder_name")
def _(docs, lang):
    for ent in docs["system"]["entries"].values():
        ent["folders"]["Alien Creature Tables"] = "异形生物表"

@mut("M6_item_pack_mule")
def _(docs, lang):
    g, n = find(docs["corerules"], "items", "Pack Mule")
    n["name"] = "驮马 Pack Mule"

@mut("M7_prefix_filter")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical Injuries on Xenomorphs")
    n["name"] = "异形重伤表"

@mut("M8_skill_stunt_tail")
def _(docs, lang):
    g, n = find(docs["system"], "items", "Close Combat")
    n["name"] = "近战 Close Combat"

@mut("M9_lang_padded")
def _(docs, lang):
    lang["ALIENRPG"]["Skillcomtech"] = " 科技"

@mut("M10_sentinel_translated")
def _(docs, lang):
    ref = next(r for r in REG["sections"]["actor_table_refs"]["refs"]
               if r["pack"] == "corerules" and r["is_sentinel"])
    g, n = find(docs["corerules"], "actors", ref["actor"])
    n[ref["field"].split(".")[-1]] = "无"

@mut("M11_ref_mismatch")
def _(docs, lang):
    ref = next(r for r in REG["sections"]["actor_table_refs"]["refs"]
               if r["pack"] == "corerules" and not r["is_sentinel"] and r["field"].endswith("rTables"))
    g, n = find(docs["corerules"], "actors", ref["actor"])
    n[ref["field"].split(".")[-1]] = "某张别的表"

@mut("M12_table_key_missing")
def _(docs, lang):
    ref = next(r for r in REG["sections"]["actor_table_refs"]["refs"]
               if r["pack"] == "corerules" and not r["is_sentinel"])
    for ent in docs["corerules"]["entries"].values():
        t = ent.get("tables")
        if isinstance(t, dict) and ref["value"] in t:
            t["表名被改成了中文键"] = t.pop(ref["value"])

@mut("M13_bad_json")
def _(docs, lang):
    pass   # handled specially

# ---- crit-gate mutations
@mut("C1_lang_not_string")
def _(docs, lang):
    lang["ALIENRPG"]["OneShift"] = ["一班"]

@mut("C2_lang_padded")
def _(docs, lang):
    lang["ALIENRPG"]["Yes"] = "是 "

@mut("C3_split_shape_fullwidth_colon")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["44-44"]["description"] = n["results"]["44-44"]["description"].replace("致命: ", "致命：")

@mut("C4_cell_mismatch_fatal")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["44-44"]["description"] = n["results"]["44-44"]["description"].replace(">是 ", ">对 ")

@mut("C5_cell_mismatch_timelimit")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["45-45"]["description"] = n["results"]["45-45"]["description"].replace(">一班 ", ">一个班次 ")

@mut("C6_shift_translated")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "EV - Critical Injuries")
    n["results"]["14-14"]["description"] = n["results"]["14-14"]["description"].replace("Shift", "班次")

@mut("C7_roll_shape")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["44-44"]["description"] = n["results"]["44-44"]["description"].replace("[[1d6]] 天", "掷 [[1d6]] 天")

@mut("C8_permanent_mismatch")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["54-54"]["description"] = n["results"]["54-54"]["description"].replace("永久", "长期")

@mut("C9_endash_to_hyphen")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["52-52"]["description"] = n["results"]["52-52"]["description"].replace("\u2013", "-")

@mut("C10_lost_colon_space")
def _(docs, lang):
    g, n = find(docs["corerules"], "tables", "Critical injuries")
    n["results"]["46-46"]["description"] = n["results"]["46-46"]["description"].replace("时限: ", "时限")


if __name__ == "__main__":
    good = build(os.path.join(TMP, "GOOD"))
    print("GOOD ->", good)
    for name, fn in MUT.items():
        r = build(os.path.join(TMP, name), fn)
        if name == "M13_bad_json":
            p = os.path.join(r, "r1", "compendium", "cn", PID["system"]["babele_file"])
            open(p, "a", encoding="utf-8").write("\n}}}garbage")
        print(name, "->", r)
