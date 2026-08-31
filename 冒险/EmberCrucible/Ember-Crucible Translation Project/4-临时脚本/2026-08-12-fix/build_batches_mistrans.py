# -*- coding: utf-8 -*-
"""Build translation batches for AUDIT-2026-08-12 section 3 (实质错译 + 同名异译).

Produces flat {batch_path: cn} batch files under batches/. Never writes compendium/cn.
Every rule is English-gated (gate_en / gate_en_not) so a bare Chinese word frequency
can never drive a replacement.

Usage:  python build_batches_mistrans.py [--report reports/mistrans_changes.txt]
"""
import argparse, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
EMBER = ROOT + "/1-Ember汉化插件"
CRUC = ROOT + "/2-Crucible汉化插件"
OUT = ROOT + "/4-临时脚本/2026-08-12-fix/batches"
SKIP_KEYS = {"_id", "path", "_variants", "_when"}

EMB_ADV = ["ember.adventure.json", "ember.crucible-adventure.json"]

# --------------------------------------------------------------------------
# rule helpers
def R(subs, gate_en=None, gate_en_not=None, path=None, path_re=None,
      path_not=None, regex=False, note=""):
    return dict(subs=subs, gate_en=gate_en, gate_en_not=gate_en_not, path=path,
                path_re=path_re, path_not=path_not, regex=regex, note=note)


RULES = {}          # (repo, pack) -> [rule, ...]


def add(repo, packs, *rules):
    if isinstance(packs, str):
        packs = [packs]
    for pk in packs:
        RULES.setdefault((repo, pk), []).extend(rules)


# ============ 1. Warden (宗教/文化义) 典狱长 -> 典守者 ======================
WARDEN_PATHS = (r"(journals\.Character Classes\.pages\.Cleric\.text"
                r"|journals\.Cultures\.pages\.Lumek\.text"
                r"|journals\.Deities\.pages\.Sunalin\.text"
                r"|journals\.Deities\.pages\.The Tanir\.text"
                r"|journals\.Organizations\.pages\.Cindaric Sages\.contentOverview"
                r"|journals\.Smoldering Cinders\.pages\.A Conflagration of Lumé\."
                r"|actors\.Flameguard (Crusader|Firebrand)\.biography\.private)")
add(EMBER, EMB_ADV,
    R([("炎之典狱长", "火焰典守者", None), ("火焰典狱长", "火焰典守者", None),
       ("风暴的典狱长", "风暴的典守者", None), ("典狱长、牧师、术士", "典守者、牧师、术士", None),
       ("肩负天命的典狱长", "肩负天命的典守者", None)],
      gate_en=r"[Ww]arden", path_re=WARDEN_PATHS, note="1 Warden 宗教/文化义"))

# ============ 2. Sockets 插孔 -> 萨克茨 ====================================
add(EMBER, EMB_ADV,
    R([("插孔", "萨克茨", None)], gate_en=r"Socket", note="2 Sockets 音译"),
    # the one leaf whose EN label is not 'Sockets' but whose UUID target is the god
    R([("插孔", "萨克茨", None)], path_re=r"The Bleak Archive\.pages\.Antechamber\.text",
      gate_en=r"WHj7BfBdSACYArYX", note="2 Sockets 音译 (UUID 目标为该神)"))

# ============ 3. Shard God 通称 -> 碎片之神 ================================
# 3a blanket: leaves whose English says Shard God(s) and never says Shard Goddess
add(EMBER, ["ember.adventure.json", "ember.crucible-adventure.json",
            "ember.character.json", "ember.crucible-character.json"],
    R([("碎片女神", "碎片之神", None)],
      gate_en=r"[Ss]hard[- ][Gg]ods?\b", gate_en_not=r"[Ss]hard [Gg]oddess",
      note="3 Shard God 通称(单一语义叶)"))

# 3b mixed leaves: anchored, one entry per occurrence
MIX = {
 r"journals\.Myths & Legends\.pages\.The High King\.text": [
     ("碎片女神@UUID[JournalEntry.emberDeities0000.JournalEntryPage.GIX4ViCk2Vr9NWDi]",
      "碎片之神@UUID[JournalEntry.emberDeities0000.JournalEntryPage.GIX4ViCk2Vr9NWDi]", 1),
     ("年轻的碎片女神马尔特卡萨", "年轻的碎片之神马尔特卡萨", 1)],
 r"journals\.Deities\.pages\.Lanespear\.contentGamemaster": [
     ("兰矛如何成为碎片女神", "兰矛如何成为碎片之神", 1)],
 r"journals\.Deities\.pages\.Janar\.text": [
     ("古老碎片女神", "古老碎片之神", 1), ("死去的碎片女神", "死去的碎片之神", 1)],
 r"journals\.History\.pages\.Age of Rediscovery\.contentGamemaster": [
     ("碎片女神 Mendal", "碎片之神 Mendal", 1),
     ("飞升的碎片女神", "飞升的碎片之神", 1),
     ("的碎片女神，包括", "的碎片之神，包括", 1),
     ("碎片女神 @UUID[JournalEntry.emberDeities0000.JournalEntryPage.cVcA9liudzvcYgZc]",
      "碎片之神 @UUID[JournalEntry.emberDeities0000.JournalEntryPage.cVcA9liudzvcYgZc]", 1)],
 r"journals\.History\.pages\.Night of Swords\.text": [
     ("碎片女神赫斯托尔", "碎片之神赫斯托尔", 1),
     ("濒死碎片女神", "濒死碎片之神", 1)],
 r"journals\.Smoldering Cinders\.pages\.Numinous Rendezvous\.text": [
     ("死灵术的碎片女神", "死灵术的碎片之神", 1),
     ("JournalEntryPage.6pJu6l3a7RTzcaRM]{碎片女神}交谈",
      "JournalEntryPage.6pJu6l3a7RTzcaRM]{碎片之神}交谈", 1),
     ("所谓碎片女神", "所谓碎片之神", 1),
     ("最古老的碎片女神", "最古老的碎片之神", 1)],
 r"journals\.Smoldering Cinders\.pages\.Spirited Revelations\.text": [
     ("<h4>碎片女神的分析</h4>", "<h4>碎片之神的分析</h4>", 1)],
 r"journals\.Local Color\.pages\.Outside the Lines\.text": [
     ("JournalEntryPage.6pJu6l3a7RTzcaRM]{碎片女神}",
      "JournalEntryPage.6pJu6l3a7RTzcaRM]{碎片之神}", 1)],
 r"journals\.Thorny Predicaments\.pages\.Planting a Seed\.overview": [
     ("碎片女神艾索恩", "碎片之神艾索恩", 1)],
 r"items\.Scoris' Fury\.description\.private": [
     ("碎片女神同伴贝尔", "碎片之神同伴贝尔", 1)],
}
for pr, subs in MIX.items():
    add(EMBER, EMB_ADV, R(subs, path_re=pr, note="3 Shard God 混合叶逐处"))

# ============ 4. Ancestry Elvish/Orcish 去掉「语」 ==========================
add(CRUC, ["crucible.ancestry.json", "crucible.playtest.json", "crucible.pregens.json"],
    R([("精灵语 Elvish", "精灵 Elvish", 1)], gate_en=r"^Elvish$", note="4 Elvish 祖裔"),
    R([("兽人语血统 Orcish", "兽人 Orcish", 1)], gate_en=r"^Orcish$", note="4 Orcish 祖裔"),
    R([("兽人语 Orcish", "兽人 Orcish", 1)], gate_en=r"^Orcish$", note="4 Orcish 祖裔"))

# ============ 5. Steading（季节名）庄园 -> 安居 ============================
add(EMBER, EMB_ADV,
    R([("庄园季", "安居季", None), ("{庄园}", "{安居}", None),
       ("庄园被称为工业季节", "安居被称为工业季节"),
       ("庄园那种脚踏实地", "安居那种脚踏实地"),
       ("庄园时期", "安居时期"),
       ("庄园 Steading", "安居 Steading", None),
       ("绽放、庄园、拾取", "绽放、安居、拾取", None)],
      gate_en=r"\bSteading", note="5 Steading 季节名"))

# ============ 6. Counter Riposte 说明与 talent 侧对齐 ======================
add(EMBER, "ember.crucible-adventure.json",
    R([("<p>当你用<strong>招架</strong>防御一次近战攻击时，你可以作出反应并消耗<strong>专注</strong>，"
        "以进行一次消耗降低的<strong>打击</strong>进行反击，并从<strong>暴露</strong>状态中获益。</p>",
        "<p>当你以<strong>招架</strong>对抗一次近战攻击进行防御时，你可以作出反应并消耗<strong>专注</strong>，"
        "以较低代价进行一次<strong>精准</strong>的<strong>打击</strong>反击。</p>", 1)],
      gate_en=r"which is <strong>Accurate</strong>",
      path_re=r"Counter Riposte\.actions\.counterRiposte\.description",
      note="6 Accurate 被写成 Exposed"))

# ============ 7. Aura（月亮）灵气 -> 奥拉 ==================================
AURA_MOON = [
    (r"smc0OuMxsxfcSdhR\]\{灵气", "smc0OuMxsxfcSdhR]{奥拉"),
    ("灵气同调", "奥拉同调"),
    ("灵气之石", "奥拉之石"),
    ("灵气宝石", "奥拉宝石"),
    ("灵气祭坛", "奥拉祭坛"),
    ("灵气月华共鸣", "奥拉月华共鸣"),
    ("灵气陨石", "奥拉陨石"),
    ("灵气之祝福", "奥拉之祝福"),
    ("灵气之力", "奥拉之力"),
    ("“灵气”", "“奥拉”"),
    ("灵气——", "奥拉——"),
    ("灵气之月", "奥拉之月"),
    ("灵气（风之月）", "奥拉（风之月）"),
    ("灵气（气元素）", "奥拉（气元素）"),
    ("（灵气、阿肯", "（奥拉、阿肯"),
    ("-&gt; 灵气 -&gt;", "-&gt; 奥拉 -&gt;"),
    (r"灵气 (\d)：", r"奥拉 \1："),
    ("灵气：28天", "奥拉：28天"),
]
add(EMBER, EMB_ADV + ["ember.character.json", "ember.crucible-character.json",
                      "ember.crucible-effects.json", "ember.dnd5e-effects.json",
                      "ember.crucible-affixes.json"],
    R(AURA_MOON, gate_en=r"Aura", regex=True, note="7 Aura 月名音译"))
# whole-leaf moon pages
add(EMBER, EMB_ADV,
    R([("灵气", "奥拉", None)], gate_en=r"Aura",
      path_re=r"(journals\.Cosmos\.pages\.Aura\.|journals\.Ancestries\.pages\.Zeph\.)",
      note="7 Aura 月名音译（整叶为该月条目）"),
    R([("<h2 class=\"divider\">灵气</h2>", "<h2 class=\"divider\">奥拉</h2>", None),
       ("对灵气的真正同调", "对奥拉的真正同调", None)],
      gate_en=r"True attunement to Aura",
      path_re=r"journals\.Cosmos\.pages\.Attunement\.text", note="7 Aura 月名音译"),
    R([("灵气狂野之风", "奥拉狂野之风", None), ("留在灵气之上", "留在奥拉之上", None)],
      gate_en=r"untamed winds of Aura",
      path_re=r"journals\.Deities\.pages\.Vesper\.text", note="7 Aura 月名音译"),
    R([("<li><p>灵气，补充了更多信息。", "<li><p>奥拉，补充了更多信息。", None)],
      gate_en=r"Aura, added additional information",
      path_re=r"Patch 0\.5\.0\.text", note="7 Aura 月名音译"))

# ============ 8. The Hallows 区/组织分名 ==================================
add(EMBER, EMB_ADV,
    R([("幽圣所", "圣堂区", None)], gate_en=r"Hallows",
      path_re=r"journals\.Ordain Gazetteer\.pages\.The Hallows\.",
      note="8 城区页正文统一为 name 字段的 圣堂区"),
    R([("JournalEntryPage.ThsrAwMhXhmowirO]{幽圣所}",
        "JournalEntryPage.ThsrAwMhXhmowirO]{圣堂区}", None)],
      gate_en=r"ThsrAwMhXhmowirO", note="8 链接标签按目标页（城区）"),
    R([("JournalEntryPage.TdFy82NoVNArBC9f]{圣堂区}",
        "JournalEntryPage.TdFy82NoVNArBC9f]{幽圣所}", None)],
      gate_en=r"TdFy82NoVNArBC9f", note="8 链接标签按目标页（组织）"))

# ============ 9. Ordinate 法序议会 -> 审序院 ==============================
add(EMBER, EMB_ADV,
    R([("法序议会", "审序院", None)], gate_en=r"\bOrdinate", note="9 Ordinate 统一"))

# ============ 10. River Destine 统一为 德斯廷 =============================
add(EMBER, EMB_ADV,
    R([("德斯汀", "德斯廷", None), ("天命河", "德斯廷河", None), ("命运河", "德斯廷河", None)],
      gate_en=r"\bDestine\b", note="10 River Destine 统一"))

# ============ 11. 品质五档统一 ============================================
QUAL_PACKS_E = EMB_ADV
QUAL_PACKS_C = ["crucible.equipment.json", "crucible.playtest.json",
                "crucible.pregens.json", "crucible.talent.json",
                "crucible.rules.json", "crucible.spell.json",
                "crucible.adversary-equipment.json"]
QSUB_SHODDY = R([("<strong>粗制滥造</strong>", "<strong>粗糙</strong>", None),
                 ("<strong>劣质</strong>", "<strong>粗糙</strong>", None),
                 ("<strong>粗劣</strong>", "<strong>粗糙</strong>", None),
                 ("<strong>粗制</strong>", "<strong>粗糙</strong>", None),
                 ("劣质品质", "粗糙品质", None)],
                gate_en=r"\bShoddy\b", note="11 Shoddy->粗糙")
QSUB_FINE = R([("<strong>精制</strong>", "<strong>精良</strong>", None),
               ("<strong>精良 </strong>", "<strong>精良</strong>", None),
               ("品质不高于精制", "品质不高于精良", None)],
              gate_en=r"\bFine\b", note="11 Fine->精良")
QSUB_SUP = R([("<strong>优异</strong>", "<strong>卓越</strong>", None),
              ("<strong>优良</strong>", "<strong>卓越</strong>", None),
              ("<strong>优质</strong>", "<strong>卓越</strong>", None)],
             gate_en=r"\bSuperior\b", note="11 Superior->卓越")
QSUB_MW = R([("<strong>大师工艺</strong>", "<strong>大师级</strong>", None),
             ("<strong>杰作</strong>", "<strong>大师级</strong>", None),
             ("<strong>大师之作</strong>", "<strong>大师级</strong>", None),
             ("大师工艺品质", "大师级品质", None), ("杰作品质", "大师级品质", None),
             ("精制珠宝戒指 Masterwork", "大师级珠宝戒指 Masterwork", None)],
            gate_en=r"\bMasterwork\b", note="11 Masterwork->大师级")
add(EMBER, QUAL_PACKS_E, QSUB_SHODDY, QSUB_FINE, QSUB_SUP, QSUB_MW)
add(CRUC, QUAL_PACKS_C, QSUB_SHODDY, QSUB_FINE, QSUB_SUP, QSUB_MW)

# ============ 12. 预生角色姓名栏与自述对齐（以 name 字段为准） =============
add(CRUC, "crucible.playtest.json",
    R([("贝拉多娜", "颠茄", None)], gate_en=r"Belladonna",
      path_re=r"actors\.Belladonna\.biography", note="12 Belladonna 自述对齐 name"),
    R([("杜拉特", "杜拉斯", None)], gate_en=r"Duurath",
      path_re=r"actors\.Duurath\.biography", note="12 Duurath 自述对齐 name"),
    R([("Fizzit", "菲兹特", None)], gate_en=r"Fizzit",
      path_re=r"actors\.Fizzit\.biography", note="12 Fizzit 自述补译名"))
add(CRUC, "crucible.pregens.json",
    R([("Fizzit ", "菲兹特", None)], gate_en=r"Fizzit",
      path_re=r"^Fizzit\.biography", note="12 Fizzit 自述补译名"),
    R([("Duurath ", "杜拉斯", None)], gate_en=r"Duurath",
      path_re=r"^Duurath\.biography", note="12 Duurath 自述补译名"))

# ============ 13. tokenName 机械替换产物重写 ==============================
add(CRUC, "crucible.playtest.json",
    R([("惠安娜 惠安娜", "惠安娜 Huiana", 1)], gate_en=r"^Huiana$", note="13 tokenName"),
    R([("铁砧 铁砧", "铁砧 Anvil", 1)], gate_en=r"^Anvil$", note="13 tokenName"),
    R([("奥里维奇 奥里维奇", "奥里维奇 Orivech", 1)], gate_en=r"^Orivech$", note="13 tokenName"),
    R([("疫病先驱 疫病先驱", "疫病先驱 Harbinger of Disease", 1)],
      gate_en=r"^Harbinger of Disease$", note="13 tokenName"),
    R([("疯狂先驱者 先驱者 of 疯狂", "疯狂先驱 Harbinger of Madness", 1)],
      gate_en=r"^Harbinger of Madness$", note="13 tokenName 逐词替换产物"),
    R([("疯狂先驱者 Harbinger of Madness", "疯狂先驱 Harbinger of Madness", 1)],
      gate_en=r"^Harbinger of Madness$", path_re=r"\.name$",
      note="13 与 疫病先驱/恐惧先驱 同构"),
    R([("队伍 队伍", "队伍 Party", 1)], gate_en=r"^Party$", note="13 tokenName"))
add(CRUC, "crucible.summons.json",
    R([("霜冻的创造 创造 of 霜冻", "霜冻造物 Creation of Frost", 1)],
      gate_en=r"^Creation of Frost$", note="13 tokenName 逐词替换产物"))

# ============ 14. Arcturel Upper/Lower -> 现行英文名 ======================
add(EMBER, EMB_ADV,
    R([("阿克图瑞尔上层 Arcturel Upper", "阿克图瑞尔贸易道 Arcturel Tradeway", 1)],
      gate_en=r"^Arcturel Tradeway$", note="14 上游改名 Tradeway"),
    R([("阿克图瑞尔下层 Arcturel Lower", "阿克图瑞尔矿渊 Arcturel Dives", 1)],
      gate_en=r"^Arcturel Dives$", note="14 上游改名 Dives"),
    R([("上阿克图瑞尔的聚归馆", "阿克图瑞尔的聚归馆", 1)],
      gate_en=r"Rallyhome in Arcturel", note="14 英文只作 Arcturel"),
    R([("俯临着上阿克图瑞尔", "俯临着阿克图瑞尔", 1)],
      gate_en=r"Looking out the balcony", note="14 英文只作 Arcturel"),
    R([("队伍前往下阿克图瑞尔", "队伍前往矿渊", 1)],
      gate_en=r"journeys to the Dives", note="14 英文作 the Dives"))

# ============ 15. 最终战对手英雄名未译 ====================================
add(CRUC, "crucible.playtest.json",
    R([("{Duurath}", "{杜拉斯}", None), ("<h4>Duurath</h4>", "<h4>杜拉斯</h4>", None),
       ("<strong>Duurath </strong>", "<strong>杜拉斯</strong>", None),
       ("Duurath会", "杜拉斯会", None), ("Duurath可以", "杜拉斯可以", None),
       ("{Belladonna}", "{颠茄}", None), ("<h4>Belladonna</h4>", "<h4>颠茄</h4>", None),
       ("<strong>Belladonna</strong>", "<strong>颠茄</strong>", None),
       ("Belladonna会", "颠茄会", None),
       ("{Kagura}", "{神乐}", None), ("<h4>Kagura</h4>", "<h4>神乐</h4>", None),
       ("<strong>Kagura</strong>", "<strong>神乐</strong>", None),
       ("Kagura还可以", "神乐还可以", None), ("Kagura会", "神乐会", None)],
      gate_en=r"Mirror Match|rival adventuring party",
      path_re=r"Day Six - Mirror Match\.text", note="15 对手英雄名未译"))

# ============ 16. Rallying Threshold -> 集结阈值 ==========================
add(CRUC, ["crucible.playtest.json", "crucible.pregens.json", "crucible.talent.json"],
    R([("<strong>激励</strong>阈值", "<strong>集结</strong>阈值", None),
       ("<strong>振奋</strong>阈值", "<strong>集结</strong>阈值", None),
       ("振奋阈值", "集结阈值", None), ("激励阈值", "集结阈值", None)],
      gate_en=r"<strong>Rallying</strong> threshold|Rallying Threshold",
      note="16 Rallying Threshold 对齐角色卡"))
add(EMBER, "ember.crucible-adventure.json",
    R([("<strong>Rallying</strong>阈值", "<strong>集结</strong>阈值", None)],
      gate_en=r"<strong>Rallying</strong> threshold",
      note="16 Rallying Threshold 对齐角色卡"))


# --------------------------------------------------------------------------
def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = ".".join(path)
        out.append((p[len("entries."):] if p.startswith("entries.") else p, en,
                    cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=ROOT + "/4-临时脚本/2026-08-12-fix/reports/mistrans_changes.txt")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    rep = open(a.report, "w", encoding="utf-8")
    summary = []

    for (repo, pack), rules in sorted(RULES.items()):
        enp = os.path.join(repo, "compendium", "en", pack)
        cnp = os.path.join(repo, "compendium", "cn", pack)
        if not os.path.isfile(enp) or not os.path.isfile(cnp):
            print(f"!! missing {repo}/{pack}")
            continue
        en = json.load(open(enp, encoding="utf-8"))
        cn = json.load(open(cnp, encoding="utf-8"))
        leaves = []
        walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], leaves)
        batch, hits = {}, {}
        for bp, e, c in leaves:
            if not c:
                continue
            new = c
            for r in rules:
                if r["path"] and bp != r["path"]:
                    continue
                if r["path_re"] and not re.search(r["path_re"], bp):
                    continue
                if r["path_not"] and re.search(r["path_not"], bp):
                    continue
                if r["gate_en"] and not re.search(r["gate_en"], e):
                    continue
                if r["gate_en_not"] and re.search(r["gate_en_not"], e):
                    continue
                for sub in r["subs"]:
                    old, rep_, exp = (sub + (None,))[:3] if len(sub) == 2 else sub
                    if r["regex"]:
                        n = len(re.findall(old, new))
                        if n:
                            new = re.sub(old, rep_, new)
                    else:
                        n = new.count(old)
                        if n:
                            new = new.replace(old, rep_)
                    if n:
                        hits[(r["note"], old)] = hits.get((r["note"], old), 0) + n
                    if exp is not None and n != exp:
                        rep.write(f"?? COUNT {repo}|{pack}|{bp}|{old!r} expected {exp} got {n}\n")
            if new != c:
                batch[bp] = new
        if not batch:
            continue
        fn = os.path.join(OUT, f"sem-mistrans__{os.path.basename(repo)}__{pack}")
        json.dump(batch, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        summary.append((repo, pack, len(batch), fn))
        rep.write(f"\n===== {repo} / {pack}  entries={len(batch)}\n")
        for k, v in sorted(hits.items()):
            rep.write(f"   [{k[0]}] {k[1]!r} x{v}\n")
    rep.close()
    for repo, pack, n, fn in summary:
        print(f"{os.path.basename(repo):<22} {pack:<34} {n:>4} -> {fn}")
    print(f"\nreport: {a.report}")


if __name__ == "__main__":
    main()
