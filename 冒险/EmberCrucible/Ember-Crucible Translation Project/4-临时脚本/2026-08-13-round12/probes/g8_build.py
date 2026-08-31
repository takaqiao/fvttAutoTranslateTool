# -*- coding: utf-8 -*-
"""Build the G8 batch (Deities journal) from targeted, count-asserted replacements."""
import json, os, sys, re, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CN = os.path.join(ROOT, "1-Ember汉化插件", "compendium", "cn", "ember.adventure.json")
OUT_DIR = os.path.join(ROOT, "4-临时脚本", "2026-08-13-round12", "batches")
PREFIX = "Ember Early Access.journals.Deities."

doc = json.load(open(CN, encoding="utf-8"))
J = doc["entries"]["Ember Early Access"]["journals"]["Deities"]


def get(key):
    node = J
    for part in key.split("."):
        node = node[part]
    return node


WARN = ("对深渊、深渊裂斩、深渊恐魔、外神等的提及，通常并不是玩家角色会在游戏开始前了解的事物，"
        "除非是在极为特定的背景情境下。对凡人而言，深渊在世界内更常见的称呼是“帷幕”，"
        "而深渊怪物则常被称为玛尔沃恩。")
QUOTE = ("如果你珍视自己心智的完整，就不要看这页。这里的实体与概念并不存在于织幕之内；它们存在于其外。"
         "甚至仅仅是为其命名，或将其说出口，都会提供一条系索，一根纤细而阴险的丝线，让它们能够回望你。"
         "我见过学者在研究深渊仅仅一个小时后，就失去了说出自己名字的能力。这不是知识；这是对心灵与灵魂的腐化。")

# leaf-key -> list of (old, new, expected_hits)
EDITS = collections.OrderedDict()


def add(key, *subs):
    EDITS.setdefault(key, []).extend(subs)


# ---------- E. "Influence and Followers" -> 影响与信徒 (32 vs 8) ----------
for k in ["pages.Avilla.text", "pages.Finor.contentGamemaster", "pages.Godkiller Orinoth.text",
          "pages.Laeora.text", "pages.Lorgi.text", "pages.Pathwalker.contentGamemaster",
          "pages.Slinerak.text", "pages.Soul of Fashion.text"]:
    add(k, (">影响与追随者<", ">影响与信徒<", 1))

# ---------- F. under-construction boilerplate ----------
add("pages.Nethehepticas.text", ("此页面当前仍在开发中", "此页面目前正在开发中", 1))

# ---------- H. WARNING + Laeora quote unification (canonical = Kinalathi) ----------
for k in ["pages.Kyasifer.contentGamemaster", "pages.Outer Gods.contentGamemaster",
          "pages.Sha-Xotha.contentGamemaster", "pages.Vhismara.contentGamemaster"]:
    add(k, ("__WARN__", WARN, 1), ("__QUOTE__", QUOTE, 1))

# ---------- G. "Pantheons: None" spacing ----------
for k in ["pages.Kinalathi.contentGamemaster", "pages.Sha-Xotha.contentGamemaster",
          "pages.Vhismara.contentGamemaster"]:
    add(k, ("<strong>万神殿：</strong> 无", "<strong>万神殿：</strong>无", 1))
add("pages.Lumé.text", ("<strong>万神殿： </strong>无", "<strong>万神殿：</strong>无", 1))

# ---------- I/J. Domains ----------
add("pages.Lorgi.text", ("<strong>领域： </strong>火元素", "<strong>领域： </strong>火焰", 1))
add("pages.Ven'avé.text", ("<strong>领域： </strong>其他", "<strong>领域：</strong>其他", 1))
add("pages.Vesper.text", ("<strong>领域： </strong>未知", "<strong>领域：</strong>未知", 1))

# ---------- K. Mortality & Destruction ----------
add("pages.Shard Gods.text", (">凡俗性与毁灭<", ">死亡与毁灭<", 1))

# ---------- L. spear symbol cell ----------
add("pages.The Nineteen.text", (">一支刺穿尖牙之口的长矛<", ">一支刺穿獠牙巨口的长矛<", 1))
add("pages.The Tanir.text", (">一柄刺穿獠牙巨口的长矛<", ">一支刺穿獠牙巨口的长矛<", 1))

# ---------- N. 十九神 -> 十九人 ----------
add("pages.Aythorn.contentOverview", ("十九神", "十九人", 2))
add("pages.Lanespear.contentOverview", ("十九神", "十九人", 1))
add("pages.Lanespear.text", ("十九神", "十九人", 1))
add("pages.Shard Gods.text", ("{十九神}", "{十九人}", 1))
add("pages.Sigil.text", ("十九神", "十九人", 1))
add("pages.Sokali.text", ("十九神", "十九人", 2))
add("pages.The Nineteen.contentOverview", ("十九神", "十九人", 3))

# ---------- R. Shard Gods heading in The Deities ----------
add("pages.The Deities.text", ("<h4>碎片之神</h4>", "<h4>碎片诸神</h4>", 1))

# ---------- B/C/D. Godkiller Orinoth ----------
add("pages.Godkiller Orinoth.text",
    ("弑神者奥里诺斯的谕令", "弑神者奥里诺斯的训谕", 1),
    ("类别：神祇般的存在（利维坦）", "类别：类神存在（利维坦）", 1),
    ("关键词：蛇，巨口，饥饿", "关键词：蛇、巨口、饥饿", 1),
    ("关注领域：知识，力量，饥饿", "关注领域：知识、力量、饥饿", 1),
    ("其他名称：弑神者，深渊中的饥饿", "其他名称：弑神者、深渊中的饥饿", 1),
    ("有些人甚至说她们曾是恋人", "有些人甚至说二人曾是恋人", 1))
add("pages.Godkiller Orinoth.contentOverview", ("祂是一个凶残而堕落的存在", "它是一个凶残而堕落的存在", 1))
add("pages.Godkiller Orinoth.contentGamemaster", ("就一定会杀了他。", "就一定会杀了它。", 1))

# ---------- O. pronoun consistency ----------
add("pages.Aythorn.contentGamemaster",
    ("但他们讲的每个故事", "但祂讲的每个故事", 1),
    ("而他们是如何设法骗它", "而祂是如何设法骗它", 1),
    ("对方给了他们一张地图", "对方给了祂一张地图", 1),
    ("这些都不是他们历史或起源的准确版本", "这些都不是祂历史或起源的准确版本", 1),
    ("很多年来，他们四处游历", "很多年来，祂四处游历", 1),
    ("他们离开了那片地区", "祂离开了那片地区", 1),
    ("二十六年前，他们抵达了", "二十六年前，祂抵达了", 1),
    ("而他们的最终目标是抵达", "而祂的最终目标是抵达", 1))
add("pages.Shard Gods.contentOverview",
    ("祂们每一位都拥有鲜明的个性", "他们每一位都拥有鲜明的个性", 1),
    ("祂们也和崇敬祂们的凡人一样容易犯错", "他们也和崇敬他们的凡人一样容易犯错", 1))
add("pages.Thayloc.contentOverview",
    ("祂曾一度被崇拜为", "他曾一度被崇拜为", 1),
    ("此后，祂被一股燃烧的怒火所吞没", "此后，他被一股燃烧的怒火所吞没", 1))
add("pages.Elder Gods.contentOverview",
    ("它们是维系现实根本概念的创造之原初力量", "他们是维系现实根本概念的创造之原初力量", 1),
    ("它们诞生于创造之时", "他们诞生于创造之时", 1),
    ("如今，它们维系着寰宇的根本现实", "如今，他们维系着寰宇的根本现实", 1),
    ("尽管它们的力量浩瀚无边", "尽管他们的力量浩瀚无边", 1))
add("pages.Wild Gods.contentOverview",
    ("通过祂们合一的力量", "通过它们合一的力量", 1),
    ("祂们是自然世界的守护者", "它们是自然世界的守护者", 1))
add("pages.The Deities.text",
    ("<p>荒野诸神是生命力的古老塑造者，创造了自然世界，并通过他们合而为一的力量——荒歌——维持着庞大而复杂的生态系统。"
     "他们是自然世界的守护者，帮助维持寰宇的平衡，并常常与碎片之神的野心发生冲突。荒野诸神体现自然概念（生长、季节、天气）</p>",
     "<p>荒野诸神是生命力的古老塑造者，创造了自然世界，并通过它们合一的力量——被称为荒歌——维系着庞大而复杂的生态系统。"
     "它们是自然世界的守护者，并帮助维持寰宇的平衡，因此常常与碎片诸神的野心发生冲突。荒野诸神体现着自然概念（生长、四季、天气）</p>", 1))

# ---------- P. 她们 with mixed-gender antecedent ----------
add("pages.Malae.text",
    ("据说她们在远古时代极为亲密", "据说二人在远古时代极为亲密", 1))
add("pages.Thoma.text",
    ("她们却建立起一种独特的通信方式", "二者却建立起一种独特的通信方式", 1),
    ("写下她们对生命以及当下种种问题的观察", "写下彼此对生命以及当下种种问题的观察", 1))
add("pages.Sokali.text",
    ("不过，她们关系的真实本质", "不过，二者关系的真实本质", 1))

# ---------- Q. Moria/Morian -> 莫伊拉/莫伊兰 ----------
add("pages.Thayloc.text",
    ("他也常常受到莫里亚人民的盛赞", "他也常常受到莫伊拉人民的盛赞", 1),
    ("并逃离了莫里亚诸王国", "并逃离了莫伊兰诸王国", 1),
    ("他和许多莫里亚人还是被", "他和许多莫伊兰人还是被", 1),
    ("也成为许多逃往其他王国与国家的幸存莫里亚人的信标", "也成为许多逃往其他王国与国家的幸存莫伊兰人的信标", 1))

# ---------- S. enumeration separators ----------
add("pages.Areyter.text",
    ("<strong>关注领域： </strong>水，耐心", "<strong>关注领域： </strong>水、耐心", 1),
    ("<strong>典型信徒： </strong>提菲克，水元素生物", "<strong>典型信徒： </strong>提菲克、水元素生物", 1))
add("pages.Nymbohr.text",
    ("<strong>关注领域： </strong>风，和平", "<strong>关注领域： </strong>风、和平", 1),
    ("<strong>典型信徒： </strong>泽夫，气元素生物", "<strong>典型信徒： </strong>泽夫、气元素生物", 1))
add("pages.Obrisire.text",
    ("<strong>关注领域： </strong>野兽，生存", "<strong>关注领域： </strong>野兽、生存", 1),
    ("<strong>典型信徒： </strong>怪物猎人，侦察兵", "<strong>典型信徒： </strong>怪物猎人、侦察兵", 1),
    ("<strong>领域： </strong>秩序，其他", "<strong>领域： </strong>秩序、其他", 1))
add("pages.Raineka.text",
    ("<strong>关注领域： </strong>雨，治疗", "<strong>关注领域： </strong>雨、治疗", 1),
    ("<strong>典型信徒： </strong>治疗者，外交家", "<strong>典型信徒： </strong>治疗者、外交家", 1),
    ("<strong>领域： </strong>生命，和平", "<strong>领域： </strong>生命、和平", 1))
add("pages.Spectra.text", ("学者<strong>、 </strong>施法者", "学者<strong>、</strong>施法者", 1))
add("pages.Solaru.text", ("<td><p>所有人，领袖</p>", "<td><p>所有人、领袖</p>", 1))

# ---------- U. Pantheons list separators -> 、 ----------
add("pages.Lantyr.text", ("}，@UUID[", "}、@UUID[", 5))
add("pages.Obrisire.text", ("}，@UUID[", "}、@UUID[", 1))
for k, n in [("pages.Janar.text", 1), ("pages.Malae.text", 1), ("pages.Sockets.text", 5),
             ("pages.Spectra.text", 4), ("pages.Ven'avé.text", 1)]:
    add(k, ("}, @UUID[", "}、@UUID[", n))

# ---------- V. Alar: non-word 暴光 + unsourced insertion ----------
add("pages.Alar.contentGamemaster",
    ("把它描绘成夜空中不断闪烁的金色与黑色魔法暴光", "把它描绘成夜空中不断闪现的金色与黑色魔法强光", 1),
    ("目睹真正毫无思考、纯粹存在的邪恶后", "目睹真正的、毫无思考的邪恶后", 1))

# ---------- T. Monkier -> 称号 ----------
add("pages.Sentina.text", ("<td><p>别称</p>", "<td><p>称号</p>", 1))

# ---------- A. untranslated pantheon-table names ----------
add("pages.Sentina.text",
    ("<td><p>Kyra</p>", "<td><p>凯拉</p>", 1),
    ("<td><p>Sha'lune</p>", "<td><p>沙露恩</p>", 1),
    ("<td><p>Zor-Thal</p>", "<td><p>佐尔-萨尔</p>", 1),
    ("<td><p>Aethela</p>", "<td><p>艾瑟拉</p>", 1),
    ("<td><p>Shin'loa</p>", "<td><p>辛洛亚</p>", 1),
    ("<td><p>Noxarin</p>", "<td><p>诺克萨林</p>", 1))
add("pages.Solaru.text",
    ("<td><p>Kestra'sul</p>", "<td><p>凯斯特拉苏尔</p>", 1),
    ("<td><p>Khasu</p>", "<td><p>卡苏</p>", 1),
    ("<td><p>Virim</p>", "<td><p>维里姆</p>", 1),
    ("<td><p>Thunderis</p>", "<td><p>桑德里斯</p>", 1),
    ("<td><p>Vae'oris the Gilded</p>", "<td><p>鎏金者维奥里斯</p>", 1),
    ("<td><p>Elar'vai</p>", "<td><p>埃拉尔瓦伊</p>", 1),
    ("<td><p>Valen the Fair</p>", "<td><p>美貌者瓦伦</p>", 1),
    ("<td><p>The Moon Wraith</p>", "<td><p>月之幽魂</p>", 1),
    ("<td><p>Thalric the Blind</p>", "<td><p>盲眼者塔尔里克</p>", 1),
    ("<td><p>Kalaru</p>", "<td><p>卡拉鲁</p>", 1),
    ("<td><p>Red Sulina</p>", "<td><p>赤色苏莉娜</p>", 1),
    ("<td><p>Nexaaris</p>", "<td><p>内克萨里斯</p>", 1),
    ("<td><p>Mor'festara</p>", "<td><p>莫尔费斯塔拉</p>", 1),
    ("<td><p>Solcastra the Worm</p>", "<td><p>蠕虫索尔卡斯特拉</p>", 1),
    ("<td><p>Lifestealer</p>", "<td><p>窃生者</p>", 1))


# ---------- apply ----------
def apply_all():
    out, problems = {}, []
    for key, subs in EDITS.items():
        cur = get(key)
        assert isinstance(cur, str), key
        for old, new, n in subs:
            if old == "__WARN__":
                # replace whatever WARNING paragraph is there with the canonical one
                m = re.search(r"(<h4>警告</h4>\s*<p>)(.*?)(</p>)", cur, re.S)
                if not m:
                    problems.append(f"{key}: WARNING block not found")
                    continue
                cur = cur[:m.start(2)] + new + cur[m.end(2):]
                continue
            if old == "__QUOTE__":
                m = re.search(r"(<blockquote><div><p>)(.*?)(</p>)", cur, re.S)
                if not m:
                    problems.append(f"{key}: quote block not found")
                    continue
                cur = cur[:m.start(2)] + new + cur[m.end(2):]
                continue
            c = cur.count(old)
            if c != n:
                problems.append(f"{key}: '{old[:40]}' found {c}x, expected {n}")
                continue
            cur = cur.replace(old, new)
        out[PREFIX + key] = cur
    return out, problems


batch, problems = apply_all()
for p in problems:
    print("PROBLEM:", p)
print(f"leaves={len(batch)} problems={len(problems)}")

os.makedirs(OUT_DIR, exist_ok=True)
for pack in ["ember.adventure.json", "ember.crucible-adventure.json"]:
    fn = os.path.join(OUT_DIR, f"G8.1.{pack}")
    open(fn, "w", encoding="utf-8").write(json.dumps(batch, ensure_ascii=False, indent=1))
    print("wrote", fn)
