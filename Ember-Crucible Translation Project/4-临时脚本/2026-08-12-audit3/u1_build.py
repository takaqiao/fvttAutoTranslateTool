# -*- coding: utf-8 -*-
"""U1: build the batch for findings [0,76) of uuid_swap.json (ember.adventure.json).

Every edit is an exact string replacement against the LIVE compendium/cn value,
so the batch value is the whole leaf with one (or a few) spots changed and every
byte of markup otherwise untouched.

EDITS: path -> list of (old, new, expected_count, tag)
  tag "label"  = the @UUID label the finding is about
  tag "prose"  = same wrong proper noun sitting bare in the SAME leaf; fixing the
                 label alone would leave 瑟洛克…瑟罗克 side by side in one paragraph
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
REPO = "1-Ember汉化插件"
PACK = "ember.adventure.json"
OUT = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches/U1__ember__ember.adventure.json"
J = "Ember Early Access.journals."

E = {}


def add(path, *reps):
    E.setdefault(path, []).extend(reps)


# [0] EN link is bare; CN invented a label AND got the name wrong (page = 采石场奔道)
add(J + "A Brush With Death.pages.Locating Kel Kornan.text",
    ("D0AAT8TNsytqy7Hm]{采石场之门}", "D0AAT8TNsytqy7Hm]{采石场奔道}", 1, "label"))
# [1][2] item name field = 符文刻印箭头
add(J + "A Brush With Death.pages.The Situation in Skybrush.text",
    ("符文标记箭头", "符文刻印箭头", 4, "label+prose"))
# [9] page name = 大破裂 The Shattering
add(J + "Ancestries.pages.Thornling.contentGamemaster",
    ("GlwUGpaqvUorPF6q]{破碎}", "GlwUGpaqvUorPF6q]{大破裂}", 1, "label"))
# [11] the crucible-side twin leaf ALREADY says 突变学派斥候 for this same EN label,
#      the org page is 突变学派, and Mutagist is 突变学派 in 147 leaves vs 变异学者 in 7.
#      Only the actor `name` field still says 变异学者斥候 (flagged for 主控).
add(J + "Ancient Paths.pages.Missing Scouts.text",
    ("dDzLM6gOY0EKzF1F]{变异学者斥候}", "dDzLM6gOY0EKzF1F]{突变学派斥候}", 1, "label"))
# [17][18][19] item name fields
add(J + "Arcturel Dives.pages.Arvoda's Elixirs.text",
    ("]{恒开花}", "]{永绽花}", 1, "label"),
    ("]{抗凝补剂}", "]{抗凝药剂}", 1, "label"),
    ("]{溶解补剂}", "]{溶解药剂}", 1, "label"))
# [20] crucible.equipment name field = 开锁工具 Lockpicks
add(J + "Arcturel Dives.pages.The Lockers.text",
    ("lockpicks0000000]{开锁器}", "lockpicks0000000]{开锁工具}", 1, "label"))
# [21] deity page name = 尼梅尔 Nimaelle;  Casia bestiary page name = 卡西娅
add(J + "Arctus Plateau Gazetteer.pages.Bloodwoods.text",
    ("2FA1FO5P0G83sjt9]{尼玛艾拉}", "2FA1FO5P0G83sjt9]{尼梅尔}", 1, "label"),
    ("卡西亚·@UUID", "卡西娅·@UUID", 1, "prose"))
# [24] page name = 碎片诸神 Shard Gods (plural label)
add(J + "Bestiary.pages.Cruel Dragons.contentGamemaster",
    ("6pJu6l3a7RTzcaRM]{碎片之神}", "6pJu6l3a7RTzcaRM]{碎片诸神}", 1, "label"))
# [25][26] item name = 阿克图里安呼吸器 (阿克图里安 already ruled 2026-08-12b)
add(J + "Chapter 1 Events.pages.Pollen Storm.text",
    ("阿克图里亚呼吸器", "阿克图里安呼吸器", 2, "label"))
# [27] anchor #reviled-magic; that section's CN heading is 遭憎魔法
add(J + "Character Classes.pages.Classes Overview.text",
    ("#reviled-magic]{受憎魔法}", "#reviled-magic]{遭憎魔法}", 1, "label"))
# [31][42][54][74] moon page name = 奥拉 Aura (decision 2026-08-12b)
add(J + "Character Classes.pages.Warlock.text",
    ("smc0OuMxsxfcSdhR]{灵气}", "smc0OuMxsxfcSdhR]{奥拉}", 1, "label"))
add(J + "Cultures.pages.Kithil.text",
    ("smc0OuMxsxfcSdhR]{灵气}", "smc0OuMxsxfcSdhR]{奥拉}", 1, "label"))
add(J + "Deities.pages.Nymbohr.text",
    ("smc0OuMxsxfcSdhR]{灵气}", "smc0OuMxsxfcSdhR]{奥拉}", 1, "label"))
add(J + "History.pages.Abyssal Shear.contentGamemaster",
    ("smc0OuMxsxfcSdhR]{灵光}", "smc0OuMxsxfcSdhR]{奥拉}", 1, "label"),
    # [75] page name = 星之盾 The Starshield
    ("Op5BD9XSlmb7l6NN]{星盾}", "Op5BD9XSlmb7l6NN]{星之盾}", 1, "label"))
# [33] EN link is bare and the word "talent" is OUTSIDE it; CN swallowed the talent
#      name and left only 天赋 -> "任何使用天赋的角色" loses which talent it is
add(J + "Corpin Sanctuary.pages.Evesso's Chamber.text",
    ("recognizespellcr]{天赋}", "recognizespellcr]{识别施法}天赋", 1, "label"))
# [34] plural label
add(J + "Cosmos.pages.Ascendancy.contentGamemaster",
    ("6pJu6l3a7RTzcaRM]{碎片之神}", "6pJu6l3a7RTzcaRM]{碎片诸神}", 1, "label"))
# [39] item name = 埃维索的便笺
add(J + "Crumbling Sanctuary.pages.Corpin Investigation.text",
    ("5sDslKMjZXB18yNm]{埃维索的便笺}", "5sDslKMjZXB18yNm]{埃维索的便笺}", 0, "noop"),
    ("5sDslKMjZXB18yNm]{埃维索的字条}", "5sDslKMjZXB18yNm]{埃维索的便笺}", 1, "label"),
    # [32] said Haxim = 哈克西姆 (item name 哈克西姆的十九个夜晚); this leaf is one of the
    # four that dropped the 克, and it is already being rewritten for [39]
    ("《哈西姆的十九夜》", "《哈克西姆的十九夜》", 1, "label(32-family)"))
# [43][55] page name = 深渊恐魔 Abyssal Horrors
add(J + "Cultures.pages.Niethun.text",
    ("nChUneUkEF2XBgxc]{深渊恐怖}", "nChUneUkEF2XBgxc]{深渊恐魔}", 1, "label"))
add(J + "Deities.pages.Outer Gods.contentGamemaster",
    ("nChUneUkEF2XBgxc]{深渊恐怖}", "nChUneUkEF2XBgxc]{深渊恐魔}", 1, "label"))
# [44][45] page names = 十九人 The Nineteen / 萨克茨 Sockets
add(J + "Cultures.pages.Ordani.text",
    ("J8wfDAZmky3f5VvV]{十九神}", "J8wfDAZmky3f5VvV]{十九人}", 1, "label"),
    ("WHj7BfBdSACYArYX]{插孔}", "WHj7BfBdSACYArYX]{萨克茨}", 1, "label"))
# [47][51]
add(J + "Cultures.pages.Waerd.text",
    ("J8wfDAZmky3f5VvV]{十九神}", "J8wfDAZmky3f5VvV]{十九人}", 1, "label"))
add(J + "Deities.pages.Aythorn.text",
    ("J8wfDAZmky3f5VvV]{十九神}", "J8wfDAZmky3f5VvV]{十九人}", 1, "label"))
# [50][58] 萨克茨 (decision 2026-08-12b: Sockets is a proper noun, 插孔 is MT)
add(J + "Deities.pages.Auris Bor.text",
    ("WHj7BfBdSACYArYX]{插孔}", "WHj7BfBdSACYArYX]{萨克茨}", 1, "label"))
add(J + "Deities.pages.The Tanir.text",
    ("WHj7BfBdSACYArYX]{插孔}", "WHj7BfBdSACYArYX]{萨克茨}", 1, "label"))
# [52] page name = 瑟洛克 Theroch; the same paragraph then says 瑟罗克 bare
add(J + "Deities.pages.Finor.contentGamemaster",
    ("瑟罗克", "瑟洛克", 3, "label+prose"))
# [53] second link only; the later 碎片之神 in prose is a different sentence
add(J + "Deities.pages.Godkiller Orinoth.text",
    ("6pJu6l3a7RTzcaRM]{碎片之神}", "6pJu6l3a7RTzcaRM]{碎片诸神}", 1, "label"))
# [56][57][59] page name = 卡西娅 Casia
for pg in ("Spectra", "Taryakel", "Vesper"):
    add(J + "Deities.pages.%s.text" % pg,
        ("U39JWvasyCgtyXQI]{卡西亚}", "U39JWvasyCgtyXQI]{卡西娅}", 1, "label"))
# [63][64] page name = 画廊 Gallery
add(J + "Disgraced House.pages.The Marlstone Gala.text",
    ("mLzBFHXratqh3i0p]{陈列库}", "mLzBFHXratqh3i0p]{画廊}", 2, "label"))
# [66][67] dnd5e side, library majority 华服 17:2 / 旅行服 9:2
add(J + "Ember's Bounty.pages.Kasta's Alterations.text",
    ("phbagClothesFine]{华美衣服}", "phbagClothesFine]{华服}", 1, "label"),
    ("phbagClothesTrav]{旅行者服装}", "phbagClothesTrav]{旅行服}", 1, "label"))
# [68] the anchored section's own CN heading is 与卡夫托尔·布伦克战斗; actor = 卡夫托尔·布伦克
add(J + "Forgotten Cistern.pages.Cistern Stockpiles.text",
    ("卡夫托及其仆从", "卡夫托尔及其仆从", 1, "prose"),
    ("#fighting-kaftor]{与卡夫托交战}", "#fighting-kaftor]{与卡夫托尔战斗}", 1, "label"))
# [70][71] journal name = 一再坠落 / actor name = 辉耀
add(J + "Gamemaster's Guide.pages.Patch 0.2.0.text",
    ("gUbcqJg4YBHbqjyi]{一坠再坠}", "gUbcqJg4YBHbqjyi]{一再坠落}", 1, "label"),
    ("OYdqWs17RvL7ntbf]{卢森特}", "OYdqWs17RvL7ntbf]{辉耀}", 1, "label"))
# [72] dnd5e babele patch renders Nondetection 回避侦测; library majority 回避侦测术 4:2
add(J + "Glitter in the Dark.pages.A Troubled Tradeway.text",
    ("aU62xVUBYkAQWIHv]{防探测}", "aU62xVUBYkAQWIHv]{回避侦测术}", 1, "label"))
# [73] actor name = 总管 Chamberlain; same leaf then says 管家 bare
add(J + "Glitter in the Dark.pages.The Story So Far.text",
    ("管家", "总管", 4, "label+prose"),
    # only leaf in the whole repo where EN `Rider` has no CN rendering; actors are
    # named 辉耀 Lucent / 骑手 Rider
    ("指引Lucent和Rider前往", "指引辉耀和骑手前往", 1, "prose"))


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def resolve(root, path):
    naive = path.split(".")
    if get_at(root, naive) is not None:
        return naive
    parts, node, rest = [], root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + ".")]
            if cands:
                k = max(cands, key=len)
                parts.append(k)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition(".")
        parts.append(head)
        node = get_at(node, [head])
    return parts


def main():
    cn = json.load(open(os.path.join(P, REPO, "compendium", "cn", PACK), encoding="utf-8"))["entries"]
    batch, bad = {}, 0
    for path, reps in E.items():
        cur = get_at(cn, resolve(cn, path))
        if not isinstance(cur, str):
            print("MISSING", path)
            bad += 1
            continue
        new = cur
        for old, rep, exp, tag in reps:
            n = new.count(old)
            if n != exp:
                print(f"COUNT {n}!={exp}  {path}  {old!r}  [{tag}]")
                bad += 1
            new = new.replace(old, rep)
        if new == cur:
            print("NOOP", path)
            bad += 1
            continue
        batch[path] = new
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"leaves={len(batch)} problems={bad} -> {OUT}")


if __name__ == "__main__":
    main()
