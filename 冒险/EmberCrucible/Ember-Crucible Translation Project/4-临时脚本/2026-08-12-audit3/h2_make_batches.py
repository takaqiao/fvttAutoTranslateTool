"""H2: build the crucible batches for the FR-crosscheck findings.

Every value is derived from the CURRENT compendium/cn value by an exact string
replacement, so nothing outside the flagged term can drift.
"""
import json
import os

CN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\cn"
OUT = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"

# pack -> [(path, [(old, new), ...])]
EDITS = {
    "crucible.playtest.json": [
        # B-1  The Ring of Valor is the gladiatorial ARENA, not a finger ring.
        ("Playtest 1 - The Ring of Valor.name",
         [("试玩测试 Playtest 1 - The Ring of Valor",
           "试玩测试 1 - 英勇角斗场 Playtest 1 - The Ring of Valor")]),
        ("Playtest 1 - The Ring of Valor.description",
         [("英勇之戒", "英勇角斗场"), ("一支英雄队伍", "一支角斗士队伍")]),
        ("Playtest 1 - The Ring of Valor.caption", [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.folders.The Ring of Valor",
         [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.name",
         [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Introduction.text",
         [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day One - The Feral Scramble.text",
         [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Two - Hammer and Anvil.text",
         [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Three - The Skeletal Army.text",
         [("英勇之戒", "英勇角斗场"), ("{第4天 - Koy Rent}", "{第4天 - 科伊·伦特}")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Five - Void Harbingers.text",
         [("英勇之戒", "英勇角斗场")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Six - Mirror Match.text",
         [("英勇之戒", "英勇角斗场")]),
        # B-3  page titles whose Chinese half stops at the day number
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Four - Koy Rent.name",
         [("第4天 Day Four", "第4天——科伊·伦特 Day Four")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Five - Void Harbingers.name",
         [("第5天 Day Five", "第5天——虚空先驱者 Day Five")]),
        ("Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor.pages.Day Six - Mirror Match.name",
         [("第6天 Day Six", "第6天——镜像对决 Day Six")]),
        # B-8 / B-9  transliterated common nouns
        ("Playtest 1 - The Ring of Valor.actors.Duurath.items.Testudo.name",
         [("泰斯图多", "龟甲阵")]),
        ("Playtest 1 - The Ring of Valor.actors.Zarajah.items.Iramancer.name",
         [("艾拉曼瑟", "怒术师")]),
    ],
    "crucible.rules.json": [
        ("Welcome To Crucible.pages.What is Crucible.text",
         [("试玩测试 1 - 英勇之戒", "试玩测试 1 - 英勇角斗场")]),
    ],
    "crucible.equipment.json": [
        ("Spellband.name", [("法术带", "法术指环")]),
        ("Mealkit.name", [("餐包", "餐具包")]),
    ],
    "crucible.adversary-talents.json": [
        ("Lightning Burst.name", [("电爆", "闪电爆发")]),
        ("Lightning Burst.actions.lightningBurst.name", [("电爆", "闪电爆发")]),
    ],
    "crucible.summons.json": [
        ("Frost Sprite.tokenName", [("霜冻造物 Creation of Frost", "霜冻的创造")]),
    ],
    "crucible.talent.json": [
        ("Testudo.name", [("泰斯图多", "龟甲阵")]),
        ("Iramancer.name", [("艾拉曼瑟", "怒术师")]),
    ],
    "crucible.pregens.json": [
        ("Duurath.items.Testudo.name", [("泰斯图多", "龟甲阵")]),
        ("Zarajah.items.Iramancer.name", [("艾拉曼瑟", "怒术师")]),
    ],
}


def get_at(node, path):
    parts = path.split(".")
    # longest-key-first walk, same tolerance as apply_translations.split_path
    rest = path
    node2 = node
    out = []
    while rest:
        if isinstance(node2, dict):
            cands = [k for k in node2 if rest == k or rest.startswith(k + ".")]
            if cands:
                k = max(cands, key=len)
                out.append(k)
                node2 = node2[k]
                rest = rest[len(k) + 1:]
                continue
        return None
    return node2


os.makedirs(OUT, exist_ok=True)
for pack, edits in EDITS.items():
    data = json.load(open(os.path.join(CN, pack), encoding="utf-8-sig"))["entries"]
    batch = {}
    for path, subs in edits:
        cur = get_at(data, path)
        assert isinstance(cur, str), f"MISSING {pack} :: {path} -> {cur!r}"
        new = cur
        for old, rep in subs:
            assert old in new, f"NOT FOUND {old!r} in {pack} :: {path}"
            new = new.replace(old, rep)
        assert new != cur
        batch[path] = new
    fn = os.path.join(OUT, f"H2__crucible__{pack[:-5]}.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=1)
    print(f"wrote {fn}  ({len(batch)} leaves)")
