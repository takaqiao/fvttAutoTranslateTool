# -*- coding: utf-8 -*-
"""Generate the crucible mechanical-blocker batches (2026-08-12 audit, 第 1 节).

Every edit is an *asserted exact* string replacement against the current
compendium/cn value, so the produced batch differs from the shipped text only
in the bytes listed here. Nothing is written to compendium/cn -- this only
emits batch files for the controller.
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(ROOT, "2-Crucible汉化插件")
OUT = os.path.join(ROOT, "4-临时脚本", "2026-08-12-fix", "batches")

# canonical trigger clause, copied from the entries that already have it
# (crucible.talent.json::Pyromancer / Inspirator / Kineturge / Thoughtbinder)
TRIG_DMG = "当你施放一个涉及此符文的法术、或使用一个涉及此符文的动作时，你造成伤害的暴击"
TRIG_HP = "当你施放一个涉及此符文的法术、或使用一个涉及此符文的动作时，你恢复生命值的暴击"

# path -> list of (old, new) applied in order; each must match exactly once
EDITS = {
    "crucible.rules.json": {
        "Character Creation.pages.Level Advancement.text": [
            # upstream 0.10.1 raised 2 -> 3 Talent Points (0.9.1 baseline says 2)
            ("<li>2点天赋点，可用于购买新的角色天赋。</li>",
             "<li>3点天赋点，可用于购买新的角色天赋。</li>"),
            # typo
            ("你的队伍完完成了", "你的队伍完成了"),
            # EN <strong>Level</strong>; the same leaf renders it 等级 twice more
            ("获得高于1的<strong>级</strong>时", "获得高于1的<strong>等级</strong>时"),
        ],
        "Character Mechanics.pages.Ability Scores.text": [
            # EN: "Half of the Health pool they would have possessed instead
            #      contributes as a bonus to their Morale pool."
            ("它们原本应拥有的任何生命值都会改为计入其<strong>士气</strong>池。",
             "它们原本应拥有的生命值池的一半会转而作为加值计入其<strong>士气</strong>池。"),
            # EN: "Half of the Morale pool ... as a bonus to their Health pool."
            ("它们原本应拥有的任何士气都会改为计入其<strong>生命值</strong>池。",
             "它们原本应拥有的士气池的一半会转而作为加值计入其<strong>生命值</strong>池。"),
            # @UUID label left in English; the SAME target is already {资源}
            # in crucible.rules.json::Spellcraft.pages.Runes.text
            ("JournalEntryPage.Resources0000000]{Resource} 池",
             "JournalEntryPage.Resources0000000]{资源} 池"),
        ],
        "Character Mechanics.pages.Resistances and Vulnerability.text": [
            # EN: "a volatile Fire Elemental"; taxonomy name is 火元素 Fire Elemental
            ("假设一个不稳定的火焰对<strong>寒冷</strong>伤害",
             "假设一个不稳定的火元素对<strong>寒冷</strong>伤害"),
            # EN <strong>Critical Hit</strong>; PROJECT.md 第 8 节 Critical Hit -> 暴击
            ("通常是由于一次<strong>重击</strong>导致的",
             "通常是由于一次<strong>暴击</strong>导致的"),
        ],
    },
    "crucible.talent.json": {
        "Dustbinder.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
            # EN: deals HALF the ability score ... as Acid damage to Health
            ("并且每轮会造成等同于你感知属性值的<strong>酸液</strong><strong>伤害</strong>，作用于<strong>生命值</strong>。",
             "并且每轮对<strong>生命值</strong>造成相当于你<strong>感知</strong>属性值一半的<strong>强酸</strong>伤害。"),
        ],
        "Mender.description": [
            ("使用此符文的法术在造成暴击时会施加", TRIG_HP + "会施加"),
        ],
        "Mesmer.description": [
            ("使用该符文的法术在暴击时会造成 @Condition[confused]。",
             TRIG_DMG + "会施加 @Condition[confused] 状态。"),
        ],
        "Necromancer.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
            # EN: deals HALF the ability score the Rune of Death scales with
            #     (Rune: Death scales using Presence -> 存在, not 智力)
            ("并且每轮对<strong>生命值</strong>造成等同于你的<strong>智力</strong>属性值的<strong>腐化</strong>伤害。",
             "并且每轮对<strong>生命值</strong>造成相当于你<strong>存在</strong>属性值一半的<strong>腐化</strong>伤害。"),
        ],
        "Rimecaller.description": [
            ("使用此符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
        ],
        "Surgeweaver.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
        ],
        "Voidcaller.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
            # EN: half the ability score the Rune of Oblivion scales with
            #     (Rune: Oblivion scales using Intellect -> 智力) as Void damage
            #     to HEALTH (not Morale)
            ("造成相当于你<strong>存在</strong>属性一半的<strong>虚空</strong><strong>士气</strong>伤害，",
             "对<strong>生命值</strong>造成相当于你<strong>智力</strong>属性一半的<strong>虚空</strong>伤害，"),
        ],
        # EN: "Move forward at least 3 feet and up to your stride" -- the CN
        # invented a fixed 6 feet cap.
        "Flying Kick.actions.flyingKick.description": [
            ("<p><strong>移动</strong>向前 3 至 6 英尺，沿直线前进，并以一次强化<strong>打击</strong>踢向一名敌人。</p>",
             "<p>沿直线向前<strong>移动</strong>至少 3 英尺、至多不超过你的步幅，并以一次强化<strong>打击</strong>踢向一名敌人。</p>"),
        ],
        # EN: "Flame: Dragons and Fire Elementals"; the other three elemental
        # rows in the same list kept 元素.
        "Gesture: Sense.description": [
            ("<li><p><strong>火焰</strong>：巨龙和火焰</p></li>",
             "<li><p><strong>火焰</strong>：巨龙和火元素</p></li>"),
        ],
    },
    "crucible.playtest.json": {
        "Playtest 1 - The Ring of Valor.actors.Eliorwen.items.Surgeweaver.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
        ],
        "Playtest 1 - The Ring of Valor.actors.Fizzit.items.Mesmer.description": [
            ("使用该符文的法术在暴击时会造成@Condition[confused]。",
             TRIG_DMG + "会施加@Condition[confused]状态。"),
        ],
        "Playtest 1 - The Ring of Valor.actors.Fizzit.items.Rimecaller.description": [
            ("使用此符文的法术在暴击时会造成@Condition[freezing]状态。",
             TRIG_DMG + "会施加@Condition[freezing]状态。"),
        ],
        "Playtest 1 - The Ring of Valor.actors.Fizzit.items.Surgeweaver.description": [
            ("使用此符文的法术在暴击时会造成 @Condition[shocked] 状态。",
             TRIG_DMG + "会施加 @Condition[shocked] 状态。"),
        ],
        "Playtest 1 - The Ring of Valor.actors.Kagura.items.Dustbinder.description": [
            ("使用该符文的法术在暴击时会造成 @Condition[corroding] 状态。",
             TRIG_DMG + "会施加 @Condition[corroding] 状态。"),
            ("腐蚀效果会持续<strong>3轮</strong>，并且每轮对<strong>生命</strong>造成等同于你的<strong>感知</strong>属性值的<strong>酸液</strong>伤害。",
             "腐蚀效果会持续<strong>3 轮</strong>，并且每轮对<strong>生命值</strong>造成相当于你<strong>感知</strong>属性值一半的<strong>强酸</strong>伤害。"),
        ],
        "Playtest 1 - The Ring of Valor.actors.Kagura.items.Mender.description": [
            ("使用此符文的法术在暴击时会施加", TRIG_HP + "会施加"),
        ],
        # untranslated 1 Round / Presence / Radiant; the pregens twin is fully
        # translated -- align to it verbatim.
        "Playtest 1 - The Ring of Valor.actors.Zarajah.items.Lightbringer.description": [
            ("<p>你极其擅长编织照明符文。使用此符文的法术在造成暴击时会施加 @Condition[irradiated] 状态。</p><p>辐照效果持续<strong>1 Round</strong>，并对<strong>生命值</strong>和<strong>士气</strong>造成等同于你<strong>Presence</strong>数值的<strong>Radiant</strong>伤害。</p>",
             "<p>你极其擅长编织照明符文。" + TRIG_DMG + "会施加 @Condition[irradiated] 状态。</p><p>辐照效果会持续<strong>1 轮</strong>，并对<strong>生命值</strong>和<strong>士气</strong>各造成等同于你的<strong>照明符文</strong>所依据属性值的<strong>光耀</strong>伤害。</p>"),
        ],
    },
    "crucible.pregens.json": {
        "Eliorwen.items.Surgeweaver.description": [
            ("使用此符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
        ],
        "Fizzit.items.Mesmer.description": [
            ("使用该符文的法术在暴击时会造成@Condition[confused]。",
             TRIG_DMG + "会施加@Condition[confused]状态。"),
        ],
        "Fizzit.items.Rimecaller.description": [
            ("使用此符文的法术在暴击时会造成@Condition[freezing]状态。",
             TRIG_DMG + "会施加@Condition[freezing]状态。"),
        ],
        "Fizzit.items.Surgeweaver.description": [
            ("使用此符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
        ],
        "Kagura.items.Dustbinder.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_DMG + "会施加"),
            ("并且每轮都会造成等同于你<strong>感知</strong>属性值的<strong>强酸</strong>伤害到<strong>生命值</strong>。",
             "并且每轮对<strong>生命值</strong>造成相当于你<strong>感知</strong>属性值一半的<strong>强酸</strong>伤害。"),
        ],
        "Kagura.items.Mender.description": [
            ("使用该符文的法术在造成暴击时会施加", TRIG_HP + "会施加"),
        ],
    },
}


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            node = node[int(p)]
        else:
            return None
    return node


def resolve(root, path):
    """Longest-key-first walk, same tolerance as apply_translations.split_path."""
    naive = path.split(".")
    if get_at(root, naive) is not None:
        return get_at(root, naive)
    node, rest = root, path
    while rest:
        if isinstance(node, dict):
            cands = [k for k in node if rest == k or rest.startswith(k + ".")]
            if cands:
                k = max(cands, key=len)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        head, _, rest = rest.partition(".")
        node = node.get(head) if isinstance(node, dict) else None
    return node


os.makedirs(OUT, exist_ok=True)
fail = 0
for pack, edits in EDITS.items():
    cn = json.load(open(os.path.join(REPO, "compendium", "cn", pack), encoding="utf-8"))
    batch = {}
    for path, ops in edits.items():
        cur = resolve(cn["entries"], path)
        if not isinstance(cur, str):
            print(f"!! {pack}::{path}  no current CN string")
            fail += 1
            continue
        new = cur
        for old, rep in ops:
            n = new.count(old)
            if n != 1:
                print(f"!! {pack}::{path}  pattern occurs {n}x: {old[:60]!r}")
                fail += 1
                break
            new = new.replace(old, rep)
        else:
            if new == cur:
                print(f"!! {pack}::{path}  no-op")
                fail += 1
            else:
                batch[path] = new
    fn = os.path.join(OUT, "mech-" + pack.replace(".json", "") + ".batch.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{pack}: {len(batch)} entries -> {fn}")

print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
