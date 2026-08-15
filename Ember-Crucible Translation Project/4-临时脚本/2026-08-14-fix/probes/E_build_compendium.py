# -*- coding: utf-8 -*-
"""分片 E 里落点在 compendium 的两条 finding：
  · cross_layer_term_split（正文用词 vs UI 枚举分叉）—— 按复核 verdict 的 scope_note 派工
  · pronoun_split 的第 3 处（crucible.rules Providing Feedback 的「您」）

整叶回写批次，逐条替换都带断言：命中数必须等于预期，且 `id="` 计数不减。
"""
import json, os, re, sys, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "4-临时脚本", "2026-08-14-fix", "batches")

EMBER = "1-Ember汉化插件"
CRU = "2-Crucible汉化插件"


def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{p}.{i}" if p else str(i))
    elif isinstance(o, str):
        yield p, o


_cache = {}


def leaf(repo, pack, path):
    key = (repo, pack)
    if key not in _cache:
        _cache[key] = dict(walk(json.load(open(os.path.join(ROOT, repo, "compendium", "cn", pack),
                                              encoding="utf-8-sig"))))
    return _cache[key][path]


def apply(repo, pack, path, reps):
    """reps: [(old, new, count)]"""
    s = leaf(repo, pack, path)
    before_id = s.count('id="')
    for old, new, n in reps:
        hit = s.count(old)
        assert hit == n, f"{pack} | {path} | {old!r} 命中 {hit}，预期 {n}"
        s = s.replace(old, new)
    assert s.count('id="') == before_id, f"{pack} | {path} | id= 计数变了"
    return s


BATCHES = {}


def put(repo, pack, path, reps):
    BATCHES.setdefault((repo, pack), {})[path] = apply(repo, pack, path, reps)


# ============================================================ crucible.rules
R = "crucible.rules.json"

# 1. Character Creation / Finishing Touches —— 7 处
put(CRU, R, "entries.Character Creation.pages.Finishing Touches.text", [
    ("在 传记 标签页中", "在 生平 标签页中", 1),                 # ACTOR.TABS.biography = 生平
    ("<strong>代称</strong>", "<strong>代词</strong>", 1),        # ...biography.pronouns.label
    ("<strong>年代</strong>", "<strong>年龄</strong>", 1),        # Age，硬错译（年代=era）
    ("<strong>重量</strong>", "<strong>体重</strong>", 1),        # Weight（用于人）
    ("<strong>缩放价格</strong>", "<strong>按比例定价</strong>", 1),  # ITEM.SHEET.ScaledPrice
    ("<strong>公开（Public）传记</strong>", "<strong>公开传记</strong>", 1),
    ("<strong>私密（Private）传记</strong>", "<strong>私人传记</strong>", 1),
])

# 2. Character Creation / Overview —— Group 类型 1 硬处 + 2 处回指
put(CRU, R, "entries.Character Creation.pages.Overview.text", [
    ("类型为 <strong>群组</strong> 的 角色", "类型为 <strong>团队</strong> 的 角色", 1),  # TYPES.Actor.group
    ("将该群组配置为", "将该团队配置为", 1),
    ("识别哪个群组是当前活跃英雄队伍", "识别哪个团队是当前活跃英雄队伍", 1),
])

# 3. Combat / Actions —— 目标类型 Wall，4 处
put(CRU, R, "entries.Combat.pages.Actions.text", [
    ("<h4>墙</h4>", "<h4>墙壁</h4>", 1),                          # ACTION.TARGET_TYPES.Wall = 墙壁
    ("墙类型动作", "墙壁类型动作", 1),
    ("用于描述墙的宽度", "用于描述墙壁的宽度", 1),
    ("用于描述墙的中心", "用于描述墙壁的中心", 1),
])

# 4. Adversaries / Overview —— Minion×3 + Boss×4
put(CRU, R, "entries.Adversaries.pages.Overview.text", [
    ("<td><p>喽啰</p></td>", "<td><p>仆从</p></td>", 1),           # THREAT_RANKS.Minion = 仆从
    ("喽啰危险性较低", "仆从危险性较低", 1),
    ("而重要喽啰的储量", "而重要仆从的储量", 1),
    ("精英与 Boss 对手", "精英与首领对手", 1),                      # THREAT_RANKS.Boss = 首领
    ("<td><p>Boss</p></td>", "<td><p>首领</p></td>", 1),
    ("都不包含 Boss。", "都不包含首领。", 1),
    ("重要精英或重要 Boss，", "重要精英或重要首领，", 1),
])

# 5. Adversaries / Swarms —— Minion 1 处（同叶「首领」已对，不动）
put(CRU, R, "entries.Adversaries.pages.Swarms.text", [
    ("<strong>爪牙</strong>级敌人", "<strong>仆从</strong>级敌人", 1),
])

# 6. Equipment / Weapons —— 附魔阶表 4 行错 3 行
put(CRU, R, "entries.Equipment.pages.Weapons.text", [
    ("<td><p>凡俗</p></td>", "<td><p>凡品</p></td>", 1),           # ITEM.EnchantmentMundane
    ("<td><p>轻微</p></td>", "<td><p>次级</p></td>", 1),           # ITEM.EnchantmentMinor
    ("<td><p>强效</p></td>", "<td><p>高级</p></td>", 1),           # ITEM.EnchantmentMajor
])

# 7. Equipment / Armor —— 附魔阶表首行
put(CRU, R, "entries.Equipment.pages.Armor.text", [
    ("<td><p>凡俗</p></td>", "<td><p>凡品</p></td>", 1),
])

# 8. pronoun_split：全库正文唯一一处「您」
put(CRU, R, "entries.Welcome To Crucible.pages.Providing Feedback.text", [
    ("您反馈", "你反馈", 1),
])

# ========================================================= crucible.playtest
P = "crucible.playtest.json"
PP = ("entries.Playtest 1 - The Ring of Valor.journals.Playtest One: The Ring of Valor"
      ".pages.Day Three - The Skeletal Army.text")
put(CRU, P, PP, [
    ("（3级 Boss）", "（3级 首领）", 1),
    ("作为 Boss 敌人", "作为首领敌人", 1),
])

# ==================================================================== ember
# 孪生包：ember.adventure 与 ember.crucible-adventure 英文逐字相同的叶子，两包各写一份
MOON = [("<strong>盈月</strong>", "<strong>渐盈</strong>", 1),      # EMBER.MOON.PHASES.WAXING
        ("<strong>亏月</strong>", "<strong>渐亏</strong>", 1)]     # EMBER.MOON.PHASES.WANING
put(EMBER, "ember.crucible-affixes.json", "entries.Lunar Shield.description", MOON)
put(EMBER, "ember.crucible-adventure.json",
    "entries.Ember Early Access.items.Moon Ring.effects.Lunar Shield.description", MOON)

MOONSTONE = [("属于凡俗", "属于凡品", 1)]                            # ITEM.EnchantmentMundane
MATRIX = [("<h4>隐藏可选Boss</h4>", "<h4>隐藏可选首领</h4>", 1),
          ("成为一名隐藏Boss", "成为一名隐藏首领", 1)]
for pack in ("ember.adventure.json", "ember.crucible-adventure.json"):
    put(EMBER, pack,
        "entries.Ember Early Access.journals.The Winding Trail.pages.Giant Moonstone.text", MOONSTONE)
    put(EMBER, pack,
        "entries.Ember Early Access.journals.Chamber of Agaseros.pages.Matrix of Agaseros.text", MATRIX)

# ------------------------------------------------------------------- 输出
os.makedirs(OUT, exist_ok=True)
for (repo, pack), data in sorted(BATCHES.items()):
    n = "1" if repo == EMBER else "2"
    p = os.path.join(OUT, f"E.{n}.{pack}")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{p}  {len(data)} 叶")
