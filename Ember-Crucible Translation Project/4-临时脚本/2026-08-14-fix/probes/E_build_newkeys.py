# -*- coding: utf-8 -*-
"""分片 E：生成「上游 en.json 里没有的新键」的 lang/cn.json 编辑清单。

apply_lang.py 第 1 道闸（key 必须存在于上游英文）结构性地拒绝这批键 ——
它们正是「上游注入面枚举不全」类缺陷的修法：以英文原串（或上游漏声明的点号键）
当键写进 cn.json，让 Foundry 的 getProperty 命中。

所以这批走编辑清单（甲），由主控串行套用；本脚本负责：
  1. 生成 old / new 文本
  2. 就地模拟套用 -> 校验仍是合法 JSON
  3. 复刻 Foundry 的 expandObject + getProperty，逐键验证「真的查得到」
"""
import json, os, sys, copy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EMBER = "1-Ember汉化插件"
CRU = "2-Crucible汉化插件"

# ----------------------------------------------------------------- crucible
CRU_NEW = {
    # 动作标签组 tooltip（const/action.mjs 5 处裸串 -> data-tooltip -> tooltip-manager has()）
    "Spell Tags": "法术标签",
    "Strikes": "打击",
    "Reload": "装填",
    "Skill Tags": "技能标签",
    "Weapon Tags": "武器标签",
    # 通用页脚提交按钮（crucible 合集来源设置窗 + 核心 av-config 也用同串）
    "Save Changes": "保存更改",
    # 字面量直接进 localize
    "None": "无",
    "Unknown": "未知",
    # enricher 占位 tooltip
    "Spell tooltips are still TO-DO.": "法术悬浮提示尚未实现。",
    # 键族缺格：useMove 动态拼 TOKEN.MOVEMENT.ACTIONS.<id>.description
    "TOKEN.MOVEMENT.ACTIONS.blink.description":
        "以传送方式瞬间抵达目标位置。不沿路径行进，因此无视地形消耗与其他生物的阻挡，每个动作的移动距离也没有上限。",
    "TOKEN.MOVEMENT.ACTIONS.displace.description":
        "由外部效果强制施加的传送位移，无视墙壁与地形。移动消耗始终为零。",
}

# -------------------------------------------------------------------- ember
EMBER_NEW = {
    # game.settings.register 的 name/hint（配置设置 -> Ember 一整块）
    "Gazetteer Location Journal Entries": "地名志地点日志条目",
    "Additional Journal Entries which provide custom gazetteer Location pages that should be added to the Ember environment.":
        "额外的日志条目，其中提供应加入余烬环境的自定义地名志地点页面。",
    "Standalone Event Journal Entries": "独立事件日志条目",
    "Additional Journal Entries which contain Standalone Event pages which should be added to the Ember event engine.":
        "额外的日志条目，其中包含应加入余烬事件引擎的独立事件页面。",
    "Clock Time Format": "时钟时间格式",
    "The clock format used to display the in-world time of day.": "用于显示游戏世界内当日时刻的时钟格式。",
    "Custom Cursors": "自定义光标",
    "Use custom Ember stylized mouse cursors instead of default browser cursors?":
        "使用余烬风格的自定义鼠标光标，而不是浏览器默认光标？",
    # game.keybindings.register 的 name/hint（配置控制项 -> Ember 一整块）
    "Flip Vista Placement": "翻转远景摆放",
    "When placing an asset in the Vista Configuration screen, flip it horizontally":
        "在远景配置界面中放置资源时，将其水平翻转",
    "Lock Vista Placement Elevation": "锁定远景摆放高度",
    "When placing an asset in the Vista Configuration screen, lock its elevation so it can be moved vertically.":
        "在远景配置界面中放置资源时，锁定其高度，使其可以垂直移动。",
    # 通用页脚提交按钮（actor-flags.hbs:52 + ember.mjs:51613 令牌工坊）
    "Save Changes": "保存更改",
    # RegionBehavior 三个子类型的表单（核心 RegionBehaviorConfig 以 localize=true 渲染）
    # -- ember.trapTrigger
    "Once": "仅一次",
    "Does the trigger automatically disable after firing once?": "触发器在触发一次后是否自动停用？",
    "Locked": "已锁定",
    "Can the trigger be disarmed? If locked, the trigger mechanism cannot be disarmed.":
        "该触发器能否被解除？若已锁定，则此触发机关无法被解除。",
    "Discovered": "已发现",
    "Has the trap been discovered?": "该陷阱是否已被发现？",
    "Triggered Behaviors": "被触发的行为",
    "Provide the UUIDs of Behaviors in this or other Regions which should be triggered by this trap.":
        "填写本区域或其他区域中应由此陷阱触发的行为的 UUID。",
    "Script": "脚本",
    "Custom JavaScript to execute when the trap is triggered.": "陷阱被触发时执行的自定义 JavaScript。",
    "Trigger Text": "触发文本",
    "Text of the scrolling message which displays when the trap is triggered.": "陷阱被触发时显示的滚动消息文本。",
    "Pause Game": "暂停游戏",
    "Automatically pause the game when the trap is triggered?": "陷阱被触发时是否自动暂停游戏？",
    # -- ember.areaEffect
    "Chat Message Description": "聊天消息描述",
    "HTML description text which appears in a chat message when this area effect is applied.":
        "此范围效果生效时，在聊天消息中显示的 HTML 描述文本。",
    "Ability Score": "属性值",
    "Save DC": "豁免 DC",
    "Damage Formula": "伤害公式",
    "Define an array of damage formula parts with format {type: string, formula: string}.":
        "定义一组伤害公式部件，格式为 {type: string, formula: string}。",
    "Effect Data": "效果数据",
    "An array of ActiveEffect data which is applied to Actors affected by this area effect.":
        "一组 ActiveEffect 数据，将应用于受此范围效果影响的角色。",
    # -- ember.footstepSurface
    "Material": "材质",
    "The material type of the surface.": "该地表的材质类型。",
    "Grass": "草地",
    "Metal": "金属",
    "Stone": "石头",
    "Water": "水",
    "Wood": "木头",
}


# ------------------------------------------------------- Foundry 查找语义复刻
def set_property(obj, key, value):
    if not key:
        return
    target = obj
    if "." in key:
        parts = key.split(".")
        key = parts.pop()
        for p in parts:
            if not isinstance(target.get(p), dict):
                target[p] = {}
            target = target[p]
    target[key] = value


def expand_object(o):
    out = {}
    for k, v in o.items():
        set_property(out, k, v)
    return out


def get_property(obj, key):
    if key in obj:
        return obj[key]
    node = obj
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def build(repo, anchor_key, new_keys, indent):
    path = os.path.join(ROOT, repo, "lang", "cn.json")
    raw = open(path, encoding="utf-8-sig").read()
    doc = json.loads(raw)
    dup = [k for k in new_keys if k in doc]
    assert not dup, f"{repo}: 这些键已经存在，别重复加: {dup}"
    tail = json.dumps(anchor_key, ensure_ascii=False) + ": " + json.dumps(doc[anchor_key], ensure_ascii=False)
    old = tail + "\n}"
    assert raw.count(old) == 1, f"{repo}: 锚点不唯一 ({raw.count(old)})"
    pad = " " * indent
    lines = [pad + json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False)
             for k, v in new_keys.items()]
    new = tail + ",\n" + ",\n".join(lines) + "\n}"
    # 模拟套用 + 校验
    patched = raw.replace(old, new)
    doc2 = json.loads(patched)               # 合法 JSON
    assert len(doc2) == len(doc) + len(new_keys)
    tr = expand_object(doc2)
    bad = [k for k, v in new_keys.items() if get_property(tr, k) != v]
    assert not bad, f"{repo}: 这些键 Foundry 查不到: {bad}"
    print(f"[{repo}] 新增 {len(new_keys)} 键，模拟套用后 JSON 合法、逐键 getProperty 命中")
    return {"file": f"{repo}/lang/cn.json", "old": old, "new": new}


edits = []
edits.append(build(CRU, "TEMPERATURE_TIERS.Gelid", CRU_NEW, 2))
edits.append(build(EMBER, "SPELL.INFLECTIONS.SignaraAdj", EMBER_NEW, 1))

out = os.path.join(ROOT, "4-临时脚本", "2026-08-14-fix", "probes", "E_newkey_edits.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(edits, f, ensure_ascii=False, indent=1)
print("->", out)
