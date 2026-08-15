# -*- coding: utf-8 -*-
"""分片 E：构建 lang/cn.json 回写批次（扁平点号键，禁止嵌套）。

只处理「键已存在于上游 en.json」的条目 —— apply_lang.py 第 1 道闸要求如此。
新增键（英文原串当键 / 上游 en.json 自己都没有的键）走编辑清单，不在这里。
"""
import json, os, sys, io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "4-临时脚本", "2026-08-14-fix", "lang")

def cnpath(repo):
    return os.path.join(ROOT, repo, "lang", "cn.json")

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)

EMBER = "1-Ember汉化插件"
CRU = "2-Crucible汉化插件"

ecn = load(cnpath(EMBER))
ccn = load(cnpath(CRU))

# ---------------------------------------------------------------- crucible
c = {}

# 符文形容词一族（5 条）
c["SPELL.RUNES.LifeAdj"] = "生机的"
c["SPELL.RUNES.StormAdj"] = "电击的"
c["SPELL.RUNES.KinesisAdj"] = "念力的"
c["SPELL.RUNES.IlluminationAdj"] = "照亮的"
c["SPELL.RUNES.DeathAdj"] = "死亡的"

# 占位符语义倒置
c["ADVANCEMENT.MilestoneTooltip"] = "升至下一级需 {required} 个里程碑，已达成 {progress} 个"

# 紧凑消耗标签单字方案
c["ACTION.TAG.CostHealth"] = "{health}生"

# 同英文多中文
c["BASE_EFFECT.FIELDS.dot.element.restoration.label"] = "恢复"
c["ACTION.FIELDS.effects.element.result.all.label"] = "全部"
c["ITEM.FIELDS.invested.label"] = "已注入"
c["SKILL.PageSheet"] = "Crucible 技能"
c["ACTOR.FIELDS.movement.stride.tooltip"] = "每消耗 1 动作点的移动距离。"
c["TAXONOMY.FIELDS.movement.stride.tooltip"] = "每消耗 1 动作点的移动距离。"

# 人称
c["ACTION.ACTIONS.DeleteConfirm"] = ccn["ACTION.ACTIONS.DeleteConfirm"].replace("您", "你")

# Token -> 令牌
for k, v in ccn.items():
    if "指示物" in v:
        c[k] = v.replace("指示物", "令牌")

# ---------------------------------------------------------------- ember
e = {}

# 同英文多中文
e["EMBER.ANCESTRY.FIELDS.color.label"] = "界面颜色"
e["EMBER.COSMOS.FIELDS.color.label"] = "界面颜色"
e["EMBER.BIOME.SECTIONS.visibility"] = "可见性"
e["EMBER.LORE.FIELDS.banner.img.label"] = "横幅图像"
e["EMBER.ORGANIZATION.FIELDS.caption.hint"] = "显示在横幅图像下方的可选说明文字。"

# attunement 定译「同调」
for k in ("EMBER.COSMOS.FIELDS.color.hint", "EMBER.COSMOS.FIELDS.icon.hint"):
    e[k] = ecn[k].replace("宇宙调谐", "宇宙同调")

# 人称
e["EMBER.EVENT.MESSAGES.CHANGE_OUTCOME"] = ecn["EMBER.EVENT.MESSAGES.CHANGE_OUTCOME"].replace("您", "你")

# Token -> 令牌
for k, v in ecn.items():
    if "指示物" in v:
        e[k] = v.replace("指示物", "令牌")

for name, data in (("E.1.json", e), ("E.2.json", c)):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(p, len(data))
