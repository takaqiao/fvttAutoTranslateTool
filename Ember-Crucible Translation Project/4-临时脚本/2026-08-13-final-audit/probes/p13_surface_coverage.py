# -*- coding: utf-8 -*-
"""
p13_surface_coverage.py  —— 「入口钩子/闸覆盖不到整个界面面」这一类缺陷的机械判据

背景实例（已记录，不重复报）：聊天卡片面没有 renderChatMessageHTML 入口，
同调奖励卡表头与「激活同调」描述进聊天栏时是英文。

抽象成判据：
  上游（ember / crucible）把**用户可见的英文字面量**写进某个「界面面」；
  本汉化模块的替换层只有两个入口 + 一道闸：
      Hooks.on("renderApplicationV2" | "renderApplication")  →  handler
      闸：  /ember/i.test(root.className) || /^Ember/.test(app.constructor.name)
            例外：DialogV2（或 class 含 dialog）只翻 .window-title 后 return
  凡是**不经这两个钩子派发**、或**经过但被闸整体挡掉**的界面面，
  该面上的英文就永远不会被替换层看到。

本脚本做三件事：
  1) 从插件源码里解析出替换层的入口与闸（coverage set），不写死。
  2) 在上游源码里按「文本汇点」（sink）抓用户可见英文字面量，并归到界面面。
  3) 打印每个面的 coverage 判定与英文字面量条数。

判定依据（Foundry v14 client 源码实证）：
  - ApplicationV2#_doEvent → #callHooks 会对**原型链上每个类**派发 render<ClassName>
    (applications/api/application.mjs:1724-1728)，所以 renderApplicationV2 覆盖所有 AppV2。
  - Notifications 不是 Application（applications/ui/notifications.mjs:30 `class Notifications`），
    整个文件 0 处 Hooks.call → 没有任何入口。
  - TooltipManager 同样不是 Application（helpers/interaction/tooltip-manager.mjs:25）。
  - ContextMenu 同样不是（applications/ux/context-menu.mjs:67）。
  - 聊天卡只经 Hooks.callAll("renderChatMessageHTML") 派发
    (documents/chat-message.mjs:393/435) —— 已记录，脚本里标 REPORTED。

已知假阳性模式（必须逐条人工核实，脚本只给候选）：
  a) 字面量其实是 lang key（EMBER.xxx / DICE.xxx）→ 已用 ^[A-Z][A-Z0-9_]*\\. 过滤，
     但形如 "Show Tracks" 这种既像文案又可能是 key 的短语过滤不掉。
  b) 只有开发模式 / 只有 GM 调试路径才会执行的分支（如 locateBrokenLinks、
     PreciseText("Daylight Color") 这类光照调试叠加层）。
  c) 数据配置里的 label/title 未必渲染（可能只做内部标识）。
  d) 正则无法解析动态拼接的 surface；跨函数的间接注入抓不到。

只读，不写任何库文件。
"""

import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
CLIENT = r"C:\Program Files\Foundry Virtual Tabletop\resources\app\client"

UPSTREAM = {
    "ember.mjs": os.path.join(FVTT, r"modules\ember\scripts\ember.mjs"),
    "dnd5e-async.mjs": os.path.join(FVTT, r"modules\ember\scripts\dnd5e-async.mjs"),
    "crucible-async.mjs": os.path.join(FVTT, r"modules\ember\scripts\crucible-async.mjs"),
    "crucible-compiled.mjs": os.path.join(FVTT, r"systems\crucible\crucible-compiled.mjs"),
}

PLUGINS = [
    os.path.join(ROOT, r"1-Ember汉化插件\register.js"),
    os.path.join(ROOT, r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"),
    os.path.join(ROOT, r"2-Crucible汉化插件\babele-register.js"),
]

LANGKEY = re.compile(r"^[A-Z][A-Z0-9_]*\.[A-Za-z0-9_.]+$")
HAS_ALPHA = re.compile(r"[A-Za-z]")
PATHY = re.compile(r"(/|\\|\.(svg|png|webp|ogg|hbs|json|mjs|js)$|^fa-|^#|^\.)")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def lines(text):
    return text.split("\n")


def is_user_facing(s):
    """粗筛：像给玩家看的英文句子/短语。"""
    s = s.strip()
    if len(s) < 4:
        return False
    if not HAS_ALPHA.search(s):
        return False
    if LANGKEY.match(s):
        return False
    if PATHY.search(s):
        return False
    # 至少两个词，或首字母大写的多字词
    if " " not in s:
        return False
    return True


# ---------------------------------------------------------------- 1. 覆盖集

def parse_coverage():
    hooks = set()
    gates = []
    for p in PLUGINS:
        if not os.path.exists(p):
            continue
        t = read(p)
        for m in re.finditer(r"Hooks\.(?:on|once)\(\s*[\"']([^\"']+)[\"']", t):
            hooks.add(m.group(1))
        for m in re.finditer(r"if\s*\(\s*!\/(.+?)\/i?\.test", t):
            gates.append(m.group(1))
    return sorted(hooks), gates


# ---------------------------------------------------------------- 2. 汇点扫描

SINKS = [
    # (sink id, 正则, 取第几组做字面量, 界面面, 覆盖判定)
    ("notify", re.compile(r"ui\.notifications\.(?:info|warn|error|success|notify)\(\s*[\"'`]([^\"'`]{4,200})"),
     1, "Notifications 通知条", "NO-HOOK"),
    ("tooltip", re.compile(r"game\.tooltip\.activate\("), 0, "Tooltip 悬浮框", "NO-HOOK"),
    ("canvas", re.compile(r"PreciseText\(\s*[\"'`]([^\"'`]{3,80})[\"'`]"), 1, "Canvas PIXI 文本", "NO-DOM"),
    ("settings", re.compile(r"game\.settings\.register\([^)]*?\{[^}]*?name:\s*[\"']([^\"']{4,120})[\"']", re.S),
     1, "SettingsConfig 设置面板", "GATE-BLOCKED"),
    ("settings_hint", re.compile(r"game\.settings\.register\([^)]*?\{[^}]*?hint:\s*[\"']([^\"']{4,240})[\"']", re.S),
     1, "SettingsConfig 设置面板", "GATE-BLOCKED"),
    ("keybind", re.compile(r"game\.keybindings\.register\([^)]*?\{\s*name:\s*[\"']([^\"']{4,120})[\"']", re.S),
     1, "KeybindingsConfig 控制面板", "GATE-BLOCKED"),
    ("dialog_content", re.compile(r"content:\s*[\"'`]([^\"'`]{8,300})[\"'`]"), 1, "DialogV2 正文", "SELECTOR-TITLE-ONLY"),
    ("dialog_btn", re.compile(r"ok:\s*\{\s*label:\s*[\"']([^\"']{2,60})[\"']"), 1, "DialogV2 按钮", "SELECTOR-TITLE-ONLY"),
    ("chat", re.compile(r"ChatMessage\.(?:create|implementation\.create)\("), 0, "聊天卡片", "REPORTED"),
]

CORE_RENDER_HOOKS = re.compile(
    r"Hooks\.on\(\s*[\"'](render(?!Ember)[A-Z][A-Za-z]*)[\"']")


def scan_upstream():
    out = {}
    for label, path in UPSTREAM.items():
        if not os.path.exists(path):
            continue
        t = read(path)
        ls = lines(t)
        for sid, rx, grp, surface, verdict in SINKS:
            for m in rx.finditer(t):
                lit = m.group(grp) if grp else ""
                if grp and not is_user_facing(lit):
                    continue
                ln = t.count("\n", 0, m.start()) + 1
                rec = out.setdefault((surface, verdict), [])
                rec.append({"file": label, "line": ln, "sink": sid,
                            "text": (lit or ls[ln - 1].strip())[:200]})
        # 上游自己往核心应用里注入 DOM 的入口
        for m in CORE_RENDER_HOOKS.finditer(t):
            ln = t.count("\n", 0, m.start()) + 1
            out.setdefault(("上游挂在核心应用上的 render 钩子", "GATE-BLOCKED"), []).append(
                {"file": label, "line": ln, "sink": "core-render-hook", "text": m.group(1)})
    return out


def main():
    hooks, gates = parse_coverage()
    print("=== 替换层入口 ===")
    print("hooks:", hooks)
    print("gates:", gates)
    print()
    res = scan_upstream()
    summary = []
    for (surface, verdict), items in sorted(res.items(), key=lambda kv: -len(kv[1])):
        summary.append({"surface": surface, "verdict": verdict, "n": len(items),
                        "sample": items[:400]})
        print(f"[{verdict:20s}] {surface:34s} n={len(items)}")
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p13_surface_coverage.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("\n->", dst)


if __name__ == "__main__":
    main()
