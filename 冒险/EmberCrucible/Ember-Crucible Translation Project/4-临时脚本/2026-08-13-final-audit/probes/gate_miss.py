# -*- coding: utf-8 -*-
"""
gate_miss.py  —— 「闸/选择器失配」类判据探针（只读）

背景（已确认实例）：ember 把一个 toFormGroup 造出来的表单组插进核心 NoteConfig，
而 ember-hardcoded-cn.mjs 的 patchRenderedApplications 闸只放行
  root.className 含 /ember/i  或  app.constructor.name 以 Ember 开头
两者都不满足时直接 return（DialogV2 只额外放行 .window-title 一行）。
=> 凡是 ember 把英文 UI 注入到「非 Ember 宿主 App」的地方，都跑不进替换。

本探针把这条抽象成三个可机械化的子判据：

  A) 宿主渲染钩子注入：ember 源码里 Hooks.on/registerHook("render<X>")，
     且 <X> 不以 Ember 开头 —— 这些宿主的根元素 class 与类名都不带 ember。
  B) DialogV2 调用点：DialogV2.{prompt,confirm,wait,input,query} 的
     content / ok.label / buttons[].label / 各 label 字段里的裸英文。
     闸对 DialogV2 只翻 .window-title，正文与按钮永远翻不到。
  C) 逃逸出所有 App 根的输出：ui.notifications.* 与 game.tooltip.activate
     —— 它们根本不在任何 renderApplicationV2 的 root 里。

判据的已知假阳性模式：
  * 字符串是 i18n 键（形如 EMBER.FOO.Bar / ACTOR.CONTROLS.X）时不算缺陷，
    走 lang/cn.json 通道 —— 脚本用正则 ^[A-Z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+$ 排除，
    但仍会漏判「看起来像键其实是英文句子」的情况，需人工复核。
  * 只在 dnd5e 世界执行的分支（initialize$1 内注册的钩子）与 dev-mode 分支
    对 crucible 世界不可达 —— 脚本标注但不自动排除。
  * 纯 CSS class / 选择器 / 文件路径字符串会被 EN_LIT 正则误收，人工剔除。
"""
import json, os, re, sys

ROOT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CN   = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
LANG_DIR = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\lang"

src = open(ROOT, encoding="utf-8").read()
lines = src.splitlines()

I18N = re.compile(r"^[A-Z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+$")
ENG  = re.compile(r"^[A-Z][A-Za-z0-9'’\-]*(?:[ ,:!?/&\-][A-Za-z0-9'’\-\.]+)*[.!?]?$")

def is_i18n(s):
    return bool(I18N.match(s.strip()))

def looks_english(s):
    s = s.strip()
    if not s or len(s) < 3: return False
    if is_i18n(s): return False
    if re.search(r"[\u4e00-\u9fff]", s): return False
    if s.startswith(("fa-", ".", "#", "/", "modules/", "systems/")): return False
    if not re.search(r"[A-Za-z]{3}", s): return False
    return True

out = {"A_host_render_hooks": [], "B_dialogs": [], "C_escapes": []}

# ---------- A ----------
for m in re.finditer(r'(?:Hooks\.(?:on|once)|this\.registerHook)\(\s*"(render[A-Za-z0-9_]+)"', src):
    name = m.group(1)
    ln = src[:m.start()].count("\n") + 1
    host = name[len("render"):]
    out["A_host_render_hooks"].append({
        "line": ln, "hook": name, "host": host,
        "host_is_ember": host.startswith("Ember"),
        "snippet": lines[ln-1].strip()
    })

# ---------- B ----------
for m in re.finditer(r"DialogV2\.(prompt|confirm|wait|input|query)\s*\(", src):
    ln = src[:m.start()].count("\n") + 1
    # 取调用点起 60 行做上下文（够覆盖所有实际调用）
    ctx = "\n".join(lines[ln-1: ln+59])
    # 只保留到括号配平处
    depth = 0; end = 0; started = False
    for i, ch in enumerate(ctx):
        if ch == "(":
            depth += 1; started = True
        elif ch == ")":
            depth -= 1
            if started and depth == 0:
                end = i; break
    body = ctx[:end+1] if end else ctx
    lits = re.findall(r'"([^"\\\n]{2,120})"|`([^`\\\n]{2,120})`', body)
    lits = [a or b for a, b in lits]
    eng = sorted({s for s in lits if looks_english(s)})
    keys = sorted({s for s in lits if is_i18n(s)})
    out["B_dialogs"].append({"line": ln, "kind": m.group(1), "english": eng, "i18n_keys": keys,
                             "body_head": body[:400]})

# ---------- C ----------
for m in re.finditer(r'ui\.notifications\.(warn|error|info|notify)\(\s*(?:"([^"\\\n]{2,160})"|`([^`\\\n]{2,160})`)', src):
    ln = src[:m.start()].count("\n") + 1
    s = m.group(2) or m.group(3)
    if looks_english(s):
        out["C_escapes"].append({"line": ln, "kind": "notification", "text": s})
for m in re.finditer(r'game\.tooltip\.activate\(', src):
    ln = src[:m.start()].count("\n") + 1
    out["C_escapes"].append({"line": ln, "kind": "tooltip.activate", "text": lines[ln-1].strip()})

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_miss.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("A host render hooks:", len(out["A_host_render_hooks"]),
      "| non-Ember hosts:", sum(1 for x in out["A_host_render_hooks"] if not x["host_is_ember"]))
print("B dialog sites:", len(out["B_dialogs"]),
      "| with english:", sum(1 for x in out["B_dialogs"] if x["english"]))
print("C escapes:", len(out["C_escapes"]))
print("->", dst)
