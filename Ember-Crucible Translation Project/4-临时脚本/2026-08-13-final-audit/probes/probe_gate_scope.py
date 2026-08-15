# -*- coding: utf-8 -*-
r"""
probe_gate_scope.py  ——「闸/选择器失配导致整块界面不进入替换」的全库机械判据（只读）

已确认实例：crucible 英雄卡「同调」整页被 patchRenderedApplications 的类名闸挡在门外。
本探针把那个实例抽象成一句话判据：

    翻译层的每一个「入口条件」都是一道闸。凡是**上游会产生用户可见英文**、
    而该英文所在的 DOM/数据**不满足任何一道闸**的，就是同一类缺陷。

ember-hardcoded-cn.mjs 一共只有四道闸（= 四个入口）：
  G-render  Hooks.on("renderApplicationV2"|"renderApplication")，且
            /ember/i.test(根元素 className) || /^Ember/.test(app.constructor.name)
            —— 例外分支：DialogV2 / class 含 dialog 的窗口，只翻 `.window-title` 一行
  G-enrich  CONFIG.TextEditor.enrichers 中 pattern 源码命中关键字白名单的那些
  G-config  globalThis.crucible.CONFIG 的 ["languages","knowledge"] 两个组
  G-cal     CONFIG.time.worldCalendarConfig / game.time.calendar 的 ["months","days"]

于是「闸外」= 以下四种，逐一枚举：
  A. DialogV2 —— 标题进得去、正文/按钮进不去
  B. ember 注入到**外部宿主 App** 的 DOM（闸看根元素类名，注入块自己带 .ember 也没用）
  C. ember 写进 crucible.CONFIG 但不在白名单里的组
  D. 根本不经过 Application 渲染钩子的界面（ui.notifications / ChatMessage）

假阳性模式（务必人工复核后再报）：
  - A 里 content 可能是变量，本探针只能标 UNKNOWN，需人工回读赋值处；
  - A/B 里的英文可能只在开发者模式或特定场景出现（如 Show Tracks 只在矿车场景）；
  - D 里 ui.notifications 有大量 `err` / `err.message` 变量参数，不是字面量，已剔除；
  - 本探针是**文本**探针，上游若改用 _loc() 而字符串还在，会误判为未翻。
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
MODCN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"

src = open(EMBER, encoding="utf-8").read()
lines = src.split("\n")
cn = open(MODCN, encoding="utf-8").read()


def table(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, cn, re.S)
    return set(re.findall(r'"([^"]+)":\s*"', m.group(1))) if m else set()


EXACT = table("EXACT")
COVERED = EXACT | table("ATTUNEMENTS") | table("LANGUAGES") | table("KNOWLEDGE") | table("MOODS")
PATTERNS = [re.compile(p) for p in
            [r"^Result of (.+)$", r"^Award Attunement: (.+)$", r"^Revoke Attunement: (.+)$",
             r"^Activate Attunement: (.+)$", r"^Day (\d+)\b(.*)$"]]


def covered(s):
    s = s.strip()
    if s in COVERED:
        return True
    for p in ["Attunement", "Language", "Knowledge", "Music Mood"]:
        if s.startswith(p + ": "):
            return True
    return any(p.match(s) for p in PATTERNS)


def block_from(i):
    """从第 i 行的 `(` 起做括号配对，返回整块源码。"""
    txt = "\n".join(lines[i:i + 60])
    start = txt.index("(")
    depth, j = 0, start
    while j < len(txt):
        if txt[j] in "([{":
            depth += 1
        elif txt[j] in ")]}":
            depth -= 1
            if depth == 0:
                return txt[start:j + 1]
        j += 1
    return txt


print("=" * 78)
print("A. DialogV2 —— G-render 例外分支只翻 .window-title，正文/按钮在闸外")
print("=" * 78)
dlg = [i for i, l in enumerate(lines) if re.search(r"DialogV2(?:\$\d)?\.(confirm|prompt|input|wait)\s*\(", l)]
# 另有以 static DEFAULT_CONFIG.dialog 声明的（EmberSwitch 系列）
decl = [i for i, l in enumerate(lines) if re.match(r"\s*dialog:\s*\{\s*$", l)]
hits = 0
for i in dlg:
    b = block_from(i)
    mt = re.search(r'title:\s*[`"]([^`"]+)[`"]', b)
    title = mt.group(1) if mt else "<变量>"
    # 正文与按钮里的英文字面量
    eng = []
    for m in re.finditer(r'content:\s*[`"]([^`"]{6,})[`"]', b):
        eng.append(("content", m.group(1)))
    for m in re.finditer(r'label:\s*[`"]([^`"]+)[`"]', b):
        eng.append(("button", m.group(1)))
    eng = [(k, v) for k, v in eng if re.search(r"[A-Za-z]{3}", v) and "${" not in v.split("}")[0][:0] or True]
    eng = [(k, v) for k, v in eng if re.search(r"[A-Za-z]{3}", v)]
    if not eng:
        continue
    tstat = "标题已翻" if covered(title) else "标题也未翻"
    hits += 1
    print(f"\n  ember.mjs:{i+1}  title={title!r}  [{tstat}]")
    for k, v in eng:
        print(f"      {k:8s} {v[:110]}")
for i in decl:
    b = block_from(i - 0) if "(" in lines[i] else "\n".join(lines[i:i + 14])
    mt = re.search(r'title:\s*"([^"]+)"', b)
    if not mt:
        continue
    labels = re.findall(r'label:\s*"([^"]+)"', b)
    content = re.findall(r'content:\s*"([^"]+)"', b)
    if not labels and not content:
        continue
    hits += 1
    print(f"\n  ember.mjs:{i+1}  DEFAULT_CONFIG.dialog title={mt.group(1)!r}  "
          f"[{'标题已翻' if covered(mt.group(1)) else '标题也未翻'}]")
    for v in content:
        print(f"      content  {v[:110]}")
    for v in labels:
        print(f"      button   {v}")
print(f"\n  → DialogV2 调用点 {len(dlg)} 个 + DEFAULT_CONFIG.dialog {len(decl)} 个，"
      f"其中带英文正文/按钮的 {hits} 个")

print()
print("=" * 78)
print("B. ember 注入外部宿主 App 的 DOM —— G-render 只看根元素类名，一律挡死")
print("=" * 78)
for i, l in enumerate(lines):
    m = re.search(r'(?:Hooks\.on|Hooks\.once|registerHook)\(\s*"(render[A-Za-z0-9_]+)"', l)
    if not m:
        continue
    hook = m.group(1)
    host = hook[len("render"):]
    verdict = "PASS(宿主是 Ember 自己的类)" if host.startswith("Ember") else "BLOCKED(外部宿主)"
    print(f"  ember.mjs:{i+1:6d}  {hook:42s} {verdict}")
print("  另：getSceneControlButtons 也是注入外部宿主（SceneControls）")
for i, l in enumerate(lines):
    if 'getSceneControlButtons' in l and ('Hooks.on' in l or 'registerHook' in l):
        print(f"  ember.mjs:{i+1:6d}  getSceneControlButtons")

print()
print("=" * 78)
print("C. ember 写进 crucible.CONFIG 的组 vs G-config 白名单 [languages, knowledge]")
print("=" * 78)
WHITELIST = {"languages", "knowledge"}
seen = {}
for i, l in enumerate(lines):
    for m in re.finditer(r"crucible\.CONFIG\.([A-Za-z0-9_]+)", l):
        g = m.group(1)
        if "=" not in l and "Object.assign" not in l:
            continue          # 只看写入点
        seen.setdefault(g, []).append(i + 1)
for g, ls in sorted(seen.items()):
    mark = "白名单内" if g in WHITELIST else "★ 白名单外"
    print(f"  crucible.CONFIG.{g:22s} 写入行 {ls}   {mark}")

print()
print("=" * 78)
print("D. 完全不经过 Application 渲染钩子的界面")
print("=" * 78)
lit = []
for i, l in enumerate(lines):
    m = re.search(r'ui\.notifications\.(info|warn|error)\(\s*([`"])', l)
    if m:
        s = re.search(r'ui\.notifications\.(?:info|warn|error)\(\s*[`"]([^`"]+)', l)
        if s and re.search(r"[A-Za-z]{3}", s.group(1)) and not s.group(1).startswith("EMBER."):
            lit.append((i + 1, s.group(1)))
print(f"  ui.notifications 字面量英文 {len(lit)} 条（v14 的 Notifications 不是 Application，无任何 render 钩子）")
for ln, s in lit[:12]:
    print(f"     ember.mjs:{ln}: {s[:100]}")
print(f"     ...（共 {len(lit)} 条）")

print()
chat = [i + 1 for i, l in enumerate(lines) if re.search(r'ChatMessage"?\)?\.create\(|cls\.create\(\{', l)]
print(f"  ChatMessage 创建点 {len(chat)} 个：{chat}"
      f"\n     （聊天卡片只走 renderChatMessageHTML，本模块不监听该钩子）")
