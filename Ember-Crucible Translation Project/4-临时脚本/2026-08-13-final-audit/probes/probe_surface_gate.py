# -*- coding: utf-8 -*-
"""
probe_surface_gate.py  —  「闸/选择器失配：替换层入口够不到整个界面面」的机械化判据

背景实例（已被报告，本探针不再重复）：
    ui.notifications 不是 Application → renderApplicationV2/renderApplication 都不触发
    → ember-hardcoded-cn.mjs 唯一的运行时入口结构上够不到这一面。

抽象成判据：
    一条**字面英文**（literal，不是 i18n key）要在中文世界里被译出来，必须至少落进
    本项目三层替换层之一：
      L1 babele      —— 只覆盖 compendium 文档字段
      L2 Foundry i18n—— 只覆盖 game.i18n.localize/format 取的 key（含 Foundry 自行本地化的
                        title/label 字段）；字面英文**天然不在**这一层
      L3 ember-hardcoded-cn.mjs 的 DOM 遍历
                     —— 入口只有 Hooks.on("renderApplicationV2"|"renderApplication")，
                        且入口内还有一道闸：
                            root.className 含 "ember"  或  app.constructor.name 以 "Ember" 开头
                        否则只在 DialogV2/.dialog 这一档翻 `.window-title`，其余直接 return。

    因此判据 = 找出满足下面全部条件的上游字面英文：
      (a) 是 literal，不是 i18n key（不匹配 ^[A-Z0-9_.]+$ 这类 key 形态，且不在
          _loc()/game.i18n.localize()/format() 的第一个实参位置）
      (b) 会进入玩家可见 DOM / 画布
      (c) 它所在的「面」满足下列任一：
          C1 宿主根本不是 Application（Notifications / ContextMenu / 画布 PIXI / 聊天条目）
          C2 宿主是 Application 但根 class 不含 ember 且类名不以 Ember 开头（闸拒绝）
          C3 宿主是 DialogV2（闸只放行 `.window-title`，正文与按钮够不到）
          C4 宿主是 crucible 侧的任何界面（crucible-cn 仓库里**没有任何运行时替换脚本**，
             L3 这一层在 crucible 侧整体不存在）

假阳性模式（必须人工复核，本脚本只出候选）：
  - 英文字面量其实是 CSS 类名 / 选择器 / 文件路径 / 数据键 / 控制台日志 / 注释里的英文
  - 英文字面量是 dnd5e 侧（ember.adventure）内容 —— 项目已定「先不管」
  - 英文字面量在 GM 永不可见的死代码分支（如 _activateDevelopmentModeHooks）
  - Handlebars 模板里 {{localize}} 包着的不是字面量

只读，不写任何库文件。
"""
import json
import os
import re
import sys

FOUNDRY_APP = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
EMBER = os.path.join(DATA, "modules", "ember")
CRUCIBLE = os.path.join(DATA, "systems", "crucible")
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- helpers

WORDY = re.compile(r"[A-Za-z]{2,}")
# 看着像界面文案：至少两个英文单词，或一个首字母大写的词且长度>=4
def looks_like_ui_text(s):
    s = s.strip()
    if not s or len(s) < 3 or len(s) > 300:
        return False
    if not WORDY.search(s):
        return False
    # 排除路径 / 选择器 / 模板路径 / URL / 数据键
    if re.search(r"[/\\]|^\.|^#|^&|^\{\{|^https?:|\.hbs$|\.svg$|\.webp$|\.png$|\.json$|\.mjs$", s):
        return False
    # 排除 i18n key 形态：CRUCIBLE.X.Y / EMBER.X / DICE.Foo / ACTOR.CONTROLS.Bar
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(\.[A-Za-z0-9_]+)+", s):
        return False
    # 排除 CSS / fa 图标 / 单个小写标识符 / camelCase 标识符
    if re.fullmatch(r"[a-z][a-zA-Z0-9]*", s):
        return False
    if s.startswith("fa-") or s.startswith("--"):
        return False
    if re.fullmatch(r"[a-z0-9-]+", s):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z']*", s)
    if len(words) >= 2:
        return True
    return bool(re.fullmatch(r"[A-Z][a-z]{3,}", s))


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def line_of(text, idx):
    return text.count("\n", 0, idx) + 1


STR_RE = re.compile(r"""(?P<q>['"`])(?P<v>(?:\\.|(?!(?P=q))[^\\])*)(?P=q)""", re.S)


def strings_in(snippet):
    for m in STR_RE.finditer(snippet):
        yield m.group("v")


# ---------------------------------------------------------------- sinks

def scan_call_sink(text, pattern, span=400):
    """找到 pattern 的每次出现，取其后 span 个字符里的字符串字面量。"""
    out = []
    for m in re.finditer(pattern, text):
        seg = text[m.start(): m.start() + span]
        # 收缩到括号平衡处，避免吃到下一句
        depth = 0
        end = len(seg)
        started = False
        for i, ch in enumerate(seg):
            if ch == "(":
                depth += 1
                started = True
            elif ch == ")":
                depth -= 1
                if started and depth == 0:
                    end = i + 1
                    break
        seg = seg[:end]
        out.append((line_of(text, m.start()), m.group(0), seg))
    return out


def report(title, rows, fh):
    fh.write("\n" + "=" * 78 + "\n" + title + f"   （{len(rows)} 条候选）\n" + "=" * 78 + "\n")
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    outpath = os.path.join(OUT, "surface_gate_candidates.txt")
    fh = open(outpath, "w", encoding="utf-8")

    ember_js = os.path.join(EMBER, "scripts", "ember.mjs")
    cruc_js = os.path.join(CRUCIBLE, "crucible-compiled.mjs")
    ember_t = read(ember_js)
    cruc_t = read(cruc_js)

    # ---- C1a: ContextMenu 条目（ContextMenu 类无任何 Hooks.call） ----
    rows = []
    for src, t in (("ember", ember_t), ("crucible", cruc_t)):
        for ln, head, seg in scan_call_sink(t, r"(getSceneContextOptions|ContextOptions|ContextMenu)\b", 900):
            lits = [s for s in strings_in(seg) if looks_like_ui_text(s)]
            if lits:
                rows.append({"src": src, "line": ln, "head": head, "literals": lits[:8]})
    report("C1a  ContextMenu（非 Application，零 Hooks）里的字面英文", rows, fh)

    # ---- C1b: 聊天条目（renderChatMessageHTML 面；ChatLog 根 class 也过不了闸） ----
    rows = []
    for src, t in (("ember", ember_t), ("crucible", cruc_t)):
        for ln, head, seg in scan_call_sink(t, r"ChatMessage(\.implementation)?\.create|renderGroupCheckCard|createChatMessage", 1200):
            lits = [s for s in strings_in(seg) if looks_like_ui_text(s)]
            if lits:
                rows.append({"src": src, "line": ln, "head": head, "literals": lits[:8]})
    report("C1b  聊天消息内容里的字面英文", rows, fh)

    # ---- C1c: 画布 PIXI 文本 ----
    rows = []
    for src, t in (("ember", ember_t), ("crucible", cruc_t)):
        for ln, head, seg in scan_call_sink(t, r"PreciseText|new PIXI\.Text|createScrollingText", 500):
            lits = [s for s in strings_in(seg) if looks_like_ui_text(s)]
            if lits:
                rows.append({"src": src, "line": ln, "head": head, "literals": lits[:8]})
    report("C1c  画布 PIXI 文本里的字面英文", rows, fh)

    # ---- C2: 往**核心/非 Ember 应用**里注入 DOM（闸按宿主根元素判，必然拒绝） ----
    # 先找出 ember 注册在核心应用 render 钩子上的处理器
    core_render_hooks = re.findall(r'Hooks\.on\("(render[A-Za-z]+)"', ember_t)
    rows = [{"hook": h} for h in sorted(set(core_render_hooks))]
    report("C2-0  ember 注册的所有 render* 钩子（宿主是不是 Ember 自己的应用，需逐个判）", rows, fh)

    # 注入型 DOM 构造：innerHTML / insertAdjacentHTML / createElement + textContent
    rows = []
    for ln, head, seg in scan_call_sink(ember_t, r"\.innerHTML\s*=|insertAdjacentHTML\(", 900):
        lits = [s for s in strings_in(seg) if looks_like_ui_text(s)]
        # 只留下 HTML 片段里带可见文本的
        keep = []
        for s in lits:
            vis = re.sub(r"<[^>]*>", " ", s)
            vis = re.sub(r"\$\{[^}]*\}", " ", vis)
            for chunk in re.split(r"\s{2,}|\n", vis):
                chunk = chunk.strip()
                if looks_like_ui_text(chunk):
                    keep.append(chunk)
        if keep:
            rows.append({"src": "ember", "line": ln, "literals": sorted(set(keep))[:10]})
    report("C2-1  ember 用 innerHTML/insertAdjacentHTML 注入的可见字面英文", rows, fh)

    # ---- C3: DialogV2 —— 闸只翻 .window-title，正文/按钮够不到 ----
    rows = []
    for src, t in (("ember", ember_t), ("crucible", cruc_t)):
        for ln, head, seg in scan_call_sink(t, r"DialogV2\$?\d*\.(prompt|confirm|wait|input|query)\s*\(", 2000):
            # 分别抽 content 与按钮 label
            body = []
            mc = re.search(r"content\s*:\s*", seg)
            if mc:
                sub = seg[mc.end(): mc.end() + 900]
                for s in strings_in(sub):
                    vis = re.sub(r"<[^>]*>", " ", s)
                    vis = re.sub(r"\$\{[^}]*\}", " ", vis)
                    for chunk in re.split(r"\s{2,}|\n", vis):
                        chunk = chunk.strip()
                        if looks_like_ui_text(chunk):
                            body.append(chunk)
            labels = []
            for lm in re.finditer(r"label\s*:\s*(['\"`])((?:\\.|(?!\1)[^\\])*)\1", seg):
                v = lm.group(2)
                if looks_like_ui_text(v):
                    labels.append(v)
            titles = []
            for tm in re.finditer(r"title\s*:\s*(['\"`])((?:\\.|(?!\1)[^\\])*)\1", seg):
                v = tm.group(2)
                if looks_like_ui_text(v):
                    titles.append(v)
            if body or labels or titles:
                rows.append({"src": src, "line": ln, "title": sorted(set(titles)),
                             "body": sorted(set(body))[:8], "buttons": sorted(set(labels))[:8]})
    report("C3  DialogV2 正文/按钮里的字面英文（闸只翻 window-title）", rows, fh)

    # ---- C4: crucible 侧整体（crucible-cn 无任何运行时替换脚本） ----
    # 4a: crucible 的 .hbs 模板里的可见字面英文
    rows = []
    tdir = os.path.join(CRUCIBLE, "templates")
    for root, _d, files in os.walk(tdir):
        for fn in files:
            if not fn.endswith(".hbs"):
                continue
            p = os.path.join(root, fn)
            t = read(p)
            # 去掉 handlebars 表达式与标签，留下裸文本节点
            stripped = re.sub(r"\{\{[^}]*\}\}", " ", t)
            stripped = re.sub(r"<[^>]*>", "\n", stripped)
            hits = []
            for chunk in stripped.split("\n"):
                chunk = chunk.strip()
                if looks_like_ui_text(chunk):
                    hits.append(chunk)
            # 属性里的 data-tooltip / placeholder / aria-label 字面量
            for am in re.finditer(r'(data-tooltip|placeholder|aria-label|title)\s*=\s*"([^"{}]+)"', t):
                if looks_like_ui_text(am.group(2)):
                    hits.append(f"[{am.group(1)}] {am.group(2)}")
            if hits:
                rows.append({"file": os.path.relpath(p, CRUCIBLE), "literals": sorted(set(hits))[:12]})
    report("C4a  crucible .hbs 模板里的可见字面英文", rows, fh)

    # 4b: ember 的 .hbs 模板（宿主是 Ember 应用 → 闸放行，但要看 DOM 遍历表里有没有）
    rows = []
    tdir = os.path.join(EMBER, "templates")
    for root, _d, files in os.walk(tdir):
        for fn in files:
            if not fn.endswith(".hbs"):
                continue
            p = os.path.join(root, fn)
            t = read(p)
            stripped = re.sub(r"\{\{[^}]*\}\}", " ", t)
            stripped = re.sub(r"<[^>]*>", "\n", stripped)
            hits = []
            for chunk in stripped.split("\n"):
                chunk = chunk.strip()
                if looks_like_ui_text(chunk):
                    hits.append(chunk)
            for am in re.finditer(r'(data-tooltip|placeholder|aria-label|title)\s*=\s*"([^"{}]+)"', t):
                if looks_like_ui_text(am.group(2)):
                    hits.append(f"[{am.group(1)}] {am.group(2)}")
            if hits:
                rows.append({"file": os.path.relpath(p, EMBER), "literals": sorted(set(hits))[:12]})
    report("C4b  ember .hbs 模板里的可见字面英文", rows, fh)

    fh.close()
    print("written:", outpath)


if __name__ == "__main__":
    main()
