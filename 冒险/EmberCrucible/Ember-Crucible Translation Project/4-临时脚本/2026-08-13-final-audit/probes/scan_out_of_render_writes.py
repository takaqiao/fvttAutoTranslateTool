# -*- coding: utf-8 -*-
"""
探针：目标字符串在渲染钩子之外被重写（钩子选错时机）
=====================================================

已确认实例（日历条 `Day N - HH:MM`）抽象出的判据：

    汉化模块对「硬编码英文」只有两条通道：
      (A) DOM 后处理  —— Hooks.on("renderApplicationV2"/"renderApplication") 之后
          translateNode(root) 递归改文本节点与 tooltip 属性；
      (B) 一次性数据补丁 —— ready 时改 crucible.CONFIG / CONFIG.time.worldCalendarConfig。

    两条通道都只在**某一时刻**生效。凡是上游在那一时刻**之后**还会把同一处
    重新写一遍的代码路径，汉化就会被冲掉。

    机械化判据（本脚本）：
      1. 枚举上游 ember.mjs 里所有**命令式 DOM 写**点
         （innerText / textContent / innerHTML / outerHTML / insertAdjacentHTML /
          replaceChildren / setAttribute / dataset.tooltip* / .title= / .ariaLabel=）；
      2. 用括号配平定位每个写点的**外层方法**；
      3. 若外层方法属于 ApplicationV2 渲染生命周期
         （_renderHTML/_replaceHTML/_onRender/_onFirstRender/_prepareContext/_preparePartContext
          /_preSyncPartState/_syncPartState/_attachPartListeners/_attachFrameListeners），
         则 renderApplicationV2 钩子会在其后触发 —— translateNode 覆盖得到，判为 SAFE；
         否则判为 OUT-OF-RENDER 候选；
      4. 对每个候选，抓写入表达式里可能出现的英文串，与汉化表的目标串取交集。

假阳性模式（必须人工核实）：
  * 写入的内容是纯数值/样式/图标路径（style.top、img.src、CSS 类）—— 与文案无关；
  * 写入的内容来自 game.i18n.localize / 已被 babele 翻过的文档字段 —— 已经是中文；
  * 写点虽在生命周期之外，但该方法只在 _onRender 里被调用**一次**且随后就触发
    render 钩子（首帧路径）—— 需要看调用方；
  * 方法定位靠缩进+括号配平，压缩过的行会失准。

只读，不写库。
"""

import io
import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_MJS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CRUC_MJS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"
CN_HARDCODED = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")

LIFECYCLE = {
    "_renderHTML", "_replaceHTML", "_onRender", "_onFirstRender", "_prepareContext",
    "_preparePartContext", "_preSyncPartState", "_syncPartState",
    "_attachPartListeners", "_attachFrameListeners", "_renderFrame", "_onFirstRenderHTML",
}

WRITE_PATTERNS = [
    ("innerText", re.compile(r"\.innerText\s*=")),
    ("textContent", re.compile(r"\.textContent\s*=")),
    ("innerHTML", re.compile(r"\.innerHTML\s*=")),
    ("outerHTML", re.compile(r"\.outerHTML\s*=")),
    ("insertAdjacentHTML", re.compile(r"\.insertAdjacentHTML\s*\(")),
    ("replaceChildren", re.compile(r"\.replaceChildren\s*\(")),
    ("setAttribute", re.compile(r"\.setAttribute\s*\(")),
    ("dataset.tooltip", re.compile(r"\.dataset\.tooltip\w*\s*=")),
    ("elem.title", re.compile(r"\.title\s*=\s*[`\"']")),
    ("ariaLabel", re.compile(r"\.ariaLabel\s*=")),
    ("append/appendChild-text", re.compile(r"\.(append|appendChild|prepend)\s*\(\s*[`\"']")),
]

METHOD_RE = re.compile(
    r"^(\s*)(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?([#\w$]+)\s*\([^;]*\)\s*\{\s*$"
)
CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([\w$]+)")
FUNC_RE = re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?function\s+([\w$]+)\s*\(")


def load_targets():
    """从汉化模块提取所有目标英文串 / 目标前缀。"""
    src = io.open(CN_HARDCODED, encoding="utf-8").read()
    targets = set()
    prefixes = set()
    # EXACT / 各表的键
    for block in ("ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
                  "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"):
        m = re.search(r"const %s = \{(.*?)\n\};" % block, src, re.S)
        if not m:
            continue
        for k in re.findall(r'"([^"]+)"\s*:', m.group(1)):
            targets.add(k)
    # PREFIXED 的英文前缀
    for en in re.findall(r'\{\s*en:\s*"([^"]+)"', src):
        prefixes.add(en + ": ")
    # PATTERNS 的字面前缀
    for lit in ("Result of ", "Award Attunement: ", "Revoke Attunement: ",
                "Activate Attunement: ", "Day "):
        prefixes.add(lit)
    return targets, prefixes


def enclosing_context(lines, idx):
    """向上找最近的、缩进更浅的方法/函数定义；再向上找 class。"""
    # 当前行缩进
    cur = lines[idx]
    cur_indent = len(cur) - len(cur.lstrip())
    method = None
    method_line = None
    for j in range(idx - 1, max(-1, idx - 900), -1):
        line = lines[j]
        if not line.strip():
            continue
        m = METHOD_RE.match(line)
        if m and len(m.group(1)) < cur_indent:
            name = m.group(2)
            if name in ("if", "for", "while", "switch", "catch", "try", "else", "return",
                        "function", "do"):
                continue
            method = name
            method_line = j + 1
            break
        f = FUNC_RE.match(line)
        if f and len(f.group(1)) < cur_indent:
            method = f.group(2)
            method_line = j + 1
            break
    klass = None
    start = method_line - 1 if method_line else idx
    for j in range(start, max(-1, start - 4000), -1):
        c = CLASS_RE.match(lines[j])
        if c:
            klass = c.group(1)
            break
    return klass, method, method_line


def scan(path, label, targets, prefixes):
    lines = io.open(path, encoding="utf-8").read().split("\n")
    rows = []
    for i, line in enumerate(lines):
        kinds = [name for name, rx in WRITE_PATTERNS if rx.search(line)]
        if not kinds:
            continue
        klass, method, mline = enclosing_context(lines, i)
        safe = method in LIFECYCLE if method else False
        # 采集写入表达式附近 12 行内的英文字面串，供交集判断
        ctx = "\n".join(lines[max(0, i - 12): i + 4])
        hit_targets = sorted(t for t in targets if t and re.search(
            r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", ctx))
        hit_prefix = sorted(p for p in prefixes if p.strip() and p in ctx)
        rows.append({
            "file": label,
            "line": i + 1,
            "kinds": kinds,
            "class": klass,
            "method": method,
            "method_line": mline,
            "in_lifecycle": safe,
            "code": line.strip()[:220],
            "targets_nearby": hit_targets[:12],
            "prefix_nearby": hit_prefix[:12],
        })
    return rows


def main():
    targets, prefixes = load_targets()
    rows = scan(EMBER_MJS, "ember.mjs", targets, prefixes)
    if os.path.exists(CRUC_MJS) and "--crucible" in sys.argv:
        rows += scan(CRUC_MJS, "crucible-compiled.mjs", targets, prefixes)
    out = {
        "n_targets": len(targets),
        "n_prefixes": len(prefixes),
        "n_write_sites": len(rows),
        "n_out_of_render": sum(1 for r in rows if not r["in_lifecycle"]),
        "rows": rows,
    }
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_of_render_writes.json")
    io.open(dest, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("targets=%d prefixes=%d write_sites=%d out_of_render=%d" %
          (len(targets), len(prefixes), len(rows), out["n_out_of_render"]))
    print("->", dest)
    for r in rows:
        if r["in_lifecycle"]:
            continue
        print("-" * 70)
        print("%s:%d  [%s]  %s#%s (def@%s)" % (r["file"], r["line"], ",".join(r["kinds"]),
                                               r["class"], r["method"], r["method_line"]))
        print("   ", r["code"])
        if r["targets_nearby"]:
            print("    targets:", r["targets_nearby"])
        if r["prefix_nearby"]:
            print("    prefix :", r["prefix_nearby"])


main()

# ============================================================================
# 2026-08-13 人工核实结论（80 个 out-of-render 候选逐条过）
#
# 判为「同一类缺陷」的（本轮上报）：
#   ember.mjs:50994 EmberDynamicTokenConfig##refresh  —— await this.render() 之后写 tooltip
#   ember.mjs:24637 EmberCalendarNavigation##refreshWeather —— animate() 链路，天气/风 tooltip
#   ember.mjs:23750 HTMLCoefficientTagsElement#_refresh —— 值变化时重写 tag（附带，GM 编辑面）
#
# 判为假阳性的三大族（不上报）：
#   (1) 18 处是 enricher 函数体 —— 走通道 B（patchEnrichers 包返回值），已覆盖；
#       12 个 enricher 的 pattern 逐条核对全部命中过滤正则（ember.mjs:129405-129479 / 123660）。
#   (2) 11 处是 toEmbed / 正文 HTML —— 内容来自 compendium 字段，babele 已翻。
#   (3) 其余多为 i18n key（phaseLabel/EMBER.CALENDAR.*/EMBER.CODEX.*）、样式、图标路径、
#       dnd5e 侧（项目已定不管）。
#
# 另外核实并**排除**的两条最像但其实安全的路径：
#   * JournalEntrySheet 分页内容：_onRender 里 await _renderPageViews()，
#     _doEvent 在 handler resolve 之后才 dispatch（foundry.mjs:31588-31593），钩子在其后。
#   * DialogV2 标题：_updateFrame 在 _onRender 之前（foundry.mjs:30491），
#     <h1 class="window-title"> 无子元素（30754），CN 的 !children.length 判定成立。
#   * ember / crucible 均未 override _postRender —— Foundry 唯一在钩子之后跑的生命周期钩子是空的。
# ============================================================================
