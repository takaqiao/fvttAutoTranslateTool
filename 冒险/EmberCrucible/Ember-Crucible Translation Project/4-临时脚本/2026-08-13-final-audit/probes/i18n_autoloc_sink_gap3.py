# -*- coding: utf-8 -*-
r"""
probe: i18n_autoloc_sink_gap3 —— 收官版，**括号配平定域**（只读）

v1 属性名匹配 → 噪声 5336；v2 字符窗口定域 → 88 但窗口会越界。
v3 用括号/花括号配平，把每个自动本地化面的**参数对象**精确切出来，再取里面的字面量。

四个面（core v14 源码证据，路径 C:\Program Files\Foundry Virtual Tabletop\resources\app）：

  F1 game.settings.register(ns, key, {config, name, hint, type})
     client/applications/settings/config.mjs:126-127
       data.field.label ||= _loc(setting.name ?? "");
       data.field.hint  ||= _loc(setting.hint ?? "");
     注意 config.mjs:70 `if ( !setting.config ) continue;` —— config:false 的设置**不进设置面板**，
     必须剔除（本脚本自动标注 config 值）。
     另注意 `||=`：若 type 是自带 label 的 DataField，setting.name 会被忽略（本脚本标注 has_type_label）。

  F2 game.keybindings.register(ns, key, {name, hint})
     client/applications/sidebar/apps/controls-config.mjs:154  label: _loc(action.name)
     client/applications/sidebar/apps/controls-config.mjs:158  _loc(action.hint)

  F3 DialogV2 窗口标题与按钮
     client/applications/api/application.mjs:320  get title(){ return _loc(this.options.window.title); }
     client/applications/api/dialog.mjs:249       span.innerText = _loc(label);
     client/applications/api/dialog.mjs:240       button.setAttribute("aria-label", _loc(tooltip));

  F4 ApplicationV2 窗口头部控件
     client/applications/api/application.mjs:910  span.innerText = _loc(control.label);
     （DEFAULT_OPTIONS.window.controls[] / _getHeaderControls() 的返回）

差集口径与前两支一致：不在 core/crucible/ember 三张 en.json 拍平键里、不形如 A.B.C、
不在本项目 cn.json 顶层键 / ember-hardcoded-cn.mjs 查表键里 → 永远英文。

假阳性模式：
  · F1 的 config:false（脚本标注，人工剔）；
  · dnd5e 分支（ember.adventure）项目所有者已定「先不管」；
  · 死代码 / 开发者宏路径；
  · EXACT 表以外还有 PATTERNS 正则兜底，脚本没模拟正则，可能把已覆盖的报成缺口（人工核）。

只读，不写库。
"""
import io
import json
import os
import re
import sys

CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_autoloc_sink_gap3.json")

JS = {"crucible": os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs"),
      "ember": os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs")}
EN_JSONS = [os.path.join(CORE, "public", "lang", "en.json"),
            os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
            os.path.join(FVTT, "modules", "ember", "lang", "en.json")]
CN_JSONS = [os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
            os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json")]
CN_JS = [os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
         os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
         os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js")]


def flat(o, p=""):
    s = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = f"{p}.{k}" if p else k
            s.add(q)
            s |= flat(v, q)
    return s


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def balance(s, i, opens="({[", closes=")}]"):
    """从 s[i] 处的开括号开始，返回配平后的闭括号下标（含）。粗略跳过字符串/模板串。"""
    depth = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in "\"'`":
            q = c
            i += 1
            while i < n:
                if s[i] == "\\":
                    i += 2
                    continue
                if s[i] == q:
                    break
                i += 1
        elif c in opens:
            depth += 1
        elif c in closes:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return min(n - 1, i)


PROP = re.compile(r"(?<![\w$.])(label|title|tooltip|name|hint)\s*:\s*([\"'])((?:[^\"'\\\n]|\\.)*)\2")
KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$\-]+)+$")


def main():
    en = set()
    for p in EN_JSONS:
        en |= flat(json.load(io.open(p, encoding="utf-8")))
    cn = set()
    for p in CN_JSONS:
        cn |= flat(json.load(io.open(p, encoding="utf-8")))
    cnlit = set()
    for p in CN_JS:
        if os.path.exists(p):
            s = read(p)
            cnlit |= set(re.findall(r"[\"'`]((?:[^\"'`\\]|\\.){2,120}?)[\"'`]\s*:", s))
            for m in re.finditer(r"game\.i18n\.translations(?:\.([\w$]+)|\[[\"']([^\"']+)[\"']\])", s):
                cn.add(m.group(1) or m.group(2))

    def gap(t):
        t = (t or "").strip()
        if not t or len(t) > 110:
            return False
        if t in en or t in cn or t in cnlit:
            return False
        if KEYISH.match(t) or "${" in t:
            return False
        return bool(re.search(r"[A-Za-z]{2,}", t)) and bool(re.search(r"[A-Z]", t) or " " in t)

    out = {}
    for who, path in JS.items():
        s = read(path)
        rows = []

        def add(face, prop, txt, pos, extra=""):
            if not gap(txt):
                return
            rows.append({"face": face, "prop": prop, "text": txt,
                         "line": s[:pos].count("\n") + 1, "extra": extra,
                         "ctx": s[max(0, pos - 140):pos + 110].replace("\n", " ⏎ ")})

        # ---- F1 / F2 -------------------------------------------------------
        for m in re.finditer(r"game\.(settings\.register(?:Menu)?|keybindings\.register)\s*\(", s):
            j = s.index("(", m.start())
            k = balance(s, j)
            blk = s[j:k + 1]
            face = "F2 keybinding" if "keybindings" in m.group(1) else "F1 setting"
            cfg = re.search(r"(?<![\w$.])config\s*:\s*(true|false)", blk)
            typelab = "label:" in blk.split("type:", 1)[-1][:200] if "type:" in blk else False
            extra = f"config={cfg.group(1) if cfg else 'ABSENT(=false)'}"
            if typelab:
                extra += " typeHasLabel?"
            for pm in PROP.finditer(blk):
                if pm.group(1) in ("name", "hint", "label"):
                    add(face, pm.group(1), pm.group(3), j + pm.start(), extra)

        # ---- F3 DialogV2 ---------------------------------------------------
        for m in re.finditer(r"DialogV2[\w$]*\s*\.\s*(prompt|confirm|wait|input|query)\s*\(", s):
            j = s.index("(", m.end() - 1) if s[m.end() - 1] == "(" else s.index("(", m.start())
            k = balance(s, j)
            blk = s[j:k + 1]
            # window.title
            for wm in re.finditer(r"window\s*:\s*\{", blk):
                we = balance(blk, blk.index("{", wm.end() - 1))
                for pm in PROP.finditer(blk, wm.start(), we + 1):
                    if pm.group(1) == "title":
                        add("F3 dialog.title", "title", pm.group(3), j + pm.start(), m.group(1))
            # ok / buttons / yes / no
            for bm in re.finditer(r"(?<![\w$.])(ok|yes|no|buttons)\s*:\s*[\{\[]", blk):
                be = balance(blk, blk.index("{", bm.end() - 1) if blk[bm.end() - 1] == "{"
                             else blk.index("[", bm.end() - 1))
                for pm in PROP.finditer(blk, bm.start(), be + 1):
                    if pm.group(1) in ("label", "tooltip"):
                        add("F3 dialog.button", pm.group(1), pm.group(3), j + pm.start(), bm.group(1))

        # ---- F4 window.title / window.controls 在 DEFAULT_OPTIONS 等处 -------
        for wm in re.finditer(r"window\s*:\s*\{", s):
            we = balance(s, s.index("{", wm.end() - 1))
            seg = s[wm.start():we + 1]
            for pm in PROP.finditer(seg):
                if pm.group(1) == "title":
                    add("F4 window.title", "title", pm.group(3), wm.start() + pm.start())
            for cm in re.finditer(r"controls\s*:\s*\[", seg):
                ce = balance(seg, seg.index("[", cm.end() - 1))
                for pm in PROP.finditer(seg, cm.start(), ce + 1):
                    if pm.group(1) in ("label", "tooltip"):
                        add("F4 headerControl", pm.group(1), pm.group(3), wm.start() + pm.start())
        for gm in re.finditer(r"_getHeaderControls\s*\(\s*\)\s*\{", s):
            ge = balance(s, s.index("{", gm.end() - 1))
            for pm in PROP.finditer(s, gm.start(), ge + 1):
                if pm.group(1) in ("label", "tooltip"):
                    add("F4 headerControl", pm.group(1), pm.group(3), gm.start() + pm.start())

        # 去重
        seen = set()
        uniq = []
        for r in rows:
            k = (r["face"], r["prop"], r["text"], r["line"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        uniq.sort(key=lambda r: (r["face"], r["line"]))
        out[who] = uniq
        print("=" * 100)
        print(f"[{who}] 配平定域后的无键裸英文 = {len(uniq)}")
        for r in uniq:
            print(f"  {r['face']:<20} {r['prop']:<8} L{r['line']:<7} {r['text']!r:<58} {r['extra']}")
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
