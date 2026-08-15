# -*- coding: utf-8 -*-
"""
probe_gate_reach.py  —— 「闸/选择器失配」这一类的机械判据

种子实例：Ember 往 PlaylistDirectory 注入的音乐面板整块被 ember 闸挡死。
抽象成判据：

    Ember 在运行时产生的**可见英文文本**，落在 `patchRenderedApplications` 的
    DOM 遍历**够不到的宿主**里 → 永远是英文。

`patchRenderedApplications` 的可达域（ember-hardcoded-cn.mjs:445-475）只有两条：
  A. `renderApplicationV2` / `renderApplication` 触发，且根元素 className 含 "ember"
     或 app.constructor.name 以 "Ember" 开头  → translateNode(整棵子树)
  B. 同上两个钩子，root class 含 "dialog" 或类名 == "DialogV2"
     → **只译 `.window-title` 一个节点**，其余整棵子树不碰

于是三种「够不到」：
  M1  宿主根元素不带 ember 类、类名不以 Ember 开头（注入到外部宿主）      —— 种子实例
  M2  宿主进得来但选择器太窄（DialogV2 只译标题，body/按钮不译）
  M3  根本不触发 renderApplicationV2/renderApplication 的渲染通道
      （ChatMessage 走 renderChatMessageHTML；ui.notifications 逐条 append；
        Handlebars PART 模板由宿主自己渲染）

本脚本只读上游源码，不改本库任何文件。
"""
import json
import re
import sys
from pathlib import Path

EMBER = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")
SCRIPTS = EMBER / "scripts"
TEMPLATES = EMBER / "templates"
OUT = Path(__file__).with_name("gate_reach.json")

# ember.mjs 里 rollup 拼接出的模块分界（@module 注释）——用来区分 dnd5e 分支（本轮不管）
DND5E_RANGE = None  # 由 detect_ranges() 填


def read(p):
    return p.read_text(encoding="utf-8", errors="replace")


SRC = {p.name: read(p) for p in SCRIPTS.glob("*.mjs")}
LINES = {n: s.splitlines() for n, s in SRC.items()}


def lineno(text, idx):
    return text.count("\n", 0, idx) + 1


# ---------------------------------------------------------------- 英文判据
I18N_KEY = re.compile(r"^[A-Z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
PATHY = re.compile(r"^[\w./#\-\[\]=\"' :,>*]+$")  # css 选择器 / 文件路径
HAS_WORDS = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")
CJK = re.compile(r"[\u4e00-\u9fff]")


def is_visible_english(s: str) -> bool:
    """字面量看起来是给玩家看的英文句子/词组，而不是 i18n 键、选择器、路径。"""
    t = s.strip()
    if not t or CJK.search(t):
        return False
    if I18N_KEY.match(t):
        return False
    if t.startswith(("modules/", "systems/", "icons/", "fa-", "http", ".", "#")):
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    # 至少两个英文词，或者一个首字母大写的独立词（"Change" / "Active"）
    if HAS_WORDS.search(t):
        return not PATHY.match(t) or " " in t
    return bool(re.match(r"^[A-Z][a-z]+$", t))


# ---------------------------------------------------------------- 平衡取块
def balanced(text, start, open_ch="(", close_ch=")"):
    """从 text[start] 处的 open_ch 起取到配对的 close_ch，跳过字符串与模板串。"""
    i = start
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            q = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q:
                    break
                # 模板串里的 ${...} 允许嵌套引号，简单跳过
                if q == "`" and text[i] == "$" and i + 1 < n and text[i + 1] == "{":
                    d2 = 0
                    i += 1
                    while i < n:
                        if text[i] == "{":
                            d2 += 1
                        elif text[i] == "}":
                            d2 -= 1
                            if d2 == 0:
                                break
                        i += 1
                i += 1
            i += 1
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i
        i += 1
    return text[start:], n


# ---------------------------------------------------------------- M1/M2: DialogV2
DLG = re.compile(r"DialogV2\s*\.\s*(confirm|prompt|wait|query)\s*\(")


def scan_dialogs():
    out = []
    for name, text in SRC.items():
        for m in DLG.finditer(text):
            blk, _ = balanced(text, m.end() - 1)
            ln = lineno(text, m.start())
            classes = re.search(r"classes\s*:\s*\[([^\]]*)\]", blk)
            cls = classes.group(1) if classes else ""
            gated = "ember" in cls.lower()          # 含 ember → 闸放行、整树翻译
            # 抓 content / label / 各种按钮文案
            hits = []
            for key in ("content", "label", "title", "hint", "placeholder"):
                for lm in re.finditer(key + r"\s*:\s*([`\"'])", blk):
                    q = lm.group(1)
                    j = lm.end() - 1
                    k = j + 1
                    buf = []
                    while k < len(blk):
                        if blk[k] == "\\":
                            buf.append(blk[k:k + 2]); k += 2; continue
                        if blk[k] == q:
                            break
                        buf.append(blk[k]); k += 1
                    val = "".join(buf)
                    # 剥掉 ${...} 与 html 标签，只留纯文本
                    plain = re.sub(r"\$\{[^}]*\}", " ", val)
                    plain = re.sub(r"<[^>]+>", " ", plain)
                    plain = re.sub(r"\s+", " ", plain).strip()
                    if is_visible_english(plain):
                        hits.append({"key": key, "text": plain[:220]})
            if hits:
                out.append({
                    "mech": "M2" if not gated else "OK",
                    "file": name, "line": ln, "call": m.group(1),
                    "classes": cls, "gate_passes_whole_tree": gated,
                    "english": hits,
                })
    return out


# ---------------------------------------------------------------- M3: ui.notifications
NOTI = re.compile(r"ui\.notifications\s*\.\s*(info|warn|error|notify)\s*\(\s*([`\"'])")


def scan_notifications():
    out = []
    for name, text in SRC.items():
        for m in NOTI.finditer(text):
            q = m.group(2)
            k = m.end()
            buf = []
            while k < len(text):
                if text[k] == "\\":
                    buf.append(text[k:k + 2]); k += 2; continue
                if text[k] == q:
                    break
                buf.append(text[k]); k += 1
            val = "".join(buf)
            plain = re.sub(r"\$\{[^}]*\}", " ", val)
            plain = re.sub(r"\s+", " ", plain).strip()
            # `{localize: true}` / i18n 键的不算
            tail = text[k:k + 80]
            if I18N_KEY.match(val.strip()) or "localize" in tail:
                continue
            if is_visible_english(plain):
                out.append({"mech": "M3", "file": name, "line": lineno(text, m.start()),
                            "level": m.group(1), "text": plain[:200]})
    return out


# ---------------------------------------------------------------- M3: ChatMessage
CHAT = re.compile(r'(getDocumentClass\("ChatMessage"\)|ChatMessage)\s*\.\s*create\s*\(')


def scan_chat():
    out = []
    for name, text in SRC.items():
        for m in CHAT.finditer(text):
            ln = lineno(text, m.start())
            # 往前 60 行找 content 的定义
            start = max(0, m.start() - 6000)
            ctx = text[start:m.end()]
            eng = []
            for lm in re.finditer(r"`([^`]{0,1500})`", ctx):
                val = lm.group(1)
                if "<" not in val and "${" not in val:
                    continue
                plain = re.sub(r"\$\{[^}]*\}", " ", val)
                plain = re.sub(r"<[^>]+>", " ", plain)
                plain = re.sub(r"\s+", " ", plain).strip()
                if is_visible_english(plain):
                    eng.append(plain[:200])
            out.append({"mech": "M3", "file": name, "line": ln, "english_in_ctx": eng[-3:]})
    return out


# ---------------------------------------------------------------- M1: 注入外部宿主的 PART / 模板
def scan_foreign_parts():
    out = []
    for name, text in SRC.items():
        for m in re.finditer(r"(\w+)\.PARTS\.(\w+)\s*=\s*\{([^}]*)\}", text):
            tpl = re.search(r'template\s*:\s*"([^"]+)"', m.group(3))
            out.append({"mech": "M1", "file": name, "line": lineno(text, m.start()),
                        "host_class_expr": m.group(1), "part": m.group(2),
                        "template": tpl.group(1) if tpl else None})
    return out


# ---------------------------------------------------------------- 模板里的硬编码英文
TAGTEXT = re.compile(r">([^<>{}]+)<")
ATTRTEXT = re.compile(r'(aria-label|data-tooltip|placeholder|title|alt)\s*=\s*"([^"{}]+)"')


def scan_templates():
    out = []
    for p in sorted(TEMPLATES.rglob("*.hbs")) + sorted(TEMPLATES.rglob("*.html")):
        t = read(p)
        hits = []
        for m in TAGTEXT.finditer(t):
            s = m.group(1).strip()
            if is_visible_english(s) or re.match(r"^[A-Z][a-z]+( [A-Za-z]+)*$", s):
                hits.append({"kind": "text", "text": s[:120],
                             "line": t.count("\n", 0, m.start()) + 1})
        for m in ATTRTEXT.finditer(t):
            s = m.group(2).strip()
            if is_visible_english(s) or re.match(r"^[A-Z][a-z]+( [A-Za-z]+)*$", s):
                hits.append({"kind": m.group(1), "text": s[:120],
                             "line": t.count("\n", 0, m.start()) + 1})
        if hits:
            out.append({"template": str(p.relative_to(EMBER)).replace("\\", "/"),
                        "hits": hits})
    return out


def main():
    res = {
        "dialogs": scan_dialogs(),
        "notifications": scan_notifications(),
        "chat_messages": scan_chat(),
        "foreign_parts": scan_foreign_parts(),
        "templates": scan_templates(),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:16s} {len(v)}")
    print("->", OUT)


if __name__ == "__main__":
    main()
