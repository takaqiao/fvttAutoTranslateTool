# -*- coding: utf-8 -*-
"""
探针：死补丁 / 够不到的规则 —— 把「日历月名」那一条抽象成可机械化判据
=====================================================================

判据（两个方向，互为镜像）：

  D1 「我们写了规则，上游没有对应的东西」
     —— 替换表里的键 / 补丁指向的成员 / CSS 选择器 / i18n 键，
        在**当前上游版本**里根本不出现 → 这条规则永远不会触发。

  D2 「上游会拼出一个界面字符串，我们没有任何规则能吃掉它」
     —— 上游 JS 里 `x.innerHTML = \`Prefix: ${...}\`` 这类**运行时拼串**，
        babele 与 i18n 都够不到，只能靠 ember-hardcoded-cn.mjs 的
        translateText 兜。凡是 translateText 原样返回的，就是漏网。

  D1 与 D2 是同一个错误的两面：**对「上游到底会显示什么」的判断是错的**。
     日历那条是 D1（表存在、上游不读）；本探针额外覆盖 D2。

做法：
  1. 从 ember-hardcoded-cn.mjs 里解析出 EXACT / PREFIXED / PATTERNS / 各张表，
     用 python 复刻 translateText 的匹配顺序（顺序必须与 JS 一致，否则结论不可复现）。
  2. 从 ember.mjs 里正则抓出所有「带英文字面前缀的运行时拼串」：
        innerHTML / innerText / textContent = `...${...}...`
        title: `...`  /  label: "..."  的模板字面量
  3. 用一个占位中文替换 `${...}`，喂给复刻的 translateText。
  4. 原样返回 && 串里含英文单词 → 候选漏网。

假阳性模式（必须知道，判定要人工做）：
  - 抓到的串可能只在 dnd5e 分支执行（`game.system.id === "dnd5e"`），
    Crucible 世界永远不显示 → 不是漏网。探针会把命中行的上下文一并打印。
  - 抓到的串可能是内部数据（日志、className、data-* 值），不上屏。
  - 占位符替换会改变文本，`^...$` 类正则可能因此不匹配 → 需人工确认。
  - 上游还可能通过别的通道（system 特定实现）覆写，探针看不见。

只读；不写库。
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MOD = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"

src = open(MOD, encoding="utf-8").read()
ember = open(EMBER, encoding="utf-8").read()
elines = ember.split("\n")


# ---------------------------------------------------------------- 解析我们的规则集
def parse_table(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src, re.S)
    if not m:
        return {}
    body = m.group(1)
    out = {}
    for k, v in re.findall(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"', body):
        out[k.replace('\\"', '"')] = v
    return out


EXACT = parse_table("EXACT")
ATTUNEMENTS = parse_table("ATTUNEMENTS")
LANGUAGES = parse_table("LANGUAGES")
KNOWLEDGE = parse_table("KNOWLEDGE")
MOODS = parse_table("MOODS")
RESULTS = parse_table("RESULTS")

PREFIXED = [(en, cn) for en, cn in re.findall(r'\{\s*en:\s*"([^"]+)",\s*cn:\s*"([^"]+)"', src)]
PATTERN_RES = re.findall(r"\{\s*re:\s*/(.+?)/,\s*cn:", src)

print("规则集：EXACT %d  ATTUNEMENTS %d  LANGUAGES %d  KNOWLEDGE %d  MOODS %d  RESULTS %d  PREFIXED %d  PATTERNS %d"
      % (len(EXACT), len(ATTUNEMENTS), len(LANGUAGES), len(KNOWLEDGE), len(MOODS), len(RESULTS),
         len(PREFIXED), len(PATTERN_RES)))
print("PREFIXED 前缀：", [p[0] for p in PREFIXED])
print("PATTERNS：", PATTERN_RES)


def js_re_to_py(r):
    return r.replace("(?<", "(?P<")


COMPILED = [re.compile(js_re_to_py(r)) for r in PATTERN_RES]


def translate_text(text):
    """复刻 ember-hardcoded-cn.mjs 的 translateText 匹配顺序。"""
    raw = text.strip()
    if not raw:
        return text
    if raw in EXACT:
        return text.replace(raw, EXACT[raw])
    for en, cn in PREFIXED:
        if raw.startswith(en + ": "):
            return text.replace(raw, cn + "：" + raw[len(en) + 2:])
    for rx in COMPILED:
        if rx.match(raw):
            return "<PATTERN>"
    return text


# ---------------------------------------------------------------- D2：上游拼串
# innerHTML / innerText / textContent 赋值，或 title:/label: 的模板字面量
ASSIGN = re.compile(r"(innerHTML|innerText|textContent)\s*=\s*`([^`]{2,160})`")
TITLED = re.compile(r"\btitle:\s*`([^`]{2,160})`")

cands = {}
for i, line in enumerate(elines, 1):
    for m in ASSIGN.finditer(line):
        cands.setdefault(m.group(2), []).append(i)
    for m in TITLED.finditer(line):
        cands.setdefault(m.group(1), []).append(i)

print("\n抓到运行时拼串 %d 条（去重后）" % len(cands))

PLACEHOLDER = "\u4e2d\u6587"          # 中文
WORD = re.compile(r"[A-Za-z]{3,}")
HTMLTAG = re.compile(r"<[a-zA-Z/][^>]*>")

uncovered = []
for tpl, lines in sorted(cands.items(), key=lambda kv: kv[1][0]):
    # 去掉 HTML 标签，只留可见文本
    visible = HTMLTAG.sub("", tpl)
    probe = re.sub(r"\$\{[^}]*\}", PLACEHOLDER, visible).strip()
    if not probe or not WORD.search(probe):
        continue
    out = translate_text(probe)
    if out == probe:
        uncovered.append((lines[0], probe, tpl, lines))

print("\n===== D2 候选：translateText 吃不下的可见拼串 %d 条 =====" % len(uncovered))
for ln, probe, tpl, lines in uncovered:
    ctx = ""
    lo = max(0, ln - 6)
    seg = "\n".join(elines[lo:ln + 2])
    if 'system.id === "dnd5e"' in seg:
        ctx = "  [附近有 dnd5e 分支]"
    if 'system.id === "crucible"' in seg:
        ctx += "  [附近有 crucible 分支]"
    print(f"L{ln}{ctx}\n    模板: {tpl[:150]}\n    探针: {probe[:150]}\n    出现行: {lines[:6]}")

# ---------------------------------------------------------------- D1：我们有键、上游无字面量
print("\n===== D1：表里的英文键在 ember.mjs 里搜不到字面量 =====")
for name, tbl in [("EXACT", EXACT), ("ATTUNEMENTS", ATTUNEMENTS), ("LANGUAGES", LANGUAGES),
                  ("KNOWLEDGE", KNOWLEDGE), ("MOODS", MOODS), ("RESULTS", RESULTS)]:
    miss = [k for k in tbl if k not in ember]
    print(f"{name}: {len(tbl)} 键，其中 {len(miss)} 个在 ember.mjs 里没有字面量 -> {miss}")

# ---------------------------------------------------------------- D3：内容里调用的增强器 id 上游注册了吗
#
# 「表里补了键，但上游根本走不到那一步」的另一种形态：
# 内容里写着 `[[/language borel]]`，而 borel 不在 crucible.CONFIG.languages 里，
# 增强器 `if (!language) return new Text(match)` 直接原样吐回 —— 屏幕上是markup 本身，
# 不是英文词，任何「英文残留」判据都抓不到，任何替换表也补不上。
import glob

CRUC = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"
cruc = open(CRUC, encoding="utf-8").read()

def registry_keys(text, anchor):
    m = re.search(anchor + r"\s*=\s*\{(.*?)\n\};", text, re.S)
    if not m:
        return set()
    return set(re.findall(r"^\s{2,4}(\w+):\s*\{", m.group(1), re.M))

lang_ids = registry_keys(cruc, r"const LANGUAGES")
know_ids = registry_keys(cruc, r"const DEFAULT_KNOWLEDGE")
# ember 增补
m = re.search(r"Object\.assign\(crucible\.CONFIG\.languages,\s*\{(.*?)\n  \}\);", ember, re.S)
if m:
    lang_ids |= set(re.findall(r"^\s+(\w+):\s*\{", m.group(1), re.M))
m = re.search(r"Object\.assign\(crucible\.CONFIG\.knowledge,\s*\{(.*?)\n  \}\);", ember, re.S)
if m:
    know_ids |= set(re.findall(r"^\s+(\w+):\s*\{", m.group(1), re.M))
know_ids.discard("outsiders")   # ember 明确 delete 掉

print("\n===== D3：内容调用的增强器 id vs 上游注册表 =====")
print(f"上游 languages 注册 {len(lang_ids)} 个；knowledge {len(know_ids)} 个")
CN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium\cn"
for kind, ids in (("language", lang_ids), ("knowledge", know_ids)):
    used = {}
    for f in glob.glob(os.path.join(CN, "*.json")):
        for tok in re.findall(r"\[\[/%s ([^\]]+)\]\]" % kind, open(f, encoding="utf-8").read()):
            used[tok] = used.get(tok, 0) + 1
    bad = {k: v for k, v in used.items() if k not in ids}
    print(f"  [[/{kind} …]] 用到 {len(used)} 个 id，其中 {len(bad)} 个上游没注册 -> {bad}")
    # \w+ 之外的字符：连增强器的正则都匹配不上
    nonword = {k: v for k, v in used.items() if not re.fullmatch(r"\w+", k)}
    print(f"      其中 {len(nonword)} 个含 \w 之外的字符（连 pattern 都不匹配）-> {nonword}")
