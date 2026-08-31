# -*- coding: utf-8 -*-
r"""
探针 P-BATCH：「替换表缺键 —— 同一批字符串只补了一半」的全库扩查。

种子实例：EmberHeroCreationSheet.STEPS 的 7 个 label 里，ember 新加的四个是裸英文
（Culture/Path/Attunement/Token），汉化 EXACT 表只补了 Culture/Path 两个。

抽象成机械判据：
  在**上游**（ember 0.6.0 / crucible 0.10.1）里找「兄弟字面量组」——
  同一个对象/数组字面量（或同一段连续代码）里若干成员共享同一个界面用键名
  （label / name / title / header / tooltip / hint / legend / placeholder / group），
  且值是**裸英文显示文本**（不是 i18n key 形态 `FOO.Bar.Baz`）。
  对每一组算汉化侧覆盖率：
      覆盖 = 该英文串在 ember-hardcoded-cn.mjs 的任一替换表里有键
            （EXACT / ATTUNEMENTS / LANGUAGES / KNOWLEDGE / MOODS / RESULTS /
              CALENDAR_MONTHS / CALENDAR_DAYS / CALENDAR_DAY_ABBR），
            或被 PATTERNS 的某条正则吃掉，或形如 `<前缀>: X` 走 PREFIXED。
  **0 < 覆盖数 < 组内裸英文数** → PARTIAL（种子那一类）
  组内还混有 i18n key 成员的，i18n 那边由 lang/cn.json 负责，另算一列
  （种子实例正是「4 个裸英文里补了 2 个 + 3 个走 i18n key」，所以这一列要单独看）。

只读，不写任何仓库文件。

已知假阳性模式（人工复核时必须逐条排掉）：
  FP1 值虽是英文但不上屏（内部 id、文件名、CSS class、sound/arrangement id、
      compendium pack 名、字段 name 属性、flag 名）。
  FP2 该组属于 crucible 本体且 crucible-cn 走 babele/lang 覆盖，与 hardcoded 表无关。
  FP3 值会被别的通道翻掉（例如出现在 compendium document 的 name 上，babele 管）。
  FP4 组的划分靠缩进+行距启发式，可能把不相干的两段并成一组或拆开一组。
"""
import io, json, os, re, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CRUC_UP = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
HC = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")

# ---------------------------------------------------------------- 汉化侧覆盖集
src = open(HC, encoding="utf-8").read()


def table_keys(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src, re.S)
    if not m:
        return set()
    return set(re.findall(r'"([^"]+)":\s*"', m.group(1)))


TABLES = ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
          "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"]
COVER = set()
PER_TABLE = {}
for t in TABLES:
    k = table_keys(t)
    PER_TABLE[t] = k
    COVER |= k

PREFIXES = ["Attunement", "Language", "Knowledge", "Music Mood"]
PAT_RE = [re.compile(r"^Result of (.+)$"), re.compile(r"^Award Attunement: (.+)$"),
          re.compile(r"^Revoke Attunement: (.+)$"), re.compile(r"^Activate Attunement: (.+)$"),
          re.compile(r"^Day (\d+)\b(.*)$")]


def covered(s):
    s = s.strip()
    if s in COVER:
        return "EXACT-ish"
    for p in PREFIXES:
        if s.startswith(p + ": "):
            return "PREFIXED"
    for r in PAT_RE:
        if r.match(s):
            return "PATTERN"
    return None


# ---------------------------------------------------------------- 语料
def collect(root, exts):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if os.path.splitext(f)[1].lower() in exts:
                p = os.path.join(dp, f)
                try:
                    out.append((p, open(p, encoding="utf-8").read()))
                except Exception:
                    pass
    return out


corpus = collect(os.path.join(EMBER_UP, "scripts"), {".mjs"})
corpus += collect(os.path.join(EMBER_UP, "templates"), {".hbs"})
corpus_cruc = collect(CRUC_UP, {".mjs", ".hbs"})

# ---------------------------------------------------------------- 提取
KEYNAMES = r"(?:label|name|title|header|heading|tooltip|hint|legend|placeholder|group|caption|text)"
LINE_RE = re.compile(r'^(\s*)(?:static\s+)?["\']?(%s)["\']?\s*:\s*"([^"\\]{2,120})"' % KEYNAMES)
# i18n key 形态：全大写段或驼峰段用点连起来，且不含空格
I18N_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
# 看着像显示文本：首字母大写，只含字母/空格/常见标点
DISPLAY_RE = re.compile(r"^[A-Z][A-Za-z0-9'’\-\u00C0-\u024F]*(?:[ /&,:'’\-][A-Za-z0-9'’\-\u00C0-\u024F]+)*[?!.]?$")
# 明显不上屏的（id / 路径 / class）
SKIP_RE = re.compile(r"[/\\]|^[a-z]|\.(hbs|mjs|json|webp|png|svg|ogg|webm)$")


def scan(files, tag):
    rows = []
    for p, c in files:
        lines = c.split("\n")
        for i, ln in enumerate(lines):
            m = LINE_RE.match(ln)
            if not m:
                continue
            indent, key, val = m.group(1), m.group(2), m.group(3)
            kind = "i18n" if I18N_RE.match(val) else ("bare" if (DISPLAY_RE.match(val) and not SKIP_RE.search(val)) else "other")
            rows.append({"file": os.path.relpath(p, os.path.dirname(EMBER_UP if tag == "ember" else CRUC_UP)),
                         "line": i + 1, "indent": len(indent), "key": key, "val": val, "kind": kind,
                         "repo": tag})
    return rows


rows_e = scan(corpus, "ember")
rows_c = scan(corpus_cruc, "crucible")
print(f"语料：ember {len(corpus)} 文件 / crucible {len(corpus_cruc)} 文件")
print(f"抽到界面键行：ember {len(rows_e)} / crucible {len(rows_c)}")

# ---------------------------------------------------------------- 分组（同文件 + 同 key 名 + 行距 <= GAP）
GAP = 30


def group(rows):
    groups = []
    cur = []
    for r in rows:
        if cur and (r["file"] == cur[-1]["file"] and r["key"] == cur[-1]["key"]
                    and r["line"] - cur[-1]["line"] <= GAP):
            cur.append(r)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = [r]
    if len(cur) >= 2:
        groups.append(cur)
    return groups


gs = group(rows_e)
print(f"ember 兄弟组（>=2 成员）：{len(gs)}")

partial, none_cov, mixed = [], [], []
for g in gs:
    bare = [r for r in g if r["kind"] == "bare"]
    i18n = [r for r in g if r["kind"] == "i18n"]
    if not bare:
        continue
    cov = [r for r in bare if covered(r["val"])]
    rec = {"file": g[0]["file"], "key": g[0]["key"],
           "lines": f'{g[0]["line"]}-{g[-1]["line"]}',
           "n_bare": len(bare), "n_cov": len(cov), "n_i18n": len(i18n),
           "covered": [r["val"] for r in cov],
           "uncovered": [r["val"] for r in bare if not covered(r["val"])]}
    if 0 < len(cov) < len(bare):
        partial.append(rec)
    elif len(cov) == 0 and i18n:
        mixed.append(rec)          # 组里有 i18n 成员被 lang 翻掉，裸英文一个没补
    elif len(cov) == 0:
        none_cov.append(rec)

print(f"\n=== A. PARTIAL（组内裸英文部分有键、部分没键）{len(partial)} 组 ===")
for r in partial:
    print(f'  {r["file"]}:{r["lines"]} .{r["key"]}  裸英文 {r["n_bare"]} 有键 {r["n_cov"]} i18n {r["n_i18n"]}')
    print(f'     有键: {r["covered"]}')
    print(f'     缺键: {r["uncovered"]}')

print(f"\n=== B. MIXED（组内有 i18n 成员走 lang，裸英文成员一个都没补）{len(mixed)} 组 ===")
for r in mixed:
    print(f'  {r["file"]}:{r["lines"]} .{r["key"]}  裸英文 {r["n_bare"]} i18n {r["n_i18n"]}')
    print(f'     缺键: {r["uncovered"]}')

# ---------------------------------------------------------------- crucible.CONFIG 写入面
print("\n=== C. ember 往 crucible.CONFIG.<组> 里写 label 的分组 vs patchCrucibleConfig 覆盖 ===")
patched = re.search(r'for \(const \[key, table\] of \[(.*?)\]\)', src, re.S)
print("  patchCrucibleConfig 覆盖的组：", patched.group(1).strip() if patched else "??")
for p, c in corpus:
    for m in re.finditer(r'crucible\.CONFIG\.([A-Za-z]+)(?:\.([A-Za-z]+))?\s*(?:=|\))', c):
        pass
for p, c in corpus:
    for m in re.finditer(r'crucible\.CONFIG\.([A-Za-z]+)', c):
        pass
cfggroups = {}
for p, c in corpus:
    for m in re.finditer(r'crucible\.CONFIG\.([A-Za-z]+)((?:\.[A-Za-z]+)*)\s*(=|,)', c):
        grp = m.group(1)
        # 只看后面 400 字符里带裸英文 label 的
        seg = c[m.start():m.start() + 900]
        labs = re.findall(r'label:\s*"([^"]+)"', seg)
        if labs:
            cfggroups.setdefault(grp, set()).update(labs)
for grp, labs in sorted(cfggroups.items()):
    miss = sorted(l for l in labs if not covered(l))
    print(f'  crucible.CONFIG.{grp}: label 数 {len(labs)}  缺键 {len(miss)}')
    if miss:
        print("      缺:", miss[:20])

out = os.path.join(os.path.dirname(HC).replace(os.path.join("1-Ember汉化插件", "scripts"), ""),
                   "")
outp = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-final-audit\findings\p_batch_halfkey.json"
json.dump({"partial": partial, "mixed": mixed, "none": none_cov,
           "cfggroups": {k: sorted(v) for k, v in cfggroups.items()}},
          open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote", outp)
