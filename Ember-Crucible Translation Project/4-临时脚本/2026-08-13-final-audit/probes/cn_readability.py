# -*- coding: utf-8 -*-
"""只读探针 #2：纯中文可读性机械筛。不比对英文，只看中文本身。
每一类都在 FP_NOTE 里写明假阳性模式。
"""
import re, io, os, sys, json, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cn_corpus import load_all, plain, stratum, HAN

SENT_END = "。！？…"
CLAUSE_SEP = "，。！？；：、…—（）()《》〈〉「」“”‘’\n"

def sentences(t):
    buf = []
    cur = []
    for ch in t:
        cur.append(ch)
        if ch in SENT_END:
            buf.append("".join(cur)); cur = []
    if cur:
        buf.append("".join(cur))
    return [b.strip() for b in buf if b.strip()]

def clauses(t):
    return [c for c in re.split("[" + re.escape(CLAUSE_SEP) + "]", t) if c.strip()]

CHECKS = collections.OrderedDict()

def check(name, fpnote):
    def deco(fn):
        CHECKS[name] = (fn, fpnote)
        return fn
    return deco

# ---------------------------------------------------------------- 的的不休
@check("de_pileup", "并列长定语在中文里合法；『的』在专名内（如『魔法的』书名）也会计入。只报同一小句内 >=4 个且平均间距 <=6 字的。")
def c_de(t):
    hits = []
    for s in sentences(t):
        idx = [i for i, ch in enumerate(s) if ch == "的"]
        if len(idx) < 4:
            continue
        # 找最密的窗口
        for a in range(len(idx) - 3):
            b = a + 3
            span = idx[b] - idx[a]
            if span <= 24 and len(HAN.findall(s)) >= 12:
                hits.append(s)
                break
    return hits

# ---------------------------------------------------------------- 超长小句
@check("long_clause", "表格单元格、列表项、机械公式常无逗号但短；此处只看 >=45 汉字的无停顿小句。数字/宏被展平会略微拉长。")
def c_longclause(t):
    return [c for c in clauses(t) if len(HAN.findall(c)) >= 45]

# ---------------------------------------------------------------- 超长整句
@check("long_sentence", "叙述性长句在文学文本中可接受；只报 >=110 汉字的单句。")
def c_longsent(t):
    return [s for s in sentences(t) if len(HAN.findall(s)) >= 110]

# ---------------------------------------------------------------- 被字堆叠
@check("bei_pileup", "『被』在『被动』『被褥』『被称为』等词内也算；已排除这些词形。")
def c_bei(t):
    out = []
    for s in sentences(t):
        n = len(re.findall(r"被(?!动|褥|子|服|单)", s))
        if n >= 3:
            out.append(s)
    return out

# ---------------------------------------------------------------- 叠字
DUP = "的了是在和与或也都会被将对从而但这那它使把给让"
@check("dup_char", "『渐渐』『常常』等叠词合法，已限定在虚词集合内；跨句边界（上句末+下句首）会误报，已按小句切分。")
def c_dup(t):
    out = []
    for c in clauses(t):
        for ch in DUP:
            if ch + ch in c:
                out.append((ch, c))
    return out

# ---------------------------------------------------------------- 半角标点夹汉字
@check("halfwidth_punct", "英文缩写/单位/URL 相邻处会误报；已要求两侧都是汉字。")
def c_half(t):
    return re.findall(r"[一-鿿][,;:!?][一-鿿]", t) + \
           re.findall(r"[一-鿿]\.[一-鿿]", t)

# ---------------------------------------------------------------- 汉字间空格
@check("space_in_han", "中文人名『·』分隔、诗歌排版可能有意留空；宏展平后也可能引入空格，已在 plain() 中把宏替换成单空格，故此项只在原串上跑。")
def c_space(t):
    return re.findall(r"[一-鿿] [一-鿿]", t)

# ---------------------------------------------------------------- 标点前空格 / 重复标点
@check("punct_anomaly", "无。")
def c_punct(t):
    return re.findall(r"\s[，。；：！？、]", t) + re.findall(r"[，。；：]{2,}", t) + \
           re.findall(r"（\s*）|「\s*」", t)

# ---------------------------------------------------------------- 一个 密度（英语冠词直译）
@check("yige_density", "『一个』在计数语境合法；只报单叶 >=6 次且每百汉字 >=2.5 次的。")
def c_yige(t):
    n = t.count("一个")
    h = len(HAN.findall(t))
    if n >= 6 and h and n * 100.0 / h >= 2.5:
        return ["一个x%d/%d字" % (n, h)]
    return []

# ---------------------------------------------------------------- 机翻腔固定串
MT_PAT = [
    (r"进行一个[一-鿿]{1,4}检定", "进行一个X检定"),
    (r"做出一个", "做出一个"),
    (r"作为一个结果", "作为一个结果"),
    (r"这是因为的", "这是因为的"),
    (r"在[一-鿿]{1,8}的[一-鿿]{1,8}上[，。]", "在…的…上，"),
    (r"对于[一-鿿]{1,10}来说[，。]", "对于…来说"),
    (r"是被[一-鿿]{1,6}的", "是被…的"),
    (r"们们", "们们"),
    (r"[一-鿿]们们", "叠们"),
    (r"不能够不", "不能够不"),
    (r"有着一种", "有着一种"),
    (r"一个的", "一个的"),
    (r"的一个的", "的一个的"),
    (r"和和|或或", "连词叠"),
]
@check("mt_idiom", "这些串多数在正常中文里也可能出现，需人工复核；纯作候选池。")
def c_mt(t):
    out = []
    for pat, lab in MT_PAT:
        for m in re.findall(pat, t):
            out.append((lab, m if isinstance(m, str) else str(m)))
    return out

# ---------------------------------------------------------------- 数字风格
UNITS = ["点", "级", "轮", "回合", "英尺", "尺", "码", "天", "小时", "分钟", "格", "次", "名", "个", "枚", "面骰"]
@check("num_style", "『三』在成语/专名内（三叶草、三重）会误报，已要求后接量词且前不接汉字数字之外字符。")
def c_num(t):
    out = []
    for u in UNITS:
        for m in re.findall(r"(?<![一-鿿])([一二三四五六七八九十百千]{1,3})" + u, t):
            out.append(("han", m + u))
        for m in re.findall(r"(\d{1,4})\s?" + u, t):
            out.append(("ara", m + u))
    return out

# ---------------------------------------------------------------- 数字与单位之间的空格是否统一
@check("num_space", "无。统计 `数字+空格+单位` 与 `数字+单位` 两种写法的分布。")
def c_numspace(t):
    out = []
    for u in ["级", "点", "轮", "回合", "英尺", "格"]:
        for m in re.findall(r"\d " + u, t):
            out.append(("spaced", m))
        for m in re.findall(r"\d" + u, t):
            out.append(("tight", m))
    return out

# ---------------------------------------------------------------- 您/你 混用
@check("nin_ni", "对话内 NPC 用敬语『您』合法。此项按叶输出，需看是否在 blockquote/readaloud 内。")
def c_nin(t):
    if "您" in t:
        return ["您x%d 你x%d" % (t.count("您"), t.count("你"))]
    return []

def main():
    leaves = load_all()
    for L in leaves:
        L["plain"] = plain(L["s"])
        L["stratum"] = stratum(L)
    res = collections.defaultdict(list)
    for L in leaves:
        t = L["plain"]
        if len(HAN.findall(t)) < 8:
            continue
        for name, (fn, _) in CHECKS.items():
            src = L["s"] if name in ("space_in_han", "punct_anomaly") else t
            try:
                hits = fn(src)
            except Exception as e:
                hits = []
            for h in hits:
                res[name].append((L["file"], L["path"], L["stratum"], h))
    outdir = os.environ.get("OUTDIR", ".")
    summary = {}
    for name in CHECKS:
        summary[name] = len(res[name])
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    with io.open(os.path.join(outdir, "readability_hits.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in res.items()}, f, ensure_ascii=False)
    print("wrote", os.path.join(outdir, "readability_hits.json"))

if __name__ == "__main__":
    main()
