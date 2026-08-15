# -*- coding: utf-8 -*-
"""块级 DOM **树形**对照 —— 标记闸看不见的那一层。

标记闸比的是标签名多重集：<ul><li>a</li></ul><p>b</p> 与 <ul><li>a</li><p>b</p></ul>
在它眼里完全一样（都是 1 ul / 1 li / 1 p）。本探针把两侧解析成树，
逐节点比较「从根到该节点的块级标签路径」序列。

  T1  块级路径序列不一致  -> 某个块被挪进/挪出了别的容器（多一层、少一层、换了父）
  T2  lxml(recover=False) 在中文侧报解析错误而英文侧不报
  T3  非 ASCII 标签名 <中文>（html.parser 会当纯文本吞掉，浏览器当自定义元素，内容仍显示但样式全丢）
  T4  属性值内出现裸 <（html.parser 与浏览器的分词点不同，会截断标签）
  T5  <section class="secret"> 的开合与英文侧不一致（秘密块错位＝把 GM 内容暴露给玩家，或反过来）
"""
import json, re, sys, collections
from pathlib import Path
from html.parser import HTMLParser

from lxml import etree, html as lhtml

BLOCK = {"p", "ul", "ol", "li", "table", "thead", "tbody", "tfoot", "tr", "td", "th",
         "h1", "h2", "h3", "h4", "h5", "h6", "section", "div", "blockquote",
         "figure", "figcaption", "dl", "dt", "dd", "hr", "aside", "article",
         "header", "footer", "caption"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def block_paths(s):
    """用 lxml 的 HTML 解析（与浏览器规则最接近）构块级路径序列。"""
    try:
        root = lhtml.fragment_fromstring(s, create_parent="root")
    except Exception:
        return None
    out = []

    def walk(el, path):
        for ch in el:
            if not isinstance(ch.tag, str):
                continue
            t = ch.tag.lower()
            np = path + [t] if t in BLOCK else path
            if t in BLOCK:
                out.append("/".join(np))
            walk(ch, np)
    walk(root, [])
    return out


def secret_shape(s):
    """<section class="…secret…"> 的位置序列（用块级路径表示）。"""
    try:
        root = lhtml.fragment_fromstring(s, create_parent="root")
    except Exception:
        return None
    out = []

    def walk(el, path):
        for ch in el:
            if not isinstance(ch.tag, str):
                continue
            t = ch.tag.lower()
            cls = (ch.get("class") or "")
            np = path + [t] if t in BLOCK else path
            if t == "section":
                out.append("/".join(np) + "|" + " ".join(sorted(cls.split())))
            walk(ch, np)
    walk(root, [])
    return out


def lxml_strict_err(s):
    p = etree.HTMLParser(recover=False)
    try:
        etree.fromstring(f"<div>{s}</div>", p)
        return None
    except Exception as ex:
        return str(ex)[:200]


NONASCII_TAG = re.compile(r"</?\s*([^\x00-\x7F][^\s<>/]*)")
ATTR_RAW_LT = re.compile(r'<[a-zA-Z][a-zA-Z0-9]*\s[^<>]*?=\s*"[^"]*<')

counts = collections.Counter()
rows = []
scanned = 0
for repo in sys.argv[1:]:
    repo = Path(repo)
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in cn.items():
            e = en.get(p, "")
            if "<" not in s and "<" not in e:
                continue
            scanned += 1
            # T3 / T4 纯文本层面
            for m in NONASCII_TAG.finditer(s):
                if not NONASCII_TAG.search(e):
                    counts["T3"] += 1
                    rows.append(("T3", repo.name, f.name, p, m.group(0)[:80], ""))
                break
            m = ATTR_RAW_LT.search(s)
            if m and not ATTR_RAW_LT.search(e):
                counts["T4"] += 1
                rows.append(("T4", repo.name, f.name, p, m.group(0)[:120], ""))
            # T2 lxml 严格
            ec, ee = lxml_strict_err(s), lxml_strict_err(e)
            if ec and not ee:
                counts["T2"] += 1
                rows.append(("T2", repo.name, f.name, p, ec, e[:150]))
            # T1 块级树形
            bc, beh = block_paths(s), block_paths(e)
            if bc is None or beh is None:
                counts["T1_parsefail"] += 1
                continue
            if bc != beh:
                # 多重集相同但顺序/层级不同 -> 标记闸完全看不见
                same_multiset = collections.Counter(bc) == collections.Counter(beh)
                counts["T1_hidden" if same_multiset else "T1_visible"] += 1
                if same_multiset:
                    dif = [(i, a, b) for i, (a, b) in enumerate(zip(bc, beh)) if a != b][:3]
                    rows.append(("T1_hidden", repo.name, f.name, p,
                                 f"{len(bc)} 个块级节点，首个差异 {dif}", ""))
            # T5 secret 块
            sc, se = secret_shape(s), secret_shape(e)
            if sc is not None and se is not None and sc != se:
                counts["T5"] += 1
                rows.append(("T5", repo.name, f.name, p, f"CN={sc[:6]}", f"EN={se[:6]}"))

print("扫描含标签的叶:", scanned)
print(counts)
seen = collections.Counter()
for tag, rn, pack, p, det, ctx in rows:
    seen[tag] += 1
    if seen[tag] > 20:
        continue
    print("-" * 96)
    print(f"[{tag}] {rn} {pack} | {p}")
    print("    det:", str(det)[:300])
    if ctx:
        print("    ctx:", str(ctx)[:250])
