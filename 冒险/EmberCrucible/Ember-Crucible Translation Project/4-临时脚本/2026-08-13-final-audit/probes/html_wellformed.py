# -*- coding: utf-8 -*-
"""译文 HTML 结构合法性探针（只读，不改库）。

标记闸（scan_markup_drift.py）只比 <tag> 名字的**多重集**，看不见结构：
`<p><strong>a</p></strong>` 与 `<p><strong>a</strong></p>` 在它眼里完全一样。
本探针用真正的解析器逐叶解析，并且**英文侧同样解析一遍做对照**——
只有「CN 坏 / EN 好」才是译文缺陷，「两边都坏」是上游问题。

口径（--why 打印）：
  P1 STRUCT_MISMATCH   结束标签与栈顶不符（交叉嵌套）
  P2 STRUCT_STRAY_END  结束标签没有对应的开始标签
  P3 STRUCT_UNCLOSED   到叶末仍未闭合（已按 HTML5 可省略结束标签规则自动闭合 p/li/td/tr/…）
  P4 NEST_ILLEGAL      非法嵌套：<p> 里套块级、<ul>/<ol> 直接含文本、<li>/<tr> 脱离父容器
  P5 UNKNOWN_TAG       标签名不在 HTML 已知集合内（浏览器会当自定义元素，内容样式全丢）
  P6 ATTR_MALFORMED    属性引号未闭合 / 属性值里混入裸 < > &
  P7 ENTITY_DRIFT      & 的转义与英文侧不一致（多转义 / 少转义）
  P8 FULLWIDTH_IN_MARKUP 全角标点混进标记内部（标签内、@UUID[]{} 的括号、[[ ]] 的括号）
  P9 BRACKET_UNBALANCED  enricher 的 [ ] { } 配对数与英文侧不一致
  PA DUP_ID            同一 journal page 内 id 重复（锚点只会跳到第一个）

用法：
  python html_wellformed.py --repo <仓库路径> [--kind P1,P4] [--out report.json] [--limit N]
  python html_wellformed.py --repo <仓库路径> --side en    # 只看英文侧本身坏不坏
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}

# HTML5 里结束标签可省略的元素 -> 遇到这些开始标签时自动闭合
AUTO_CLOSE = {
    "p": {"address", "article", "aside", "blockquote", "details", "div", "dl",
          "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
          "h4", "h5", "h6", "header", "hr", "main", "menu", "nav", "ol", "p",
          "pre", "section", "table", "ul"},
    "li": {"li"},
    "dt": {"dt", "dd"},
    "dd": {"dt", "dd"},
    "td": {"td", "th", "tr"},
    "th": {"td", "th", "tr"},
    "tr": {"tr"},
    "thead": {"tbody", "tfoot"},
    "tbody": {"tbody", "tfoot"},
    "option": {"option", "optgroup"},
}
# 父容器闭合时可以隐式闭合的子元素
IMPLICIT_ON_PARENT_CLOSE = {
    "ul": {"li"}, "ol": {"li"}, "menu": {"li"},
    "table": {"tr", "td", "th", "thead", "tbody", "tfoot", "caption"},
    "thead": {"tr", "td", "th"}, "tbody": {"tr", "td", "th"}, "tfoot": {"tr", "td", "th"},
    "tr": {"td", "th"},
    "dl": {"dt", "dd"},
    "select": {"option", "optgroup"},
    # 任何块级父元素关闭时都能隐式关掉未闭合的 p
    "div": {"p"}, "section": {"p"}, "blockquote": {"p"}, "figure": {"p"},
    "figcaption": {"p"}, "li": {"p"}, "td": {"p"}, "th": {"p"}, "article": {"p"},
    "aside": {"p"}, "header": {"p"}, "footer": {"p"}, "main": {"p"}, "form": {"p"},
    "details": {"p"}, "fieldset": {"p"},
}

KNOWN_TAGS = {
    "a", "abbr", "address", "area", "article", "aside", "audio", "b", "base", "bdi",
    "bdo", "blockquote", "body", "br", "button", "canvas", "caption", "cite", "code",
    "col", "colgroup", "data", "datalist", "dd", "del", "details", "dfn", "dialog",
    "div", "dl", "dt", "em", "embed", "fieldset", "figcaption", "figure", "footer",
    "form", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr",
    "html", "i", "iframe", "img", "input", "ins", "kbd", "label", "legend", "li",
    "link", "main", "map", "mark", "menu", "meta", "meter", "nav", "noscript",
    "object", "ol", "optgroup", "option", "output", "p", "param", "picture", "pre",
    "progress", "q", "rp", "rt", "ruby", "s", "samp", "script", "section", "select",
    "slot", "small", "source", "span", "strong", "style", "sub", "summary", "sup",
    "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time",
    "title", "tr", "track", "u", "ul", "var", "video", "wbr",
    # Foundry / ProseMirror 里实际出现过的自定义元素，视为已知
    "enriched-content", "document-embed", "secret-block",
}

BLOCK_IN_P = {"div", "p", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6",
              "blockquote", "section", "article", "aside", "figure", "form", "hr",
              "pre", "dl", "header", "footer", "main", "nav"}


class Walker(HTMLParser):
    """带栈的解析器；按 HTML5 可省略结束标签规则自动闭合，剩下的才算真错。"""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack = []          # [(tag, pos)]
        self.errors = []         # (code, detail)
        self.tags_seen = Counter()
        self.ids = []            # (id_value, pos)
        self.text_ctx = []       # (parent_tag, text)

    def _pos(self):
        l, c = self.getpos()
        return f"L{l}C{c}"

    def _auto_close(self, new_tag):
        while self.stack:
            top = self.stack[-1][0]
            if new_tag in AUTO_CLOSE.get(top, ()):
                self.stack.pop()
            else:
                break

    def handle_starttag(self, tag, attrs):
        self.tags_seen[tag] += 1
        if tag not in KNOWN_TAGS:
            self.errors.append(("P5", f"未知标签 <{tag}> @{self._pos()}"))
        self._auto_close(tag)
        # 非法嵌套
        if self.stack:
            parent = self.stack[-1][0]
            if parent == "p" and tag in BLOCK_IN_P:
                self.errors.append(("P4", f"<p> 内出现块级 <{tag}> @{self._pos()}"))
            if parent in ("ul", "ol") and tag not in ("li", "script", "template", "ul", "ol"):
                self.errors.append(("P4", f"<{parent}> 直接含 <{tag}>（应先有 <li>）@{self._pos()}"))
            if tag == "li" and parent not in ("ul", "ol", "menu"):
                self.errors.append(("P4", f"<li> 的父元素是 <{parent}> 而非 ul/ol @{self._pos()}"))
            if tag == "tr" and parent not in ("table", "thead", "tbody", "tfoot"):
                self.errors.append(("P4", f"<tr> 的父元素是 <{parent}> @{self._pos()}"))
            if tag in ("td", "th") and parent != "tr":
                self.errors.append(("P4", f"<{tag}> 的父元素是 <{parent}> 而非 tr @{self._pos()}"))
        else:
            if tag == "li":
                self.errors.append(("P4", f"<li> 在顶层（无 ul/ol）@{self._pos()}"))
            if tag in ("td", "th", "tr"):
                self.errors.append(("P4", f"<{tag}> 在顶层（无 table）@{self._pos()}"))
        for k, v in attrs:
            if k == "id" and v:
                self.ids.append((v, self._pos()))
        if tag not in VOID:
            self.stack.append((tag, self._pos()))

    def handle_startendtag(self, tag, attrs):
        self.tags_seen[tag] += 1
        if tag not in KNOWN_TAGS:
            self.errors.append(("P5", f"未知自闭合标签 <{tag}/> @{self._pos()}"))
        for k, v in attrs:
            if k == "id" and v:
                self.ids.append((v, self._pos()))

    def handle_endtag(self, tag):
        self.tags_seen[f"/{tag}"] += 1
        if tag in VOID:
            return
        # 父元素闭合时隐式关掉可省略的子元素
        while self.stack and self.stack[-1][0] != tag and \
                self.stack[-1][0] in IMPLICIT_ON_PARENT_CLOSE.get(tag, set()):
            self.stack.pop()
        if not self.stack:
            self.errors.append(("P2", f"多余的 </{tag}> @{self._pos()}"))
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
            return
        # 栈里往下找；找得到说明是交叉嵌套，找不到说明是孤立结束标签
        depth = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                depth = i
                break
        if depth is None:
            self.errors.append(("P2", f"多余的 </{tag}>（栈顶是 <{self.stack[-1][0]}>）@{self._pos()}"))
        else:
            unclosed = [t for t, _ in self.stack[depth + 1:]]
            self.errors.append(("P1", f"</{tag}> 与栈顶 <{self.stack[-1][0]}> 交叉；"
                                      f"跨过未闭合的 {unclosed} @{self._pos()}"))
            del self.stack[depth:]

    def handle_data(self, data):
        if data.strip():
            parent = self.stack[-1][0] if self.stack else None
            if parent in ("ul", "ol", "table", "thead", "tbody", "tfoot", "tr", "select", "dl"):
                self.errors.append(("P4", f"<{parent}> 里出现裸文本 {data.strip()[:40]!r} @{self._pos()}"))

    def finish(self):
        for tag, pos in reversed(self.stack):
            if tag in ("p", "li", "td", "th", "tr", "tbody", "thead", "tfoot",
                       "dt", "dd", "option", "colgroup", "rt", "rp"):
                continue  # 结束标签可省略，浏览器会自己收口
            self.errors.append(("P3", f"<{tag}> 未闭合（开于 {pos}）"))
        return self.errors


# ---------- 非解析器类检查 ----------

TAG_SRC = re.compile(r"<[a-zA-Z/][^<>]*>")
# 只挑「真的像开始标签」的片段来查属性，避免把 `a < b` 之类误当标签
OPEN_TAG_SRC = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:\s[^<>]*)?)/?>")
ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{0,30}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});")
# Foundry / dnd5e 的 enricher 用裸 & 起头：&Reference[…] &amp;Reference[…]
ENRICHER_AMP = re.compile(r"&(?:amp;)?[A-Za-z]+\[")
FULLWIDTH = "，。；：！？、（）【】《》「」『』［］｛｝“”‘’￥…—·　"
# 标签内部的全角字符（属性值里的中文是合法的，只查标签名与属性名区域）
CJK_PUNCT_ANY = re.compile("[" + re.escape(FULLWIDTH) + "]")

UUID_LIKE = re.compile(r"@[A-Za-z]+\s*[\[［]")
INLINE_ROLL = re.compile(r"\[\[")


def attr_problems(s):
    """查属性引号未闭合 / 属性值里裸 < > &。返回 [(code, detail)]"""
    out = []
    for m in OPEN_TAG_SRC.finditer(s):
        tag, attrs = m.group(1), m.group(2) or ""
        if not attrs.strip():
            continue
        # 引号奇偶
        if attrs.count('"') % 2:
            out.append(("P6", f"<{tag}> 属性里双引号个数为奇数：{m.group(0)[:120]!r}"))
        # 属性值里裸 &（不是合法实体、也不是 enricher 前缀）
        for am in re.finditer(r'=\s*"([^"]*)"', attrs):
            val = am.group(1)
            for pos in (i for i, ch in enumerate(val) if ch == "&"):
                rest = val[pos:]
                if ENTITY.match(rest) or ENRICHER_AMP.match(rest):
                    continue
                out.append(("P6", f"<{tag}> 属性值里裸 &：{val[:80]!r}"))
                break
            if "<" in val or ">" in val:
                out.append(("P6", f"<{tag}> 属性值里裸 <>：{val[:80]!r}"))
    # 标签名/属性名区域出现全角标点
    for m in TAG_SRC.finditer(s):
        body = m.group(0)
        # 抠掉带引号的属性值再查
        stripped = re.sub(r'"[^"]*"', '""', body)
        stripped = re.sub(r"'[^']*'", "''", stripped)
        if CJK_PUNCT_ANY.search(stripped):
            out.append(("P8", f"标签内部（引号外）有全角标点：{body[:120]!r}"))
    return out


def raw_amp_count(s):
    """数「不是合法实体、也不是 enricher 前缀」的裸 &。"""
    n = 0
    for i, ch in enumerate(s):
        if ch != "&":
            continue
        rest = s[i:]
        if ENTITY.match(rest) or ENRICHER_AMP.match(rest):
            continue
        n += 1
    return n


def bracket_sig(s):
    """enricher 括号配对签名。只数 ASCII 方括号/花括号。"""
    return (s.count("["), s.count("]"), s.count("{"), s.count("}"))


def fullwidth_in_enricher(s):
    out = []
    # @Xxx［ 或 @Xxx[…］ —— 全角方括号
    for m in re.finditer(r"@[A-Za-z]+[\[［][^\]］]{0,200}[\]］]?", s):
        seg = m.group(0)
        if "［" in seg or "］" in seg:
            out.append(("P8", f"enricher 用了全角方括号：{seg[:100]!r}"))
    # {标签} 的花括号被打成全角
    for m in re.finditer(r"(?:@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\])\s*[｛{][^｝}]{0,120}[｝}]", s):
        seg = m.group(0)
        if "｛" in seg or "｝" in seg:
            out.append(("P8", f"enricher 标签用了全角花括号：{seg[:100]!r}"))
    # [[ ]] 内联指令被打成全角
    if "［［" in s or "］］" in s:
        out.append(("P8", "内联指令用了全角双方括号 ［［/］］"))
    return out


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def analyze(s):
    """返回 (errors, ids, tags)"""
    w = Walker()
    try:
        w.feed(s)
        w.close()
    except Exception as e:  # noqa
        w.errors.append(("P0", f"解析器抛异常：{type(e).__name__}: {e}"))
    w.finish()
    errs = list(w.errors)
    errs += attr_problems(s)
    errs += fullwidth_in_enricher(s)
    return errs, w.ids, w.tags_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out")
    ap.add_argument("--kind")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--side", default="cn", choices=["cn", "en"])
    args = ap.parse_args()
    kinds = set(args.kind.split(",")) if args.kind else None

    repo = Path(args.repo)
    findings = []
    stats = Counter()
    n_leaves = n_html = 0

    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        primary, control = (cn, en) if args.side == "cn" else (en, cn)
        # ---- 逐叶 ----
        for p, s in primary.items():
            n_leaves += 1
            if "<" not in s and "&" not in s and "[" not in s:
                continue
            n_html += 1
            other = control.get(p, "")
            errs, ids, _ = analyze(s)
            oerrs, oids, _ = analyze(other) if other else ([], [], Counter())
            ocodes = Counter(c for c, _ in oerrs)
            mycodes = Counter(c for c, _ in errs)
            for code, detail in errs:
                # 同类错误英文侧也有同样多 -> 上游问题，标注但降级
                upstream = ocodes.get(code, 0) >= mycodes.get(code, 0) and other
                stats[code] += 1
                if kinds and code not in kinds:
                    continue
                findings.append({
                    "code": code, "pack": f.name, "path": p,
                    "detail": detail, "upstream": bool(upstream),
                    "cn": s[:400], "en": other[:400],
                })
            # ---- 实体转义对照 ----
            if other:
                a_en, a_cn = raw_amp_count(other), raw_amp_count(s)
                e_en, e_cn = other.count("&amp;"), s.count("&amp;")
                if (a_en > 0 or a_cn > 0 or e_en != e_cn) and (a_en, e_en) != (a_cn, e_cn):
                    stats["P7"] += 1
                    if not kinds or "P7" in kinds:
                        findings.append({
                            "code": "P7", "pack": f.name, "path": p,
                            "detail": f"裸& EN={a_en} CN={a_cn} / &amp; EN={e_en} CN={e_cn}",
                            "upstream": False, "cn": s[:400], "en": other[:400]})
                b_en, b_cn = bracket_sig(other), bracket_sig(s)
                if b_en != b_cn:
                    stats["P9"] += 1
                    if not kinds or "P9" in kinds:
                        findings.append({
                            "code": "P9", "pack": f.name, "path": p,
                            "detail": f"[ ] {{ }} 计数 EN={b_en} CN={b_cn}",
                            "upstream": False, "cn": s[:400], "en": other[:400]})
        # ---- 同页 id 重复 ----
        by_page = defaultdict(list)
        for p, s in primary.items():
            if "id=" not in s:
                continue
            _, ids, _ = analyze(s)
            for v, pos in ids:
                by_page[p].append(v)
        for p, vals in by_page.items():
            dup = [v for v, n in Counter(vals).items() if n > 1]
            if dup:
                stats["PA"] += 1
                if not kinds or "PA" in kinds:
                    findings.append({"code": "PA", "pack": f.name, "path": p,
                                     "detail": f"同叶内重复 id：{dup}", "upstream": False,
                                     "cn": primary[p][:400], "en": control.get(p, "")[:400]})

    print(f"[{args.side}] 扫描 {n_leaves} 个字符串叶，其中含 <&[ 的 {n_html} 个")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"  命中 findings: {len(findings)}")
    if args.out:
        Path(args.out).write_text(json.dumps(findings, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"  -> {args.out}")
    if args.limit:
        for x in findings[:args.limit]:
            print(json.dumps(x, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
