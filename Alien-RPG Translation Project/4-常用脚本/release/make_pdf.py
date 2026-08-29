#!/usr/bin/env python3
"""把某个已译期刊页导出成 PDF（走 Chrome 的打印引擎）。

  python make_pdf.py --pack starterset --journal "Starter Set Rules" [--out <file.pdf>]
  python make_pdf.py --list                       # 列出可导出的页

为什么用 Chrome 而不是 weasyprint / wkhtmltopdf
-----------------------------------------------
本机没装后两者，而 Chrome 在（headless 打印）。更重要的是**这些页面依赖真正的
浏览器排版**：`.basesymbol` / `.stresssymbol` 是 CSS 背景图，`arpgtable` 用百分比列宽，
`evexamplebox` 用 border-image SVG。wkhtmltopdf 的 WebKit 太旧，weasyprint 不支持
border-image。Chromium 是这些页面在 Foundry 里本来就跑的引擎，所见即所得。

资源怎么解析（三条路径规则，缺一不可）
--------------------------------------
1. 正文里的 `src="systems/…"` / `src="modules/…"` 是**相对 Foundry 数据根**的，
   所以 `<base href="file:///…/Data/">`。
2. CSS 里的 `url("../images/…")` 是**相对 CSS 文件自身**的，所以样式表必须用
   **绝对 file:// 路径 <link>** 进来，不能内联——内联之后 `../` 会相对 HTML 解析，
   36 个背景图与 OCR-A 字体全部 404，而且**不报错**，只是符号变成空白方块。
3. `@font-face` 里 Changa / Roboto / Wallpoet / Kosugi / Blinker 指向 fonts.gstatic.com。
   离线时 Chrome 回落系统字体，版式略有出入但不影响可读性。OCR-A 是本地的，正常。

⚠ starterset 模块自带的 css/starterset.css 是 **0 字节**，样式全部来自
   systems/alienrpg/css/alienrpg.css。别以为漏了哪个样式表。
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data"
SYS_CSS = DATA + "/systems/alienrpg/css/alienrpg.css"

PACKS = {
    "system": ("1-系统汉化插件", "alienrpg.alien-rpg-system.json", "Alien RPG System"),
    "starterset": ("2-新手包汉化插件",
                   "alien-evolved-starterset.alien-evolved-starter-set.json",
                   "Alien Evolved Starter Set"),
    "corerules": ("3-核心书汉化插件",
                  "alien-evolved-corerules.alien-evolved-core-rules.json",
                  "Alien Evolved Core Rules"),
}

CHROMES = [
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]

PRINT_CSS = """
/* ── 打印版式 ───────────────────────────────────────────────── */
@page { size: A4; margin: 14mm 12mm 16mm 12mm; }

html, body {
  background: #fff !important;
  /* 中文正文字体栈。前两个是 Windows 自带，第三个给装了思源的机器。 */
  font-family: "Microsoft YaHei", "微软雅黑", "Source Han Sans SC",
               "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt;
  line-height: 1.75;
  color: #111;
}

/* Foundry 的页面容器在纸上不需要固定高度与滚动 */
.evolvedpage { height: auto !important; overflow: visible !important; }

/* 分页控制：这三类框整块不许被切开 —— 规则示例与朗读框被腰斩最影响可读性 */
.evexamplebox, .evblockquoteCenter, figure, tr, .arpgtable thead {
  break-inside: avoid; page-break-inside: avoid;
}
h1, h2, h3, .evchapterheading, .evolvedsubheading {
  break-after: avoid; page-break-after: avoid;
}
.evchapterheading { break-before: page; page-break-before: page; }
.evchapterheading:first-of-type { break-before: auto; page-break-before: auto; }

img { max-width: 100% !important; height: auto !important; }
table { width: 100% !important; }

/* 链接在纸上不需要颜色与下划线，但保留可辨识度 */
a, a.ev-content-link, a.content-link {
  color: #111 !important; text-decoration: none !important;
  background: none !important; border: none !important; padding: 0 !important;
}

/* 骰面符号是 CSS 背景图，打印时必须保留背景 */
* { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
"""


def jload(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pack(pack):
    repo, fname, adv = PACKS[pack]
    p = os.path.join(ROOT, repo, "compendium", "cn", fname)
    if not os.path.exists(p):
        sys.exit("译文不存在：%s\n（该包还没翻）" % p)
    return jload(p)["entries"][adv]


def cmd_list():
    for pack in PACKS:
        repo, fname, adv = PACKS[pack]
        p = os.path.join(ROOT, repo, "compendium", "cn", fname)
        if not os.path.exists(p):
            print("  [%s] 未翻译" % pack)
            continue
        entry = jload(p)["entries"][adv]
        print("  [%s]" % pack)
        for jn, j in entry.get("journals", {}).items():
            for pn, pg in (j.get("pages") or {}).items():
                t = pg.get("text") or ""
                if t.strip():
                    vis = len(re.sub(r"<[^>]+>", "", t))
                    print("     --journal %-34r  可见 %6d 字" % (jn, vis))
                    break
    return 0


# 我们自己的模块通常**没装进 Foundry**（部署在 VPS，本机只是冒烟机），
# 所以 <base> 解析不到 `modules/<我们的模块>/…`。这里把这些前缀重定向到项目目录。
# ⚠ 只映射我们自己的三个模块；上游资源仍然走 <base>，不许改。
MODULE_OVERRIDES = {
    "modules/alien-evolved-starterset-cn/": "2-新手包汉化插件/",
    "modules/alien-evolved-corerules-cn/": "3-核心书汉化插件/",
    "modules/alienrpg-cn/": "1-系统汉化插件/",
}


def resolve_local_modules(html):
    """把指向我们自己模块的 src 换成绝对 file:// 路径，并报告解析不到的。

    ⚠ 两个坑，都实测踩过：
    1. **路径里有空格必须 URL 编码**。项目目录叫 `Alien-RPG Translation Project`，
       未编码的空格会让 Chrome 静默拿不到图 —— 不报错，只是破图，而且
       `grep src=` 看上去完全正常。用 pathname2url 一次性解决（它同时处理
       中文目录名，虽然 Chrome 对未编码的 UTF-8 通常也认）。
    2. 别手工拼 `%s/%s` —— repo 常量带尾斜杠，拼出来是 `…//images/…`。
    """
    from urllib.request import pathname2url
    missing = []
    for prefix, repo in MODULE_OVERRIDES.items():
        for m in set(re.findall(r'src="(%s[^"]+)"' % re.escape(prefix), html)):
            rel = m[len(prefix):]
            p = os.path.normpath(os.path.join(ROOT, repo, rel.replace("/", os.sep)))
            if not os.path.exists(p):
                missing.append(m)
                continue
            html = html.replace('src="%s"' % m, 'src="file:%s"' % pathname2url(p))
    if missing:
        print("  ⚠ 这些资源在项目里找不到，PDF 里会是破图：")
        for m in missing:
            print("     " + m)
    return html


def build_html(entry, journal, title):
    j = entry["journals"].get(journal)
    if j is None:
        sys.exit("没有这个日志：%r\n可用：%s" % (journal, list(entry["journals"])))
    parts = []
    for pn, pg in (j.get("pages") or {}).items():
        t = pg.get("text") or ""
        if t.strip():
            parts.append(t)
    if not parts:
        sys.exit("该日志没有正文页（可能是纯图片日志）")
    body = resolve_local_modules("\n".join(parts))

    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<base href="file:///%s/">
<title>%s</title>
<link rel="stylesheet" href="file:///%s">
<style>%s</style>
</head>
<body class="alienrpg">
%s
</body>
</html>
""" % (DATA.replace("\\", "/"), title, SYS_CSS.replace("\\", "/"), PRINT_CSS, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", choices=sorted(PACKS))
    ap.add_argument("--journal")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-html", action="store_true", help="保留中间 HTML 供排查")
    a = ap.parse_args()

    if a.list or not a.pack:
        return cmd_list()
    if not a.journal:
        sys.exit("需要 --journal，先用 --list 看有哪些")

    entry = load_pack(a.pack)
    title = a.journal
    html = build_html(entry, a.journal, title)

    out_dir = os.path.join(ROOT, "6-工作区", "pdf")
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^\w-]", "_", a.journal)
    html_path = os.path.join(out_dir, safe + ".html")
    pdf_path = a.out or os.path.join(out_dir, safe + ".pdf")
    io.open(html_path, "w", encoding="utf-8", newline="\n").write(html)
    print("中间 HTML：%s  (%d 字节)" % (html_path, os.path.getsize(html_path)))

    chrome = next((c for c in CHROMES if os.path.exists(c)), None)
    if not chrome:
        sys.exit("找不到 Chrome/Edge，无法打印。手动方案：用浏览器打开上面那个 HTML，Ctrl+P 存为 PDF。")

    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer",
           "--virtual-time-budget=30000",       # 等图片与字体加载
           "--print-to-pdf=" + pdf_path,
           "file:///" + html_path.replace("\\", "/")]
    print("渲染中（%s）…" % os.path.basename(chrome))
    # ⚠ Chrome 往 stderr 写的是 UTF-8，而 Windows 的 text=True 默认按 GBK 解码 —— 会抛
    #   UnicodeDecodeError 并**盖掉真正的错误信息**。显式指定编码并容错。
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    err = r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8", "replace")
    if not os.path.exists(pdf_path):
        print(out[-2000:])
        print(err[-2000:], file=sys.stderr)
        sys.exit("Chrome 没有产出 PDF。")

    print("PDF：%s  (%.1f MB)" % (pdf_path, os.path.getsize(pdf_path) / 1048576))
    if not a.keep_html:
        pass  # 保留 HTML —— 排查排版问题时比 PDF 有用得多
    return 0


if __name__ == "__main__":
    sys.exit(main())
