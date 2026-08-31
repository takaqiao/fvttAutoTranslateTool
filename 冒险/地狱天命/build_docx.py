"""
地狱天命 markdown → docx 装配脚本

读取 translated/page_*.md，解析中文部分（跳过 <details>英文原文），
输出 output/地狱天命_中文版.docx。

每页一个 Heading 2；保留 markdown ###/####, 表格, 引用块, 加粗。
页之间分页符。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
TRANSLATED = ROOT / "translated"
OUTPUT_DIR = ROOT / "output"

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
DETAILS_OPEN_RE = re.compile(r"<details[\s>]")
DETAILS_CLOSE_RE = re.compile(r"</details>")


def _set_cjk_font(run, name: str = "Microsoft YaHei") -> None:
    """Ensure CJK glyphs use the given font."""
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)


def parse_inline(text: str):
    """Yield (chunk, bold) tuples — splits on **bold** markers."""
    pos = 0
    for m in BOLD_RE.finditer(text):
        if m.start() > pos:
            yield text[pos : m.start()], False
        yield m.group(1), True
        pos = m.end()
    if pos < len(text):
        yield text[pos:], False


def add_inline(paragraph, text: str, *, bold_all: bool = False, font: str = "Microsoft YaHei") -> None:
    for chunk, bold in parse_inline(text):
        if not chunk:
            continue
        run = paragraph.add_run(chunk)
        run.bold = bold or bold_all
        _set_cjk_font(run, font)


def is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and "|" in s[1:-1]


def is_table_sep(line: str) -> bool:
    return bool(TABLE_SEP_RE.match(line.strip()))


def parse_md(text: str):
    """Return list of (type, payload) blocks.

    Types: h1/h2/h3/h4, p, quote, table.
    Skips <details>...</details> regions entirely.
    """
    lines = text.split("\n")
    blocks = []
    i = 0
    in_details = False

    while i < len(lines):
        line = lines[i]

        if DETAILS_OPEN_RE.search(line):
            in_details = True
            i += 1
            continue
        if in_details:
            if DETAILS_CLOSE_RE.search(line):
                in_details = False
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Headings
        for prefix, kind in (("#### ", "h4"), ("### ", "h3"), ("## ", "h2"), ("# ", "h1")):
            if stripped.startswith(prefix):
                blocks.append((kind, stripped[len(prefix) :].strip()))
                i += 1
                break
        else:
            # Table
            if is_table_row(line):
                rows = []
                while i < len(lines) and is_table_row(lines[i]):
                    if is_table_sep(lines[i]):
                        i += 1
                        continue
                    inner = lines[i].strip().strip("|")
                    cells = [c.strip() for c in inner.split("|")]
                    rows.append(cells)
                    i += 1
                if rows:
                    blocks.append(("table", rows))
                continue

            # Quote block (consecutive `> ` lines)
            if stripped.startswith(">"):
                qlines = []
                while i < len(lines) and lines[i].strip().startswith(">"):
                    qtext = lines[i].strip()[1:].lstrip()
                    qlines.append(qtext)
                    i += 1
                blocks.append(("quote", qlines))
                continue

            # Paragraph (single line — source files use blank-line-separated paragraphs)
            blocks.append(("p", line.rstrip()))
            i += 1

    return blocks


def _ensure_style(doc: Document, name: str, fallback: str) -> str:
    for s in doc.styles:
        if s.name == name:
            return name
    return fallback


def render_blocks(doc: Document, blocks, *, quote_style: str, table_style: str) -> None:
    for kind, payload in blocks:
        if kind in ("h1", "h2", "h3", "h4"):
            level = int(kind[1])
            p = doc.add_heading("", level=level)
            add_inline(p, payload)
        elif kind == "p":
            p = doc.add_paragraph()
            add_inline(p, payload)
        elif kind == "quote":
            for qline in payload:
                p = doc.add_paragraph(style=quote_style)
                if qline:
                    add_inline(p, qline)
        elif kind == "table":
            rows = payload
            ncols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=ncols)
            try:
                table.style = table_style
            except KeyError:
                pass
            for r_idx, row in enumerate(rows):
                for c_idx, cell_text in enumerate(row):
                    cell = table.rows[r_idx].cells[c_idx]
                    cell.text = ""
                    para = cell.paragraphs[0]
                    add_inline(para, cell_text, bold_all=(r_idx == 0))
            # blank line after table
            doc.add_paragraph()


def build_title_page(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("地狱天命")
    run.bold = True
    run.font.size = Pt(40)
    _set_cjk_font(run)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Hell's Destiny")
    run.italic = True
    run.font.size = Pt(20)
    _set_cjk_font(run)

    code = doc.add_paragraph()
    code.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = code.add_run("PZO15223  ·  Pathfinder 2E Adventure Path")
    run.font.size = Pt(12)
    _set_cjk_font(run)

    desc = doc.add_paragraph()
    desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = desc.add_run("10–20 级  ·  全 258 页中文译本")
    run.font.size = Pt(12)
    _set_cjk_font(run)

    doc.add_paragraph()
    doc.add_paragraph()
    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = note.add_run("（民间翻译，仅供学习交流，请支持 Paizo 正版）")
    run.font.size = Pt(10)
    _set_cjk_font(run)


def build(pages: list[int] | None, output_path: Path) -> None:
    doc = Document()

    # Default font
    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(11)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    quote_style = _ensure_style(doc, "Intense Quote", "Normal")
    table_style = _ensure_style(doc, "Light Grid Accent 1", "Table Grid")

    build_title_page(doc)
    doc.add_page_break()

    files = sorted(TRANSLATED.glob("page_*.md"))
    if pages:
        wanted = {f"page_{n:03d}.md" for n in pages}
        files = [f for f in files if f.name in wanted]

    total = len(files)
    print(f"装配 {total} 页 → {output_path}")

    for idx, pf in enumerate(files, 1):
        text = pf.read_text(encoding="utf-8")
        blocks = parse_md(text)
        render_blocks(doc, blocks, quote_style=quote_style, table_style=table_style)
        if idx < total:
            doc.add_page_break()
        if idx % 25 == 0 or idx == total:
            print(f"  {idx}/{total} ({pf.name})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"完成: {output_path}  ({output_path.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=str, default=None, help="逗号分隔页号 (e.g. 1,10,50)，默认全量")
    ap.add_argument("--out", type=str, default=None, help="输出 docx 路径")
    args = ap.parse_args()

    pages = None
    if args.pages:
        pages = [int(x) for x in args.pages.split(",") if x.strip()]

    if args.out:
        output_path = Path(args.out)
    elif pages:
        output_path = OUTPUT_DIR / f"地狱天命_sample_{'_'.join(str(p) for p in pages)}.docx"
    else:
        output_path = OUTPUT_DIR / "地狱天命_中文版.docx"

    build(pages, output_path)


if __name__ == "__main__":
    main()
