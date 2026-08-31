"""Dump structure of the Chinese Book 1 docx."""
from docx import Document

d = Document('Season of Ghosts GM Guide - Book 1_zh.docx')
with open('book1_zh_dump.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(d.paragraphs):
        style = p.style.name if p.style else '?'
        text = p.text
        f.write(f'[{i}] {style} | {text}\n')
print('done')
