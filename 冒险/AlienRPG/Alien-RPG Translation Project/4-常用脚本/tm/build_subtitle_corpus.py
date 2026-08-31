#!/usr/bin/env python3
"""把 AlienRPG/ 下的双语字幕规整成对齐的 en\tzh TSV 语料。

  python build_subtitle_corpus.py [--src <dir>] [--out <dir>]

为什么要这一步
--------------
勘察轮只挖了 CnSCG 那四部（同一条翻译血脉，算一票）。业主补的
Romulus 2024 / Covenant 2017 / Alien:Earth 2025 / Prometheus 2012
是**独立血脉**，能把若干中置信度术语升为高置信度——但前提是先对齐。

.ass/.ssa 的双语行把中英塞在同一个 Dialogue 事件里、用 \\N 分隔（中文在前），
.srt 则是相邻两行。两种都不需要时间戳匹配。

繁体文件默认跳过（--keep-trad 保留）：Romulus 的繁简两份除字形外内容相同，
挖出来会让计数虚高一倍。
"""
import argparse
import json
import os
import re
import sys
from glob import glob

HAN = re.compile(r'[\u4e00-\u9fff]')
LAT = re.compile(r'[A-Za-z]')
ENCODINGS = ('utf-8-sig', 'utf-8', 'gb18030', 'cp936', 'utf-16')


def read_any(path):
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc) as fh:
                return fh.read(), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None, None


def clean(s):
    s = re.sub(r'\{[^}]*\}', '', s)          # ASS override tags
    s = re.sub(r'<[^>]+>', '', s)            # html / srt styling
    s = s.replace('\\N', '\n').replace('\\n', '\n').replace('\\h', ' ')
    s = re.sub(r'^[\s\-–—]+', '', s)         # leading dialogue dashes
    return s.strip()


def split_bilingual(segments):
    zh = [s for s in segments if HAN.search(s)]
    en = [s for s in segments if not HAN.search(s) and LAT.search(s)]
    return (en[0], zh[0]) if zh and en else None


def pairs_from_ass(txt):
    out = []
    for line in txt.splitlines():
        if not line.startswith('Dialogue:'):
            continue
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue
        segs = [x for x in (clean(p) for p in parts[9].split('\n')) if x]
        # the split above only works after clean() turned \N into a newline
        segs = [x for seg in segs for x in seg.split('\n') if x.strip()]
        pair = split_bilingual(segs)
        if pair:
            out.append(pair)
    return out


def pairs_from_srt(txt):
    out = []
    for blk in re.split(r'\n\s*\n', txt):
        lines = [clean(l) for l in blk.strip().splitlines()]
        lines = [l for l in lines if l and not re.fullmatch(r'\d+', l) and '-->' not in l]
        pair = split_bilingual(lines)
        if pair:
            out.append(pair)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r'C:/Users/Taka/Desktop/fvtt/AlienRPG')
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..',
        '7-其他内容', 'reference', 'subtitles'))
    ap.add_argument('--keep-trad', action='store_true',
                    help='保留繁体文件（默认跳过，避免同片繁简重复计数）')
    ap.add_argument('--min-pairs', type=int, default=20)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for ext in ('ass', 'ssa', 'srt'):
        files += glob(os.path.join(args.src, '**', '*.' + ext), recursive=True)

    manifest, total = {}, 0
    for path in sorted(files):
        rel = os.path.relpath(path, args.src).replace(os.sep, '/')
        if not args.keep_trad and ('繁体' in rel or 'Traditional' in rel or '繁體' in rel):
            continue
        txt, enc = read_any(path)
        if txt is None:
            print(f'  !! undecodable: {rel}', file=sys.stderr)
            continue
        pairs = (pairs_from_ass(txt) if path.lower().endswith(('.ass', '.ssa'))
                 else pairs_from_srt(txt))
        if len(pairs) < args.min_pairs:
            continue
        key = os.path.basename(path)
        tsv = re.sub(r'[^\w.-]', '_', key) + '.tsv'
        with open(os.path.join(out_dir, tsv), 'w', encoding='utf-8') as fh:
            for en, zh in pairs:
                fh.write(f'{en}\t{zh}\n')
        manifest[rel] = {'pairs': len(pairs), 'encoding': enc, 'tsv': tsv}
        total += len(pairs)
        print(f'{len(pairs):>5} pairs  [{enc}]  {rel}')

    with open(os.path.join(out_dir, '_manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(f'\n{len(manifest)} files, {total} aligned bilingual pairs -> {out_dir}')


if __name__ == '__main__':
    main()
