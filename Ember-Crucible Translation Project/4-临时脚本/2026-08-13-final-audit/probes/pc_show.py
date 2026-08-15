# -*- coding: utf-8 -*-
"""人物一致性镜头：按 path 正则 dump 对齐叶子（只读）。"""
import os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all

def main():
    rx = re.compile(sys.argv[1])
    en_rx = re.compile(sys.argv[2], re.I) if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    trunc = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n = 0
    for repo, pack, path, en, cn in load_all():
        if not rx.search(path):
            continue
        if en_rx and not en_rx.search(en):
            continue
        n += 1
        print('=' * 100)
        print(f'{repo} / {pack} / {path}')
        print('--EN--'); print(en[:trunc] if trunc else en)
        print('--CN--'); print((cn or '')[:trunc] if trunc else (cn or ''))
    print(f'\n({n} leaves)')

if __name__ == '__main__':
    main()
