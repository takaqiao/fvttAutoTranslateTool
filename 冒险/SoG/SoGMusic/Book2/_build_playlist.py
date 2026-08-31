# -*- coding: utf-8 -*-
"""由 _manifest.tsv 生成 书2 的 Foundry playlist 导出 JSON。

清单已按 Book 2 叙事顺序排好（第1章 -> 第2章 -> 第3章 -> 通用），
本脚本原样保序输出，不再自己排序。

schema 完全照抄现有导出（fvtt-Playlist-书1-…json）：同一个 folder、
sorting="a"、channel="music"、mode=-1（音板模式，逐条点播，与 书1 一致）。
因为 sorting="a" 是按 name 排序，而每条 name 都以两位序号开头，
所以清单顺序 == Foundry 侧栏显示顺序。

清单第 5 列 kind：
  new   -> 文件在本目录，第 7 列是 YouTube id，投放后路径为 assets/SoG/Book2/<文件名>
  reuse -> 库里早有的文件，第 7 列直接就是 VPS 上的现成路径，不搬不复制

_id 由路径哈希确定性生成，重跑不变。
"""
import json, os, sys, csv, hashlib, urllib.parse

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER = 'lzGARFzSnT2g6RTu'          # 与其余 12 个播放列表同一目录
SERVER_DIR = 'assets/SoG/Book2'      # VPS 上的投放位置
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def fid(seed: str) -> str:
    """确定性生成 16 位 Foundry 风格 id。"""
    h = hashlib.sha256(seed.encode('utf-8')).digest()
    return ''.join(ALPHA[b % len(ALPHA)] for b in h[:16])


def enc(path: str) -> str:
    """按 Foundry 导出的方式编码路径：逐段编码，保留 /。

    与现有导出实测一致： ( ) ! 不编码； ' -> %27 、 [ -> %5B 、 & -> %26 、
    # -> %23 、 , -> %2C 、空格 -> %20 。
    """
    return '/'.join(urllib.parse.quote(seg, safe="()!") for seg in path.split('/'))


def sound(name: str, path: str) -> dict:
    return {
        'path': path,
        'name': name,
        '_id': fid(path),
        'channel': '',
        'playing': False,
        'repeat': True,
        'volume': 0.5,
        'sort': 0,
        'flags': {},
        'pausedTime': None,
    }


def main():
    rows = [r for r in csv.reader(open(os.path.join(BASE, '_manifest.tsv'), encoding='utf-8'),
                                  delimiter='\t')
            if r and not r[0].startswith('#')]

    sounds, missing = [], []
    n_new = n_reuse = 0
    for num, ch, zh, en, kind, src, ref, note in rows:
        name = f'{num} {ch}·{zh} {en}'
        if kind == 'new':
            fname = f'{num} {zh} - {en}.ogg'
            if not os.path.exists(os.path.join(BASE, fname)):
                missing.append(fname)
                continue
            path = f'{SERVER_DIR}/{enc(fname)}'
            n_new += 1
        else:
            path = ref            # 清单里已经是编码好的 VPS 路径，原样使用
            n_reuse += 1
        sounds.append(sound(name, path))

    doc = {
        'folder': FOLDER,
        'name': '书2',
        'sounds': sounds,
        'channel': 'music',
        'mode': -1,
        'playing': False,
        'sorting': 'a',
        'ownership': {'default': 0},
        'flags': {},
        '_stats': {
            'compendiumSource': None,
            'duplicateSource': None,
            'coreVersion': '14.365',
            'systemId': 'pf2e',
            'systemVersion': '7.12.2',
        },
        'description': '肆季鬼志 第二幕：任叶飘落 —— 按第1/2/3章叙事顺序排列。'
                       '社区补充曲目（Arch1v3 / lilithilu / Auroroth）+ 库内已有 Book 2 曲目重新点位。',
        'fade': None,
        'seed': None,
    }

    out = os.path.join(BASE, 'fvtt-Playlist-书2.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f'写出 {out}')
    print(f'  新曲目 {n_new} 条 + 重新点位 {n_reuse} 条 = 共 {len(sounds)} 条')
    if missing:
        print(f'  !! 缺 {len(missing)} 个 .ogg，未写进播放列表：')
        for m in missing:
            print('     -', m)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
