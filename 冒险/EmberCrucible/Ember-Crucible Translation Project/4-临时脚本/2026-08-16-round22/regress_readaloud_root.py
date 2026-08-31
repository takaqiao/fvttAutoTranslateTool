#!/usr/bin/env python3
"""真库回测（磁盘副本 + 完整主闸入口），不是内存模拟。

做法：把两仓 compendium / lang / module.json 与 5-其他内容、PROJECT.md 复制到副本树，
用 `assert_resolutions.py --root <副本>` 跑三遍：
  ① 原样      → R-readaloud-coverage 必须**绿**（特异度）
  ② 删空      → 必须**红**（灵敏度：把那段 311 字的中文朗读正文整段删掉）
  ③ 砍掉一半  → 报或不报都要如实记下来（判据边界）
每一步都打印「改了几处」，改 0 处就是空转，直接判定本回测失败。
"""
import json
import os
import re
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
SCRIPT = os.path.join(P, '3-常用脚本', 'qa', 'assert_resolutions.py')
COPY = os.path.join(os.environ.get('TEMP', HERE), 'ec-round22-root')

TARGET_PAGE = 'Dusktide Destruction'
NEEDLE = '暮潮'          # 只用来找那一段，真正的定位靠 readaloud= 参数本体


def build_copy():
    if os.path.isdir(COPY):
        shutil.rmtree(COPY)
    os.makedirs(COPY)
    for rel in ('1-Ember汉化插件', '2-Crucible汉化插件'):
        for sub in ('compendium', 'lang'):
            s = os.path.join(P, rel, sub)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(COPY, rel, sub))
        mj = os.path.join(P, rel, 'module.json')
        if os.path.exists(mj):
            shutil.copy2(mj, os.path.join(COPY, rel, 'module.json'))
    shutil.copytree(os.path.join(P, '5-其他内容'), os.path.join(COPY, '5-其他内容'))
    for f in ('PROJECT.md',):
        if os.path.exists(os.path.join(P, f)):
            shutil.copy2(os.path.join(P, f), os.path.join(COPY, f))
    print(f'副本建好：{COPY}')


RA = re.compile(r'(readaloud=")([^"]*)(")')


def mutate(mode, min_len=300):
    """在副本的 cn/*.json 里改中文朗读正文。返回改动处数。

    ⚠ **必须在 JSON 解码后的字符串上改，不能在文件原文上正则替换** —— 库里的
    `readaloud="…"` 在 JSON 源码里是 `readaloud=\\"…\\"`（引号被转义），
    第一版直接对原文跑 `readaloud="` 匹配到 **0 处**、回测当场空转。
    这就是任务书里点名的第 5 条：验证工具本身也会空转，所以每一步都要报「实改几处」。
    """
    n = [0]

    def fix(s):
        def repl(m):
            v = m.group(2)
            if len(v) < min_len:
                return m.group(0)
            new = '' if mode == 'wipe' else v[:len(v) // 2]
            n[0] += 1
            print(f'  改一处：{len(v)} 字 -> {len(new)} 字　{v[:18]}…')
            return m.group(1) + new + m.group(3)
        return RA.sub(repl, s)

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return fix(o)
        return o

    cn_dir = os.path.join(COPY, '1-Ember汉化插件', 'compendium', 'cn')
    for fn in sorted(os.listdir(cn_dir)):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(cn_dir, fn)
        d = json.load(open(p, encoding='utf-8'))
        d2 = walk(d)
        if d2 != d:
            json.dump(d2, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return n[0]


def run(tag):
    r = subprocess.run([sys.executable, SCRIPT, '--root', COPY, '--verbose', '--max-show', '4'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    lines = r.stdout.splitlines()
    mine = [i for i, l in enumerate(lines) if 'R-readaloud-coverage' in l]
    print(f'\n===== {tag} =====')
    for i in mine:
        for l in lines[i:i + 6]:
            print('  ' + l.strip()[:300])
    tail = [l for l in lines if l.startswith('通过 ')]
    print('  ' + (tail[-1] if tail else '(没读到总计行)'))
    verdict = 'FAIL' if any(l.strip().startswith('FAIL  R-readaloud-coverage') for l in lines) else 'ok'
    return verdict


def main():
    build_copy()
    base = run('① 原样（特异度：必须 ok）')
    n1 = mutate('wipe')
    print(f'\n[删空] 实改 {n1} 处 —— 改 0 处就是空转')
    assert n1 > 0, '一处都没改到，回测本身空转了'
    wipe = run('② 中文朗读正文删空（灵敏度：必须 FAIL）')

    shutil.rmtree(COPY)
    build_copy()
    n2 = mutate('half')
    print(f'\n[砍一半] 实改 {n2} 处')
    assert n2 > 0, '一处都没改到，回测本身空转了'
    half = run('③ 中文朗读正文砍掉后一半')

    print('\n结论：原样=' + base + ' · 删空=' + wipe + ' · 砍一半=' + half)
    print('期望：原样 ok / 删空 FAIL / 砍一半 FAIL')
    shutil.rmtree(COPY)


if __name__ == '__main__':
    main()
