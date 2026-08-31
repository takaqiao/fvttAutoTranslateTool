#!/usr/bin/env python3
"""
Wave 4 conservative prose simplification for FotRP bestiary + addons.

After analyzing the candidate corpus:
- ALL p1 (对X进行Y) candidates live inside `.items.X.description` and are mechanic
  (e.g., 对触及内一个生物进行一次拳头打击 / 对该生物进行推撞 / 对它进行了敌对行为).
- ALL p2 (执行Y, non-者) candidates live in `.items.X.description` and are mechanic
  (执行这个动作 / 执行精确动作 / 执行带有「操作」特征的动作).
- 82 of 84 p3 (通过X来Y) candidates live in `.items.X.description` and are mechanic
  (通过攻击触手来尝试脱离 / 通过维持该法术的一部分来命令它 / 通过10轮专注来影响 ...).
- Only 2 p3 candidates live in `publicNotes` and are true prose:
    * `通过祭祀来安抚它们` (Tamikan publicNotes)
    * `通过信息素和气味腺来维系着紧密的家族结构` (Dromornis publicNotes)

Conservative wave: ONLY simplify p3 inside non-mechanic prose paragraphs of non-item paths
(publicNotes / system.details.publicNotes / etc), AND apply paragraph-level mechanic guard.
"""

import json
import re
import sys
import os
import shutil
from datetime import datetime

FILES = [
    r'FotRP/需要翻译/pf2e.fists-of-the-ruby-phoenix-bestiary.json',
    r'FotRP/需要翻译/fist-of-the-ruby-phoenix-addons.fist-of-the-ruby-phoenix-addons.json',
]

# Path-level skip: any path containing these segments is treated as inherently mechanic.
MECHANIC_PATH_SEGMENTS = [
    '.items.',                # actor.items.X.description - spell/ability/feat mechanics
    '.system.actions.',
    '.system.attack',
    '.system.damage',
    '.system.proficiency',
    '.system.saves',
    '.system.skills',
    '.system.combat',
]

# Paragraph-level keywords that flag a paragraph as mechanic.
MECHANIC_KEYWORDS = [
    '豁免', '检定', '反应', '抗力', '弱点', '免疫', '打击', '攻击骰',
    '反应动作', '自由动作', '集中力', '专注力', '维持动作',
    '基础伤害', '伤害骰', '影响范围', '范围内', '一次',
    '@UUID', '@Check', '@Damage', '@Template', '@Localize',
    '攻击加值', '攻击加成', '专长',
    'DC', '动作', '法术', '回合',
]

HAS_MECHANIC_INLINE_RE = re.compile(
    r'@UUID|@Check|@Damage|@Template|@Localize|@Compendium|@Embed|'
    r'DC\s*\d+|\d+d\d+|\d+点|\+\d+|−\d+|\b\d+ft\b|尺范围|尺锥形|尺爆发',
    re.IGNORECASE,
)


def is_mechanic_path(path):
    for seg in MECHANIC_PATH_SEGMENTS:
        if seg in path:
            return True
    return False


def is_mechanic_paragraph(text):
    if HAS_MECHANIC_INLINE_RE.search(text):
        return True
    for k in MECHANIC_KEYWORDS:
        if k in text:
            return True
    return False


stats = {
    'p3_通过来_applied': 0,
    'p3_通过来_skipped_path': 0,
    'p3_通过来_skipped_paragraph_keyword': 0,
    'paths_processed': 0,
    'paths_skipped_by_path': 0,
}


def simplify_p3_in_paragraph(text):
    cnt = 0
    def repl(m):
        nonlocal cnt
        cnt += 1
        return '以' + m.group(1) + '来' + m.group(2)
    new = re.sub(r'通过([^,，。;；!?!?\"\\\n]{1,14}?)来([^,，。;；!?!?\"\\\n]{1,14})', repl, text)
    return new, cnt


def walk_text(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                yield (path + '.' + k, v, lambda nv, _d=obj, _k=k: _d.__setitem__(_k, nv))
            else:
                yield from walk_text(v, path + '.' + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                yield (path + f'[{i}]', v, lambda nv, _l=obj, _i=i: _l.__setitem__(_i, nv))
            else:
                yield from walk_text(v, path + f'[{i}]')


def split_paragraphs(text):
    chunks = re.split(
        r'(</?p>|</?li>|</?div>|</?h[1-6]>|</?td>|</?th>|\n\n+)',
        text,
    )
    return chunks


def process_text(path, text):
    if not text or not isinstance(text, str):
        return text, False
    if '通过' not in text:
        return text, False
    # paragraph-level processing
    chunks = split_paragraphs(text)
    changed = False
    new_chunks = []
    for chunk in chunks:
        if not chunk:
            new_chunks.append(chunk)
            continue
        if re.match(r'^</?(p|li|div|h[1-6]|td|th)>$|^\n\n+$', chunk):
            new_chunks.append(chunk)
            continue
        if '通过' not in chunk:
            new_chunks.append(chunk)
            continue
        if is_mechanic_paragraph(chunk):
            c = len(re.findall(r'通过[^,，。;；!?!?\"\\\n]{1,14}?来', chunk))
            stats['p3_通过来_skipped_paragraph_keyword'] += c
            new_chunks.append(chunk)
            continue
        new_chunk, c3 = simplify_p3_in_paragraph(chunk)
        stats['p3_通过来_applied'] += c3
        if new_chunk != chunk:
            changed = True
        new_chunks.append(new_chunk)
    return ''.join(new_chunks), changed


def process_file(path, dry_run=False):
    lines = [f'=== {path} ===']
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    file_changed = 0
    samples = []
    for p, v, setter in walk_text(data):
        if not isinstance(v, str):
            continue
        if is_mechanic_path(p):
            # count would-be hits for stats
            if '通过' in v:
                c = len(re.findall(r'通过[^,，。;；!?!?\"\\\n]{1,14}?来', v))
                stats['p3_通过来_skipped_path'] += c
            stats['paths_skipped_by_path'] += 1
            continue
        stats['paths_processed'] += 1
        new_v, ch = process_text(p, v)
        if ch:
            file_changed += 1
            for i, (a, b) in enumerate(zip(v, new_v)):
                if a != b:
                    s = max(0, i - 40)
                    e = min(len(v), i + 80)
                    samples.append((p, v[s:e], new_v[s:e]))
                    break
            if not dry_run:
                setter(new_v)
    if not dry_run and file_changed > 0:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = 'FotRP/_backup/wave4_simplify_' + ts
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        # validate
        with open(path, 'r', encoding='utf-8') as fh:
            json.load(fh)
        lines.append(f'  written, backup: {backup_dir}/{os.path.basename(path)}')
    lines.append(f'  text values changed: {file_changed}')
    for p, before, after in samples:
        lines.append(f'  sample: {p}')
        lines.append(f'    before: ...{before}...')
        lines.append(f'    after:  ...{after}...')
    return lines


def main():
    dry = '--dry' in sys.argv
    log_path = 'FotRP/_simplify_prose_dry.report.txt' if dry else 'FotRP/_simplify_prose_apply.report.txt'
    out = []
    for f in FILES:
        out.extend(process_file(f, dry_run=dry))
    out.append('')
    out.append('=== STATS ===')
    for k, v in stats.items():
        out.append(f'  {k}: {v}')
    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out))
    sys.stdout.write(f'Report: {log_path}\n')
    sys.stdout.write(f'Stats: {stats}\n')


if __name__ == '__main__':
    main()
