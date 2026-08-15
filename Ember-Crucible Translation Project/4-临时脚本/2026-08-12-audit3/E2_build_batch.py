#!/usr/bin/env python3
"""E2 单元的批次构造器：在**现有中文**上做定点替换，其余字节原样保留。

`apply_translations.py` 是整叶覆盖，所以批次里必须是完整叶值。整页重写会洗掉
已校对的译文（阶段 23 的教训），因此这里以 compendium/cn 的现值为底，
只按 (old, new, 期望次数) 三元组做替换，次数不符就报错退出 —— 宁可不落盘，
也不要在看不见的地方改坏别的字。
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

REPO = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件"
PACK = "ember.crucible-adventure.json"
OUT = (r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/"
       r"e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches/"
       r"E2__ember__ember.crucible-adventure.json")

# path -> [(旧串, 新串, 期望替换次数), ...]
EDITS = {
    # [101] 远征挑战 Overview：三个专名整段没译，库内别处早有定译
    #       Skywarders→天行者(9:3 且 Fernis 传记与 An Auspicious Acquaintance 都用它)
    #       Veron Longspear→维隆·长矛(14 处)，短称 维隆(既有用法)
    "Ember Early Access.journals.The Expedition Challenge.pages.Overview.text": [
        # 正文里不加英文尾巴：同一战役的 An Auspicious Acquaintance 与
        # Fernis Ossa.biography 写的就是裸「天行者」「维隆·长矛」；
        # 内联双语并列在本包的 journal 正文里不是惯例（name 字段才是）
        ('她的队伍“Skywarders”一同夺得优胜', '她的队伍“天行者”一同夺得优胜', 1),
        ('“Skywarders”又赢得了许多其他荣誉', '“天行者”又赢得了许多其他荣誉', 1),
        ('“Skywarders”最终解散', '“天行者”最终解散', 1),
        ('时任公会会长 Veron Longspear 的天然继任者',
         '时任公会会长维隆·长矛的天然继任者', 1),
        # 英文这里用姓氏简称 Longspear；中文裸「长矛」会被读成兵器，故用全名
        ('她直接向 Longspear 汇报', '她直接向维隆·长矛汇报', 1),
        ('Veron Longspear 却在神秘的情形下遇害', '维隆·长矛却在神秘的情形下遇害', 1),
        ('以揭开 Veron 之死', '以揭开维隆之死', 1),
    ],
    # [51][65] Decorators：定角色的那一页（Casing the Joint，队伍在那里挑伪装身份）
    #          与本包多数写作「装饰工」11:5，这两叶是少数派
    "Ember Early Access.journals.Marlstone Manor.pages.Garden Fountains.text": [
        ('<strong>装饰人员</strong>', '<strong>装饰工</strong>', 1),
    ],
    "Ember Early Access.journals.Marlstone Manor.pages.Gallery.text": [
        ('<strong>装饰人员</strong>', '<strong>装饰工</strong>', 1),
    ],
    # [58] Restricted：规则页 Area Overview 自己把 Restricted Rooms 写作「受限房间」，
    #      65 个通行等级框里 14 处作「受限」，这叶的「限制进入」对不上规则页的名词
    "Ember Early Access.journals.Marlstone Manor.pages.Storage.text": [
        ('<strong>限制进入</strong>', '<strong>受限</strong>', 1),
    ],
}


def load_json(path):
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def get(node, parts):
    for p in parts:
        if isinstance(node, list):
            node = node[int(p)]
        else:
            node = node[p]
    return node


def main():
    cn = load_json(os.path.join(REPO, 'compendium', 'cn', PACK))['entries']
    out = {}
    bad = 0
    for path, edits in EDITS.items():
        cur = get(cn, path.split('.'))
        new = cur
        for old, rep, want in edits:
            got = new.count(old)
            if got != want:
                print(f'!! {path}\n   期望 {want} 次，实测 {got} 次: {old[:60]}')
                bad += 1
                continue
            new = new.replace(old, rep)
        if new == cur:
            print(f'-- {path} 无变化，跳过')
            continue
        out[path] = new
        print(f'OK {path}  {len(cur)} -> {len(new)} 字符')
    if bad:
        print(f'\n有 {bad} 条替换匹配失败，未写出批次')
        sys.exit(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n写出 {len(out)} 条 -> {OUT}')


if __name__ == '__main__':
    main()
