#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**enricher 的参数取值**是否还在上游的取值表里 —— 2026-08-14 第十四轮新增。

这一维度此前 qa/ 下 32 个脚本无一覆盖
------------------------------------
* `scan_markup_targets.py` 的 docstring 写着「everything between the brackets is
  machinery」，但实现只有 `if CJK.search(body)` —— 只查中文有没有漏进方括号，
  从不校验参数值是否存在于任何取值表；
* `scan_markup_drift.py` 比的是中英**标记签名**是否一致，非法参数只要中英一致
  就必然全绿；
* 全 qa/ 目录 grep `statusEffects|SKILLS|ACTION.TAGS|RUNES` 只命中
  `restore_enrichers.py:69` 的一句注释，没有任何脚本内置取值表。

后果是：`[[/knowledge legends]]` 被译成 `[[/knowledge 传说]]`、
`[[/skill athletics 15]]` 被译成 `[[/skill 运动 15]]` 这一类改动，三道闸全过、
所有扫描全绿，而 enricher 在运行时查不到这个键，玩家看到的是裸文本。

判据
----
1. **上游权威取值表**直接读系统的 `lang/en.json`（纯 JSON，比解析
   `crucible-compiled.mjs` 稳）：
       skill / skillCheck / skillcheck  ->  SKILL.LABELS 的键
       knowledge                        ->  KNOWLEDGE 的键（小写比）
       rune= (spell / counterspell)     ->  SPELL.RUNES 的键（去掉 *Adj，小写比）
       gesture=                         ->  SPELL.GESTURES 的键（小写比）
       inflection=                      ->  SPELL.INFLECTIONS 的键（去 *Adj，小写比）
2. **英文侧自用词表**：其余动词（`attunement` / `ancestry` / `culture` / `path` /
   `eventState` / `outcome` / `language` / `check` / `save` / `tool` …）的取值是
   各自模块里的文档 id，上游没有集中表可读，就拿**英文基线里实际用过的取值**当表。
3. EN / CN 两侧都跑。CN 侧出现、而权威表里没有的取值：
   * 英文侧也用了同一个取值 -> `UPSTREAM`，上游自己就这么写的，不报（`--show-upstream` 可看）；
   * 英文侧也没有            -> `REPORT`，是译文里改坏的。

**dnd5e 孪生包必须单独分支**：`ember.adventure` / `ember.character` /
`ember.dnd5e-*` 是 dnd5e 侧的同源副本，它们里的 `[[/skill history]]`
`[[/skill perception]]` 用的是 **dnd5e 的技能表**，拿 crucible 的 `SKILL.LABELS`
去比会一次报出 3800+ 条假阳性。这些包一律退回「英文基线自用词表」。
（顺带一提：英文基线自己就有 `awarness` / `athlestics` / `stealth14` / `diplomacy13`
这类越表取值，那是上游的错，判为 `UPSTREAM` 不进 REPORT。）

**不做**路径字符串替换：路径一律走 `os.path.join`（本项目吃过 `'/cn/'->'/en/'`
在 Windows 上静默失配的亏）。

用法：
  python scan_enricher_vocab.py --repo 1-Ember汉化插件 --repo 2-Crucible汉化插件 \
         [--system <crucible 系统目录>] [--out <json>] [--show-upstream]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

# [[/verb  body ]]  —— body 里既有位置参数也有 key=value
CMD = re.compile(r'\[\[/([A-Za-z][\w.-]*)([^\]]*)\]\]')
PARAM = re.compile(r'([A-Za-z][\w-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s\]]+))')
NUMERIC = re.compile(r'^[\d.+\-*/dD@\s]+$')

DEFAULT_SYSTEM = os.path.expandvars(r'%LOCALAPPDATA%\FoundryVTT\Data\systems\crucible')

# 第一个位置参数是词表取值的动词。
POSITIONAL_VERBS = {
    'skill', 'skillCheck', 'skillcheck', 'knowledge', 'language', 'attunement',
    'ancestry', 'culture', 'path', 'eventState', 'eventstate', 'outcome',
    'date', 'check', 'save', 'ability', 'tool', 'attack', 'soundscape', 'spell',
}
# 第一个位置参数是骰式/数字，没有词表可比。
FORMULA_VERBS = {'damage', 'roll', 'r', 'gmroll', 'gmr', 'heal', 'healing',
                 'award', 'hazard', 'item', 'Item', 'talent'}
# 要校验的命名参数。
NAMED_PARAMS = {'rune', 'gesture', 'inflection'}

# 动词 -> 权威表名（在 lang/en.json 里的位置由 authoritative_tables 决定）
VERB_TABLE = {
    'skill': 'skills', 'skillCheck': 'skills', 'skillcheck': 'skills',
    'knowledge': 'knowledge',
}
PARAM_TABLE = {'rune': 'runes', 'gesture': 'gestures', 'inflection': 'inflections'}

# dnd5e 侧的孪生包：同一段冒险的 dnd5e 版本，enricher 参数走 dnd5e 的表。
DND5E_PACK = re.compile(r'^ember\.(adventure|character|dnd5e-.*)\.json$')


def family(pack: str) -> str:
    return 'dnd5e' if DND5E_PACK.match(pack) else 'crucible'


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def authoritative_tables(system_dir):
    """从 <system>/lang/en.json 读出上游硬编码的取值表（键名，不是 label）。"""
    lang = os.path.join(system_dir, 'lang', 'en.json')
    if not os.path.exists(lang):
        return {}, lang
    d = load(lang)
    spell = d.get('SPELL', {})

    def keys(node, drop_adj=False):
        ks = [k for k in (node or {})]
        if drop_adj:
            ks = [k for k in ks if not k.endswith('Adj')]
        return {k.lower() for k in ks}

    return {
        'skills': keys(d.get('SKILL', {}).get('LABELS', {})),
        'knowledge': keys(d.get('KNOWLEDGE', {})),
        'runes': keys(spell.get('RUNES', {}), drop_adj=True),
        'gestures': keys(spell.get('GESTURES', {})),
        'inflections': keys(spell.get('INFLECTIONS', {}), drop_adj=True),
    }, lang


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str) and node:
        out.append(('.'.join(path), node))


def extract(text):
    """产出 (动词, 槽位, 取值)。槽位是 'arg0' 或命名参数名。"""
    for m in CMD.finditer(text):
        verb, body = m.group(1), m.group(2)
        for pm in PARAM.finditer(body):
            name = pm.group(1)
            val = pm.group(2) or pm.group(3) or pm.group(4) or ''
            if name in NAMED_PARAMS and val and not NUMERIC.match(val):
                yield verb, name, val
        if verb in FORMULA_VERBS or verb not in POSITIONAL_VERBS:
            continue
        stripped = PARAM.sub(' ', body)
        toks = [t for t in stripped.split() if t]
        if not toks:
            continue
        val = toks[0]
        if NUMERIC.match(val):
            continue
        yield verb, 'arg0', val


def collect(repos, side):
    """side='en'|'cn' -> list of (repo, pack, path, verb, slot, value)."""
    rows = []
    for repo in repos:
        d = os.path.join(repo, 'compendium', side)
        if not os.path.isdir(d):
            continue
        for pack in sorted(os.listdir(d)):
            if not pack.endswith('.json'):
                continue
            leaves = []
            walk(load(os.path.join(d, pack)), [], leaves)
            for path, text in leaves:
                if '[[/' not in text:
                    continue
                for verb, slot, val in extract(text):
                    rows.append((os.path.basename(os.path.normpath(repo)), pack,
                                 path, verb, slot, val))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--system', default=DEFAULT_SYSTEM)
    ap.add_argument('--out')
    ap.add_argument('--show-upstream', action='store_true',
                    help='连「英文侧也这么写」的越表取值一起打印（默认只记数）')
    ap.add_argument('--selftest', action='store_true',
                    help='往 CN 侧注入 3 条已知样本（2 条该报、1 条不该报）并断言判据确实会响')
    a = ap.parse_args()

    tables, lang_path = authoritative_tables(a.system)
    if not tables:
        print(f'! 读不到上游取值表：{lang_path}\n  用 --system 指向 Foundry 的 crucible 系统目录。')
        raise SystemExit(2)
    print('上游权威取值表（来自 %s）：' % lang_path)
    for k, v in sorted(tables.items()):
        print(f'  {k:<12}{len(v):>4}  {sorted(v)[:6]}')

    en_rows = collect(a.repo, 'en')
    cn_rows = collect(a.repo, 'cn')

    SELFTEST = [
        ('(selftest)', 'crucible.rules.json', 'x.text', 'knowledge', 'arg0', '传说', True),
        ('(selftest)', 'crucible.rules.json', 'x.text', 'skill', 'arg0', '运动', True),
        ('(selftest)', 'crucible.rules.json', 'x.text', 'knowledge', 'arg0', 'legends', False),
    ]
    if a.selftest:
        cn_rows = cn_rows + [row[:6] for row in SELFTEST]
    # 英文侧自用词表：(动词, 槽位) -> 取值集合
    en_vocab = {}
    for _r, pack, _path, verb, slot, val in en_rows:
        en_vocab.setdefault((family(pack), verb, slot), set()).add(val)
    print(f'\n英文侧 enricher 取值 {len(en_rows)} 处 / {len(en_vocab)} 个 (系统,动词,槽位) 词表')
    print(f'中文侧 enricher 取值 {len(cn_rows)} 处')

    reported, upstream = [], []
    for repo, pack, path, verb, slot, val in cn_rows:
        fam = family(pack)
        table_name = None
        if fam == 'crucible':
            table_name = VERB_TABLE.get(verb) if slot == 'arg0' else PARAM_TABLE.get(slot)
        if table_name:
            ok = val.lower() in tables[table_name]
            source = f'upstream table `{table_name}`'
        else:
            ok = val in en_vocab.get((fam, verb, slot), set())
            source = f'EN baseline vocabulary ({fam})'
        if ok:
            continue
        rec = {'repo': repo, 'pack': pack, 'family': fam, 'path': path, 'verb': verb,
               'slot': slot, 'value': val, 'checkedAgainst': source}
        if val in en_vocab.get((fam, verb, slot), set()):
            rec['verdict'] = 'UPSTREAM'      # 英文侧自己也这么写，不是译文改坏的
            upstream.append(rec)
        else:
            rec['verdict'] = 'REPORT'
            reported.append(rec)

    if a.selftest:
        got = {(r['verb'], r['value']) for r in reported}
        ok = True
        for _repo, _pack, _path, verb, _slot, val, should in SELFTEST:
            hit = (verb, val) in got
            flag = 'OK ' if hit == should else 'FAIL'
            ok = ok and hit == should
            print(f'  selftest {flag}  [{verb}] {val!r}  期望{"报出" if should else "不报"}'
                  f'  实际{"报出" if hit else "不报"}')
        print(f'  selftest: {"PASS" if ok else "FAILED"}')
        if not ok:
            raise SystemExit(3)

    print(f'\nCN 侧越表取值：REPORT {len(reported)} 处 | UPSTREAM(英文侧也如此，不报) {len(upstream)} 处')
    for r in reported[:60]:
        print(f'  [{r["verb"]} {r["slot"]}] {r["value"]}\n      {r["pack"]}::{r["path"][:100]}\n'
              f'      比对: {r["checkedAgainst"]}')
    if len(reported) > 60:
        print(f'  ... 还有 {len(reported) - 60} 处')
    if a.show_upstream:
        print('\nUPSTREAM（上游原文就越表，本项目不动）：')
        seen = set()
        for r in upstream:
            k = (r['verb'], r['slot'], r['value'])
            if k in seen:
                continue
            seen.add(k)
            print(f'  [{r["verb"]} {r["slot"]}] {r["value"]}   e.g. {r["pack"]}')

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'_meta': {'report': len(reported), 'upstream': len(upstream),
                                 'langFile': lang_path,
                                 'tables': {k: sorted(v) for k, v in tables.items()}},
                       'report': reported, 'upstream': upstream},
                      f, ensure_ascii=False, indent=2)
        print(f'\n-> {a.out}')
    raise SystemExit(1 if reported else 0)


if __name__ == '__main__':
    main()
