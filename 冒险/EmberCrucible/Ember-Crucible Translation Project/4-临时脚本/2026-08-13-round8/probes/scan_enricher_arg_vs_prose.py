#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""enricher 参数（不可译，照抄）与它旁边／同表的中文术语对不上。

判据
----
`[[/skillCheck awareness 14]]` / `@Condition[weakened]` / `[[/knowledge shent]]`
这类增强器，**方括号里的参数是照抄不译的**，但它在玩家屏幕上渲染出来的是
一个**中文词** —— 这个中文词由 `lang/cn.json`（或 Ember 的运行时补丁表）唯一确定，
和 compendium 正文一个字都不沾边。两条通道各译各的，就会同屏打架：
玩家读到「……受拘束的生物可以挣脱」，而同一句里的链接写着「受缚」。

既有判据为什么全盲
------------------
  * `scan_markup_drift` 为了让 `{标签}` 可译，特地把花括号剥掉，只比方括号；
  * `scan_uuid_swap` 只认 `@UUID`；
  * `scan_cross_channel` 的 B 段只比 `.mjs` ↔ lang 的**键**，不看正文；
  * `scan_label_vs_name` 比的是 `@UUID{标签}` ↔ 目标文档 name。
  **参数渲染出来的词 ↔ compendium 中文正文**，这一对从来没有人比过。

两种模式
--------
`--mode adjacent`（默认）：增强器**邻域**。英文侧该增强器前后 W 个**可见**字符里
    明写了该术语 → 中文侧同一处前后 W2 个可见字符里必须有那个定译。
`--mode cell`：**术语清单表**。表格/列表里一格就是一个术语（玩家指南的
    「知识领域列表」、crucible.rules 的祖裔抗性表），英文那格就是术语本身，
    零歧义。要求同叶同族至少 `--min-list` 格才算清单表（否则怪物数据块会淹掉判据）。

渲染链路是从系统源码读出来的，不是猜的
--------------------------------------
    `[[/skillCheck X]]`  → crucible `SYSTEM.SKILLS[X].label`      → SKILL.LABELS.X
    `[[/skill X]]`       → `DND5E_SKILL_MAPPING[X]` → 同上（5e 写法映射到 crucible 技能）
    `@Condition[X]`      → `enrichRule("condition."+X)`           → statuses.mjs 的 name 键
    `[[/knowledge X]]`   → `ACTOR.KnowledgeSpecific{knowledge}`   → KNOWLEDGE.* / Ember 补丁表
    `[[/language X]]`    → `ACTOR.LanguageSpecific{language}`     → LANGUAGES.* / Ember 补丁表
    `[[/hazard N d r t]]`→ DEFENSES.* / RESOURCES.* / DAMAGE.*（tooltip）
    `[[/damage F t]]`    → DAMAGE.*
    `[[/attunement X]]`  → `Attunement: ${ATTUNEMENTS[X].label}`  → ember-hardcoded-cn 补丁表
英文名取 crucible 本体 `lang/en.json`，中文名取本项目 `lang/cn.json`
（Ember 新增的语言/知识/同调取 `scripts/ember-hardcoded-cn.mjs` 的运行时补丁表）。

两个必须踩过的坑（都是实测踩出来的）
------------------------------------
1. **窗口只能装可见字符。** 第一版直接切原串，英文闸被隔壁兄弟增强器的参数拼写喂饱了 ——
   `[[/skill investigation 13]]` 旁边就摆着 `[[/skillCheck awareness 13]]`，
   字面量里那个 `awareness` 让闸误以为「英文正文写了 Awareness」，2703 条里绝大多数这么来的。
   剔掉 HTML 标签、方括号参数、上游遗留的 `&amp;Reference[...]` 坏标记与残缺增强器之后剩 36 条。
2. **窗口边界要留余量**，否则 `[[/damage 1d6 Radiant]]` 前面的「光耀之火」被切成「耀之火」，白报一条。

已知假阳性模式（报出来要人过一遍，不能直接落盘）
------------------------------------------------
adjacent 模式的主要假阳性是「英文术语同时也是普通英文词」：
Legends / Trade / Machines / Crime / Common / Performance / Poison / Elementals / Undeath。
英文正文里那个词是普通名词（"the legends about her stories"、"common reverence"），
不是术语，中文自然不会用定译。36 条里约 26 条属于这一类。
cell 模式没有这个问题（英文格里孤零零一个词），实测 0 假阳性。

用法
----
  python scan_enricher_arg_vs_prose.py --repo <repoDir> [--repo <另一个>]
         [--mode adjacent|cell] [--out findings.json] [--w-en 90] [--w-cn 50]
         [--only skillCheck,condition] [--min-list 5] [--rival-only]

`--sys-crucible` / `--mod-ember` 指向已安装的本体（默认 Foundry 默认路径），
用来读渲染链路的常量与英文名；**只读**，本脚本不写 compendium/ 与 lang/。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEF_SYS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"
DEF_MOD = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"

# ------------------------------------------------------------------ 遍历

SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            yield from walk(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f'{path}.{i}')
    elif isinstance(obj, str) and obj:
        yield path, obj


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f'{p}.{k}' if p else k)
    elif isinstance(o, str):
        yield p, o


# ------------------------------------------------------------------ 术语表

def js_table(src, start_marker, end_marker='}'):
    """从 .mjs 里抠一段 `key: "value"` 表（只做粗解析，够用即可）。"""
    i = src.find(start_marker)
    if i < 0:
        return {}
    j = src.find(end_marker, i)
    return src[i:j]


def build_registry(sys_dir, mod_dir, repos):
    """返回 {family: {param: {'key':..,'en':..,'cn':..}}}"""
    sys_en = dict(flat(load(os.path.join(sys_dir, 'lang', 'en.json'))))

    cn = {}
    for repo in repos:                       # 两个仓库的 lang/cn.json 合并
        p = os.path.join(repo, 'lang', 'cn.json')
        if os.path.exists(p):
            cn.update(load(p))

    reg = collections.defaultdict(dict)

    def add(fam, param, key, en_override=None, cn_override=None):
        en = en_override if en_override is not None else sys_en.get(key)
        zh = cn_override if cn_override is not None else cn.get(key)
        if not en or not zh:
            return
        reg[fam][param] = {'key': key, 'en': en, 'cn': zh}

    # --- 技能：crucible/module/const/skills.mjs 里的 SKILLS
    skills_src = open(os.path.join(sys_dir, 'module', 'const', 'skills.mjs'),
                      encoding='utf-8').read()
    for m in re.finditer(r'\n  (\w+): \{\n    label: "(SKILL\.LABELS\.\w+)"', skills_src):
        add('skillCheck', m.group(1), m.group(2))
    # 5e 技能别名 -> crucible 技能（crucible/module/enrichers.mjs 的 DND5E_SKILL_MAPPING）
    enr_src = open(os.path.join(sys_dir, 'module', 'enrichers.mjs'), encoding='utf-8').read()
    blk = js_table(enr_src, 'const DND5E_SKILL_MAPPING = {')
    dnd5e_map = dict(re.findall(r'(\w+):\s*"(\w+)"', blk))

    # --- 知识领域：skills.mjs 里的 KNOWLEDGE（label 指向 KNOWLEDGE.*）
    for m in re.finditer(r'\n  (\w+): \{label: "(KNOWLEDGE\.\w+)"', skills_src):
        add('knowledge', m.group(1), m.group(2))

    # --- 状态：statuses.mjs 的 id -> name 键
    st_src = open(os.path.join(sys_dir, 'module', 'const', 'statuses.mjs'),
                  encoding='utf-8').read()
    for m in re.finditer(r'\n  (\w+): \{\n    id: "\w+",\n    name: "(ACTIVE_EFFECT\.STATUSES\.\w+)"',
                         st_src):
        add('condition', m.group(1), m.group(2))

    # --- 伤害类型 / 防御 / 资源：attributes.mjs
    at_src = open(os.path.join(sys_dir, 'module', 'const', 'attributes.mjs'),
                  encoding='utf-8').read()
    for fam, pat in (('damage', r'\n  (\w+): \{[^}]*?label: "(DAMAGE\.\w+)"'),
                     ('defense', r'\n  (\w+): \{[^}]*?label: "(DEFENSES\.\w+)"'),
                     ('resource', r'\n  (\w+): \{[^}]*?label: "(RESOURCES\.\w+)"')):
        for m in re.finditer(pat, at_src, re.S):
            add(fam, m.group(1), m.group(2))

    # --- 语言：crucible 本体只有 Common / Sign
    add('language', 'common', 'LANGUAGES.Common')
    add('language', 'sign', 'LANGUAGES.Sign')

    # --- Ember 新增的语言/知识：英文 label 写死在 ember.mjs，
    #     中文写死在本项目 scripts/ember-hardcoded-cn.mjs 的运行时补丁表
    ember_lang, ember_know = {}, {}
    ep = os.path.join(mod_dir, 'scripts', 'ember.mjs')
    if os.path.exists(ep):
        esrc = open(ep, encoding='utf-8', errors='replace').read()
        b = js_table(esrc, 'crucible.CONFIG.languages, {')
        ember_lang = dict(re.findall(r"(\w+): \{label: \"([^\"]+)\"", b))
        b = js_table(esrc, 'crucible.CONFIG.knowledge, {')
        ember_know = dict(re.findall(r"(\w+): \{label: \"([^\"]+)\"", b))
        # aliases: outsiders -> abyssals（Ember 把 outsiders 删了并做成别名）
        for m in re.findall(r'(\w+): \{label: "[^"]+", skill: "\w+", aliases: \[([^\]]+)\]', b):
            for al in re.findall(r'"(\w+)"', m[1]):
                ember_know[al] = ember_know.get(m[0], '')

    cn_tables = {}
    for repo in repos:
        hp = os.path.join(repo, 'scripts', 'ember-hardcoded-cn.mjs')
        if not os.path.exists(hp):
            continue
        hsrc = open(hp, encoding='utf-8').read()
        for name in ('LANGUAGES', 'KNOWLEDGE', 'ATTUNEMENTS'):
            b = js_table(hsrc, f'const {name} = {{')
            cn_tables[name] = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', b))

    for pid, en_label in ember_lang.items():
        zh = cn_tables.get('LANGUAGES', {}).get(en_label)
        if zh:
            reg['language'][pid] = {'key': f'<ember.languages.{pid}>', 'en': en_label, 'cn': zh}
    for pid, en_label in ember_know.items():
        zh = cn_tables.get('KNOWLEDGE', {}).get(en_label)
        if zh:
            reg['knowledge'][pid] = {'key': f'<ember.knowledge.{pid}>', 'en': en_label, 'cn': zh}

    # --- 同调（ember.CONST.ATTUNEMENTS，id 与 identifier 两种写法都在正文里出现）
    ap = os.path.join(mod_dir, 'scripts', 'dnd5e-async.mjs')
    if os.path.exists(ap):
        asrc = open(ap, encoding='utf-8', errors='replace').read()
        b = js_table(asrc, 'const ATTUNEMENTS = {', '\n};')
        for m in re.finditer(r'(\w+): \{id: "\w+", identifier: "(\w+)", label: "([^"]+)"\}', b):
            zh = cn_tables.get('ATTUNEMENTS', {}).get(m.group(3))
            if not zh:
                continue
            for pid in (m.group(1), m.group(2)):
                reg['attunement'][pid] = {'key': f'<ember.ATTUNEMENTS.{m.group(1)}>',
                                          'en': m.group(3), 'cn': zh}

    return reg, dnd5e_map


# ------------------------------------------------------------------ 增强器切分

# family -> (正则, 取参数的函数)  参数函数返回 [(family_for_lookup, param, span)]
RX_SKILLCHECK = re.compile(r'\[\[/skillCheck,? +([^\]]+)\]\]')
RX_SKILL5E = re.compile(r'\[\[/skill +([^\]]+)\]\]')
RX_KNOW = re.compile(r'\[\[/knowledge +(\w+)\]\]')
RX_LANG = re.compile(r'\[\[/language +(\w+)\]\]')
RX_COND = re.compile(r'@Condition\[(\w+)\]')
RX_HAZ = re.compile(r'\[\[/hazard +([^\]]+)\]\]')
RX_DMG = re.compile(r'\[\[/damage +([^\]]+)\]\]')
RX_ATTUNE = re.compile(r'\[\[/attunement +(\w+)')


def enricher_terms(text, reg, dnd5e_map, families):
    """产出 (family, param, term, match_span, literal)"""
    out = []

    def emit(fam, param, m):
        t = reg.get(fam, {}).get(param)
        if t:
            out.append((fam, param, t, m.span(), m.group(0)))

    if 'skillCheck' in families:
        for m in RX_SKILLCHECK.finditer(text):
            emit('skillCheck', m.group(1).split()[0], m)
    if 'skill5e' in families:
        for m in RX_SKILL5E.finditer(text):
            sid = m.group(1).split()[0]
            mapped = dnd5e_map.get(sid) or dnd5e_map.get(sid.lower())
            if mapped:
                t = reg['skillCheck'].get(mapped)
                if t:
                    out.append(('skill5e', sid, t, m.span(), m.group(0)))
    if 'knowledge' in families:
        for m in RX_KNOW.finditer(text):
            emit('knowledge', m.group(1), m)
    if 'language' in families:
        for m in RX_LANG.finditer(text):
            emit('language', m.group(1), m)
    if 'condition' in families:
        for m in RX_COND.finditer(text):
            emit('condition', m.group(1), m)
    if 'hazard' in families:
        for m in RX_HAZ.finditer(text):
            for tok in m.group(1).split()[1:]:
                for fam in ('defense', 'resource', 'damage'):
                    t = reg.get(fam, {}).get(tok)
                    if t:
                        out.append((f'hazard:{fam}', tok, t, m.span(), m.group(0)))
                        break
    if 'damage' in families:
        for m in RX_DMG.finditer(text):
            for tok in m.group(1).split():
                t = reg['damage'].get(tok.lower())
                if t:
                    out.append(('damage', tok.lower(), t, m.span(), m.group(0)))
                    break
    if 'attunement' in families:
        for m in RX_ATTUNE.finditer(text):
            emit('attunement', m.group(1), m)
    return out


ALL_FAMILIES = ['skillCheck', 'skill5e', 'knowledge', 'language',
                'condition', 'hazard', 'damage', 'attunement']

# ------------------------------------------------------------------ 可见文本窗口
#
# 关键：窗口必须只含**玩家看得见的字**。第一版直接切原串，结果英文闸被
# 隔壁那条兄弟增强器的参数拼写喂饱了 —— `[[/skill investigation 13]]` 旁边
# 就摆着 `[[/skillCheck awareness 13]]`，字面量里那个 awareness 让闸误以为
# 「英文正文写了 Awareness」，2703 条里绝大多数是这么来的。
# 所以先把 HTML 标签与方括号参数剔掉，只留正文与 `{标签}` 里的字（那才是渲染出来的）。

RX_TAG = re.compile(r'<[^>]*>')
RX_ENR = re.compile(r'(?:@\w+\[[^\]]*\]|\[\[[^\]]*\]\])(\{[^}]*\})?')
# 上游 5e→crucible 移植时留下的坏标记，两侧都有，**不是**正文：
#   `&amp;Reference[prone]` / `&reference[grappled]`（Foundry 里会原样显示成一串英文垃圾）
RX_REF = re.compile(r'&(?:amp;)?[Rr]eference\[[^\]]*\]')
# 缺一个方括号的残缺增强器：`[/skill arcana 13]]`、`[[/skillCheck awareness 13`（没有收尾）。
# 只在正常增强器都吃掉之后再跑，所以命中的都是残缺件；宁可多吃两个词也不要让参数拼写混进英文闸。
RX_BROKEN = re.compile(r'\[{1,2}/\w+(?:\s+[-\w=@.]+){0,5}\]{0,2}')


def segments(text):
    """把一段文本切成 (可见?, 原串起点, 原串终点, 可见文字) 的序列。"""
    marks = []
    for m in RX_TAG.finditer(text):
        marks.append((m.start(), m.end(), ''))          # 标签整段不可见
    for m in RX_REF.finditer(text):
        marks.append((m.start(), m.end(), ''))
    taken = [(m.start(), m.end()) for m in RX_ENR.finditer(text)]
    for m in RX_BROKEN.finditer(text):
        if not any(s <= m.start() < e for s, e in taken):
            marks.append((m.start(), m.end(), ''))
    for m in RX_ENR.finditer(text):
        lab = m.group(1)
        # 方括号部分不可见；`{标签}` 里的字是渲染出来的，算可见
        if lab:
            marks.append((m.start(), m.end() - len(lab), ''))
            marks.append((m.end() - len(lab), m.end(), lab[1:-1]))
        else:
            marks.append((m.start(), m.end(), ''))
    marks.sort()
    out, pos = [], 0
    for s, e, vis in marks:
        if s < pos:
            continue
        if s > pos:
            out.append((True, pos, s, text[pos:s]))
        out.append((bool(vis), s, e, vis))
        pos = e
    if pos < len(text):
        out.append((True, pos, len(text), text[pos:]))
    return out


def visible_window(text, span, w, pad=0):
    """以 span 为中心，向两侧各取 w(+pad) 个**可见**字符。span 自身完全排除。

    `pad` 是给「窗口边界把词切成两半」留的余量 —— 不给的话
    `[[/damage 1d6 Radiant]]` 前面的「光耀之火」会被切成「耀之火」，白报一条。
    """
    segs = segments(text)
    before = ''.join(t for vis, s, e, t in segs if e <= span[0] and vis)
    after = ''.join(t for vis, s, e, t in segs if s >= span[1] and vis)
    k = w + pad
    return (before[-k:] if before else ''), (after[:k] if after else '')


# ------------------------------------------------------------------ B 模式：整格对整词
#
# A 模式（增强器邻域）只能抓到「链接和它旁边那句话打架」。但同一类错还有一个更整齐的
# 落点：**表格里把术语单列一格**。玩家指南的「知识领域列表 / 技能列表 / 状态列表」
# 就是这种，一格一个术语，英文那格就是术语本身，没有任何歧义 ——
# 中文那格必须与 lang 渲染出来的那个词逐字相同，否则角色卡上写 A、指南上写 B。
# 这一档 0 歧义（英文格子里就一个词），所以假阳性极低。

RX_CELL = re.compile(r'<(td|li)\b[^>]*>(.*?)</\1>', re.S)


def cells(text):
    # 注意：**不能**在这里按长度过滤 —— 英文格 42 字、中文格 20 字的时候
    # 两边格数就不等了，整张表被 `格数不等` 丢掉（玩家指南那张知识领域表就是这么漏的）。
    return [(m.start(), RX_TAG.sub('', m.group(2)).strip())
            for m in RX_CELL.finditer(text)]


def nth_literal_pos(text, lit, n):
    """第 n 次出现 lit 的 span；找不到返回 None"""
    i = -1
    for _ in range(n + 1):
        i = text.find(lit, i + 1)
        if i < 0:
            return None
    return (i, i + len(lit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--sys-crucible', default=DEF_SYS)
    ap.add_argument('--mod-ember', default=DEF_MOD)
    ap.add_argument('--w-en', type=int, default=90)
    ap.add_argument('--w-cn', type=int, default=50)
    ap.add_argument('--only', help='逗号分隔的 family 子集')
    ap.add_argument('--rival-only', action='store_true',
                    help='只报「窗口里出现了同族别的术语定译」的那一档')
    ap.add_argument('--min-list', type=int, default=5,
                    help='cell 模式：同叶同族至少多少格才算「术语清单表」')
    ap.add_argument('--mode', default='adjacent', choices=['adjacent', 'cell'],
                    help='adjacent=增强器邻域；cell=表格/列表里整格就是一个术语')
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=60)
    a = ap.parse_args()

    families = [f.strip() for f in a.only.split(',')] if a.only else ALL_FAMILIES
    reg, dnd5e_map = build_registry(a.sys_crucible, a.mod_ember, a.repo)

    # 同族定译反查表：中文 -> 参数名（用来找 rival）
    fam_cn = collections.defaultdict(dict)
    for fam, d in reg.items():
        for p, t in d.items():
            fam_cn[fam].setdefault(t['cn'], p)
    fam_cn['skill5e'] = fam_cn['skillCheck']
    for f in ('defense', 'resource', 'damage'):
        fam_cn[f'hazard:{f}'] = fam_cn[f]

    stats = collections.Counter()
    hits = []

    for repo in a.repo:
        en_dir = os.path.join(repo, 'compendium', 'en')
        cn_dir = os.path.join(repo, 'compendium', 'cn')
        if not os.path.isdir(en_dir):
            continue
        for pack in sorted(os.listdir(en_dir)):
            if not pack.endswith('.json') or pack.startswith('_'):
                continue
            cp = os.path.join(cn_dir, pack)
            if not os.path.exists(cp):
                continue
            en = dict(walk(load(os.path.join(en_dir, pack)).get('entries', {})))
            cnm = dict(walk(load(cp).get('entries', {})))
            for path, ev in en.items():
                cv = cnm.get(path)
                if not cv:
                    continue
                stats['叶对'] += 1
                if a.mode == 'cell':
                    ec, cc = cells(ev), cells(cv)
                    if len(ec) != len(cc):
                        stats['格数不等（跳过）'] += 1
                        continue
                    # 先按 family 归堆：只有当**同一叶里同族术语占了 >=N 格**时，
                    # 这张表才是「术语清单表」。否则一格 Fire、一格 Armor 的怪物数据块
                    # 会把判据淹掉（未加这道闸时 689 条，几乎全是数据块）。
                    bucket = collections.defaultdict(list)
                    for (_, etx), (_, ctx) in zip(ec, cc):
                        for fam, d in reg.items():
                            for param, term in d.items():
                                if etx.strip() == term['en']:
                                    bucket[fam].append((param, term, etx, ctx))
                    for fam, items in bucket.items():
                        if len(items) < a.min_list:
                            stats[f'不是术语清单表（{fam}，{len(items)} 格）'] += 1
                            continue
                        for param, term, etx, ctx in items:
                            stats[f'候选格 {fam}'] += 1
                            if term['cn'] in ctx:
                                stats['中文一致'] += 1
                                continue
                            stats['**不一致**'] += 1
                            hits.append({
                                'repo': os.path.basename(repo.rstrip('\\/')),
                                'pack': pack, 'path': path, 'batch_path': path,
                                'family': f'cell:{fam}', 'param': param,
                                'lang_key': term['key'], 'term_en': term['en'],
                                'term_cn': term['cn'], 'en_capitalized': True,
                                'rival': [], 'enricher': '<清单格>',
                                'en_ctx': etx, 'cn_ctx': ctx,
                            })
                    continue
                seen = collections.Counter()
                for fam, param, term, span, lit in enricher_terms(ev, reg, dnd5e_map, families):
                    ordinal = seen[lit]
                    seen[lit] += 1
                    stats[f'候选 {fam}'] += 1
                    # --- 英文闸：英文侧窗口里必须明写该术语
                    ea, eb = visible_window(ev, span, a.w_en, pad=len(term['en']))
                    rx = re.compile(r'(?<![A-Za-z])' + re.escape(term['en']) + r'(?![A-Za-z])',
                                    re.IGNORECASE)
                    mm = rx.search(ea) or rx.search(eb)
                    if not mm:
                        stats['英文侧没写该术语（不判）'] += 1
                        continue
                    capped = mm.group(0)[:1].isupper()
                    stats['过英文闸'] += 1
                    # --- 中文侧：定位同一个增强器
                    cspan = nth_literal_pos(cv, lit, ordinal)
                    if cspan is None:
                        stats['中文侧找不到同一处增强器（标记漂移，另有判据）'] += 1
                        continue
                    ca, cb = visible_window(cv, cspan, a.w_cn, pad=len(term['cn']))
                    if term['cn'] in ca or term['cn'] in cb:
                        stats['中文一致'] += 1
                        continue
                    # --- 找 rival：窗口里有没有同族别的术语的定译
                    rival = []
                    for zh, other in fam_cn.get(fam, {}).items():
                        if other == param or zh == term['cn']:
                            continue
                        if zh in ca or zh in cb:
                            rival.append(f'{zh}({other})')
                    stats['**不一致**'] += 1
                    if a.rival_only and not rival:
                        continue
                    hits.append({
                        'repo': os.path.basename(repo.rstrip('\\/')),
                        'pack': pack,
                        'path': path,
                        'batch_path': path,
                        'family': fam,
                        'param': param,
                        'lang_key': term['key'],
                        'term_en': term['en'],
                        'term_cn': term['cn'],
                        'en_capitalized': capped,
                        'rival': rival,
                        'enricher': lit,
                        'en_ctx': (ea + '⟦' + lit + '⟧' + eb).replace('\n', ' '),
                        'cn_ctx': (ca + '⟦' + lit + '⟧' + cb).replace('\n', ' '),
                    })

    print('统计：')
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f'  {k:46s} {v}')
    print(f'\n报出 {len(hits)} 条'
          + ('（只含 rival）' if a.rival_only else ''))
    byfam = collections.Counter(h['family'] for h in hits)
    for k, v in byfam.most_common():
        print(f'   {k:18s} {v}')
    for h in hits[:a.show]:
        print(f'\n- {h["pack"]} :: {h["path"][-70:]}')
        print(f'  {h["family"]}/{h["param"]}  应为「{h["term_cn"]}」 rival={h["rival"]}')
        print(f'  EN {h["en_ctx"][:200]}')
        print(f'  CN {h["cn_ctx"][:200]}')

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'stats': dict(stats),
                       'registry_size': {k: len(v) for k, v in reg.items()},
                       'hits': hits}, f, ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
