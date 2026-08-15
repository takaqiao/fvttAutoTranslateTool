#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代词性别错配 —— 英文侧的人称/性别线索与中文侧的代词不一致。

为什么需要单独一个闸
--------------------
既有判据全部是**单叶自身**的形态检查（标记、class、数字、外来文字、死键）或
**键集合**检查（覆盖率、孪生包）。代词性别是**语义**层的，而且：
  * 它不改变任何标记，不改变任何数字，中文侧看上去完全通顺；
  * 它只有把英中并排读才看得出来 —— 正是「全绿的库」里最容易活下来的一类。
历史实证：`Shard God` 一律译成「碎片女神」，造成「碎片女神贾纳尔……**他**所斩杀」，
那次靠人读出来，至今没有判据。

七条臂（每条独立开关，独立统计）
--------------------------------
A `SAME_EN_SPLIT`  同一条英文串（去空白后逐字相同）被译出**性别不同**的中文代词。
B `LEAF_HARD`      单叶硬冲突：英文只出现女性代词（男性 0 次）而中文只出现单数「他」
                   （「她」0 次）；反向同理。
C `ACTOR_SUBTREE`  actor 子树内英文性别单一（≥3 次且反向 0 次），而中文某叶用了反向代词。
D `NOUN_PRONOUN`   中文侧性别名词与中文代词打架（「女神…他」/「国王…她」）。
E `TITLE_CLASH`    英文亲属/爵称/神祇称谓的性别与中文的性别**互斥地**相反：
                   英文只出现该类的男性词、中文只出现该类的女性词（反之亦然）。
F `INANIMATE`      英文通篇只用 it/its/itself（无 he/she/they），中文却用单数「他」。
G `ENTITY_ANCHOR`  ★ 唯一真正有产出的一条。先在**英文侧**给每个专名定性别
                   （句级投票 + 原文自带的 `(NG, Ordani Keth, he/him)` 声明，全库 977 处），
                   再回中文侧看代词，并要求该代词的**最近先行词**就是这个人。
                   2026-08-13 首测：A/B/C/D/E/F 六条臂合计 13 报 3 真（还都是 G 也报了的），
                   G 报 6 条 0 假阳性（3 处缺陷 × 孪生两包）。

各臂实测假阳性率（2026-08-13，全库 39941 条有译文的叶）
------------------------------------------------------
  A 2 报 0 真（2 条是同串译文的性别写法分叉，够呛算缺陷，归 uncertain）
  B 3 报 1 真   C 1 报 0 真   D 1 报 0 真   E 0 报   F 6 报 2 真   G 6 报 6 真
B/C/D/F 的假阳性全是同一个成因：**一片叶里有好几个人**，中文代词指的是另一个人。
只有 G 的最近先行词约束能把这件事判对，所以真要复用，先跑 G。

三个必须先踩过的坑（都是本项目实测，凭空拟规则一定翻车）
--------------------------------------------------------
1. 裸 `他` 会把 **其他 / 他们 / 他人 / 吉他 / 排他** 全数扫进来：清洗前全库「他」15449 次，
   清洗后只剩 10699 次，**三成是「其他」**。`它` 同理要排 `其它`。
   不清洗时 `EN[F] CN[他]` 有 4 条，清洗后 0 条 —— 4 条全是「其他」。
2. **「他们」是中性复数**，不能算作男性证据。第一版把 `他们` 当男性，B 臂 19 条里
   有 12 条是 `she brought them → 便将他们带到` 这种完全正确的译文。只有**单数「他」**
   才携带性别；「她们」才反过来携带（断言全女）。
3. E 臂**不能做共现检测**。第一版「英文有 brother && 中文有 姐妹」报了 160 条，
   brother 26 / sister 26 / mother 24 / father 22 的对称计数一眼就能看出
   全是「兄弟姐妹」「父母」同叶共现。必须改成**互斥**：英文只有该类男性词、
   中文只有该类女性词。改完 160 -> 见下方统计。

用法:
  python scan_pronoun_gender.py --repo <仓库目录> [--repo <另一个>] --out <json>
  python scan_pronoun_gender.py --repo ... --arm A,B --show 60
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SKIP_KEYS = {"_id", "path", "_variants", "_when"}

# ---------------------------------------------------------------- 文本清洗
TAG = re.compile(r'<[^>]*>')
ENRICHER = re.compile(r'@\w+\[[^\]]*\]')      # @UUID[...] / @Check[...] 的方括号目标
ROLL = re.compile(r'\[\[[^\]]*\]\]')          # [[/skill ...]] / [[lookup @name]]
REF = re.compile(r'@ref\[[^\]]*\]')


def strip_en(s: str) -> str:
    s = ROLL.sub(' ', s)
    s = REF.sub(' ', s)
    s = ENRICHER.sub(' ', s)
    return TAG.sub(' ', s)


def strip_cn(s: str) -> str:
    return strip_en(s)


# ---------------------------------------------------------------- 代词
EN_FEM = re.compile(r'\b(she|her|hers|herself)\b', re.I)
EN_MASC = re.compile(r'\b(he|him|his|himself)\b', re.I)
EN_NEUT = re.compile(r'\b(they|them|their|theirs|themselves|themself)\b', re.I)
EN_INAN = re.compile(r'\b(it|its|itself)\b', re.I)

# 「他」的非代词用法：其他/他们/他人/他日/他处/他方/他乡/他国/他者/他律/吉他/排他/利他/维他
CN_M = re.compile(r'(?<![其吉排利维])他(?![们人日处方乡国族物者律])')
CN_MP = re.compile(r'(?<!其)他们')
CN_F = re.compile(r'她(?!们)')
CN_FP = re.compile(r'她们')
CN_IT = re.compile(r'(?<!其)它(?!们)')
CN_ITP = re.compile(r'(?<!其)它们')
CN_DEITY = re.compile(r'祂')


def sig_cn(c: str) -> str:
    """中文侧的**性别代词签名**（只看他/她单复数，不看它/祂）。"""
    return ''.join(k for k, rx in (('他', CN_M), ('她', CN_F),
                                   ('他们', CN_MP), ('她们', CN_FP))
                   if rx.search(c))


# ---------------------------------------------------------------- D 臂词表
# 中文性别名词（只收**性别无歧义**的；「主人」「大师」「祭司」这类中性词不收）
CN_FEM_NOUN = re.compile(
    r'女神|女王|王后|女士|夫人|母亲|妈妈|姐姐|妹妹|姐妹|女儿|妻子|老婆|祖母|外婆|奶奶|'
    r'姑妈|姨妈|婶婶|侄女|外甥女|寡妇|少女|女子|女人|女孩|女性|女祭司|女修士|女伯爵|'
    r'公主|皇后|太后|女主人|女爵|女猎手|女武神|女教师|女英雄')
CN_MASC_NOUN = re.compile(
    r'(?<!女)国王|父亲|爸爸|哥哥|弟弟|兄弟|儿子|丈夫|老公|祖父|外公|爷爷|'
    r'伯父|叔叔|舅舅|侄子|外甥|鳏夫|少年|男子|男人|男孩|男性|男爵(?!夫人)|'
    r'王子|皇帝|太子|男主人|先生|绅士')

# ---------------------------------------------------------------- E 臂词表
# 每一类给出 (英文男性词, 英文女性词, 中文男性词, 中文女性词)。
# 判据是**互斥**的：英文只出现该类的一性、中文只出现该类的另一性。
# 共现（「兄弟姐妹」「父母双亲」）因此天然不报。
TITLE_CATS = {
    '同胞': (r'\bbrothers?\b', r'\bsisters?\b',
             r'兄弟|哥哥|弟弟|兄长|长兄|胞兄|胞弟',
             r'姐妹|姊妹|姐姐|妹妹|胞姐|胞妹'),
    '亲代': (r'\bfathers?\b|\bdads?\b|\bpapa\b', r'\bmothers?\b|\bmoms?\b|\bmama\b',
             r'父亲|爸爸|father(?!)|爹', r'母亲|妈妈|娘亲'),
    '子代': (r'\bsons?\b', r'\bdaughters?\b', r'儿子', r'女儿'),
    '配偶': (r'\bhusbands?\b', r'\bwi(?:fe|ves)\b', r'丈夫|夫君|老公', r'妻子|夫人|老婆'),
    '王室': (r'\bkings?\b|\bprinces?\b|\bemperors?\b', r'\bqueens?\b|\bprincess(?:es)?\b|\bempress(?:es)?\b',
             r'国王|王子|皇帝', r'女王|王后|公主|女皇|皇后'),
    '祖辈': (r'\bgrand(?:father|pa)s?\b', r'\bgrand(?:mother|ma)s?\b',
             r'祖父|外公|爷爷', r'祖母|外婆|奶奶'),
    '旁系': (r'\bunc(?:le)s?\b|\bnephews?\b', r'\baunts?\b|\bnieces?\b',
             r'伯父|叔叔|舅舅|侄子|外甥(?!女)', r'姑妈|姨妈|婶婶|侄女|外甥女'),
    '神职': (r'\bpriests?\b|\bmonks?\b', r'\bpriestess(?:es)?\b|\bnuns?\b',
             r'男祭司|僧侣', r'女祭司|修女'),
    '神祇': (r'\bgods?\b', r'\bgoddess(?:es)?\b', r'之神|男神', r'女神'),
}
TITLE_CATS_C = {k: (re.compile(a, re.I), re.compile(b, re.I),
                    re.compile(c), re.compile(d))
                for k, (a, b, c, d) in TITLE_CATS.items()}


# ---------------------------------------------------------------- 遍历
def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            sub = cn.get(k) if isinstance(cn, dict) else None
            walk(v, sub, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        p = '.'.join(path)
        out.append({'path': p,
                    'batch_path': p[len('entries.'):] if p.startswith('entries.') else p,
                    'en': en, 'cn': cn if isinstance(cn, str) else None})


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def collect(repo, repo_tag):
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    rows = []
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith('.json') or fn.startswith('_'):
            continue
        cp = os.path.join(cn_dir, fn)
        en = load(os.path.join(en_dir, fn))
        cn = load(cp) if os.path.isfile(cp) else {}
        sub = []
        walk(en.get('entries', {}), cn.get('entries', {}), ['entries'], sub)
        for r in sub:
            r['pack'] = fn
            r['repo'] = repo_tag
        rows.extend(sub)
    return rows


# ---------------------------------------------------------------- G 臂：实体性别表
# 英文原文在 NPC 名字后面**直接声明代词**：`Hob Korell (CG, Waerd Keth, he/him)`、
# `<span class="tag">Chaotic Good, Ordani Human, she/her</span>`。全库 977 处。
# 这是比任何推断都硬的证据，优先采信。
DECL = re.compile(r'\b(he/him|she/her|they/them)\b', re.I)
BILINGUAL_TAIL = re.compile(r'\s+[^一-鿿　-〿＀-￯]+$')
# 全库只有 Deities / Notable Figures 两本日志的页是人物；其余 38 本都是地点/事件。
PERSON_JOURNAL = re.compile(r'\.journals\.(Deities|Notable Figures)\.pages\.')
PERSON_NAME = re.compile(r"^[A-Z][a-z'’À-ÿ-]+"
                         r"(?:\s+(?:of|the|von|van|de|Ben-)?\s*"
                         r"[A-Z][A-Za-z'’À-ÿ-]+){0,2}$")
# 明显不是人名的通用词（章节标题、地名后缀等）
NOT_PERSON = re.compile(
    r'\b(Overview|Details|Room|Hall|Path|Shore|Tunnel|Tower|Gate|Temple|District|'
    r'Shrine|Chamber|Camp|Bridge|Market|Guild|House|Order|Keep|Manor|Plaza|Ruins|'
    r'Forest|Mountain|Lake|River|Island|Event|Encounter|Challenge|Quest|Scene|Map|'
    r'Level|Stage|Table|List|Notes|Summary|Rules|Options|Features|Traits|Actions|'
    r'Spell|Weapon|Armor|Item|Effect|Ability|Skill|Check|Damage|Attack|Ancestry|'
    r'Culture|Background|Talent|Archetype|Taxonomy|Access|Early|Area|Gameplay)\b')


def cn_head(s: str) -> str:
    return BILINGUAL_TAIL.sub('', s).strip() or s.strip()


def build_entities(rows):
    """从 `.name` 叶里取人名，返回 {en_name: cn_head}。

    单词名（`Sionia`）必须**在全库英文里从不以小写出现**才算专名 ——
    否则 `Time` 会被当成一个叫「时间」的人（第一版就这么翻车了，
    `Eveis Brightstone` 的 `Quadruple Arms` 被判成「时间」性别错）。
    """
    # lower_seen 必须在**剥掉 UUID/标记之后**统计：`Mioroth` 的小写形态只出现在
    # `@UUID[...mioroth...]` 之类的 id 串里，用原始串统计会把他误判成普通词剔掉。
    lower_seen = set()
    for r in rows:
        for w in re.findall(r'\b[a-z]{3,}\b', strip_en(r['en'])):
            lower_seen.add(w)
    ents = {}
    for r in rows:
        if not r['path'].endswith('.name') or not r['cn']:
            continue
        p = r['batch_path']
        # **结构性**人物过滤：只认 Actor 文档**自身**的 name，以及
        # Deities / Notable Figures 两本人物志的页名。
        #  * 第一版用「被代词指称过」当人物过滤，结果 `The Bleak Archive`（黯淡秘库）、
        #    `Fogbound Caverns`（雾缚洞窟）这些地名从邻近句子蹭到 she/her 票，
        #    被当成女性人物报了 4 条假阳性。
        #  * 第二版只查 `.actors.` 在不在路径里，把 `actors.Mioroth.archetype.name`
        #    = `Shent Seer`（原型标签，一个女性传奇人物）也收成了实体，
        #    害得米奥罗斯本人的正确译文被报错。必须要求 name 是 actor 自身的。
        is_actor_self = re.search(r'\.actors\.[^.]+\.name$', p)
        if not (is_actor_self or PERSON_JOURNAL.search(p)):
            continue
        en = r['en'].strip()
        if len(en) < 4 or not PERSON_NAME.match(en) or NOT_PERSON.search(en):
            continue
        if ' ' not in en and en.lower() in lower_seen:
            continue                      # `Time` / `Spark` / `Hope` 这类普通词
        h = cn_head(r['cn'])
        # 中文头必须是纯中文人名长度（2-12 字），且不含标点
        if not (2 <= len(h) <= 12) or not re.fullmatch(r'[一-鿿·•]+', h):
            continue
        ents.setdefault(en, h)
    return ents


def actor_scope(batch_path: str):
    """`... .actors.<Name>. ...` -> 该 actor 的作用域键；不是 actor 叶则 None。"""
    seg = batch_path.split('.')
    for i, s in enumerate(seg):
        if s == 'actors' and i + 1 < len(seg):
            return '.'.join(seg[:i + 2])
    return None


def norm_en(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def excerpt(s: str, rx: re.Pattern, w=70, limit=3):
    outs = []
    for m in list(rx.finditer(s))[:limit]:
        outs.append(s[max(0, m.start() - w):m.end() + w].replace('\n', ' '))
    return outs


# ---------------------------------------------------------------- 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--arm', default='A,B,C,D,E,F,G')
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=30)
    ap.add_argument('--min-actor-evidence', type=int, default=3)
    ap.add_argument('--max-dist', type=int, default=60,
                    help='G 臂：代词与先行词的最大字距')
    a = ap.parse_args()
    arms = {x.strip().upper() for x in a.arm.split(',') if x.strip()}

    rows = []
    for repo in a.repo:
        tag = os.path.basename(os.path.normpath(repo))
        rows.extend(collect(repo, tag))

    pairs = [r for r in rows if r['cn']]
    stats = collections.Counter()
    stats['叶总数'] = len(rows)
    stats['有中文的叶'] = len(pairs)
    stats['英文字符数'] = sum(len(r['en']) for r in rows)

    for r in pairs:
        e, c = strip_en(r['en']), strip_cn(r['cn'])
        r['_e'], r['_c'] = e, c
        r['_f'] = len(EN_FEM.findall(e))
        r['_m'] = len(EN_MASC.findall(e))
        r['_n'] = len(EN_NEUT.findall(e))
        r['_i'] = len(EN_INAN.findall(e))
        r['_cm'] = len(CN_M.findall(c))
        r['_cf'] = len(CN_F.findall(c))
        r['_cmp'] = len(CN_MP.findall(c))
        r['_cfp'] = len(CN_FP.findall(c))
    stats['含性别代词的英文叶'] = sum(1 for r in pairs if r['_f'] or r['_m'])
    stats['含他/她的中文叶'] = sum(1 for r in pairs if r['_cm'] or r['_cf'])

    findings = []

    # ---------------- A: 同英文串译出不同性别 ----------------
    if 'A' in arms:
        groups = collections.defaultdict(list)
        for r in pairs:
            # 只看英文里出现了人称代词（含单数 they）的串，且串够长（短串同形太多）
            if (r['_f'] or r['_m'] or r['_n']) and len(r['en']) >= 40:
                groups[norm_en(r['en'])].append(r)
        stats['A: 可比较的英文串组'] = sum(1 for v in groups.values() if len(v) > 1)
        for key, g in groups.items():
            if len(g) < 2:
                continue
            sigs = collections.Counter(sig_cn(r['_c']) for r in g)
            if len(sigs) < 2:
                continue
            # 只在**性别轴**上分叉才报：签名里含 他/他们 的一支 vs 含 她/她们 的一支
            has_m = {s for s in sigs if '他' in s}
            has_f = {s for s in sigs if '她' in s}
            if not (has_m and has_f):
                continue
            major, mcount = sigs.most_common(1)[0]
            for r in g:
                s = sig_cn(r['_c'])
                if s == major:
                    continue
                findings.append({
                    'arm': 'A', 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'同一条英文串共 {len(g)} 份译文，性别代词签名分叉为 '
                           f'{dict(sigs)}；本叶 [{s}] 与多数派 [{major}]({mcount}/{len(g)}) 不同',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'group_size': len(g), 'sigs': dict(sigs),
                    'majority': major,
                    'siblings': [x['batch_path'] for x in g if x is not r][:8],
                })

    # ---------------- B: 单叶硬冲突 ----------------
    # 「他们」是中性复数，**不算**男性证据；只有单数「他」算。「她们」断言全女，算女性证据。
    if 'B' in arms:
        for r in pairs:
            hit = None
            if r['_f'] and not r['_m'] and r['_cm'] and not (r['_cf'] or r['_cfp']):
                hit = ('英文只出现女性代词', r['_f'], CN_M)
            elif r['_m'] and not r['_f'] and (r['_cf'] or r['_cfp']) and not r['_cm']:
                hit = ('英文只出现男性代词', r['_m'], CN_F)
            if hit:
                findings.append({
                    'arm': 'B', 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'{hit[0]}（{hit[1]} 次，反向 0 次），中文侧却只有反向代词',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'cn_ctx': excerpt(r['_c'], hit[2]),
                })

    # ---------------- C: actor 子树性别单一 vs 中文反向 ----------------
    if 'C' in arms:
        scopes = collections.defaultdict(list)
        for r in pairs:
            sc = actor_scope(r['batch_path'])
            if sc:
                scopes[(r['repo'], r['pack'], sc)].append(r)
        stats['C: actor 子树数'] = len(scopes)
        for (repo, pack, sc), g in scopes.items():
            F = sum(x['_f'] for x in g)
            M = sum(x['_m'] for x in g)
            if F >= a.min_actor_evidence and M == 0:
                gender, bad_rx, bad = 'F', CN_M, '他'
            elif M >= a.min_actor_evidence and F == 0:
                gender, bad_rx, bad = 'M', CN_F, '她'
            else:
                continue
            stats['C: 英文性别单一的 actor'] += 1
            for r in g:
                n = len(bad_rx.findall(r['_c']))
                if not n:
                    continue
                findings.append({
                    'arm': 'C', 'repo': repo, 'pack': pack,
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'actor 子树 `{sc.split(".actors.")[-1]}` 英文侧'
                           f'{"只用女性代词" if gender=="F" else "只用男性代词"}'
                           f'（F={F} / M={M}），本叶中文出现「{bad}」{n} 次',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'cn_ctx': excerpt(r['_c'], bad_rx),
                    'actor': sc,
                })

    # ---------------- D: 中文名词 vs 中文代词 ----------------
    if 'D' in arms:
        for r in pairs:
            c = r['_c']
            fem_n = CN_FEM_NOUN.findall(c)
            masc_n = CN_MASC_NOUN.findall(c)
            # 女性名词在场，中文只有「他」没有「她」，且英文侧没有男性代词
            if fem_n and r['_cm'] and not r['_cf'] and not r['_m']:
                findings.append({
                    'arm': 'D', 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'中文出现女性名词 {sorted(set(fem_n))[:4]}，同叶却用「他」'
                           f'（「她」0 次），且英文侧无男性代词（F={r["_f"]}/M={r["_m"]}）',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'cn_ctx': excerpt(c, CN_M),
                })
            if masc_n and r['_cf'] and not r['_cm'] and not r['_f']:
                findings.append({
                    'arm': 'D', 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'中文出现男性名词 {sorted(set(masc_n))[:4]}，同叶却用「她」'
                           f'（「他」0 次），且英文侧无女性代词（F={r["_f"]}/M={r["_m"]}）',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'cn_ctx': excerpt(c, CN_F),
                })

    # ---------------- E: 英文爵称/亲属/神职称谓性别 vs 中文（互斥） ----------------
    if 'E' in arms:
        for r in pairs:
            e, c = r['_e'], r['_c']
            for cat, (rx_em, rx_ef, rx_cm, rx_cf) in TITLE_CATS_C.items():
                em, ef = rx_em.search(e), rx_ef.search(e)
                cm, cf = rx_cm.search(c), rx_cf.search(c)
                if em and not ef and cf and not cm:
                    d = ('M', rx_em, rx_cf)
                elif ef and not em and cm and not cf:
                    d = ('F', rx_ef, rx_cm)
                else:
                    continue
                findings.append({
                    'arm': 'E', 'cat': cat, 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'「{cat}」类：英文只出现{"男" if d[0]=="M" else "女"}性词 '
                           f'{d[1].pattern}，中文只出现'
                           f'{"女" if d[0]=="M" else "男"}性词 {d[2].pattern}',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'en_ctx': excerpt(e, d[1]), 'cn_ctx': excerpt(c, d[2]),
                })

    # ---------------- F: 英文纯无生命 it/its，中文却用单数「他」 ----------------
    if 'F' in arms:
        for r in pairs:
            if r['_i'] and not (r['_f'] or r['_m'] or r['_n']) and r['_cm']:
                findings.append({
                    'arm': 'F', 'repo': r['repo'], 'pack': r['pack'],
                    'path': r['path'], 'batch_path': r['batch_path'],
                    'why': f'英文通篇只用 it/its/itself（{r["_i"]} 次，无 he/she/they），'
                           f'中文却用单数「他」{r["_cm"]} 次',
                    'en': r['en'][:600], 'cn': r['cn'][:600],
                    'cn_ctx': excerpt(r['_c'], CN_M),
                })

    # ---------------- G: 实体锚定（跨叶性别表） ----------------
    # 这才是 `碎片女神…他` 那一类的正面判据：先在**英文侧**为每个专名定性别，
    # 再回到中文侧看代词。为压假阳性，只在「本叶里只出现这一个已知人名」时才判。
    if 'G' in arms:
        ents = build_entities(rows)
        stats['G: 候选实体（有中英名对）'] = len(ents)
        by_cn = collections.defaultdict(list)
        for en, cn in ents.items():
            by_cn[cn].append(en)
        # 一个中文头对应多个英文名的（同名），全部弃用，避免张冠李戴
        ents = {en: cn for en, cn in ents.items() if len(by_cn[cn]) == 1}
        stats['G: 中文头唯一的实体'] = len(ents)

        en_names = sorted(ents, key=len, reverse=True)
        RX_EN_ALL = re.compile(r'\b(' + '|'.join(re.escape(n) for n in en_names) + r')\b')
        # 姓/名单字也要能匹配（`Sionia has tasked us` / `Calyn had been`）
        singles = {}
        for n in en_names:
            for w in n.split():
                if len(w) >= 4 and w[0].isupper():
                    singles.setdefault(w, set()).add(n)
        singles = {w: next(iter(v)) for w, v in singles.items() if len(v) == 1}
        RX_EN_SINGLE = re.compile(r'\b(' + '|'.join(re.escape(w) for w in
                                                    sorted(singles, key=len, reverse=True)) + r')\b')

        vote = collections.defaultdict(lambda: [0, 0, 0])   # name -> [F, M, decl]
        SENT = re.compile(r'(?<=[.!?;:])\s+|</p>|</li>|</dt>|</dd>')
        for r in rows:
            e = strip_en(r['en'])
            for s in SENT.split(e):
                if not s or len(s) > 1200:
                    continue
                hits = {singles.get(w, w) for w in RX_EN_SINGLE.findall(s)}
                hits |= set(RX_EN_ALL.findall(s))
                hits = {h for h in hits if h in ents}
                if len(hits) != 1:
                    continue
                n = next(iter(hits))
                dm = DECL.search(s)
                if dm:
                    d = dm.group(1).lower()
                    if d == 'she/her':
                        vote[n][0] += 100
                    elif d == 'he/him':
                        vote[n][1] += 100
                    vote[n][2] += 1
                vote[n][0] += len(EN_FEM.findall(s))
                vote[n][1] += len(EN_MASC.findall(s))

        # 「反向必须为 0」太严：`Alar` 是女神（she/her 130），但共现句里
        # `Ankarist keeps **his** pact with the elder goddess Alar` 这类会给她记上
        # 男性票，反向清零的规则直接把她排除，深渊页那条真缺陷就漏了。
        # 改成**压倒性多数**：总票 >=6 且主导方 >= 6 倍。
        confident = {}
        for n, (F, M, d) in vote.items():
            if F >= 3 and M == 0:
                confident[n] = ('F', F, M, d)
            elif M >= 3 and F == 0:
                confident[n] = ('M', F, M, d)
            elif F + M >= 6 and F > M and F >= 6 * max(M, 1):
                confident[n] = ('F', F, M, d)
            elif F + M >= 6 and M > F and M >= 6 * max(F, 1):
                confident[n] = ('M', F, M, d)
        stats['G: 英文侧性别确凿的实体'] = len(confident)

        # 只有**被人称代词指称过**的实体才算「人」。地名（卡伦阿克）拿不到票，
        # 因此不会把「本叶只出现一个人名」这一条卡死 —— 第一版就是因为把地名
        # 也算进去，Calyn Kariset 那条真缺陷被 `len(found)!=1` 直接吞掉了。
        persons = dict(ents)      # ents 已经是结构性筛过的人物表
        stats['G: 人物实体'] = len(persons)
        cn_of = {ents[n]: n for n in confident}
        RX_CN_ALL = re.compile('|'.join(re.escape(c) for c in
                                        sorted(set(persons.values()), key=len, reverse=True)))
        cn_conf = set(cn_of)
        # 「他」离先行词多远之内还算指它。60 是实测出来的：三条真缺陷的距离是
        # 12 / 13 / 20 字，而 160 会把 `破法者的典狱长…唯有他` 这种真正的先行词在
        # 91 字外的句子错记到更前面的「辛达里克贤者」头上（2 条假阳性）。
        MAXDIST = a.max_dist
        for r in pairs:
            c = r['_c']
            if not (r['_cm'] or r['_cf']):
                continue
            # 本叶里所有人物名的出现位置（不只性别确凿的那些）
            occ = [(m.start(), m.group(0)) for m in RX_CN_ALL.finditer(c)]
            if not occ:
                continue
            for rx, want, bad in ((CN_M, 'F', '他'), (CN_F, 'M', '她')):
                for pm in rx.finditer(c):
                    # **最近先行词**：代词往前找最近的一个人物名。
                    # `阿拉尔亲手摧毁，因为只有他能够伤害它` 里 `他` 的最近先行词是
                    # 阿拉尔（性别不明），不是同叶远处的莱欧拉 —— 第一版就是这样
                    # 把 Laeora 报成了假阳性。
                    win = [o for o in occ
                           if o[0] < pm.start() and pm.start() - o[0] <= MAXDIST]
                    if not win:
                        continue
                    # 窗口里只要**有一个**人物的性别与该代词相容，就不判 ——
                    # 同位语会骗过「最近先行词」：`{米奥罗斯}——一位申特先知残留至今的
                    # 投影。他揭示出…` 里最近的名字是「申特先知」（女），但真正的
                    # 先行词是同位语中心词米奥罗斯（男），中文「他」是对的。
                    if any(cn_of.get(x) and confident[cn_of[x]][0] != want
                           for _, x in win if x in cn_conf):
                        continue
                    pos, h = win[-1]
                    if h not in cn_conf:
                        continue      # 最近先行词性别不明 -> 不判
                    n = cn_of[h]
                    g, F, M, d = confident[n]
                    if g != want:
                        continue
                    # 英文侧若出现**反向性别**代词，说明本叶还有别人在场
                    #（第一版 Funar Cevher 那条：英文 she/her 指的是同叶的 Hephiss）
                    if (g == 'M' and r['_f']) or (g == 'F' and r['_m']):
                        continue
                    findings.append({
                        'arm': 'G', 'repo': r['repo'], 'pack': r['pack'],
                        'path': r['path'], 'batch_path': r['batch_path'],
                        'why': f'「{bad}」的最近先行词是「{h}」= {n}（相距 '
                               f'{pm.start()-pos} 字）；英文侧该人性别确凿为'
                               f'{"女" if g=="F" else "男"}'
                               f'（she/her 票 {F} / he/his 票 {M}，'
                               f'显式 he-him 式声明 {d} 处）',
                        'en': r['en'][:600], 'cn': r['cn'][:600],
                        'entity': n, 'entity_cn': h, 'gender': g,
                        'cn_ctx': [c[max(0, pos - 30):pm.start() + 60]],
                    })
                    break

    for f in findings:
        stats[f'命中 {f["arm"]}'] += 1
    stats['命中合计'] = len(findings)

    print('规模与命中：')
    for k, v in stats.items():
        print(f'  {k:28s} {v}')
    print()
    for f in findings[:a.show]:
        print(f'[{f["arm"]}] {f["pack"][:26]:28s} {f["batch_path"][-70:]}')
        print(f'    {f["why"]}')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'stats': dict(stats), 'findings': findings}, fh,
                      ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}  ({len(findings)} 条)')


if __name__ == '__main__':
    main()
