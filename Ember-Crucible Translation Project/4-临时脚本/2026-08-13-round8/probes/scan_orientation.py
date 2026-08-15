#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方位 / 方向 / 序数：译反、丢失、量词错位 —— 玩家按译文导航会走错路。

三个模式
--------
`--mode axis`   逐**对齐段**比对「对立轴」两极，只报**反向**（英文只说 A、中文只说 B）。
                轴：compass_ns / compass_ew / lr / updown_spatial / updown_ref /
                    rotation / ordinal
`--mode nameq`  名称叶里的方位限定词（North/Upper/Inner/Front/First…）在中文侧**整个丢掉**。
`--mode floor`  楼层号的**量词**：同一栋建筑的「Level N / NF / Nth Floor」在不同字段里
                被译成 第N层 / N层 / 层级N / **N级** / N楼 等多套写法；
                其中 `N级` 与 `N环` 在本库里是**角色等级/法术环级**的量词，
                拿来指建筑楼层会和角色等级混淆。

为什么需要单独一个闸
--------------------
方位词**不含数字、不含标记、不含拉丁残留**，既有全部判据（标记五项 / 方括号内标记 /
class 漂移 / 数字覆盖 / 外来文字 / 死键 / 中文侧缺键 / tokenName / 孪生分叉 / lang 三数）
对它全盲：译反了 JSON 结构完全合法，数字集合也完全一致（根本没数字），
但玩家「从北门进」会变成「从南门进」。
`scan_name_splits` 只看 `*.name` 叶子且只查「同英文→多中文」，
`--mode floor` 命中的 `scenes.X.levels.<name>` 这类叶子不在它的范围里。

═══ 三条会让这类判据全是假阳性的坑（v1 实测，务必保留这些收紧）═══
1. **整叶比对不可用**。一页 2 万字的地名志里英文某处有 `first`、中文另一处有「第3章」，
   整叶比会报一堆噪声。必须按块级标签**切段对齐**后逐段比
   （本库 10637/10750 的长叶两侧块段数完全相同，markup_drift 全绿是前提）。
2. **中文方位字必须过「方位词形」闸，不能裸数字符**。本库音译名里 `西`/`南` 极多
   （西吉尔 Sigil、西希拉 Sitheera、凯西安 Kessian、卡西娅 Cassia、纳西拉 Nathira……），
   裸数 `西` 会把它们全报出来。英文侧同理：`\\beast` 必须带**词首**边界，
   否则 `beast` / `at least` 全部命中（v1 的 57 条 compass_ew 全是这么来的）。
   同理中文「下一个/下一层」是 next 不是 below，「上一段」是 previous 不是 above，
   一律不能进上下轴。
3. **一段里同时出现上下两种关系是常态**，例如「桌子在挂毯**下方**」＝「挂毯悬在桌子**上方**」，
   两种译法都对。所以上下轴必须要求**两侧各只有一个极性词**，并限制段长；
   即便如此 updown_spatial 仍是本判据里假阳性最高的一轴，报出来必须逐条人看。

用法
----
  python scan_orientation.py --repo <repoDir> [--repo <另一个>] \
         [--mode axis|nameq|floor|all] [--axis compass_ns,lr,…] [--out x.json] [--show 40]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SKIP_KEYS = {'_id', 'path', '_variants', '_when'}

# ═══════════════════════════════════════════════════ 切段 / 归一化
SPLIT = re.compile(
    r'(?:</p>|</li>|</h[1-6]>|</td>|</th>|</tr>|</div>|</section>|</blockquote>'
    r'|</dt>|</dd>|</figcaption>|<br\s*/?>|\n)', re.I)

TAG = re.compile(r'<[^>]+>')
ENRICH_LBL = re.compile(r'@[A-Za-z]+\[[^\]]*\]\{([^}]*)\}')
ENRICH_BARE = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
INLINE_ROLL = re.compile(r'\[\[[^\]]*\]\](?:\{([^}]*)\})?')
ENTITY = re.compile(r'&[a-zA-Z#0-9]+;')


def norm(s: str) -> str:
    """`@UUID[...]` 方括号内的目标照抄不译，不该参与语义比对，只留 {标签} 与正文。"""
    s = ENRICH_LBL.sub(r' \1 ', s)
    s = ENRICH_BARE.sub(' ', s)
    s = INLINE_ROLL.sub(lambda m: ' ' + (m.group(1) or '') + ' ', s)
    s = TAG.sub(' ', s)
    return ENTITY.sub(' ', s)


def c(p):
    return re.compile(p, re.I)


# ═══════════════════════════════════════════════════ 中文方位词形闸
SUF = ('方面侧端部边向岸区半墙门角翼路段口壁塔楼梯廊厅室房厢屋亭窗阶院线滩崖坡岭峰谷'
       '城镇村堡道街巷桥台通林缘首')
PRE = '向往朝自从到正最'
CN_N = re.compile(f'(?:北[{SUF}]|[{PRE}]北|以北|[东西]北|北[东西])')
CN_S = re.compile(f'(?:南[{SUF}]|[{PRE}]南|以南|[东西]南|南[东西])')
CN_E = re.compile(f'(?:东[{SUF}]|[{PRE}]东|以东|东[南北]|[南北]东)')
CN_W = re.compile(f'(?:西[{SUF}]|[{PRE}]西|以西|西[南北]|[南北]西)')

CN_LR_PAIR = re.compile(r'左右')
CN_L, CN_R = re.compile(r'左'), re.compile(r'右')

# 只认空间义；「下一层/上一段」是 next/previous，绝不能进来
CN_UP = re.compile(r'(?:上方|上面|上层|上部|上端|上侧|楼上|头顶|之上|上空|地上|向上|往上|朝上'
                   r'|攀上|顶层|顶部|顶端|上升|高处|楼顶|上楼|上城|上区)')
CN_DN = re.compile(r'(?:下方|下面|下层|下部|下端|下侧|楼下|之下|脚下|底下|地下|向下|往下|朝下'
                   r'|底层|底部|下降|下沉|低处|地底|下楼|下城|下区)')
# 引用义（见上文 / 见下文）
CN_REF_UP = re.compile(r'(?:上文|上述|如上|以上|上表|前述|上图|上一节)')
CN_REF_DN = re.compile(r'(?:下文|下述|如下|以下|下表|下列|下图|下一节)')

# ═══════════════════════════════════════════════════ 英文侧
EN_N, EN_S = c(r'\bnorth\w*'), c(r'\bsouth\w*')
EN_E = c(r'(?:\beast(?:ern|erly|ward|wards|erner|ers)?\b|\bnorth-?east\w*|\bsouth-?east\w*)')
EN_W = c(r'(?:\bwest(?:ern|erly|ward|wards|erner|ers)?\b|\bnorth-?west\w*|\bsouth-?west\w*)')

_MOVE = (r'(?:turn|turns|turning|veer|veers|bear|bears|head|heads|go|goes|swing|swings'
         r'|branch|branches|fork|forks|curve|curves|bend|bends|lead|leads|continue|continues|run|runs)')
_SIDE = (r'(?:hand|side|most|wall|door|doorway|path|passage|passageway|branch|arm|flank|eye'
         r'|shoulder|leg|foot|turn|corner|edge|column|lane|bank|half|fork|stair|stairs'
         r'|staircase|window|alcove|niche|aisle|wing)')
_POSS = r'(?:the|your|his|her|its|their|our)'


def _lr(w):
    # 刻意**不**收 bare `the left/right`：`the right place/way/person`、`what's left`、
    # `right hand`（＝得力助手）全是非方位义，实测 33 段全假阳性。
    return c(rf'(?:(?:to|on|at|from|along|toward|towards|past)\s+{_POSS}\s+(?:immediate\s+|far\s+)?{w}\b'
             rf'|\b{w}[-\s]{_SIDE}\b|\b{w}most\b|\bfar\s+{w}\b'
             rf'|(?:top|bottom|upper|lower)-{w}\b'
             rf'|{_MOVE}\s+(?:off\s+)?(?:to\s+the\s+)?{w}\b)')


LR_L, LR_R = _lr('left'), _lr('right')

# upper/lower 必须带处所名词，否则 "raise and lower the claw"（动词）会命中
_PL = (r'(?:floors?|levels?|decks?|stor(?:y|ey|ies|eys)|chambers?|rooms?|halls?|city|district'
       r'|wards?|sections?|half|tiers?|reaches|slopes|platforms?|plazas?|bar|stairs?|waters?'
       r'|cliffs?|interiors?|paths?|shelf|rings?|areas?|regions?|quarters?|passages?|corridors?'
       r'|vaults?|cellars?|basements?|galler(?:y|ies)|balcon(?:y|ies)|walkways?|catwalks?'
       r'|landings?|terraces?|ledges?|tunnels?|caverns?|caves?)')
EN_UP = c(rf'(?:\bupstairs\b|\bupper\s+{_PL}\b|\boverhead\b|\bascend\w*|\bclimbs?\s+up\b'
          r'|\bupwards?\b|\btop\s+floor\b|\btopmost\b|\buppermost\b|\bfloor\s+above\b'
          r'|\blevel\s+above\b|\bstor(?:y|ey)\s+above\b|\bstairs?\s+up\b'
          r'|\bup\s+the\s+(?:stairs|ladder|steps|shaft))')
EN_DN = c(rf'(?:\bdownstairs\b|\blower\s+{_PL}\b|\bbeneath\b|\bunderneath\b|\bdescend\w*'
          r'|\bclimbs?\s+down\b|\bdownwards?\b|\bbottom\s+floor\b|\bbottommost\b|\blowermost\b'
          r'|\bfloor\s+below\b|\blevel\s+below\b|\bstor(?:y|ey)\s+below\b|\bstairs?\s+down\b'
          r'|\bdown\s+the\s+(?:stairs|ladder|steps|shaft)|\bsubterranean\b|\bunderground\b)')
EN_REF_UP = c(r'(?:\b(?:see|listed|described|shown|noted|mentioned|detailed|outlined|stated)'
              r'\s+(?:\w+\s+){0,2}above\b|\b(?:table|list|section|text|chart|box|rules?|entry|entries)'
              r'\s+above\b|\bas\s+above\b)')
EN_REF_DN = c(r'(?:\b(?:see|listed|described|shown|noted|mentioned|detailed|outlined|stated)'
              r'\s+(?:\w+\s+){0,2}below\b|\b(?:table|list|section|text|chart|box|rules?|entry|entries)'
              r'\s+below\b|\bas\s+below\b|\bas\s+follows\b)')

AXES = {
    'compass_ns': dict(ea=EN_N, eb=EN_S, ca=CN_N, cb=CN_S, poles=('north', 'south'), cap=0),
    'compass_ew': dict(ea=EN_E, eb=EN_W, ca=CN_E, cb=CN_W, poles=('east', 'west'), cap=0),
    'lr': dict(ea=LR_L, eb=LR_R, ca=CN_L, cb=CN_R, pair=CN_LR_PAIR, poles=('left', 'right'), cap=0),
    # 上下轴：一段里同时讲两种垂直关系是常态，故限段长 + 两侧各只许一个极性词
    'updown_spatial': dict(ea=EN_UP, eb=EN_DN, ca=CN_UP, cb=CN_DN,
                           poles=('above/upper', 'below/lower'), cap=260, unique=True),
    'updown_ref': dict(ea=EN_REF_UP, eb=EN_REF_DN, ca=CN_REF_UP, cb=CN_REF_DN,
                       poles=('see-above', 'see-below'), cap=0),
    'rotation': dict(ea=c(r'(?<!counter)(?<!counter-)(?<!anti)(?<!anti-)\bclockwise\b'),
                     eb=c(r'\b(?:counter-?clockwise|anti-?clockwise|widdershins)\b'),
                     ca=re.compile(r'顺时针'), cb=re.compile(r'逆时针'),
                     poles=('clockwise', 'counterclockwise'), cap=0),
}

EN_ORD = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
          'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10}
EN_ORD_RX = c(r'\b(' + '|'.join(EN_ORD) + r')\b')
EN_SECOND_TIME = c(r'(?:\d+|a|one|few|several|couple\s+of|per|every)\s+seconds?\b')
DIGIT = re.compile(r'\d+')
CN_ORD_RX = re.compile(r'第\s*([一二三四五六七八九十]|\d+)')
CN_ORD_MAP = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
ORD_MIN = 2   # 「first」在中文里常合法地写成 首先/最初/首个，不报

# ═══════════════════════════════════════════════════ nameq / floor
CJK = re.compile(r'[一-鿿]')
NAMEKEY = re.compile(r'(?:\.name|\.tokenName)$')
QUALS = {
    'north': '北', 'south': '南', 'east': '东', 'west': '西',
    'northern': '北', 'southern': '南', 'eastern': '东', 'western': '西',
    'northeast': '东北', 'northwest': '西北', 'southeast': '东南', 'southwest': '西南',
    'upper': '上', 'lower': '下', 'inner': '内', 'outer': '外', 'front': '前',
    'back': '后', 'rear': '后', 'top': '顶|上|首', 'bottom': '底|下',
    'left': '左', 'right': '右', 'basement': '地下', 'underground': '地下',
    'second': '二|次', 'third': '三', 'fourth': '四', 'fifth': '五',
}
# 这些是成语/固定搭配，方位字本来就不该出现
NAMEQ_IDIOM = {'Back to Back', 'Strike First'}

FLOOR_RX = re.compile(r'\b(?:Level|Floor)\s*(\d)\b|\b(\d)F\b|\bL(\d)\b'
                      r'|\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh)\s+Floor\b', re.I)
WORD_ORD = {'first': '1', 'second': '2', 'third': '3', 'fourth': '4',
            'fifth': '5', 'sixth': '6', 'seventh': '7'}
CN_NUM = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七'}
FLOOR_OK = '第N层'


# ═══════════════════════════════════════════════════ 遍历
def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str):
        if isinstance(cn, str) and cn:
            out.append(('.'.join(path), en, cn))


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def collect(repo):
    en_dir = os.path.join(repo, 'compendium', 'en')
    cn_dir = os.path.join(repo, 'compendium', 'cn')
    rows = []
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or pack.startswith('_'):
            continue
        cp = os.path.join(cn_dir, pack)
        if not os.path.isfile(cp):
            continue
        sub = []
        walk(load(os.path.join(en_dir, pack)).get('entries', {}),
             load(cp).get('entries', {}), ['entries'], sub)
        tag = os.path.basename(repo.rstrip('/\\'))
        for p, e, z in sub:
            rows.append(dict(repo=tag, pack=pack, path=p,
                             batch_path=p[len('entries.'):], en=e, cn=z))
    return rows


def segments(en, cn):
    e, z = SPLIT.split(en), SPLIT.split(cn)
    if len(e) == len(z) and len(e) > 1:
        return [(norm(a), norm(b), i) for i, (a, b) in enumerate(zip(e, z))]
    return [(norm(en), norm(cn), -1)]


def excerpt(s, rx, w=100):
    s = re.sub(r'\s+', ' ', s).strip()
    m = rx.search(s)
    if not m:
        return s[:2 * w]
    a = max(0, m.start() - w // 2)
    return ('…' if a else '') + s[a:m.end() + w].strip() + ('…' if m.end() + w < len(s) else '')


def check_axis(ax, en, cn):
    ea, eb = len(ax['ea'].findall(en)), len(ax['eb'].findall(en))
    if not (ea or eb) or (ea and eb):
        return None
    if ax.get('cap') and max(len(en), len(cn)) > ax['cap']:
        return None
    ca, cb = len(ax['ca'].findall(cn)), len(ax['cb'].findall(cn))
    if 'pair' in ax:
        n = len(ax['pair'].findall(cn))
        ca, cb = max(ca - n, 0), max(cb - n, 0)
    if ax.get('unique') and (max(ea, eb) != 1 or max(ca, cb) != 1):
        return None
    if ea and cb and not ca:
        return dict(kind='reverse', en_pole=ax['poles'][0], cn_pole=ax['poles'][1],
                    en_n=ea, cn_n=cb, _re=ax['ea'], _rc=ax['cb'])
    if eb and ca and not cb:
        return dict(kind='reverse', en_pole=ax['poles'][1], cn_pole=ax['poles'][0],
                    en_n=eb, cn_n=ca, _re=ax['eb'], _rc=ax['ca'])
    return None


def check_ordinal(en, cn):
    ens = {EN_ORD[m] for m in EN_ORD_RX.findall(EN_SECOND_TIME.sub(' TIME ', en.lower()))}
    cns = {CN_ORD_MAP.get(m) or int(m) for m in CN_ORD_RX.findall(cn)}
    if not ens or not cns:
        return None
    ed = {int(x) for x in DIGIT.findall(en)}
    cd = {int(x) for x in DIGIT.findall(cn)}
    extra = {v for v in cns if v >= ORD_MIN and v not in ens and v not in ed}
    miss = {v for v in ens if v >= ORD_MIN and v not in cns and v not in cd}
    if extra and miss:
        return dict(kind='ordinal_mismatch', en_pole=str(sorted(miss)), cn_pole=str(sorted(extra)),
                    en_n=len(ens), cn_n=len(cns), _re=EN_ORD_RX, _rc=CN_ORD_RX)
    return None


def cn_head(en, cn):
    """双语并列约定是「中文 English」。只在尾巴**恰好等于英文**时剥掉——
    绝不能用「剥掉结尾所有非汉字」，那会把『楼梯间检查点 1』的编号也吃掉（v1 教训）。"""
    s = cn.strip()
    return s[:-len(en)].strip() if s.endswith(en) else s


def run_nameq(rows):
    out, seen = [], set()
    scanned = 0
    for r in rows:
        if not (NAMEKEY.search(r['path']) or '.notes.' in r['path']) or len(r['en']) > 80:
            continue
        if r['en'] in NAMEQ_IDIOM:
            continue
        h = cn_head(r['en'], r['cn'])
        if not CJK.search(h):
            continue
        for q, acc in QUALS.items():
            if not re.search(r'\b' + q + r'\b', r['en'], re.I):
                continue
            scanned += 1
            if re.search(acc, h):
                continue
            k = (q, r['en'], r['cn'])
            if k in seen:
                continue
            seen.add(k)
            out.append(dict(repo=r['repo'], pack=r['pack'], path=r['path'],
                            batch_path=r['batch_path'], axis='nameq', kind='qualifier_lost',
                            en_pole=q, cn_pole='(缺)', en_n=1, cn_n=0,
                            en_excerpt=r['en'], cn_excerpt=r['cn']))
    return out, scanned


def floor_unit(n, cn):
    for pat, lab in ((rf'第\s*{n}\s*层', '第N层'), (rf'第\s*{CN_NUM[n]}\s*层', '第N层'),
                     (rf'{CN_NUM[n]}\s*层', 'N层'), (rf'{n}\s*层', 'N层'),
                     (rf'{CN_NUM[n]}\s*楼', 'N楼'), (rf'{n}\s*楼', 'N楼'),
                     (rf'层级\s*{n}', '层级N'), (rf'{n}\s*级', 'N级'), (rf'{n}\s*环', 'N环')):
        if re.search(pat, cn):
            return lab
    return None


def floor_other_number(n, cn):
    """英文说第 n 层，中文写的却是**别的**层号 —— 返回那个错号，否则 None。"""
    for m, cnm in CN_NUM.items():
        if m == n:
            continue
        if re.search(rf'第\s*(?:{m}|{cnm})\s*层|(?:{m}|{cnm})\s*[层楼]|层级\s*{m}', cn):
            return m
    return None


def run_floor(rows):
    """把每个「楼层号」叶子按 (仓库, 建筑) 归组，一组内出现两种以上量词就报**分裂**。

    ⚠ 刻意**不投票**：本库实测多数派常常是错的那一边 —— 破法者之塔里
    「层级N」（区域叶，86 处）是多数，而正确写法是 journal 目录用的「第N层」（14 处）。
    所以这里只列清单，方向交人判；只有 `N级`/`N环` 会额外打 `level_as_rank` 标记，
    因为「级/环」在本库是**角色等级 / 法术环级**的量词，拿来指建筑楼层必错。
    """
    fam = collections.defaultdict(list)
    mismatch = []
    scanned = 0
    for r in rows:
        if len(r['en']) > 80:
            continue
        m = FLOOR_RX.search(r['en'])
        if not m:
            continue
        n = m.group(1) or m.group(2) or m.group(3) or WORD_ORD.get((m.group(4) or '').lower())
        if not n or n not in CN_NUM:
            continue
        h = cn_head(r['en'], r['cn'])
        if not CJK.search(h):
            continue
        u = floor_unit(n, h)
        scanned += 1
        if u is None:
            bad = floor_other_number(n, h)
            if bad:
                mismatch.append(dict(repo=r['repo'], pack=r['pack'], path=r['path'],
                                     batch_path=r['batch_path'], axis='floor',
                                     kind='floor_number_mismatch',
                                     en_pole=f'floor {n}', cn_pole=f'第{bad}层',
                                     en_n=int(n), cn_n=int(bad),
                                     en_excerpt=r['en'], cn_excerpt=r['cn']))
            continue
        # 建筑 = 路径里 scenes./journals. 之后的那一段
        mm = re.search(r'\.(?:scenes|journals)\.([^.]+)', r['path'])
        b = re.sub(r'\s*\(.*?\)\s*$', '', mm.group(1)) if mm else r['pack']
        fam[(r['repo'], b)].append((u, n, r))
    out = []
    for (repo, b), items in sorted(fam.items()):
        units = collections.Counter(u for u, _, _ in items)
        if len(units) < 2:
            continue
        inv = ' / '.join(f'{u}×{k}' for u, k in units.most_common())
        for u in units:
            ex = [r for uu, _, r in items if uu == u]
            out.append(dict(repo=repo, pack=ex[0]['pack'], path=ex[0]['path'],
                            batch_path=ex[0]['batch_path'], axis='floor',
                            kind='floor_unit_' + ('level_as_rank' if u in ('N级', 'N环')
                                                  else 'split'),
                            en_pole=f'{b} 楼层号', cn_pole=f'{u}（本建筑量词清单：{inv}）',
                            en_n=units[u], cn_n=len(units), building=b,
                            sample_paths=[x['path'] for x in ex[:4]],
                            en_excerpt=ex[0]['en'], cn_excerpt=ex[0]['cn']))
    return mismatch + out, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--mode', default='all', choices=['axis', 'nameq', 'floor', 'all'])
    ap.add_argument('--axis', default='all')
    ap.add_argument('--out')
    ap.add_argument('--show', type=int, default=60)
    a = ap.parse_args()

    rows = []
    for repo in a.repo:
        rows += collect(repo)
    axes = (list(AXES) + ['ordinal']) if a.axis == 'all' else a.axis.split(',')

    findings, stats = [], collections.Counter()
    segs = chars = 0
    if a.mode in ('axis', 'all'):
        for r in rows:
            chars += len(r['en'])
            for es, cs, si in segments(r['en'], r['cn']):
                if not es.strip():
                    continue
                segs += 1
                for ax in axes:
                    hit = check_ordinal(es, cs) if ax == 'ordinal' else check_axis(AXES[ax], es, cs)
                    if not hit:
                        continue
                    re_, rc = hit.pop('_re'), hit.pop('_rc')
                    stats[f'{ax}/{hit["kind"]}'] += 1
                    findings.append(dict(repo=r['repo'], pack=r['pack'], path=r['path'],
                                         batch_path=r['batch_path'], axis=ax, seg=si,
                                         en_excerpt=excerpt(es, re_),
                                         cn_excerpt=excerpt(cs, rc), **hit))
    nq_n = fl_n = 0
    if a.mode in ('nameq', 'all'):
        got, nq_n = run_nameq(rows)
        for g in got:
            stats['nameq/' + g['kind']] += 1
        findings += got
    if a.mode in ('floor', 'all'):
        got, fl_n = run_floor(rows)
        for g in got:
            stats['floor/' + g['kind']] += 1
        findings += got

    findings.sort(key=lambda f: (f['axis'], f['repo'], f['pack'], f['path']))
    scale = dict(leaf_pairs=len(rows), aligned_segments=segs, en_chars=chars,
                 nameq_qualifier_occurrences=nq_n, floor_number_leaves=fl_n)
    print('scanned: ' + '  '.join(f'{k}={v}' for k, v in scale.items()))
    for k, v in sorted(stats.items()):
        print(f'  {k:<34} {v}')
    print(f'TOTAL raw hits={len(findings)}')
    for f in findings[:a.show]:
        print('-' * 72)
        print(f"[{f['axis']}/{f['kind']}] EN={f['en_pole']} -> CN={f['cn_pole']}"
              f"  {f['repo']} {f['pack']}")
        print('  ' + f['path'])
        print('  EN: ' + f['en_excerpt'])
        print('  CN: ' + f['cn_excerpt'])
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'scale': scale, 'stats': dict(stats), 'findings': findings},
                      fh, ensure_ascii=False, indent=1)
        print('wrote ' + a.out)


if __name__ == '__main__':
    main()
