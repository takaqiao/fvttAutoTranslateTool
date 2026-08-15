#!/usr/bin/env python3
"""Build the project glossary `glossary_ec.json`.

Layers, highest priority last:
  1. base        - glossary_crucible_merged.json (4602 已裁决译名)
  2. harvested   - EN->CN pairs mined from the existing shipped translations by
                   walking the English baseline and the CN file in parallel
  3. (report)    - English terms present in the current baselines with no CN at
                   all, written to glossary_ec.pending.json for translators

Outputs (into 5-其他内容/glossary/):
  glossary_ec.json             flat {en: cn}   <- consumed by QA / TM tooling
  glossary_ec.provenance.json  per-term source + conflicts
  glossary_ec.pending.json     English terms still needing a Chinese name

The harvest pool is keyed on `(\u82f1\u6587, \u5b57\u6bb5\u89d2\u8272)`, never on the English alone
--------------------------------------------------------------------------
"\u8981\u4e0d\u8981\u63a5\u82f1\u6587\u5c3e\u5df4" is a FIELD-ROLE convention in this project, not a style choice:
`name` is bilingual (\u300c\u62a4\u76fe\u672f Shield\u300d) while `tokenName` / `adjective` / `levels` /
`folders` / \u5206\u7c7b\u540d / \u8868\u7ed3\u679c\u884c are bare Chinese. A pool keyed on the English alone
has no way to see that, so a full-library majority decided the form -- which is
how `Determination` (an `adjective`, wired into \u7269\u54c1\u540d\u62fc\u88c5) ended up as
\u300c\u51b3\u5fc3 Determination\u300d and `Lyla Cevher` (a `name`) lost its English tail.
The role now enters the key, the tie-break falls back to the role's convention
instead of to `sorted`'s stability, and what is still tied is reported rather
than picked.

`glossary_ec.json` stays flat `{en: cn}` (every consumer expects that shape); the
per-role table is written next to it as `glossary_ec.byrole.json`.

Babele `nameCollection` containers are harvested too (2026-08-15, 第十六轮)
--------------------------------------------------------------------------
`scenes.<X>.notes` is `{noteId: "Label"}` in English and `{"Label": "中文"}` in
Chinese, so the same-key descent never reached it: 175 long-finished map-note
translations could only ever enter the glossary through the LOWEST-priority
base layer. That is the exact failure `R-glossary-not-laundering` guards —
edit one of those notes in the pack and the glossary keeps serving the stale
base value forever. `walk_pairs` now bridges it (see its docstring), the role
being the CONTAINER's name (`notes`) because the English key is an opaque id.

Known limit: the role is the LAST NON-DIGIT path segment (same semantics as
`qa/propagate_fix.py::role_of`, imported from there so the two cannot drift).
That still merges the sub-populations of `name` -- \u8868\u7ed3\u679c\u884c `results.<k>.name` is
bare Chinese while `pages.<k>.name` is bilingual. Those are separated by
`CONVENTION_PARENTS` below when deciding a FORM, but they share one bucket.
`tm/fill_twin.py::shape_of` is the finer instrument if a full structural key is
ever needed here.

Usage:
  python build_glossary.py [--project <dir>] [--out-dir <dir>]
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict

CJK = re.compile(r'[\u4e00-\u9fff]')
LATIN = re.compile(r'[A-Za-z]')
HTML = re.compile(r'<[^>]+>')
# A glossary term is a short, tag-free, non-sentence string: names and labels.
MAX_TERM_LEN = 60

# One implementation of "\u5b57\u6bb5\u89d2\u8272", imported rather than copied.
_spec = importlib.util.spec_from_file_location(
    '_propagate_fix',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qa', 'propagate_fix.py'))
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)
role_of = _pf.role_of

# Roles whose value is rendered verbatim (bare Chinese, no English tail).
BARE_ROLES = {'tokenName', 'adjective', 'levels', 'condition', 'folders',
              'prototypeToken', 'navName'}
# Roles that carry the bilingual \u300c\u4e2d\u6587 English\u300d house format.
# `notes` = the map-note nameCollection reached by walk_pairs' nameCollection
# bridge; all 600 shipped note leaves are bilingual, so the tie-break must not
# fall through to 'prose'.
BILINGUAL_ROLES = {'name', 'label', 'title', 'notes'}
# Containers that force the bare convention on whatever role sits under them --
# `tables.X.results.<k>.name` is a table row, not an entity name.
CONVENTION_PARENTS = {'results', 'regions', 'levels', 'categories', 'tokens'}
# Collections keyed by the ENTITY's English name (Babele's nameCollection shape).
# Their leaf's last segment is the entity name, not a field name, so `role_of`
# would hand back 「Biomes」/「East Wing」 as if those were field roles -- 322
# distinct "roles" in one run, most of them one-off entity names.
NAME_KEYED = {'categories', 'folders'}

# 永久 pending 豁免表 —— 这些英文**不该**有中文，出现在 pending 里不是欠账。
# 只有一族：`ember.location` / `ember.biome` 的 `system.terrain` 枚举 id。
# `extract/mappings.mjs:235` 已把它登记为 NOT translatable（schema 用
# `terrainChoices = () => ember.scenes.region.terrain` 校验，译了就不在 choices 里，
# 页面直接构造失败）。2026-08-15 第十六轮主控裁定：永久 pending，不再当待办。
PERMANENT_PENDING = {
    'bluffs':   'ember.location/biome system.terrain 枚举 id（mappings.mjs:235 NOT translatable）',
    'canyon':   'ember.location/biome system.terrain 枚举 id（mappings.mjs:235 NOT translatable）',
    'difficult': 'ember.location/biome system.terrain 枚举 id（mappings.mjs:235 NOT translatable）',
    'extreme':  'ember.location/biome system.terrain 枚举 id（mappings.mjs:235 NOT translatable）',
    'normal':   'ember.location/biome system.terrain 枚举 id（mappings.mjs:235 NOT translatable）',
    'water':    'ember.location/biome system.terrain 枚举 id（mappings.mjs:235 NOT translatable）',
}


def convention_of(path: str, role: str) -> str:
    """'bare' | 'bilingual' | 'prose' \u2014 what FORM this slot's Chinese takes."""
    if role in BARE_ROLES:
        return 'bare'
    if set(path.split('.')) & CONVENTION_PARENTS:
        return 'bare'
    if role in BILINGUAL_ROLES:
        return 'bilingual'
    return 'prose'


def form_of(cn: str, en: str) -> str:
    """The form a candidate Chinese actually has."""
    return 'bilingual' if (cn.strip().endswith(en.strip()) or LATIN.search(cn)) else 'bare'


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


_WS = re.compile(r'\s+')


def bilingual_tail(base_cn: str, en: str, shipped: str) -> bool:
    """已发布值是不是「base 值 + 空格 + 英文」这**一个**形态 —— 唯一的 format_only。

    ⚠ 判据必须是**整串相等**，绝不能退回子串包含。第十六轮收尾实测：旧判据写作
    `(cn_s.startswith(b) and en_s in cn_s) or (b in cn_s) or (cn_s in b)`，
    那个纯子串子句让「凡 base 值是 shipped 值的子串」全部被静默判成「只是双语格式差异」，
    `baseVsShipped` 里因此吃掉了 **11 条真 dispute**（当时报 3174 条，收紧后实读 **3172**；
    ⚠ 别把这两个数当同一口径的前后值 —— 差的 2 条是 `Ooze` / `Elemental`，
    它们改对之后完全不再进 baseVsShipped）：
      `Cosmology` 宇宙→宇宙观 · `Axe` 斧→斧头 · `Buckler` 圆盾→小圆盾 ·
      `Pallid Drakeling` 幼龙→幼龙兽 · `Afflicted Pallid Drakeling` 同上 ·
      `Lyla Cevher` 莱拉→莱拉·杰夫赫尔 · `Funar Cevher` 杰夫赫→杰夫赫尔 ·
      `Dress Clothing` 礼服→礼服服装 · `Wingsuit` 飞行服→翼装飞行服 ·
      `Ooze` 软泥怪→软泥（反向子串）· `Elemental` 元素→元素生物。
    多数方向上 shipped 是对的 —— 但它们本该以 dispute 的形态**被人看见一次**再落回 base，
    而不是被判据自动消解。**「disputes 归 0」只有在判据不吃缺陷时才是个有意义的指标**；
    判据一宽，这个数就成了本项目反复吃亏的那种「静默全绿」。

    只做空白归一（`名字  English` 这种多空格算同一形态），不做任何包含式放宽。
    """
    norm = lambda s: _WS.sub(' ', s.strip())        # noqa: E731
    return norm(shipped) == f'{norm(base_cn)} {norm(en)}'


def is_term(s: str) -> bool:
    """Name-like enough to belong in a glossary."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > MAX_TERM_LEN:
        return False
    if HTML.search(s) or '\n' in s:
        return False
    # Skip sentences: a term rarely ends in punctuation or has many words.
    if s[-1] in '.!?:;,':
        return False
    return len(s.split()) <= 8


def walk_pairs(en, cn, out, path=(), root_role='name'):
    """Walk two parallel structures, yielding `(en_string, cn_string, path, role)`.

    `path` is what gives the pool its role dimension; without it every leaf in
    the two repos lands in one undifferentiated bucket (see module docstring).
    `root_role` names the role of a leaf that sits directly under the walk root,
    where the last path segment is the ENTITY's own name rather than a field
    name — that is the shape of the top-level `folders` collection.

    Two dicts are "parallel" only where they share a key. Babele's
    `nameCollection` containers do NOT: `scenes.<X>.notes` is `{noteId: "Label"}`
    on the English side and `{"Label": "中文"}` on the Chinese side, so a
    same-key descent never reaches them and 175 long-finished map-note
    translations were invisible to harvest (they only ever showed up via the
    lowest-priority base layer, which is exactly the "launder a stale value into
    authority" failure `R-glossary-not-laundering` exists to stop). The
    `elif` below is that missing bridge: when the English VALUE is itself a key
    on the Chinese side, the pair is `(en_value, cn[en_value])` and the role is
    the CONTAINER's name (`notes`), because the English-side key is an opaque id,
    not a field name.
    """
    if isinstance(en, dict) and isinstance(cn, dict):
        for k, v in en.items():
            if k in cn:
                walk_pairs(v, cn[k], out, path + (k,), root_role)
            elif isinstance(v, str) and isinstance(cn.get(v), str):
                out.append((v, cn[v], '.'.join(path + (k,)),
                            path[-1] if path else root_role))
    elif isinstance(en, list) and isinstance(cn, list):
        for i, (a, b) in enumerate(zip(en, cn)):
            walk_pairs(a, b, out, path + (str(i),), root_role)
    elif isinstance(en, str) and isinstance(cn, str):
        dotted = '.'.join(path)
        if len(path) >= 2 and path[-2] in NAME_KEYED:
            role = path[-2]
        elif len(path) > 1:
            role = role_of(dotted)
        else:
            role = root_role
        out.append((en, cn, dotted, role))


def harvest(en_dir, cn_dir, label, pairs, pending, conv, cells=None):
    """Mine EN->CN term pairs from a baseline/translation directory pair.

    `cells` is the same pool sliced one level finer, by `(模块, 约定)` -- see
    `main`'s sameRoleMissingTail block for why the role key alone cannot tell a
    dropped suffix from a convention boundary.
    """
    if not (os.path.isdir(en_dir) and os.path.isdir(cn_dir)):
        return 0
    found = 0
    for fn in sorted(f for f in os.listdir(en_dir) if f.endswith('.json') and not f.startswith('_')):
        cn_path = os.path.join(cn_dir, fn)
        try:
            en_doc = load(os.path.join(en_dir, fn))
        except Exception:
            continue
        en_entries = en_doc.get('entries', {})
        # 顶层 folders 也要采：它是玩家在侧边栏直接看到的分类名。
        # 2026-08-13 第十二轮发现，此前只走 entries，导致所有 crucible.* 包与
        # ember.character / ember.crucible-adversary / ember.crucible-items 的
        # 文件夹名永远进不了 harvest 层，只能被 base 层的陈旧值覆盖
        # （包里 folders.Monstrosities 早已是「畸怪」而词表还是「怪物」就是这么来的）。
        en_folders = en_doc.get('folders', {})

        # Every short EN string is a term candidate, translated or not.
        cand = []
        walk_pairs(en_entries, en_entries, cand)
        walk_pairs(en_folders, en_folders, cand, root_role='folders')
        for s, _cn, _path, _role in cand:
            if is_term(s):
                pending.setdefault(s, set()).add(f'{label}:{fn}')

        if not os.path.exists(cn_path):
            continue
        try:
            cn_doc = load(cn_path)
        except Exception as e:
            print(f'  ! unreadable CN file {fn}: {e}')
            continue

        got = []
        walk_pairs(en_entries, cn_doc.get('entries', {}), got)
        walk_pairs(en_folders, cn_doc.get('folders', {}), got, root_role='folders')
        for en_s, cn_s, path, role in got:
            if not is_term(en_s) or not CJK.search(cn_s):
                continue
            convention = convention_of(path, role)
            pairs[(en_s, role)][cn_s] += 1
            conv[(en_s, role)][convention] += 1
            if cells is not None:
                cells[(en_s, role)][(label, convention)][cn_s] += 1
            found += 1
    return found


def pick(key, cns, conv):
    """Winner for one `(英文, 角色)` bucket, or `(None, tied_candidates)`.

    A plain `sorted(..., key=-count)[0]` resolves a tie by whichever candidate
    was met first, i.e. by file iteration order. When the vote is tied the
    ROLE's convention decides instead; if that still cannot separate them the
    bucket is reported unresolved rather than picked.
    """
    ranked = sorted(cns.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    tied = [cn for cn, n in ranked if n == top]
    if len(tied) == 1:
        return tied[0], None
    want = conv[key].most_common(1)[0][0] if conv.get(key) else 'prose'
    match = [cn for cn in tied if form_of(cn, key[0]) == want]
    if len(match) == 1:
        return match[0], None
    return None, tied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')))
    ap.add_argument('--out-dir', default=None,
                    help='写产物的目录（缺省 <project>/5-其他内容/glossary）。'
                         '给它一个临时目录就能在不动库存产物的前提下比对新旧输出。')
    a = ap.parse_args()
    P = a.project
    OUT = a.out_dir or os.path.join(P, '5-其他内容', 'glossary')
    BASE_DIR = os.path.join(P, '5-其他内容', 'english-baseline')

    # ---- layer 1: base glossary -------------------------------------------
    base_path = r'C:\Users\Taka\Desktop\fvtt\glossary_crucible_merged.json'
    base_raw = load(base_path)
    base = {k: v for k, v in base_raw.items() if isinstance(v, str) and CJK.search(v)}
    print(f'base   : {len(base)} terms  ({os.path.basename(base_path)})')

    # ---- layer 2: harvest from shipped translations ------------------------
    pairs = defaultdict(Counter)          # (en, role) -> Counter(cn)
    conv = defaultdict(Counter)           # (en, role) -> Counter(convention)
    # (en, role) -> (模块, 约定) -> Counter(cn). Report-only: `pick` never reads
    # it, so slicing here cannot move a single glossary value.
    cells = defaultdict(lambda: defaultdict(Counter))
    pending_src: dict[str, set] = {}
    n = harvest(os.path.join(BASE_DIR, 'crucible-0.10.1'),
                os.path.join(P, '2-Crucible汉化插件', 'compendium', 'cn'),
                'crucible', pairs, pending_src, conv, cells)
    print(f'harvest: crucible {n} pair-hits')
    n = harvest(os.path.join(BASE_DIR, 'ember-0.6.0'),
                os.path.join(P, '1-Ember汉化插件', 'compendium', 'cn'),
                'ember', pairs, pending_src, conv, cells)
    print(f'harvest: ember    {n} pair-hits')

    # 每个 (英文, 角色) 桶各自定胜负；平票不靠排序稳定性，退到该角色的约定形态。
    byrole: dict[str, dict[str, str]] = defaultdict(dict)
    votes: dict[str, dict[str, int]] = defaultdict(dict)
    conflicts_by_role: dict[str, dict[str, dict]] = defaultdict(dict)
    unresolved_ties: dict[str, dict[str, list]] = defaultdict(dict)
    for (en_s, role), cns in pairs.items():
        chosen, tied = pick((en_s, role), cns, conv)
        if len(cns) > 1:
            conflicts_by_role[en_s][role] = dict(cns.most_common())
        if chosen is None:
            unresolved_ties[en_s][role] = tied
            continue
        byrole[en_s][role] = chosen
        votes[en_s][role] = cns[chosen]

    # 顶层 glossary_ec.json 仍是扁平 {en: cn}，取「主形态」：
    # 有 name/label 桶就用它（双语并列约定），否则用票数最多的那个角色桶。
    ROLE_PRIORITY = ('name', 'label', 'title', 'folders')
    harvested = {}
    primary_role = {}
    for en_s, per_role in byrole.items():
        role = next((r for r in ROLE_PRIORITY if r in per_role), None)
        if role is None:
            role = max(per_role, key=lambda r: votes[en_s][r])
        harvested[en_s] = per_role[role]
        primary_role[en_s] = role

    # 兼容旧字段：provenance 的 multiCandidate 仍按英文给出所有候选（现在带角色）。
    conflicts = {en_s: rr for en_s, rr in conflicts_by_role.items()}
    print(f'harvest: {len(harvested)} distinct terms, {len(conflicts)} with >1 candidate '
          f'in some role, {len(unresolved_ties)} unresolved ties (not written)')

    # ---- merge -------------------------------------------------------------
    glossary = {}
    provenance = {}
    base_vs_harvest = {}
    for en_s, cn_s in base.items():
        glossary[en_s] = cn_s
        provenance[en_s] = {'cn': cn_s, 'source': 'base'}
    disputes = {}
    for en_s, cn_s in harvested.items():
        if en_s in glossary:
            if glossary[en_s] != cn_s:
                b = glossary[en_s]
                # The house style is bilingual ("中文 English"). The base glossary
                # stores the bare Chinese. That -- and ONLY that -- is a FORMAT
                # difference rather than a disagreement about the translation, so
                # shipped may win silently. Anything else is a dispute; see
                # `bilingual_tail` for the 11 real disputes the old substring
                # criterion was swallowing.
                format_only = bilingual_tail(b, en_s, cn_s)
                base_vs_harvest[en_s] = {'base': b, 'shipped': cn_s,
                                         'kind': 'format' if format_only else 'conflict'}
                if format_only:
                    glossary[en_s] = cn_s
                    provenance[en_s] = {'cn': cn_s, 'source': 'shipped-bilingual-form',
                                        'base_was': b}
                else:
                    # A real disagreement: two different Chinese renderings.
                    # Do NOT pick silently. Keep shipped (status quo, what players
                    # see today) but surface it for adjudication.
                    disputes[en_s] = {'base': b, 'shipped': cn_s}
                    glossary[en_s] = cn_s
                    provenance[en_s] = {'cn': cn_s, 'source': 'DISPUTED-shipped-kept',
                                        'base_was': b}
            continue
        glossary[en_s] = cn_s
        provenance[en_s] = {'cn': cn_s, 'source': 'shipped',
                            'role': primary_role.get(en_s),
                            'byRole': dict(sorted(byrole[en_s].items()))}
        if en_s in conflicts:
            provenance[en_s]['candidates'] = conflicts[en_s]

    # ---- pending: English terms with no Chinese anywhere --------------------
    pending = {s: sorted(src) for s, src in pending_src.items() if s not in glossary}
    leftover = sorted(set(pending) - set(PERMANENT_PENDING))

    fmt = sum(1 for v in base_vs_harvest.values() if v['kind'] == 'format')
    print(f'\nglossary_ec : {len(glossary)} terms '
          f'({len(base)} base + {len(glossary) - len(base)} new from shipped)')
    print(f'base vs shipped, format-only (auto-resolved): {fmt}')
    print(f'base vs shipped, REAL disputes (need ruling) : {len(disputes)}')
    print(f'pending (no CN yet)                          : {len(pending)} '
          f'（其中永久豁免 {len(set(pending) & set(PERMANENT_PENDING))}，真待补 {len(leftover)}）')
    if leftover:
        print(f'  ⚠ 未豁免的待补词：{leftover[:12]}')
    gone = sorted(set(PERMANENT_PENDING) - set(pending))
    if gone:
        print(f'  ⚠ 豁免表里这几条已不在 pending（上游改了？该删豁免）：{gone}')

    # Internal inconsistency: the same English proper noun rendered several ways
    # inside the shipped translations themselves. Split into the two very
    # different problems this hides.
    def bare(cn, en):
        """Strip the ' English' suffix of the bilingual house format."""
        return cn[:-len(en)].strip() if cn.endswith(en) else cn

    # 有了角色维度，从前那个「392 条、不得批量归一」的大杂烩可以分开了：
    #   crossRoleFormDifference —— 同一英文在不同角色里形态不同。这是**约定正确**
    #     的表现，不是缺陷（`name` 双语并列「决心 Determination」，`adjective` 裸
    #     中文「决心」；后者被 crucible item-physical.mjs:324 拼进物品名，归一即回归）。
    #   sameRoleMissingTail —— **同一个角色内部**后缀时加时不加。这才是真漏加，可动。
    #   termInconsistency —— 剥掉后缀后中文本身就不同，必须裁决。
    cross_role_form, same_role_missing_tail, term_inconsistent = {}, {}, {}
    cross_convention_form = {}
    for en_s, per_role in byrole.items():
        forms = {r: form_of(c, en_s) for r, c in per_role.items()}
        if len(set(forms.values())) > 1:
            cross_role_form[en_s] = {r: {'cn': per_role[r], 'form': forms[r]}
                                     for r in sorted(per_role)}
    # 「同角色内漏加后缀」以前是拿 `(英文, 角色)` 直接判的 —— 那个键看不见两条
    # **真实存在的约定边界**，于是把「两群各自守规矩的叶子」报成了缺陷：
    #   模块边界 —— crucible-cn 的 `folders` 是裸中文（150/151），
    #     ember_cn 的 `folders` 是双语并列（337/355）。两个分开发布的产品，
    #     各自内部近乎全一致；把 crucible 的「护甲」和 ember 的「护甲 Armor」
    #     放一起比，比出来的是产品间风格差异，不是谁漏了后缀。
    #   约定槽边界 —— `convention_of` 早就知道 `regions.X.name` 该裸、
    #     `pages.X.name` 该双语（CONVENTION_PARENTS），但 role_of 把两者并进
    #     同一个 `name` 桶（模块 docstring 里的 known limit）。Darkness /
    #     Elevator / Ocean 三条就是这么来的：region 名裸、页/物品/场景名双语，
    #     **两边都是对的**。
    # 2026-08-15 第十七轮实测：31 条里 0 条是「一个译者在同一约定内忘了后缀」。
    # 现在按 `(模块, 约定)` 单元格判：只有**同一单元格内部**同时出现两种形态才
    # 算真漏加；跨单元格的挪到 crossConventionFormDifference，边界写在条目里，
    # 不是删掉不报。
    for en_s, per_role_cands in conflicts_by_role.items():
        for role, cands in per_role_cands.items():
            if len(cands) < 2:
                continue
            if len({bare(c, en_s) for c in cands}) != 1:
                term_inconsistent.setdefault(en_s, {})[role] = cands
                continue
            cell = cells[(en_s, role)]
            split = {f'{m}/{cv}': dict(c.most_common())
                     for (m, cv), c in sorted(cell.items()) if len(c) > 1}
            if split:
                same_role_missing_tail.setdefault(en_s, {})[role] = {
                    'candidates': cands, 'splitWithinCell': split}
            else:
                cross_convention_form.setdefault(en_s, {})[role] = {
                    'axis': '模块' if len({m for m, _ in cell}) > 1 else '约定槽',
                    'cells': {f'{m}/{cv}': dict(c.most_common())
                              for (m, cv), c in sorted(cell.items())},
                }
    print(f'shipped: 跨角色形态差异（约定正确，勿归一）    : {len(cross_role_form)}')
    print(f'shipped: 跨模块/跨约定槽形态差异（各自守规矩）: {len(cross_convention_form)}')
    print(f'shipped: 同角色同单元格内漏加双语后缀（可动）  : {len(same_role_missing_tail)}')
    print(f'shipped: same term translated differently    : {len(term_inconsistent)}')

    dump(os.path.join(OUT, 'glossary_ec.disputes.json'), {
        '_meta': {
            'count': len(disputes),
            'note': '基底术语表与已发布译文对同一英文给出了不同中文（非双语格式差异）。'
                    '当前一律先保留已发布值（玩家现在看到的），但每条都需要人工/上下文裁决。'
                    '裁决后写回 glossary_ec.json 并记入 PROJECT.md 第 8 节。',
        },
        'disputes': dict(sorted(disputes.items())),
        'crossRoleFormDifference': {
            '_note': '同一英文在不同**字段角色**里形态不同（一边双语并列、一边裸中文）。'
                     '⚠ 这是**约定正确**的表现，不是缺陷，**不得归一**：`name` 用'
                     '「护盾术 Shield」，`tokenName` / `adjective` / `levels` / `folders` / '
                     '表结果行用裸中文。adjective 归一后会被 crucible '
                     'item-physical.mjs:324 拼进物品名，玩家看到「决心 Determination护符」。'
                     '列在这里只为让人看见这条英文横跨了几个角色。',
            'count': len(cross_role_form),
            'terms': dict(sorted(cross_role_form.items())),
        },
        'crossConventionFormDifference': {
            '_note': '同一英文、同一角色，中文相同、只差双语后缀，但两种形态分别落在'
                     '**不同的约定单元格**里，各自单元格内部是一致的 —— 也就是说没人'
                     '漏加后缀，只是这个 `(英文, 角色)` 键横跨了一条真实的约定边界。'
                     '`axis` 指出是哪条：'
                     '「模块」= crucible-cn 的 `folders` 裸中文（150/151）对上 '
                     'ember_cn 的 `folders` 双语并列（337/355），两个分开发布的产品'
                     '各有各的行文风格；'
                     '「约定槽」= `regions.X.name` 该裸而 `pages.X.name` 该双语'
                     '（CONVENTION_PARENTS 早就知道，但 role_of 把它们并进同一个 '
                     '`name` 桶，见模块 docstring 的 known limit）。'
                     '⚠ **不是缺陷，不要按票数归一。** 列在这里只为让人看见边界；'
                     '真要统一，那是一次产品级裁决（同一个世界里两个包的侧边栏文件夹'
                     '会一个「护甲」一个「护甲 Armor」并排显示），不是脚本能定的。',
            'count': len(cross_convention_form),
            'terms': dict(sorted(cross_convention_form.items())),
        },
        'sameRoleMissingTail': {
            '_note': '**同一个角色、同一个 `(模块, 约定)` 单元格内部**，同一英文的中文'
                     '相同、只是双语后缀时加时不加 —— 单元格内部自相矛盾，这才是真漏加，'
                     '可按该单元格的约定统一。`splitWithinCell` 直接给出是哪个单元格裂了。'
                     '⚠ 判据从前只看 `(英文, 角色)`，那个键看不见模块边界和约定槽边界，'
                     '把两群各自守规矩的叶子报成了缺陷：2026-08-15 第十七轮把当时的 31 条'
                     '逐条落到叶子上核过，**0 条**是同一约定内忘加后缀，24 条跨边界'
                     '（已移入 crossConventionFormDifference），余下 7 条全是 ember 的'
                     '冒险文件夹（双语，290/290）对上 ember 镜像包 '
                     '`crucible-items` 的包文件夹（裸，8/8）—— 每个文件内部 100% 自洽，'
                     '一条越界的叶子都没有。这个桶的数字只有在判据不吃约定边界时才有意义。',
            'count': len(same_role_missing_tail),
            'terms': dict(sorted(same_role_missing_tail.items())),
        },
        'shippedTermInconsistency': {
            '_note': '同一专名在同一角色里被译成了不同的中文。必须裁决后统一。',
            'count': len(term_inconsistent),
            'terms': dict(sorted(term_inconsistent.items())),
        },
        'unresolvedRoleTies': {
            '_note': '该 (英文, 角色) 桶里两个及以上中文票数相同，且按角色约定的形态'
                     '也分不出唯一胜者。**没有硬选**，这些英文没有进 harvest 层。'
                     '以前是靠 sorted 的稳定性（即文件遍历顺序）默默定的。',
            'count': len(unresolved_ties),
            'terms': dict(sorted(unresolved_ties.items())),
        },
    })

    dump(os.path.join(OUT, 'glossary_ec.byrole.json'), {
        '_meta': {
            'count': len(byrole),
            'note': '按 (英文, 字段角色) 给出的译名。角色 = 叶子路径最后一段非数字键名'
                    '（与 qa/propagate_fix.py::role_of 同语义）。'
                    'fill_twin_names.py / apply_tm 这类要按角色取形态的工具查这里；'
                    'glossary_ec.json 只给「主形态」（有 name/label 桶就用它）。',
            'primaryRoleCounts': dict(Counter(primary_role.values()).most_common()),
        },
        'terms': {k: dict(sorted(v.items())) for k, v in sorted(byrole.items())},
    })

    dump(os.path.join(OUT, 'glossary_ec.json'), dict(sorted(glossary.items())))
    dump(os.path.join(OUT, 'glossary_ec.provenance.json'), {
        '_meta': {
            'baseGlossary': base_path,
            'baseTerms': len(base),
            'totalTerms': len(glossary),
            'harvestedTerms': len(harvested),
            'baseVsShippedDisagreements': len(base_vs_harvest),
            'pendingTerms': len(pending),
            'policy': 'shipped translation wins over base glossary; '
                      'base glossary wins over nothing; PF2E master glossary NOT merged',
        },
        'baseVsShipped': dict(sorted(base_vs_harvest.items())),
        'multiCandidate': dict(sorted(conflicts.items())),
        'terms': dict(sorted(provenance.items())),
    })
    dump(os.path.join(OUT, 'glossary_ec.pending.json'), {
        '_meta': {'count': len(pending),
                  'note': '当前英文基准里出现、但项目术语表中还没有中文对应的名词。'
                          '翻译时遇到即补进 glossary_ec.json。'
                          '⚠ 列在 permanentlyExempt 里的**不是欠账**，别再当待办。',
                  'exemptCount': len(set(pending) & set(PERMANENT_PENDING)),
                  'actionableCount': len(leftover)},
        'permanentlyExempt': {
            '_note': '这些英文按裁决**不该**有中文；仍出现在 terms 里是因为它们确实'
                     '存在于英文基准。2026-08-15 第十六轮主控裁定：永久 pending。',
            'terms': {k: v for k, v in sorted(PERMANENT_PENDING.items()) if k in pending},
        },
        'terms': dict(sorted(pending.items())),
    })
    print(f'\nwrote -> {OUT}')


main()
