#!/usr/bin/env python3
"""Fill the dnd5e twin pack's untranslated item/effect NAMES from two ordered sources.

  python fill_twin_names.py [--out <dir>] [--report <json>]

Sources, in the project's evidence order (强 → 弱)
--------------------------------------------------
1. **本项目库内同英文的既有译名.** If any `.name` anywhere in the two repos already
   translates this exact English, reuse it verbatim. Consistency inside our own
   library outranks any outside source.
2. **`dnd-simplified-chinese-babele-patch` 的 `dnd5e.*` 核心包.** Only the files
   named `dnd5e.*` — those are the system's own compendia, where that module is
   the localisation a dnd5e player actually sees. The other ~395 files in that
   module translate third-party content and carry no authority here.

Anything neither source covers is left alone and reported; those need a translator.

Why the library TM is keyed on `(结构路径, 英文)` and not on the English alone
---------------------------------------------------------------------------
`.name` is not one population. `journals.pages.name` / `actors.items.name` follow
the project's bilingual `中文 English` convention, while `scenes.regions.name`,
`scenes.regions.behaviors.name` and `tables.results.name` are **bare Chinese**
(they are rendered as-is into region banners and table rows). Keying the TM on
`e.strip()` alone -- which is what this script used to do -- merges them and lets
a full-library majority decide, so 532 keys resolved to a bare-Chinese value
harvested purely from bare-convention roles and would have been poured into
bilingual `.name` slots. `fill_twin.py` hit exactly this and wrote `shape_of` for
it (see its module docstring); that one implementation is imported here rather
than copied, so the two cannot drift apart.

A structural miss does NOT fall back to the English-only key -- that fallback was
deleted from `fill_twin.py` on 2026-08-14 for the same reason. A miss goes to the
translator, and what the role-blind key *would* have suggested is recorded in the
report under `roleblind_withheld`.

Why the module is not applied blindly
-------------------------------------
Comparing the module against names we already translated: **997 agree, 752 differ**
— and the differences are systematic (module 燃火术/疗伤术/纠缠术 vs ours
产生火焰/治疗伤口/缠绕). Worse, 542 of those "differences" sit in
`ember.crucible-adventure` and the `crucible.*` packs, where the English happens to
collide (Dagger, Longbow…) but the item is **Crucible's own** and the dnd5e module
has no say at all. So a blanket sweep would corrupt the Crucible side.

This script therefore only **fills empty slots**. It never overwrites an existing
translation. Whether to also re-align the ~787 already-translated twin names to the
module's wording is a separate, deliberate decision — see PROJECT.md 第 8 节.

The name format is the project's bilingual `中文 English`.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ("1-Ember汉化插件", "2-Crucible汉化插件")
DND = (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules"
       r"\dnd-simplified-chinese-babele-patch\translation\cn")
PACK = "ember.adventure.json"
CJK = re.compile(r'[一-鿿]')
ID16 = re.compile(r'[A-Za-z0-9]{16}')

# Single implementation of the structural key, imported from fill_twin.py so the
# two scripts cannot drift (its STRUCT whitelist is maintained there).
_spec = importlib.util.spec_from_file_location(
    "_fill_twin", os.path.join(P, "3-常用脚本", "tm", "fill_twin.py"))
_fill_twin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fill_twin)
shape_of = _fill_twin.shape_of

# Container segments whose child `name` is rendered verbatim (bare Chinese, no
# English tail): scene region banners, region behaviour labels, table result
# rows, level/category labels. Everything else follows the bilingual convention.
BARE_CN_PARENTS = {'regions', 'results', 'levels', 'categories', 'behaviors', 'tokens'}


def wants_bilingual(path: str) -> bool:
    """True when this `.name` slot takes the project's `中文 English` form."""
    return not (set(shape_of(path).split('.')[:-1]) & BARE_CN_PARENTS)


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def load_pairs(repo):
    d = os.path.join(P, repo, 'compendium')
    en_dir, cn_dir = os.path.join(d, 'en'), os.path.join(d, 'cn')
    for pack in sorted(os.listdir(en_dir)):
        if not pack.endswith('.json') or not os.path.exists(os.path.join(cn_dir, pack)):
            continue
        o = []
        walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
             json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
             [], o)
        for row in o:
            yield (pack,) + row


def dnd_core_tm():
    """English name -> Chinese, from the module's `dnd5e.*` files only.

    Files keyed by 16-char document ids are skipped: without the dnd5e system's
    own English side we cannot map an id back to a name, so they are unusable
    as a text TM.
    """
    tm = {}
    if not os.path.isdir(DND):
        return tm
    for f in sorted(os.listdir(DND)):
        if not (f.startswith('dnd5e.') and f.endswith('.json')):
            continue
        try:
            e = json.load(open(os.path.join(DND, f), encoding='utf-8')).get('entries')
        except Exception:
            continue
        if not isinstance(e, dict) or not e:
            continue
        keys = list(e)[:20]
        if sum(1 for k in keys if ID16.fullmatch(k)) > len(keys) * 0.6:
            continue
        for k, v in e.items():
            if isinstance(v, dict) and isinstance(v.get('name'), str) and CJK.search(v['name']):
                tm.setdefault(k.strip(), v['name'])
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(P, '5-其他内容', 'reports', 'ember'))
    ap.add_argument('--report', default=os.path.join(
        P, '5-其他内容', 'reports', 'ember', 'twin_names.report.json'))
    a = ap.parse_args()

    # 1. library names, majority wins when the same English has several renderings.
    #    键 = (结构路径形状, 英文)。**不是**英文单独作键 —— 见文件头。
    shaped = defaultdict(Counter)   # (shape, en) -> Counter(cn)
    blind = defaultdict(Counter)    # en -> Counter(cn)，只用于报告「无角色键会建议什么」
    for repo in REPOS:
        for pack, path, e, c in load_pairs(repo):
            if path.endswith('.name') and c and CJK.search(c):
                shaped[(shape_of(path), e.strip())][c] += 1
                blind[e.strip()][c] += 1
    lib = {k: v.most_common(1)[0] for k, v in shaped.items()}

    core = dnd_core_tm()

    # 2. glossary_ec —— 它是从**本项目已发布译文**挖出来的，所以代表库内用法，
    #    排在外部模块之前。但阶段 2 记录过它含机翻残留，所以过滤掉明显坏的条目：
    #    中文里混着拉丁字母、或中文与英文相同（等于没译）。
    gloss = {}
    gpath = os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json')
    if os.path.exists(gpath):
        for k, v in json.load(open(gpath, encoding='utf-8')).items():
            if not isinstance(v, str):
                continue
            zh = v.split(' ')[0].strip()
            if zh and CJK.search(zh) and not re.search(r'[A-Za-z]', zh) and zh != k:
                gloss[k.strip()] = zh
    # `Arrow` 在孪生包里是弹药（箭矢），glossary 记的「箭头」是把 arrowhead 混进来了 ——
    # 同一个词在 crucible.affixes 侧也错过一次（那里 Arrow 是法术形状，已按 adjective 改为箭形）。
    gloss.pop('Arrow', None)

    batch, stat, todo, conflicts, roleblind = {}, Counter(), [], [], []
    for pack, path, e, c in load_pairs('1-Ember汉化插件'):
        if pack != PACK or not path.endswith('.name'):
            continue
        if c and CJK.search(c):
            stat['already'] += 1
            continue
        key = e.strip()
        skey = (shape_of(path), key)
        # glossary / dnd5e 两源给的是**裸中文词条**，接不接英文尾巴由槽位角色决定：
        # regions / results 这类槽是裸中文约定，接上英文就是错的。
        def render(zh):
            return f'{zh} {e}'.strip() if wants_bilingual(path) else zh
        if skey in lib:
            batch[path] = lib[skey][0]
            stat['from_library'] += 1
        elif key in gloss:
            batch[path] = render(gloss[key])
            stat['from_glossary'] += 1
            if key in core and core[key] != gloss[key]:
                conflicts.append({'en': e, 'glossary': gloss[key], 'dnd5e': core[key]})
        elif key in core:
            batch[path] = render(core[key])
            stat['from_dnd5e'] += 1
        else:
            stat['needs_translator'] += 1
            todo.append({'path': path, 'en': e})
        # 结构键落空、但无角色的纯英文键有候选 —— 记下来，**不**据此写值。
        if skey not in lib and key in blind:
            stat['roleblind_withheld'] += 1
            if len(roleblind) < 500:
                roleblind.append({'path': path, 'shape': skey[0], 'en': key[:120],
                                  'no_role_majority': blind[key].most_common(1)[0][0][:120]})

    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, f'twinnames.{PACK}')
    json.dump(batch, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'stat': dict(stat), 'glossary_vs_dnd5e': conflicts,
               'needs_translator': todo, 'roleblind_withheld': roleblind},
              open(a.report, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f'库内既有 .name TM {len(lib)} 条（按 (结构形状, 英文) 键；'
          f'无角色的纯英文键只有 {len(blind)} 条，仅供报告参考）'
          f' | dnd5e 核心包 TM {len(core)} 条')
    print(f'\n{PACK} 的 .name：')
    print(f'  已有中文        {stat["already"]:>5}')
    print(f'  ① 库内既有译名   {stat["from_library"]:>5}')
    print(f'  ② glossary_ec    {stat["from_glossary"]:>5}   （其中与 dnd5e 模块不同的 {len(conflicts)} 条已记进报告）')
    print(f'  ③ dnd5e 核心包   {stat["from_dnd5e"]:>5}')
    print(f'  ④ 仍需译者       {stat["needs_translator"]:>5}')
    print(f'  （结构键落空、仅无角色键有候选，已拦下 {stat["roleblind_withheld"]:>5} 条'
          f' ← 见报告 roleblind_withheld）')
    print(f'\nbatch  -> {fp}')
    print(f'report -> {a.report}')


if __name__ == '__main__':
    main()
