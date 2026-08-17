# -*- coding: utf-8 -*-
"""建立「库」的双向索引：EN -> {CN...}  和  CN核心 -> {EN...}

库 = 两个插件仓的 compendium/en + compendium/cn（结构镜像，逐叶配对）
     + lang/en.json + lang/cn.json
     + 5-其他内容/glossary/glossary_ec.json

项目约定：条目名是「中文 English」双语并列。做撞车检测时要拿**中文核心**
（把并列的英文尾巴剥掉）去比，否则永远撞不上。
"""
import json, os, re, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(ROOT, r"4-临时脚本\2026-08-18-ember-061\terms")

NAME_KEYS = {'name', 'label', 'tokenName', 'title'}
CJK = re.compile(r'[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]')


def cn_core(s):
    """剥掉并列英文尾巴，取中文核心。'深渊先驱 Abyssal Harbinger' -> '深渊先驱'"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not CJK.search(s):
        return None
    # 从右往左砍掉纯 ASCII 的尾巴词
    toks = s.split(' ')
    while len(toks) > 1 and not CJK.search(toks[-1]):
        toks.pop()
    core = ' '.join(toks).strip()
    return core or None


en2cn = collections.defaultdict(collections.Counter)   # EN -> Counter(CN原文)
cn2en = collections.defaultdict(collections.Counter)   # CN核心 -> Counter(EN)
prov = collections.defaultdict(set)                    # (EN,CN) -> 出处


def record(en, cn, where):
    if not isinstance(en, str) or not isinstance(cn, str):
        return
    en = en.strip()
    cn = cn.strip()
    if not en or not cn:
        return
    en2cn[en][cn] += 1
    c = cn_core(cn)
    if c:
        cn2en[c][en] += 1
    prov[(en, cn)].add(where)


def walk(e, c, path, where):
    """并行走 en/cn 两棵同构树。字典 key 本身就是英文原名（Babele 约定）。"""
    if isinstance(e, dict) and isinstance(c, dict):
        for k, ev in e.items():
            if k not in c:
                continue
            cv = c[k]
            # folders / entries / pages ... 这类：key=英文名，value 若是字符串就是译名
            if isinstance(cv, str) and not isinstance(ev, (dict, list)):
                if k in NAME_KEYS:
                    # 值对值：ev 是英文名，cv 是中文名
                    record(ev, cv, where + path)
                else:
                    # key 对值：k 是英文名，cv 是中文名（folders 表就是这形状）
                    record(k, cv, where + path)
            else:
                walk(ev, cv, path + '/' + k, where)
    elif isinstance(e, list) and isinstance(c, list) and len(e) == len(c):
        for i, (ev, cv) in enumerate(zip(e, c)):
            walk(ev, cv, path + f'/[{i}]', where)


packs = 0
for repo in ['1-Ember汉化插件', '2-Crucible汉化插件']:
    endir = os.path.join(ROOT, repo, 'compendium', 'en')
    cndir = os.path.join(ROOT, repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(endir)):
        if not fn.endswith('.json') or fn.startswith('_'):
            continue
        cp = os.path.join(cndir, fn)
        if not os.path.exists(cp):
            continue
        e = json.load(open(os.path.join(endir, fn), encoding='utf-8'))
        c = json.load(open(cp, encoding='utf-8'))
        walk(e, c, '', repo + '/' + fn)
        packs += 1
    # lang
    le = os.path.join(ROOT, repo, 'lang', 'en.json')
    lc = os.path.join(ROOT, repo, 'lang', 'cn.json')
    if os.path.exists(le) and os.path.exists(lc):
        E = json.load(open(le, encoding='utf-8'))
        C = json.load(open(lc, encoding='utf-8'))
        for k, v in E.items():
            if k in C:
                record(v if isinstance(v, str) else str(v), C[k], repo + '/lang/cn.json')

# glossary_ec
gl = json.load(open(os.path.join(ROOT, r"5-其他内容\glossary\glossary_ec.json"), encoding='utf-8'))
for k, v in gl.items():
    record(k, v, 'glossary_ec.json')

print(f"扫了 {packs} 个 compendium 包对")
print(f"EN 键 {len(en2cn)} 个 · CN核心 键 {len(cn2en)} 个")

json.dump({k: dict(v) for k, v in en2cn.items()},
          open(os.path.join(OUT, 'lib_en2cn.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=0)
json.dump({k: dict(v) for k, v in cn2en.items()},
          open(os.path.join(OUT, 'lib_cn2en.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=0)
json.dump({f"{a}\t{b}": sorted(v) for (a, b), v in prov.items()},
          open(os.path.join(OUT, 'lib_prov.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=0)

# --- 索引自证：库里已知的几条必须查得到，且方向都对 ---
CHECKS = [
    ('Aburyx', '阿布里克斯'),
    ('Abyssal Harbinger', '深渊先驱'),
    ('Redrak Fields', None),
    ('Golden Flats', None),
    ('Arctus Plateau', None),
    ('Overrun', None),
]
for en, expect in CHECKS:
    got = dict(en2cn.get(en, {}))
    print(f"  EN[{en}] -> {got}")
    if expect:
        ok = any(cn_core(x) == expect for x in got)
        print(f"     期望中文核心 {expect} ->", "PASS" if ok else "FAIL")
print("  CN核心[冲撞] ->", dict(cn2en.get('冲撞', {})))
