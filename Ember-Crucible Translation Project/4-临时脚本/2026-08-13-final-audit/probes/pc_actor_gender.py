# -*- coding: utf-8 -*-
"""人物一致性镜头 A：**同一 actor 内部**的性别代词自洽。

做法
----
1. 按 `entries.<adventure>.actors.<ActorName>.…` 把叶子归到 actor 名下。
2. 用该 actor **英文侧全部叶子**里的 he/him/his/himself 与 she/her/hers/herself 计数，
   判定英文认定的性别（差距 >=2 且比例 >=3:1 才算「确定」，否则 UNKNOWN 不参与）。
3. 逐叶比对中文侧 他/她（先剥「其他/他人/…」等非代词用法）：
   * CN 出现与英文性别相反的代词 -> 报
   * 同一 actor 的中文侧同时出现 他 与 她 -> 报（内部分裂）

假阳性来源
----------
* actor 的 biography 里提到别的人物（配偶/上级/敌人）时会带出另一性别代词。
  -> 报告里给出 EN/CN 原文供人判读；并给出「该叶英文侧代词计数」。
* 复数「他们」已在剥词表里保留（他们 是合法的中文复数），会算作「他」。
  为降噪，单独统计 `他们/她们` 并允许 --no-plural 排除。
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all, plain, CJK

FEM = re.compile(r'\b(she|her|hers|herself)\b', re.I)
MASC = re.compile(r'\b(he|him|his|himself)\b', re.I)
THEY = re.compile(r'\b(they|them|their|theirs|themselves|themself)\b', re.I)
TA_NON = re.compile(r'其他|他人|他日|他处|他乡|他方|吉他|利他|他律|他杀|他者|排他|他山之石')
ACTOR = re.compile(r'^[^.]+\.actors\.([^.]+)(?:\.(.*))?$')


def cn_counts(s, drop_plural=True):
    s = TA_NON.sub('  ', s)
    if drop_plural:
        s = s.replace('他们', '  ').replace('她们', '  ')
    return s.count('她'), s.count('他')


def main():
    drop_plural = '--keep-plural' not in sys.argv
    rows = load_all()
    per = defaultdict(list)          # (repo, pack, actor) -> [(sub, en, cn)]
    for repo, pack, path, en, cn in rows:
        m = ACTOR.match(path)
        if m:
            per[(repo, pack, m.group(1))].append((m.group(2) or '', en, cn))

    findings = []
    stat = Counter()
    for key, leaves in sorted(per.items()):
        repo, pack, actor = key
        enf = enm = 0
        for _, en, _ in leaves:
            pe = plain(en)
            enf += len(FEM.findall(pe)); enm += len(MASC.findall(pe))
        if enf + enm == 0:
            stat['actor_no_gender'] += 1
            continue
        if enf >= max(2, 3 * enm):
            g = 'F'
        elif enm >= max(2, 3 * enf):
            g = 'M'
        else:
            stat['actor_ambiguous'] += 1
            continue
        stat['actor_' + g] += 1
        # 逐叶查中文
        tot_f = tot_m = 0
        bad = []
        for sub, en, cn in leaves:
            if not (cn and CJK.search(cn)):
                continue
            pe, pc = plain(en), plain(cn)
            cf, cm = cn_counts(pc, drop_plural)
            tot_f += cf; tot_m += cm
            wrong = (g == 'F' and cm and not cf) or (g == 'M' and cf and not cm)
            if wrong:
                bad.append({'sub': sub, 'en_f': len(FEM.findall(pe)),
                            'en_m': len(MASC.findall(pe)),
                            'en_they': len(THEY.findall(pe)),
                            'cn_f': cf, 'cn_m': cm,
                            'en': pe[:600], 'cn': pc[:600]})
        if bad or (tot_f and tot_m):
            findings.append({'repo': repo, 'pack': pack, 'actor': actor,
                             'en_gender': g, 'en_f': enf, 'en_m': enm,
                             'cn_f_total': tot_f, 'cn_m_total': tot_m,
                             'bad': bad})
    findings.sort(key=lambda r: -len(r['bad']))
    print(json.dumps(stat, ensure_ascii=False))
    print('actors flagged:', len(findings))
    for f in findings:
        print(f"\n### {f['repo']}/{f['pack']} :: {f['actor']}  EN={f['en_gender']}(f{f['en_f']}/m{f['en_m']}) CNtot(f{f['cn_f_total']}/m{f['cn_m_total']}) bad={len(f['bad'])}")
        for b in f['bad'][:6]:
            print(f"   - {b['sub'][:70]}  en(f{b['en_f']}/m{b['en_m']}/t{b['en_they']}) cn(f{b['cn_f']}/m{b['cn_m']})")
            print(f"     EN: {b['en'][:220]}")
            print(f"     CN: {b['cn'][:220]}")
    json.dump(findings, open('pc_actor_gender.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
