# -*- coding: utf-8 -*-
"""镜头「人物设定跨页一致性」探针之一：性别代词闸。

判据：对每个已译的叶字符串，剥标记后
  EN 侧：female = she/her/hers/herself；male = he/him/his/himself；
        （用词边界，且 his/her 的所有格也算）
  CN 侧：female = 她；male = 他（**先剥掉 其他/他人/吉他/他日/他处/他乡/利他 等非代词用法**）
若 EN 侧是「纯女性」而 CN 侧只出现「他」不出现「她」 → 强信号；反之亦然。
若 EN 纯一性别而 CN 两性都有 → 弱信号（同段可能提到别人）。

已知假阳性：
  * 同一叶里出现多个人物（英文只用了一种代词，中文点名另一人时带出另一代词）→ MIXED 档
  * 英文 they/them 单数（本作大量非二元/龙类角色）→ 不参与判定，单独统计
  * 中文用「其」「该角色」「祂」等替代 → 只会漏报不会误报
"""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TAG = re.compile(r'<[^>]+>')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]|&(?:amp;)?[Rr]eference\[[^\]]*\]')
UUIDLABEL = re.compile(r'@UUID\[[^\]]*\]\{([^}]*)\}')
CJK = re.compile(r'[\u4e00-\u9fff]')

FEM = re.compile(r'\b(she|her|hers|herself)\b', re.I)
MASC = re.compile(r'\b(he|him|his|himself)\b', re.I)
THEY = re.compile(r'\b(they|them|their|theirs|themselves|themself)\b', re.I)

# 「他」的非代词用法：必须先剥，否则「其他」把全库淹掉
TA_NONPRON = re.compile(r'其他|他人|他日|他处|他乡|他方|吉他|利他|他律|他杀|他者|排他|他山之石')


def plain(s):
    s = UUIDLABEL.sub(lambda m: ' ' + m.group(1) + ' ', s)
    s = MARKUP.sub(' ', s)
    s = TAG.sub(' ', s)
    return s


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out')
    ap.add_argument('--min-en', type=int, default=0)
    a = ap.parse_args()

    en_dir = os.path.join(a.repo, 'compendium', 'en')
    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    packs = sorted(f for f in os.listdir(en_dir)
                   if f.endswith('.json') and not f.startswith('_')
                   and os.path.exists(os.path.join(cn_dir, f)))
    rows = []
    stats = {'checked': 0, 'en_gendered': 0, 'they_only': 0}
    for pack in packs:
        o = []
        walk(json.load(open(os.path.join(en_dir, pack), encoding='utf-8')).get('entries', {}),
             json.load(open(os.path.join(cn_dir, pack), encoding='utf-8')).get('entries', {}),
             [], o)
        for path, e, c in o:
            if not (c and CJK.search(c)):
                continue
            pe, pc = plain(e), plain(c)
            if len(pe) < a.min_en:
                continue
            stats['checked'] += 1
            nf, nm, nt = len(FEM.findall(pe)), len(MASC.findall(pe)), len(THEY.findall(pe))
            if not (nf or nm):
                if nt:
                    stats['they_only'] += 1
                continue
            stats['en_gendered'] += 1
            pc2 = TA_NONPRON.sub('  ', pc)
            cf, cm = pc2.count('她'), pc2.count('他')
            verdict = None
            if nf and not nm:            # 英文纯女性
                if cm and not cf:
                    verdict = 'HARD_F2M'
                elif cm and cf:
                    verdict = 'MIXED_F'
            elif nm and not nf:          # 英文纯男性
                if cf and not cm:
                    verdict = 'HARD_M2F'
                elif cf and cm:
                    verdict = 'MIXED_M'
            if verdict:
                rows.append({'pack': pack, 'path': path, 'verdict': verdict,
                             'en_f': nf, 'en_m': nm, 'en_they': nt,
                             'cn_f': cf, 'cn_m': cm,
                             'en': pe[:1400], 'cn': pc[:1400]})
    order = {'HARD_F2M': 0, 'HARD_M2F': 1, 'MIXED_F': 2, 'MIXED_M': 3}
    rows.sort(key=lambda r: (order[r['verdict']], r['pack'], r['path']))
    from collections import Counter
    print(json.dumps(stats, ensure_ascii=False))
    print(Counter(r['verdict'] for r in rows))
    for r in rows[:20]:
        print(f"[{r['verdict']}] {r['pack']} :: {r['path'][-90:]}  en(f{r['en_f']}/m{r['en_m']}/t{r['en_they']}) cn(f{r['cn_f']}/m{r['cn_m']})")
    if a.out:
        json.dump({'stats': stats, 'rows': rows}, open(a.out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print('->', a.out)


if __name__ == '__main__':
    main()
