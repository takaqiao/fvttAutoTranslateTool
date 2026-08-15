#!/usr/bin/env python3
"""独立复核 G3 的假阴性：中文 @UUID{标签} 与目标文档的中文 name 不匹配的，全部列出。
判据独立于 G3 的工具：不看旧基准，直接用 ember_ids.json 解析目标现名 + 包内 CN name。
"""
import json, sys, io, os, re, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
IDS = json.load(open(os.path.join(P, '4-临时脚本/2026-08-12-fix/reports/ember_ids.json'), encoding='utf-8'))

UUID_RE = re.compile(r'@UUID\[([^\]]+)\]\{([^}]*)\}')


def leaves(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + '.' + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, p + '[%d]' % i)
    elif isinstance(o, str):
        yield p.lstrip('.'), o


def collect_cn_names(cn_root):
    """英文名 -> 中文 name（用包内任意层级的 name 叶）"""
    out = {}

    def rec(en_node, cn_node):
        if not isinstance(cn_node, dict):
            return
        for k, v in cn_node.items():
            if k == 'name' and isinstance(v, str):
                continue
            if isinstance(v, dict):
                if 'name' in v and isinstance(v['name'], str):
                    out[k] = v['name']
                rec(None, v)
    rec(None, cn_root)
    return out


def main():
    repo = os.path.join(P, '1-Ember汉化插件')
    for pack in ['ember.adventure.json', 'ember.crucible-adventure.json']:
        cn = json.load(open(os.path.join(repo, 'compendium/cn', pack), encoding='utf-8'))
        en = json.load(open(os.path.join(repo, 'compendium/en', pack), encoding='utf-8'))
        cn_names = collect_cn_names(cn)
        en_leaves = dict(leaves(en))
        bad = []
        for path, s in leaves(cn):
            for m in UUID_RE.finditer(s):
                target, label = m.group(1), m.group(2)
                tid = target.split('.')[-1]
                info = IDS.get(tid)
                if not info:
                    continue
                tname_en = info['name']
                tname_cn = cn_names.get(tname_en)
                if not tname_cn:
                    continue
                # 标签的中文部分是否出现在目标中文名里
                lab_cn = re.sub(r'[^\u4e00-\u9fff]', '', label)
                nm_cn = re.sub(r'[^\u4e00-\u9fff]', '', tname_cn)
                if not lab_cn:
                    continue
                if lab_cn in nm_cn or nm_cn in lab_cn:
                    continue
                # 英文侧同位置的标签
                en_s = en_leaves.get(path, '')
                en_lab = None
                for em in UUID_RE.finditer(en_s):
                    if em.group(1) == target:
                        en_lab = em.group(2)
                        break
                bad.append((pack, path, target, tname_en, tname_cn, label, en_lab))
        print('=' * 100)
        print(pack, '不匹配', len(bad), '处')
        for b in bad:
            print('  path=%s' % b[1])
            print('     target=%s  EN名=%r  CN名=%r' % (b[2], b[3], b[4]))
            print('     CN标签=%r   EN标签=%r' % (b[5], b[6]))


if __name__ == '__main__':
    main()
