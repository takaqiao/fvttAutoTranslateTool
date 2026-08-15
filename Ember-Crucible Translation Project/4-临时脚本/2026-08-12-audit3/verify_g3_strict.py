#!/usr/bin/env python3
"""严格子集：EN 侧该 @UUID 无标签 或 标签==目标英文名，但中文标签与目标中文名毫无重合。
这正是 G3 的判据（中文标签写着目标的旧名），用独立实现复核假阴性。"""
import json, sys, io, os, re
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


def collect_cn_names(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, dict):
                if isinstance(v.get('name'), str):
                    out.setdefault(k, v['name'])
                collect_cn_names(v, out)
            elif isinstance(v, list):
                for x in v:
                    collect_cn_names(x, out)
    return out


repo = os.path.join(P, '1-Ember汉化插件')
total = 0
for pack in ['ember.adventure.json', 'ember.crucible-adventure.json']:
    cn = json.load(open(os.path.join(repo, 'compendium/cn', pack), encoding='utf-8'))
    en = json.load(open(os.path.join(repo, 'compendium/en', pack), encoding='utf-8'))
    cn_names = collect_cn_names(cn, {})
    en_leaves = dict(leaves(en))
    out = []
    for path, s in leaves(cn):
        for m in UUID_RE.finditer(s):
            target, label = m.group(1), m.group(2)
            info = IDS.get(target.split('.')[-1])
            if not info:
                continue
            tn = info['name']
            tc = cn_names.get(tn)
            if not tc:
                continue
            lab = re.sub(r'[^\u4e00-\u9fff]', '', label)
            nm = re.sub(r'[^\u4e00-\u9fff]', '', tc)
            if not lab or lab in nm or nm in lab:
                continue
            en_s = en_leaves.get(path, '')
            el = None
            for em in UUID_RE.finditer(en_s):
                if em.group(1) == target:
                    el = em.group(2)
                    break
            if el is None or el == tn:
                out.append((path, target, tn, tc, label, el))
    total += len(out)
    print('#####', pack, '严格子集', len(out))
    for o in out:
        print('  %s' % o[0][-100:])
        print('     %s  EN名=%r  CN名=%r' % (o[1], o[2], o[3]))
        print('     CN标签=%r   EN标签=%r' % (o[4], o[5]))
print('TOTAL', total)
