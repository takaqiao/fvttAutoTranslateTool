#!/usr/bin/env python3
"""逐段回测全部 48 个 readaloud 槽：删空 / 砍一半 / 砍 51%（留 49%），各报抓获率。

现读真库（不是报告快照），每一段单独变异、单独跑 `a_enricher_text_coverage`。
反空转：先打印「摊到多少段」，摊到 0 段直接退出非零。
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(P, '3-常用脚本', 'qa'))
import assert_resolutions as AR   # noqa: E402

RULE = json.load(open(os.path.join(P, '5-其他内容', 'RESOLUTIONS.assertions.json'),
                      encoding='utf-8'))
RULE = next(a for a in RULE['assertions'] if a['id'] == 'R-readaloud-coverage')
# 单叶跑：把反空转护栏放宽，否则响的是护栏而不是被测的那件事
SOLO = dict(RULE, min_leaves=1, min_slots=1, min_gated=1,
            min_anchor_terms=0, min_anchor_slots=0, min_anchor_hits=0)
RA = re.compile(r'(readaloud=")([^"]*)(")')


def real_rows(rule, pairs):
    bad, detail = AR.a_enricher_text_coverage(rule, AR._FakeCtx(pairs=pairs))
    return [b for b in bad if b[0] != '-'], detail


def main():
    repos = {n: os.path.join(P, rel) for n, rel in AR.REPOS.items()
             if os.path.isdir(os.path.join(P, rel))}
    ctx = AR.Ctx(repos, {})
    targets = []          # (repo, pack, path, ev, cv, 第几个 readaloud)
    for repo, pack, path, ev, cv in ctx.all_pairs(None):
        k = len(RA.findall(cv))
        for i in range(k):
            targets.append((repo, pack, path, ev, cv, i))
    print(f'摊到 {len(targets)} 个中文 readaloud 槽（现读真库；0 就是空转）')
    if not targets:
        return 1

    base_bad, base_detail = real_rows(RULE, list(ctx.all_pairs(None)))
    print(f'基线（全库、原样）：违规 {len(base_bad)} 条\n  {base_detail}\n')

    def mutate(cv, idx, mode):
        out, k = [], [0]

        def repl(m):
            v = m.group(2)
            if k[0] != idx:
                k[0] += 1
                return m.group(0)
            k[0] += 1
            new = ('' if mode == 'wipe' else
                   v[:len(v) // 2] if mode == 'half' else
                   v[:int(len(v) * 0.49)])
            out.append((len(v), len(new)))
            return m.group(1) + new + m.group(3)
        return RA.sub(repl, cv), out

    for mode in ('wipe', 'half', 'p49'):
        fired = 0
        misses = []
        for repo, pack, path, ev, cv, i in targets:
            mcv, ch = mutate(cv, i, mode)
            assert ch, f'变异没生效：{path} #{i}'          # 探针空转的直接护栏
            bad, _ = real_rows(SOLO, [(repo, pack, path, ev, mcv)])
            if len(bad) > 0:
                fired += 1
            else:
                misses.append((repo, pack, path, i, ch[0]))
        print(f'[{mode}] 抓获 {fired} / {len(targets)}')
        for repo, pack, path, i, ch in misses:
            print(f'    跑掉：{repo}/{pack} {path[-58:]} #{i} （{ch[0]} 字 -> {ch[1]} 字）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
