"""Find HTML attribute NAMES that were themselves translated into Chinese.
Such an attribute is dead: the browser ignores it, so whatever it controlled is lost."""
import json, os, re, sys

sys.path.insert(0, r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\3-常用脚本\qa")
import scan_attr_text as S

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
# attribute name allowed to contain CJK/any non-ascii
BAD = re.compile(r'([^\s<>=/"\']*[\u4e00-\u9fff][^\s<>=/"\']*)\s*=\s*"([^"]*)"')

n = 0
for repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(repo, 'compendium', side)
        if not os.path.isdir(d):
            continue
        for f in S.packs_of(d):
            o = []
            S.walk(S.load_entries(os.path.join(d, f)), [], o)
            for path, s in o:
                for m in S.TAG.finditer(s):
                    for bm in BAD.finditer(m.group(1)):
                        n += 1
                        print(f'[{os.path.basename(repo)}/{side}/{f}] {path}')
                        print(f'    {bm.group(0)}   in   {m.group(0)[:160]}')
print(f'\nCJK-named attributes: {n}')
