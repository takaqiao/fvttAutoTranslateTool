# -*- coding: utf-8 -*-
import os, subprocess, sys, json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJ = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
TMP = os.path.join(os.environ['TEMP'], 'alien-dnt-fixture')
QA = os.path.join(PROJ, '4-常用脚本', 'qa')
REG = os.path.join(PROJ, '7-其他内容', 'DO-NOT-TRANSLATE.json')

REPOS = ['1-系统汉化插件', '2-新手包汉化插件', '3-核心书汉化插件']


def run(script, root, out):
    args = [sys.executable, os.path.join(QA, script), '--register', REG, '--out', out,
            '--lang', os.path.join(root, '1-系统汉化插件', 'lang', 'cn.json'),
            '--lang-en', os.path.join(root, '1-系统汉化插件', 'lang', 'en.json')]
    for r in REPOS:
        args += ['--repo', os.path.join(root, r)]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', env=env)
    return p.returncode, p.stdout, p.stderr


others = [d for d in os.listdir(TMP) if d != 'clean' and os.path.isdir(os.path.join(TMP, d))]
cases = ['clean'] + sorted(others, key=lambda s: (s[0], int(s[1:s.index('-')])))
rows = []
for case in cases:
    root = os.path.join(TMP, case)
    for script, tag in (('scan_name_lookup_traps.py', 'NAME'), ('scan_crit_lockstep.py', 'CRIT')):
        out = os.path.join(TMP, '%s.%s.json' % (case, tag))
        rc, so, se = run(script, root, out)
        res = json.load(open(out, encoding='utf-8')) if os.path.exists(out) else {}
        verdicts = {}
        for f in res.get('findings', []):
            verdicts[f['verdict']] = verdicts.get(f['verdict'], 0) + 1
        rows.append((case, tag, rc, res.get('checked'), len(res.get('findings', [])), verdicts))
        if se.strip():
            print('STDERR', case, tag, se[:2000])

print('%-28s %-5s %-4s %-8s %-6s %s' % ('fixture', 'gate', 'exit', 'checked', 'viol', 'verdicts'))
for r in rows:
    print('%-28s %-5s %-4d %-8s %-6d %s' % r)
