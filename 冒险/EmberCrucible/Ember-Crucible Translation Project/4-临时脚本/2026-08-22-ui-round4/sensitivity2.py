# -*- coding: utf-8 -*-
"""灵敏度回测（第二版）。

第一版抓到两处真漏（require 只钉名字 ⇒ 删调用不响；实现细节压根没进 require ⇒ 换回异步不响），
判据已拆成两条并加强。本版另修了测例自身的一个毛病：④⑤ 当初只改**孪生的一份**，
于是 `R-selfcheck-twin` 跟着响 —— 那是测例的错，不是判据顺带咬。现在两份一起改。
"""
import io, os, re, shutil, subprocess, sys, tempfile
sys.stdout.reconfigure(encoding='utf-8')
SRC = os.path.abspath('.')
GATE = ['python3', os.path.join('3-常用脚本', 'qa', 'assert_resolutions.py')]
TWINS = ['1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs',
         '2-Crucible汉化插件/selfcheck/cn-selfcheck.mjs']

def run(root):
    p = subprocess.run(GATE + ['--root', root], cwd=SRC, capture_output=True, text=True, encoding='utf-8')
    out = (p.stdout or '') + (p.stderr or '')
    return p.returncode, sorted(set(re.findall(r'FAIL\s+(R-[\w-]+)', out))), \
        next((l for l in out.splitlines() if l.startswith('通过 ')), '(没抓到统计行)')

def copytree(dst):
    shutil.copytree(SRC, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('4-临时脚本', '__pycache__', '.git', 'module.zip', 'release'))

def edit(root, rels, fn):
    changed = False
    for rel in rels:
        p = os.path.join(root, *rel.split('/'))
        t = io.open(p, encoding='utf-8', newline='').read()
        t2 = fn(t)
        if t2 != t:
            changed = True
            io.open(p, 'w', encoding='utf-8', newline='').write(t2)
    return changed

BR = ['2-Crucible汉化插件/babele-register.js']
LR = ['2-Crucible汉化插件/lang-reclaim.js']
CASES = [
 ("① 删掉 `registerLangReclaim();` 整行（文件还在包里，就是没人调）",
  BR, lambda t: re.sub(r'^registerLangReclaim\(\);.*$', '', t, flags=re.M), 'R-lang-reclaim-wired'),
 ("② 把调用注释掉（半吊子回退：名字还在文件里）",
  BR, lambda t: re.sub(r'^(registerLangReclaim\(\);)', r'// \1', t, flags=re.M), 'R-lang-reclaim-wired'),
 ("③ 删掉 import 行",
  BR, lambda t: t.replace("import { registerLangReclaim } from './lang-reclaim.js';", ''), 'R-lang-reclaim-wired'),
 ("④ 同步 XHR 改成异步（实测必然赶不上 i18nInit，且赶不上的样子是英文、不报错）",
  LR, lambda t: t.replace("xhr.open('GET', url, false);", "xhr.open('GET', url, true);"), 'R-lang-reclaim-mechanism'),
 ("⑤ 去掉 enumerable: false（会让 #hotReloadJSON 的 mergeObject 抛 TypeError）",
  LR, lambda t: t.replace('      enumerable: false,', '      enumerable: true,'), 'R-lang-reclaim-mechanism'),
 ("⑥ 钩子从 i18nInit 挪到 ready（太晚：那 42 条早被查过了）",
  LR, lambda t: t.replace("Hooks.once('i18nInit', () => {", "Hooks.once('ready', () => {"), 'R-lang-reclaim-mechanism'),
 ("⑦ 摘掉账本出口（面板会退化成「装了但没挂账本」）",
  LR, lambda t: t.replace('pkg.api.getReclaimState = getReclaimState;', ''), 'R-lang-reclaim-mechanism'),
 ("⑧ 把面板那一节从 checkI18n 里摘掉（函数还在、没人调 ⇒ 面板安静地少一节）·两份一起改",
  TWINS, lambda t: t.replace('  out.push(...checkLangSquat(S, table));\r\n', '').replace('  out.push(...checkLangSquat(S, table));\n', ''), 'R-lang-squat-panel'),
 ("⑨ 改掉一条探针键 ·两份一起改",
  TWINS, lambda t: t.replace('"TOKEN.MOVEMENT.ACTIONS.walk.label"', '"TOKEN.MOVEMENT.ACTIONS.walk.LABEL"'), 'R-lang-squat-panel'),
]

base = tempfile.mkdtemp(prefix='sens2-base-'); copytree(base)
rc0, reds0, stat0 = run(base)
print(f'基线副本：rc={rc0}  {stat0}  报红={reds0 or "无"}')
BASE = set(reds0)
if BASE - {'R-assertion-inputs-tracked'}:
    print('❌ 基线除了 tracked 那条（副本没 .git，必然红）还有别的红'); sys.exit(2)
ok = 0
for title, rels, fn, expect in CASES:
    d = tempfile.mkdtemp(prefix='sens2-'); copytree(d)
    changed = edit(d, rels, fn)
    rc, reds, stat = run(d)
    added = sorted(set(reds) - BASE)
    good = changed and added == [expect]
    ok += good
    print(f'  {"PASS" if good else "FAIL"}  {title}')
    print(f'        注入生效={changed}  rc={rc}  {stat}  新增报红={added or "无"}')
    shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(base, ignore_errors=True)
print(f'\n灵敏度回测：{ok}/{len(CASES)} 通过')
sys.exit(0 if ok == len(CASES) else 1)
