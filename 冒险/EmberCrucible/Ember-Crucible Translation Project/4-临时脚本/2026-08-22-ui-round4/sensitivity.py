# -*- coding: utf-8 -*-
"""灵敏度回测：往副本树里逐个注入「这条判据本该咬住」的改动，确认它真的会红。

⚠ 只证「改了会红」还不够 —— 还得证**别的判据不会跟着一起红**（否则分不清是谁咬的），
   以及**基线副本本身是全绿的**（否则整组数都不作数）。所以每一格都记：
     基线 rc / 注入后 rc / 报红的断言 id 集合。
"""
import io, os, re, shutil, subprocess, sys, tempfile
sys.stdout.reconfigure(encoding='utf-8')

SRC = os.path.abspath('.')
GATE = ['python3', os.path.join('3-常用脚本', 'qa', 'assert_resolutions.py')]

def run(root):
    p = subprocess.run(GATE + ['--root', root], cwd=SRC, capture_output=True, text=True, encoding='utf-8')
    out = (p.stdout or '') + (p.stderr or '')
    reds = sorted(set(re.findall(r'FAIL\s+(R-[\w-]+)', out)))
    tail = [l for l in out.splitlines() if l.startswith('通过 ')]
    return p.returncode, reds, (tail[-1] if tail else '(没抓到统计行)')

def copytree(dst):
    shutil.copytree(SRC, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('4-临时脚本', '__pycache__', '.git', 'module.zip', 'release'))

CASES = [
  ("① 把 registerLangReclaim() 从 babele-register.js 里删掉（文件还在包里，但没人调 ⇒ 白进包）",
   '2-Crucible汉化插件/babele-register.js',
   lambda t: re.sub(r'^.*registerLangReclaim\(\);.*$', '', t, flags=re.M),
   'R-lang-reclaim-wired'),
  ("② 把调用注释掉（半吊子回退：字面量还在文件里，require 咬不住，靠 forbid_re）",
   '2-Crucible汉化插件/babele-register.js',
   lambda t: re.sub(r'^(\s*)(registerLangReclaim\(\);)', r'\1// \2', t, flags=re.M),
   'R-lang-reclaim-wired'),
  ("③ 把同步 XHR 换回 fetch 风格（实测必然赶不上 i18nInit，且赶不上的样子是英文、不报错）",
   '2-Crucible汉化插件/lang-reclaim.js',
   lambda t: t.replace("xhr.open('GET', url, false)", "xhr.open('GET', url, true)"),
   None),
  ("④ 把面板那一节从 checkI18n 里摘掉（函数还在、没人调 ⇒ 面板安静地少一节）",
   '1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs',
   lambda t: t.replace('  out.push(...checkLangSquat(S, table));\r\n', '').replace('  out.push(...checkLangSquat(S, table));\n', ''),
   'R-lang-squat-panel'),
  ("⑤ 去掉一条探针键（TOKEN.MOVEMENT.ACTIONS.walk.label）",
   '1-Ember汉化插件/scripts/ember-cn-selfcheck.mjs',
   lambda t: t.replace('"TOKEN.MOVEMENT.ACTIONS.walk.label"', '"TOKEN.MOVEMENT.ACTIONS.walk.LABEL"'),
   'R-lang-squat-panel'),
]

base = tempfile.mkdtemp(prefix='sens-base-')
copytree(base)
rc0, reds0, stat0 = run(base)
print(f'基线副本：rc={rc0}  {stat0}  报红={reds0 or "无"}')
# 副本没有 .git（拷进来太大），R-assertion-inputs-tracked 必然报红 ——
# 这是副本本身的局限，不是被测判据的事。于是改成比**增量**：
# 注入后新多出来的报红必须恰好是期待的那一条。
BASE = set(reds0)
if BASE - {'R-assertion-inputs-tracked'}:
    print('❌ 基线副本除了 tracked 那一条还有别的红，后面的数不作数'); sys.exit(2)

ok = 0
for title, rel, mut, expect in CASES:
    d = tempfile.mkdtemp(prefix='sens-')
    copytree(d)
    p = os.path.join(d, *rel.split('/'))
    t = io.open(p, encoding='utf-8', newline='').read()
    t2 = mut(t)
    changed = (t2 != t)
    io.open(p, 'w', encoding='utf-8', newline='').write(t2)
    rc, reds, stat = run(d)
    added = sorted(set(reds) - BASE)
    hit = (added == [expect]) if expect else (rc != 0)
    verdict = 'PASS' if (changed and hit) else 'FAIL'
    if verdict == 'PASS': ok += 1
    extra = [r for r in added if r != expect]
    print(f'  {verdict}  {title}')
    print(f'        注入生效={changed}  rc={rc}  {stat}')
    print(f'        新增报红={added or "无"}' + (f'   ⚠ 顺带咬到={extra}' if extra else ''))
    shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(base, ignore_errors=True)
print(f'\n灵敏度回测：{ok}/{len(CASES)} 通过')
sys.exit(0 if ok == len(CASES) else 1)
