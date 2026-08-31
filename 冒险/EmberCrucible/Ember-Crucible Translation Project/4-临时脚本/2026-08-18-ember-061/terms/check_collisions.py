# -*- coding: utf-8 -*-
"""双向撞车判据。

上一轮栽的那次（Overrun＝围攻）只查了中文方向，没查英文方向 ——
`Overrun` 早被 Juggernaut 天赋的「冲撞」占着。所以这里两个方向都跑：

  方向 A（中文侧）：候选中文（核心，剥掉并列英文尾）在库里是否已经是**别的英文**的既定译名？
  方向 B（英文侧）：候选英文在库里是否已经被译成**别的中文**？

⚠ 前置自证：先断言判据本身能抓到已知真值 ——
   拿 Overrun/围攻 这对做阳性对照（必须报撞车），
   拿一个必然不撞的伪造串做阴性对照（必须不报）。
"""
import json, os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
lib_en2cn = json.load(open(os.path.join(HERE, 'lib_en2cn.json'), encoding='utf-8'))
lib_cn2en = json.load(open(os.path.join(HERE, 'lib_cn2en.json'), encoding='utf-8'))

CJK = re.compile(r'[㐀-鿿]')


def core(s):
    t = s.split(' ')
    while len(t) > 1 and not CJK.search(t[-1]):
        t.pop()
    return ' '.join(t).strip()


def check(en, cn):
    """返回 (dirA, dirB)。dirA=中文侧撞车列表；dirB=英文侧既有译名"""
    c = core(cn)
    a = {k: v for k, v in lib_cn2en.get(c, {}).items() if k != en}
    b = {k: v for k, v in lib_en2cn.get(en, {}).items() if core(k) != c}
    return a, b


# ---------- 前置自证 ----------
posA, posB = check('Overrun', '围攻')       # 英文侧应当报出「冲撞 Overrun」
negA, negB = check('Zzqx Nonexistent', '兹兹克斯虚构词')
ok1 = bool(posB)                             # 阳性：英文方向必须抓到
posA2, _ = check('Juggernaut Ram', '冲撞')   # 中文侧：冲撞 已被 Ram/Overrun 占
ok2 = bool(posA2)
ok3 = not negA and not negB                  # 阴性对照必须干净
print("自证-阳性(英文方向 Overrun→已有译名):", "PASS" if ok1 else "FAIL", posB)
print("自证-阳性(中文方向 冲撞→已被占):", "PASS" if ok2 else "FAIL", posA2)
print("自证-阴性(伪造串不报):", "PASS" if ok3 else "FAIL")
if not (ok1 and ok2 and ok3):
    print("!! 判据自证不过，后面的结论不作数")
    sys.exit(1)
print("-" * 70)

rows = []
for fn in ['candidates.json', 'candidates_extra.json', 'candidates_lang_hardcoded.json']:
    d = json.load(open(os.path.join(HERE, fn), encoding='utf-8'))
    for k, v in d.items():
        if k.startswith('_'):
            continue
        en = re.sub(r'（.*?）$', '', k)          # 去掉我加的角色后缀，如 Maevren（token）
        cn, role, why = v
        rows.append({'en': en, 'key': k, 'cn': cn, 'role': role, 'why': why, 'file': fn})

print(f"候选条目 {len(rows)} 条")

collisions = []
for r in rows:
    a, b = check(r['en'], r['cn'])
    if a or b:
        collisions.append({
            'en': r['en'], 'cn': r['cn'], 'role': r['role'], 'from': r['file'],
            'cn_direction_hit': a,   # 这个中文已经是别的英文的译名
            'en_direction_hit': b,   # 这个英文已经有别的中文译名
        })

print(f"撞车 {len(collisions)} 条")
for c in collisions:
    print(f"\n### {c['en']}  →  {c['cn']}   [{c['role']}]")
    if c['en_direction_hit']:
        print(f"    [英文方向] 库里 {c['en']} 已译作: {c['en_direction_hit']}")
    if c['cn_direction_hit']:
        print(f"    [中文方向] 库里「{core(c['cn'])}」已用来译: {c['cn_direction_hit']}")

json.dump(collisions, open(os.path.join(HERE, 'collisions.raw.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
