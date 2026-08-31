#!/usr/bin/env python3
"""E1 单元的批次生成器：以 compendium/cn 的现值为底，做**逐处定点替换**。

为什么不手写整叶：这些叶子里最长的有 2.4 万字符，手抄一遍必然引入不可见的字节差异，
而 `apply_translations.py` 是整叶覆盖 —— 抄错一个空格就是永久损失。
定点替换保证「没打算动的字节一个都没动」。

每条规则都要求命中次数与预期相等，不等就报错退出（防止上游变动后规则静默失配）。

场景 Level 名那一类不写死，而是**从配对英文里把 `"X" Level` 的引号内容抽出来**再回填 ——
判据：`extract/mappings.mjs` 的 Scene 只映射 name/drawings/notes/regions，**没有 levels**，
英文基准与中文包里都不存在任何一个 Level 名，所以 babele 翻不到它，Foundry 的层级下拉框
显示的永远是英文。中文正文把层级名译过去，GM 就找不到那一层。
"""
from __future__ import annotations
import json, os, re, sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project'
REPO = os.path.join(ROOT, '1-Ember汉化插件')
OUT = r'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches'
J = 'Ember Early Access.journals.'
PACKS = ['ember.crucible-adventure.json', 'ember.adventure.json']


def load(path):
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


# ------------------------------------------------------------ 定点规则
# want=None 表示「有几处改几处，可以是 0 处」（用于孪生包，两包措辞不保证一致）
SHARED = [
    # ① 祝福密室：本页 name 就是「祝福密室 Blessing Chamber」，正文却自称赐福密室
    (J + 'Lantern Roads.pages.Blessing Chamber.text', '搜查赐福密室', '搜查祝福密室', 1),

    # ② Token → 指示物（lang 的 EMBER.ACTOR_FLAGS.SHEET.TOKEN＝指示物；Token Reveal 库内 5:2）
    (J + 'Lantern Roads.pages.Impromptu Jail.text', '<h4>令牌揭示</h4>', '<h4>指示物显现</h4>', 1),
    (J + 'Lantern Roads.pages.Impromptu Jail.text',
     '令牌最初处于隐藏状态；现在把它们的令牌揭示出来。',
     '指示物最初处于隐藏状态；现在把它们的指示物显现出来。', 1),

    # ③ Rune-Marked 家族：只改「符文-已标记」这两处机翻残迹，以及库内少数派的绶带/小瓶。
    #    Band 保持「符文纹环带」——全库 13:0，是它自己的多数写法，改动面更小的是别人。
    ('Ember Early Access.items.Rune-Marked Arrowhead.name', '符文-已标记箭头', '符文刻印箭头', 1),
    ('Ember Early Access.items.Rune-Marked Vial.actions.runemarkedVial.name',
     '符文-已标记变形术', '符文刻印变形术', None),
    ('Ember Early Access.items.Rune-Marked Sash.name', '符文标记绶带', '符文刻印腰带', 1),
    ('Ember Early Access.items.Rune-Marked Vial.name', '符文标记小瓶', '符文刻印小瓶', 1),
]

CRUCIBLE_ONLY = [
    # ④ Marlstone 街区页 name 是「马尔石 Marlstone」，Gala 页把同一目标标成马尔斯通
    (J + 'Disgraced House.pages.The Marlstone Gala.text',
     'uVodk7tLiG3pZMTq]{马尔斯通}', 'uVodk7tLiG3pZMTq]{马尔石}', 1),
]

HEPHISS = ('赫菲斯', '赫菲丝')   # actor name 是「赫菲丝·万德伦 Hephiss Wandren」

# Level 名：中文引号里若已是纯 ASCII 就说明本来就没译，跳过
CN_LEVEL = re.compile(r'“([^”]{1,60})”')
EN_LEVEL_CTX = re.compile(r'"([^"]{1,60})"')


def level_fix(en, cn):
    """把中文里被译过的 Level 名换回英文；返回 (新中文, [(旧,新)…]) 或 None。"""
    if 'Level' not in en or '层级' not in cn:
        return None
    # 只取「和 Level 这个词同处一句」的引号内容，避免抓到别的引用
    en_names = []
    for sent in re.split(r'(?<=[.!?])\s+', re.sub(r'<[^>]+>', ' ', en)):
        if 'Level' not in sent:
            continue
        en_names += EN_LEVEL_CTX.findall(sent)
    # 中文侧按**出现位置**取，不用 replace —— 同一个层级名在一叶里会出现两次
    # （「本事件使用 X 与 Y 层级」＋「此时激活 Y 层级」），全局 replace 会歧义中止。
    spans = []
    pos = 0
    for sent in re.split(r'(?<=[。！？])', cn):
        if '层级' in re.sub(r'<[^>]+>', ' ', sent):
            for m in CN_LEVEL.finditer(sent):
                spans.append((pos + m.start(), pos + m.end(), m.group(1)))
        pos += len(sent)
    if not en_names or len(en_names) != len(spans):
        return None
    subs = []
    out = cn
    for (a, b, c), e in sorted(zip(spans, en_names), key=lambda t: -t[0][0]):
        if c == e or c.isascii():
            continue
        out = out[:a] + '“' + e + '”' + out[b:]
        subs.append((c, e))
    return (out, subs) if subs else None


def build(pack, extra_rules):
    E, C = {}, {}
    walk(load(os.path.join(REPO, 'compendium', 'en', pack))['entries'], [], E)
    walk(load(os.path.join(REPO, 'compendium', 'cn', pack))['entries'], [], C)
    batch, errs = {}, []

    def edit(path, old, new, want):
        cur = batch.get(path, C.get(path))
        if cur is None:
            if want is None:
                return
            errs.append(f'路径不存在: {path}')
            return
        got = cur.count(old)
        if want is not None and got != want:
            errs.append(f'命中数不符 {got}!={want}: {path} :: {old[:40]!r}')
            return
        if got:
            batch[path] = cur.replace(old, new)

    for path, old, new, want in SHARED:
        edit(path, old, new, want if pack == PACKS[0] else None)
    for path, old, new, want in extra_rules:
        edit(path, old, new, want)

    # Hephiss 统一（英文闸：这些叶子的英文全部含 Hephiss）
    n = 0
    for path, val in sorted(C.items()):
        c = val.count(HEPHISS[0])
        if not c:
            continue
        if 'Hephiss' not in (E.get(path) or ''):
            errs.append(f'英文闸未过，跳过: {path}')
            continue
        edit(path, HEPHISS[0], HEPHISS[1], c)
        n += c
    print(f'  Hephiss 赫菲斯→赫菲丝：{n} 处')

    # 场景 Level 名回填英文
    lv = 0
    for path, cn in sorted(C.items()):
        en = E.get(path)
        if not en:
            continue
        r = level_fix(en, batch.get(path, cn))
        if r:
            batch[path], subs = r
            lv += len(subs)
            for a, b in subs:
                print(f'  Level  “{a}” → “{b}”   {path.split(".pages.")[-1][:44]}')
    print(f'  场景 Level 名回填：{lv} 处')

    if errs:
        print('!! 规则失配：')
        for e in errs:
            print('   ', e)
        sys.exit(1)

    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, f'E1__ember__{pack[:-5]}.json')
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(batch, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f'  写出 {len(batch)} 条 -> {fp}')


for p, extra in ((PACKS[0], CRUCIBLE_ONLY), (PACKS[1], [])):
    print('###', p)
    build(p, extra)
