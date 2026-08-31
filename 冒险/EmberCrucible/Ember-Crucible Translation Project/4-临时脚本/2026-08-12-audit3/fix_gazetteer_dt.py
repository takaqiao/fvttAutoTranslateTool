#!/usr/bin/env python3
"""地名志 `<dt>` 条目的两类缺陷：名字错位 + NPC 名整个没译。

来历：J05 单元报出 `Ordain Gazetteer / Scholar's Nook` 的 landmark `<dt>` 名**整体错位一格**
（阻断级：玩家按地名志找任何一家店都会走错），但 `Ordain Gazetteer` **不在那 35 本的清单里**，
超出该单元 scope，所以没人修。顺着查下去发现同类页还有第二个缺陷：

  **432 个带对齐标记的 NPC `<dt>` 条目里，有 70 处（22 叶、35 个唯一人名）中文名仍是英文原名。**

这一类现有闸门全盲：叶子里有大量中文，覆盖率算它已译；名字是专名，数字覆盖检查看不见；
标记与 class 都没动，标记五项也不响。

判据：`<dt>` 的英文形如 `Name (Alignment, Ancestry, pronouns)`，而中文的名字段仍以英文原名开头。


⚠ 运行前提：本脚本**只在库里真有错时才产批次**（2026-08-15 加的幂等护栏）
================================================================================
它被列在 `PROJECT.md §5.4「发版前必跑」`，会被下一个会话「照清单全跑一遍」，
所以**必须幂等**。第一版不是：

* 旧版的 `NOOK` 是一张**链式替换表**（`秘藏书架→曲脊巷`、`墨泉书店→秘藏书架`、
  `抄写员之巢→盖德里克宅邸` …），它假定库还停在「整体错位一格」的旧状态。
* 缺陷**修好之后再跑一遍**，这张表会拿修好的值当输入**再错位一次** ——
  2026-08-15 实测：正确库上它仍报「错位修复 3」，并产出一份会把
  `Scholar's Nook` 的 7 个 `<dt>` 名重新写坏的批次。谁顺手落了那份批次谁就改坏正确数据。

现在改成**英文锚定的定位重写**（`fix_nook`）：逐个 `<dt>` 拿**同位置的英文名**去
`NOOK_EN2CN` 里查既定译名，中文对得上就一个字不动，对不上才改。
输入是「英文怎么写」而不是「中文当前是什么」，因此**跑几遍结果都一样**。
另外主流程在**三类改动合计为 0 时打印「已修复，无需再跑」并直接退出，不创建
batches 目录、不写任何批次文件** —— 清单照跑不会有任何东西可落盘。

`NAMES`（英文名→中文名，命中后英文名就没了）与 `ANCESTRY`（错写→正确写法，
正确写法不含错写模式）本来就幂等，未改判据。

回测：`4-临时脚本/2026-08-15-round16/qa/backtest_gazetteer_dt.py`，双向 **6/6 PASS**
（特异度 2：正确库上 0 叶、且不创建 batches 目录；灵敏度 4：整体错位一格能修回、
只动 `<dt>` 不动 `<dd>` 正文、落盘后重跑收敛到 0、单个 `<dt>` 改错也能定位修回）。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

S = r'C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad\audit3'
REPO = '1-Ember汉化插件'
PACKS = ['ember.crucible-adventure.json', 'ember.adventure.json']

# ---- 1. Scholar's Nook 的 landmark 名整体错位 ---------------------------------
# 英文 <dt> 顺序：Old City Library / The Crooked Spine / The Secret Shelf /
#                 The Gadrick Estate / The Inkswell Bookshop / Librarium of Spectra /
#                 Scrivener's Den
# 缺陷当初的中文：旧城区图书馆 / 秘藏书架 / 墨泉书店 / 抄写员之巢 / 余墨书店 /
#                 光谱藏书馆 / 抄写员之巢穴
# —— <dd> 正文全部对得上，只有 <dt> 名整体前移一格。锚点：`A Message from Sin` 页把
# `The Secret Shelf` 译作「秘藏书架」，证实是整体前移。
# 缺的两个名字（The Crooked Spine / The Gadrick Estate）库内此前无任何写法，2026-08-12 新定。
#
# 下表是**英文名 → 既定中文名**的对照（不是「错的 → 对的」的替换链），
# 判据因此与库当前值无关，重复运行不会累积错位。
NOOK_EN2CN = {
    'Old City Library': '旧城区图书馆',
    'The Crooked Spine': '曲脊巷',          # 蜿蜒如蛇的书店小巷，spine 兼指书脊
    'The Secret Shelf': '秘藏书架',         # 外部锚点：A Message from Sin
    'The Gadrick Estate': '盖德里克宅邸',    # 创办人埃里森·盖德里克
    'The Inkswell Bookshop': '墨泉书店',    # inkwell＝墨泉
    'Librarium of Spectra': '斯佩克特拉藏书馆',
    "Scrivener's Den": '抄写员之巢',
}

# ---- 2. 35 个未译 NPC 名（音译；库内此前无任何写法，逐个查过） ----------------
NAMES = {
    'Cavran Ellisar': '卡夫兰·埃利萨尔', 'Darissa Liliman': '达丽莎·莉莉曼',
    'Divonnay': '迪冯奈', 'Eilo Vark': '艾洛·瓦克',
    'Elden Jhonda': '埃尔登·琼达',            # 同页正文已作「琼达」
    'Elvar Andraad': '埃尔瓦·安德拉德',
    'Faral the Fearless': '无畏者法拉尔',      # the Fearless 是绰号，按意译＋人名
    'Gravin': '格拉文', 'Gregorius Prestyn': '格雷戈里乌斯·普雷斯廷',  # 正文已作「圣普雷斯廷」
    'Hestyl Drein': '赫斯提尔·德雷因', 'Jastor Riven': '贾斯托·里文',
    'Jemble Kane': '杰姆布尔·凯恩', 'Jintal Aveniri': '金塔尔·阿维尼里',
    'Jurlan Savas': '尤尔兰·萨瓦斯', 'Lirma Grell': '莉尔玛·格雷尔',
    'Lisarra Pell': '莉萨拉·佩尔', 'Marnis Vell': '马尼斯·维尔',
    'Matin Folke': '马汀·福尔克',
    'Miles Melfield': '迈尔斯·梅尔菲尔德',    # 同页正文已作「迈尔斯」
    'Myella Tarrie': '米耶拉·塔里', 'Paela Tern': '帕埃拉·特恩',
    'Pelin Dratch': '佩林·德拉奇', 'Pevin Draivel': '佩文·德莱维尔',
    'Rilan Verdane': '里兰·维尔丹', 'Shel': '谢尔',
    'Shevra Tassel': '谢芙拉·塔塞尔', 'Tazhira Dyre': '塔兹希拉·戴尔',
    'Tidwick Lassinger': '提德威克·拉辛格', 'Tomnas Grey': '托姆纳斯·格雷',
    'Trianda': '特里安达',
    # Vijin 与 Vujin 是两个不同的人（不同页、不同阵营），译名必须能区分
    'Vijin Barriq': '维金·巴里克', 'Vujin Barriq': '武金·巴里克',
    'Yerig Fenmarch': '耶里格·芬马奇', 'Zale': '扎勒', 'Zalend Ark': '扎伦德·阿克',
}

# ---- 3. 血统串里的半译 / 误译 -------------------------------------------------
# 每条都是「错写 → 正确写法」，且正确写法本身不含错写模式，重复运行是空操作。
ANCESTRY = [
    ('奥尔达尼 Altyran', '奥尔达尼阿尔提拉'),   # 祖裔页 name＝「阿尔提拉 Altyra」
    ('Arcurian科拉克', '阿克图里安科拉克'),     # 上游把 Arcturian 拼错成 Arcurian
    ('阿克图里安人人类', '阿克图里安人类'),      # 「人」后面又跟「人类」
    ('阿克图里安人赫尔格伦', '阿克图里安赫尔格伦'),
    ('阿克图里安人荆芽灵', '阿克图里安荆芽灵'),
    ('(守序中立, 奥尔达尼 费伊杰, he/him)', '（守序中立，奥尔达尼费伊杰，he/him）'),  # 半角标点
]

DT = re.compile(r'<dt>(.*?)</dt>', re.S)


def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{p}.{i}')
    elif isinstance(o, str):
        yield p, o


def plain(s: str) -> str:
    """`<p><strong>Miles Melfield </strong>(…)</p>` → `Miles Melfield (…)`。"""
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', s)).strip()


def fix_nook(cn: str, en: str):
    """英文锚定的定位重写，**幂等**。

    第 i 个 `<dt>` 的英文名若在 `NOOK_EN2CN` 里，中文第 i 个 `<dt>` 的纯文本就必须是
    对应译名：对得上（不管里面裹的是什么标签）一个字不动，对不上才整段换成
    `<p>译名</p>`。NPC 那些 `<dt>`（英文是 `Name (…)`）查不到，天然不碰。

    返回 `(新中文, 改了几个 dt, 状态)`。中英 `<dt>` 个数对不上时**拒绝改动** ——
    那说明上游改过页面结构，本表的位置假设已经失效，得人来看。
    """
    cs, es = list(DT.finditer(cn)), list(DT.finditer(en))
    if len(cs) != len(es):
        return cn, 0, f'SHAPE_MISMATCH cn={len(cs)} en={len(es)}'
    edits = []
    for c, e in zip(cs, es):
        want = NOOK_EN2CN.get(plain(e.group(1)))
        if want is None or plain(c.group(1)) == want:
            continue
        edits.append((c.start(1), c.end(1), f'<p>{want}</p>'))
    if not edits:
        return cn, 0, 'ALREADY_OK'
    buf, last = [], 0
    for a, b, rep in edits:
        buf.append(cn[last:a])
        buf.append(rep)
        last = b
    buf.append(cn[last:])
    return ''.join(buf), len(edits), 'FIXED'


def build(repo=REPO, packs=PACKS):
    """→ [(pack, {path: 新值}, stats, [告警])]，纯计算，不落盘。"""
    result = []
    for pack in packs:
        with open(f'{repo}/compendium/cn/{pack}', encoding='utf-8') as f:
            cn = dict(walk(json.load(f)['entries']))
        with open(f'{repo}/compendium/en/{pack}', encoding='utf-8') as f:
            en = dict(walk(json.load(f)['entries']))
        out, warn = {}, []
        stats = {'nook': 0, 'name': 0, 'anc': 0}
        for path, v in cn.items():
            if '<dt>' not in v:
                continue
            nv = v
            if "Scholar's Nook" in path:
                nv, n, state = fix_nook(nv, en.get(path, ''))
                stats['nook'] += n
                if state.startswith('SHAPE_MISMATCH'):
                    warn.append(f'{path}: {state} —— 结构变了，Nook 段跳过')
            # NPC 名：只动 <dt> 里「英文名 + 对齐括号」这个形态，不碰正文里的同名串。
            # 有一部分条目名外面还包着 <strong>（Scholar's Nook / Smokerie 两页），要一并认。
            for enk, zh in NAMES.items():
                pat = re.compile(r'(<dt><p>(?:<strong>)?)' + re.escape(enk)
                                 + r'(\s*(?:</strong>)?\s*[（(])')
                nv, n = pat.subn(lambda m, z=zh: m.group(1) + z + m.group(2), nv)
                stats['name'] += n
            for a, b in ANCESTRY:
                if a in nv:
                    stats['anc'] += nv.count(a)
                    nv = nv.replace(a, b)
            if nv != v:
                out[path] = nv
        result.append((pack, out, stats, warn))
    return result


def main():
    ap = argparse.ArgumentParser(description='地名志 <dt> 修复（幂等；库已正确则不产批次）')
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--out-dir', default=S, help='批次落到 <out-dir>/batches/')
    a = ap.parse_args()

    result = build(a.repo)
    total = sum(len(o) for _, o, _, _ in result)
    for pack, out, stats, warn in result:
        for w in warn:
            print(f'  ⚠ {w}')
        print(f'{pack:34s} 叶 {len(out):3d}  错位修复 {stats["nook"]}  '
              f'人名 {stats["name"]}  血统 {stats["anc"]}')
    if not total:
        print('已修复，无需再跑（三类改动合计 0 叶，未产出任何批次）')
        return 0
    bdir = os.path.join(a.out_dir, 'batches')
    os.makedirs(bdir, exist_ok=True)
    for pack, out, _, _ in result:
        if not out:
            continue
        p = os.path.join(bdir, f'N3__ember__{pack[:-5]}.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f'  -> {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
