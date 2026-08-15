# -*- coding: utf-8 -*-
"""探针：**无作用域地写入共享命名空间** —— 与种子缺陷同类，只是集合换成了
「i18n 全局翻译表」而不是「世界文档集合」。

判据（与种子同构）：
  (a) 本模块把一批键写进一个**全局共享**的集合（Foundry 的 game.i18n.translations，
      模块 lang 文件会被 mergeObject 深并进去，后加载者覆盖先加载者）；
  (b) 准入条件只看**键的形状**（「我们的 lang 里有这个键」），
      不看**这个键归谁**（Foundry 本体 / 系统 / 本模块）；
  (c) 于是本模块顺手改掉了不属于自己的字符串。

输出：本项目两个 lang/cn.json 里，**Foundry 本体 en.json 也有、而 crucible 系统与
ember 模块的 en.json 都没有**的键。这些键写进去只会盖掉本体 UI。

假阳性模式（必须人工核）：
  1. 本体键 + 我们译得也对 → 观感上无害，但仍然是越界（本体自己的中文包会被我们盖掉）；
  2. 键名同形但语义确实是系统自己的（系统故意复用本体键名）—— 需要看系统源码是否 localize 它；
  3. 只读 en.json 判断「本体有没有」，本体的 zh-CN 官方包不在此仓库，无法判断玩家实际看到什么。
只读，不写任何库文件。
"""
import json
import os
import sys

ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
CORE = r'C:\Program Files\Foundry Virtual Tabletop\resources\app\public\lang\en.json'
CRUCIBLE = r'C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\lang\en.json'
EMBER = r'C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\lang\en.json'

TARGETS = [
    ('ember_cn_unofficial', os.path.join(ROOT, '1-Ember汉化插件', 'lang', 'cn.json')),
    ('crucible-cn', os.path.join(ROOT, '2-Crucible汉化插件', 'lang', 'cn.json')),
]


def flat(obj, prefix=''):
    """把嵌套 + 点号混排的 lang 文件拍平成点号键集合（Foundry 的 mergeObject 也会先 expandObject）。"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f'{prefix}.{k}' if prefix else k))
    else:
        out[prefix] = obj
    return out


def load(p):
    if not os.path.exists(p):
        print(f'!! 缺文件 {p}')
        return {}
    with open(p, encoding='utf-8-sig') as f:
        return flat(json.load(f))


def main():
    core = load(CORE)
    cru = load(CRUCIBLE)
    emb = load(EMBER)
    print(f'core keys {len(core)} / crucible {len(cru)} / ember {len(emb)}\n')

    for name, path in TARGETS:
        ours = load(path)
        owned = set(cru) | set(emb)
        collide = sorted(k for k in ours if k in core and k not in owned)
        print(f'== {name}: {len(ours)} 键，其中撞本体且系统/模块都没有的 {len(collide)} 条')
        for k in collide:
            print(f'   {k}')
            print(f'       core EN : {core[k]!r}')
            print(f'       我们写的 : {ours[k]!r}')
        # 另一档：撞本体、系统也有（系统自己复用了本体键名）
        both = sorted(k for k in ours if k in core and k in owned)
        print(f'   （参考）撞本体但系统/模块自己也声明了同名键：{len(both)} 条')
        for k in both[:40]:
            print(f'       {k}  core={core[k]!r}  sys={ (cru.get(k) or emb.get(k))!r}  ours={ours[k]!r}')
        print()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
