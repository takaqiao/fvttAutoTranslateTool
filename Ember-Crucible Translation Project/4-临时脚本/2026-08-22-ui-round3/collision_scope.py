# -*- coding: utf-8 -*-
"""
foundry_chn 那 98 个顶层裸串，除了咬到 crucible-cn 那 2 个，还咬到谁？
—— **升报用的规模数，不是本轮工作面。**

⚠ 一个必须踩对的坑（本脚本第一版踩了）：核心 `public/lang/en.json` 里**大量键本身就是
  扁平点号键**，682 个顶层键里只有 90 个是嵌套对象。所以「核心有哪些顶层命名空间」必须按
  **展开之后**算；直接数 `isinstance(v, dict)` 会把 `TOKEN.FIELDS.…` 这一整片漏掉，
  于是得出「TOKEN 没撞上核心」的**错**结论。第一版就是这么错的，订正后 TOKEN 赫然在列（260 条叶）。

前置自证两件（只断言条数会让「切对条数、读错对象」蒙混过去）：
  A 切对条数：foundry_chn 顶层 177 键 / 裸串 98 条；
  B 切对对象：TOKEN 是裸串「指示物」、CONTROLS 是对象、已知反例 SETTINGS **不**在裸串里、
             核心侧 TOKEN 确实是命名空间而不是叶键。

跑法：python collision_scope.py
"""
import io
import json

L = lambda p: json.load(io.open(p, encoding='utf-8'))

CHN = r'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/foundry_chn/cn.json'
CORE_EN = r'C:/Program Files/Foundry Virtual Tabletop/resources/app/public/lang/en.json'
CRU_CN = r'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件/lang/cn.json'
EMB_CN = r'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/lang/cn.json'


def flatten(obj, prefix=''):
    """拍平成「点号路径 -> 叶值」。等价于 Foundry expandObject 之后再拍平。"""
    out = {}
    for k, v in obj.items():
        q = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, q))
        else:
            out[q] = v
    return out


def namespaces(flat):
    """顶层命名空间 -> 该命名空间下的叶数。"""
    out = {}
    for k in flat:
        top = k.split('.', 1)[0]
        out[top] = out.get(top, 0) + 1
    return out


chn = L(CHN)
bare = {k for k, v in chn.items() if isinstance(v, str)}

# ---- 前置自证 A：切对条数 ----
assert len(chn) == 177, len(chn)
assert len(bare) == 98, len(bare)
# ---- 前置自证 B：切对对象（含一条已知反例）----
assert chn['TOKEN'] == '指示物', chn['TOKEN']
assert isinstance(chn['CONTROLS'], dict) and len(chn['CONTROLS']) == 248
assert 'SETTINGS' not in bare, '已知反例 SETTINGS 被误判成裸串'
print('前置自证 A/B OK：foundry_chn 顶层 177 键 / 裸串 98 条 / '
      'TOKEN 是裸串「指示物」/ CONTROLS 是 248 键对象 / SETTINGS 不是裸串')

core_flat = flatten(L(CORE_EN))
core_ns = namespaces(core_flat)
print('\n核心 public/lang/en.json：展开后 %d 个顶层命名空间 / %d 条叶' % (len(core_ns), len(core_flat)))
assert 'TOKEN' in core_ns and 'TOKEN' not in core_flat, '核心侧 TOKEN 应当是命名空间而不是叶键'
print('前置自证 B OK：核心侧 TOKEN 确实是命名空间（%d 条叶）、不是叶键' % core_ns['TOKEN'])

hit_core = sorted(k for k in bare if k in core_ns and k not in core_flat)
print('\n98 个裸串里**撞上核心命名空间**的：%d 个'
      ' —— 这些核心键在装了 foundry_chn 的世界里全部走英文 fallback' % len(hit_core))
for k in hit_core:
    print('   %-14s = %-10s  核心该命名空间下 %d 条叶' % (k, chn[k], core_ns[k]))
print('   合计核心受害叶：%d 条' % sum(core_ns[k] for k in hit_core))
print('   ⚠ 这一块**本轮没修**，而且也修不到全部：其中 TOKEN.FIELDS.* 走 #localizeDataModels()，')
print('     发生在 i18nInit **之前**，lang-reclaim.js 那个时机够不到。要不要做、怎么做，是另一条裁决。')

for name, path in [('crucible-cn', CRU_CN), ('ember_cn_unofficial', EMB_CN)]:
    d = L(path)
    ns = {(k.split('.', 1)[0] if '.' in k else k) for k in d}
    hit = sorted(bare & ns)
    n = sum(1 for k in d if (k.split('.', 1)[0] if '.' in k else k) in hit)
    print('\n%s：顶层键 %d / 顶层命名空间 %d ⇒ 撞上的裸串 %s ⇒ 受害键 %d 条'
          % (name, len(d), len(ns), hit or '（无）', n))
