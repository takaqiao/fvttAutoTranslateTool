# -*- coding: utf-8 -*-
"""把「URL 怎么拼」钉进 R-lang-reclaim-mechanism，并同步四张登记表。"""
import io, os, sys, json, importlib.util
sys.stdout.reconfigure(encoding='utf-8')

RJ = '5-其他内容/RESOLUTIONS.assertions.json'
d = json.load(io.open(RJ, encoding='utf-8'))
a = next(x for x in d['assertions'] if x['id'] == 'R-lang-reclaim-mechanism')
assert len(a['require']) == 4, len(a['require'])
a['require'] += [
    "  const url = getRoute(toClientPath(declared));",
    "  return clean.startsWith(prefix) ? clean : `${prefix}${clean}`;",
]
a['min_checks'] = 6
a['title'] = "抢回器那几处**实测逼出来的**实现细节不许被改掉（同步 XHR / 非枚举 / i18nInit / 取址）"
a['why'] += (
  "｜⚑⚑ **2026-08-22 追加两条 require，是被一次真实事故逼出来的**："
  "`0.9.16` 发出去之后项目所有者在 VPS 上实测，那 42 条**仍然是英文**。"
  "面板新加的那一节一次就点名了病因："
  "`phase:\"error\" · HTTP 404 for /modules/crucible-cn/modules/crucible-cn/lang/cn.json` —— "
  "**前缀拼了两次**。成因在 Foundry 服务端（`dist/packages/package.mjs`）："
  "`languages.element.fields.path = new PackageAssetField`，而 `PackageAssetField.initialize()` "
  "对 `installed` 的包会把值解析成 **clientPath**（`relativeToPackage` 默认真 ⇒ 已含 `modules/<id>/`）；"
  "我却按清单里的写法又拼了一次。这也正是 core 自己能把 `l.path` 直接交给 "
  "`#loadTranslationFile(path)` 的原因。"
  "｜⚠ **本条第一版为什么没接住**：它钉的是 `xhr.open('GET', url, false);` 这一行，"
  "而坏掉的是 **`url` 怎么来的**——判据钉在了「会坏的那一行的隔壁」。"
  "同轮的离线复刻器同样脱靶：它验的是纯函数 `reclaimTranslations`，表**直接从磁盘读**，"
  "于是 `ownLanguagePath()` / `loadOwnTranslationsSync()` 一次都没被执行过。"
  "⇒ 这是空转形态 (h) 的又一变体：**验的对象不是会坏的那个对象**。"
  "现在两行都钉死，并新增端到端离线验 "
  "`4-临时脚本/2026-08-22-ui-round5/verify_reclaim_e2e.mjs`（从 `registerLangReclaim()` 真入口进，"
  "走完钩子→取路径→取值→回写全链，16 条断言；其中第 ③ 格是**回归对照**，"
  "断言旧拼法拼出的 URL 在假服务器上 404 —— 也就是这一验若在 0.9.16 上跑必红）。"
)
d['meta']['updated'] = "2026-08-22（UI 补漏第五轮：修 0.9.16 的取址 bug，判据补钉两行）"
io.open(RJ, 'w', encoding='utf-8', newline='\n').write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
print('规则集已更新：require 4 → 6, min_checks → 6')

# 现推地板 + JUDGED_UNITS
PY = os.path.join('3-常用脚本', 'qa', 'assert_resolutions.py')
spec = importlib.util.spec_from_file_location('ar', os.path.abspath(PY))
M = importlib.util.module_from_spec(spec); sys.modules['ar'] = M; spec.loader.exec_module(M)
derived = M._derive_payload_floors([a])['R-lang-reclaim-mechanism']
print(' 现推地板 =', derived)

s = io.open(PY, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
def fmt(sp):
    return '{' + ', '.join('"%s": ("%s", %r)' % (k, v[0], v[1]) for k, v in sorted(sp.items())) + '}'
old = [l for l in s.split(nl) if l.startswith('    "R-lang-reclaim-mechanism": {')]
assert len(old) == 1, old
s = s.replace(old[0], '    "R-lang-reclaim-mechanism": %s,' % fmt(derived))
# 规矩数：1 仓 × 6 条 require
assert s.count('    "R-lang-reclaim-mechanism": 4,') == 1
s = s.replace('    "R-lang-reclaim-mechanism": 4,', '    "R-lang-reclaim-mechanism": 6,')
io.open(PY, 'w', encoding='utf-8', newline='').write(s)
print('assert_resolutions.py 两处已同步')
