# -*- coding: utf-8 -*-
"""修 0.9.16 的路径重复 bug，并把「取值那一段」抽成可离线验的纯函数。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '2-Crucible汉化插件/lang-reclaim.js'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

OLD = nl.join([
"function loadOwnTranslationsSync() {",
"  const path = ownLanguagePath();",
"  if (!path) return null;",
"  const url = foundry.utils.getRoute(`modules/${MODULE_ID}/${path}`);",
"  const xhr = new XMLHttpRequest();",
"  xhr.open('GET', url, false); // 同步：理由见文件头「时机与取值方式」",
"  xhr.send(null);",
"  if (xhr.status !== 200) throw new Error(`HTTP ${xhr.status} for ${url}`);",
"  const json = JSON.parse(xhr.responseText);",
"  if (!json || typeof json !== 'object' || Array.isArray(json)) {",
"    throw new Error(`${url} 不是一个对象`);",
"  }",
"  return json;",
"}",
])
NEW = nl.join([
"/**",
" * **纯函数**：把 `languages[].path` 归一成「从数据根算起」的相对路径。",
" *",
" * ⚠⚠ 0.9.16 在这里栽过一次，值得把成因写死：**已安装**的包上，`languages[].path`",
" *   在运行时**已经是** `modules/<id>/lang/cn.json`，不是清单里写的 `lang/cn.json`。",
" *   成因在 Foundry 服务端（`dist/packages/package.mjs`）：",
" *       const a = e.languages.element;  a.fields.path = new PackageAssetField;",
" *   而 `PackageAssetField.initialize()` 对 `installed` 的包会把值解析成 **clientPath**",
" *   （`relativeToPackage` 默认为真 ⇒ 前缀 `modules/<id>`）；只有**未安装**（远端 manifest",
" *   读出来的）包才原样返回清单里的写法。这也正是 core 自己 `#filterLanguagePaths` 能把",
" *   `l.path` 直接交给 `#loadTranslationFile(path)` 的原因。",
" *   我当初拿清单里的形状去拼，于是拼出了 `/modules/crucible-cn/modules/crucible-cn/lang/cn.json`",
" *   —— 404、抢回器整段没跑，而屏幕上只是英文、控制台只有一条 warn。",
" *",
" * ⇒ 两种形态都接住，不挑食（与上面 `languages` 那处同一个原则）。",
" * @param {string} p            `languages[].path` 的值",
" * @param {string} [moduleId]",
" * @returns {string}            形如 `modules/crucible-cn/lang/cn.json`",
" */",
"export function toClientPath(p, moduleId = MODULE_ID) {",
"  const prefix = `modules/${moduleId}/`;",
"  const clean = String(p).replace(/^\/+/, '');",
"  return clean.startsWith(prefix) ? clean : `${prefix}${clean}`;",
"}",
"",
"/**",
" * 同步读回自家的扁平点号键翻译表。失败一律返回 `null`（调用方当没这回事，绝不影响开世界）。",
" * @param {(p: string) => string} [getRoute]  注入点：离线验要能不启动 Foundry 就跑它",
" * @param {() => XMLHttpRequest} [makeXhr]    同上",
" * @returns {Record<string, string>|null}",
" */",
"function loadOwnTranslationsSync(getRoute = foundry.utils.getRoute,",
"                                 makeXhr = () => new XMLHttpRequest()) {",
"  const declared = ownLanguagePath();",
"  if (!declared) return null;",
"  const url = getRoute(toClientPath(declared));",
"  const xhr = makeXhr();",
"  xhr.open('GET', url, false); // 同步：理由见文件头「时机与取值方式」",
"  xhr.send(null);",
"  // 报错里**两个**都带上：只报最终 URL 的话，下次形状再变还得再猜一轮。",
"  if (xhr.status !== 200) {",
"    throw new Error(`HTTP ${xhr.status} for ${url}（languages[].path 声明为 ${declared}）`);",
"  }",
"  const json = JSON.parse(xhr.responseText);",
"  if (!json || typeof json !== 'object' || Array.isArray(json)) {",
"    throw new Error(`${url} 不是一个对象`);",
"  }",
"  return json;",
"}",
])
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已修')
