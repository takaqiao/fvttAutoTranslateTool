# -*- coding: utf-8 -*-
"""收紧 crucibleDescription 的「源字段不存在」分支：不再猜成裸字符串。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '1-Ember汉化插件/babele-mappings.js'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

OLD = nl.join([
"  if (typeof value === 'string' || value === undefined || value === null) {",
"    if (isStr(translation)) return translation;",
"    if (isStr(translation?.public)) return translation.public;",
"    return value;",
"  }",
])
NEW = nl.join([
"  // ── 源**是字符串**：按字符串还回去（形状跟着源走，这是本函数的全部意义）",
"  if (typeof value === 'string') {",
"    if (isStr(translation)) return translation;",
"    if (isStr(translation?.public)) return translation.public;",
"    return value;",
"  }",
"",
"  // ── 源**不存在**（undefined / null）：形状**无从得知**，不许猜。",
"  //",
"  // ⚠⚠ 2026-08-22 收紧。旧写法把这一支与「源是字符串」合在一起，于是源不存在时",
"  //   会返回 `translation.public` 这个**裸字符串**。那是在猜，而猜错的代价刚好被",
"  //   项目所有者撞上过一次：ember 0.6.0 里 `Potion of Climbing` / `Growing Thorns` 的",
"  //   `system.description` 是**纯字符串**（349 / 1360 字符，见 english-baseline/ember-0.6.0），",
"  //   0.6.1 上游改成了 `{public, private}` 且 type 变 `consumable`；再导入时 Foundry 的",
"  //   `SchemaField._updateDiff`（common/data/fields.mjs:1231）执行",
"  //       const source = (state.source[key] ||= {});   // 旧值是非空字符串 ⇒ ||= 不替换",
"  //   随后 `state.source[\"public\"] = …` 往字符串上建属性 ⇒ **TypeError，整条文档导入失败**。",
"  //   那一次的字符串是 0.6.0 上游给的、不是本函数造的（当时源就是字符串，返回字符串是对的），",
"  //   但**同一个形状差**只要走一次「源不存在时猜成字符串」就能被我们自己造出来。",
"  //",
"  // ⇒ 改为按**译文自己的形状**还回去：译文是从某个源抽出来的，它的形状就是那个源的形状，",
"  //   比凭空猜准。译文是对象 → 还对象；是字符串 → 还字符串；都不是 → 原样不动。",
"  if (value === undefined || value === null) {",
"    if (isStr(translation)) return translation;",
"    if (translation && typeof translation === 'object') {",
"      const t = {};",
"      if (isStr(translation.public)) t.public = translation.public;",
"      if (isStr(translation.private)) t.private = translation.private;",
"      if (isStr(translation.value)) t.value = translation.value;",
"      return Object.keys(t).length ? t : value;",
"    }",
"    return value;",
"  }",
])
assert s.count(OLD) == 1, s.count(OLD)
io.open(P, 'w', encoding='utf-8', newline='').write(s.replace(OLD, NEW))
print('已收紧')
