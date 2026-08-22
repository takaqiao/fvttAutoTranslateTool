# -*- coding: utf-8 -*-
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '2-Crucible汉化插件/babele-register.js'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
a = "import { registerLangReclaim } from './lang-reclaim.js';"
assert s.count(a) == 1
s = s.replace(a, a + nl + "import { registerCrucibleHardcoded } from './crucible-hardcoded-cn.mjs';")
b = "registerLangReclaim();"
assert s.count(b) == 1
s = s.replace(b, b + nl + nl + nl.join([
 "/**",
 " * 硬编码串的汉化通道：Babele 只管合集、lang 只管上游声明过的键，两者都够不到的",
 " * 那 18 条（掷骰面板的加值/减值来源、动作卡的上下文标签 tooltip、几处 placeholder",
 " * 与创建页的加减按钮）走这里。为什么不塞进 lang/cn.json —— 两条理由写在",
 " * `crucible-hardcoded-cn.mjs` 的文件头（`R-lang-parity` 钉死键数 + 裸键是全局的）。",
 " * 同样把挂钩动作留在那个文件里，保持它顶层零副作用、可离线 import。",
 " */",
 "registerCrucibleHardcoded();",
]))
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('已接线')
