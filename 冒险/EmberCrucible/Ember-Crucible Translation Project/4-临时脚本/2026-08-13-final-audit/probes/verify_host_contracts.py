# -*- coding: utf-8 -*-
"""
verify_host_contracts.py — 「宿主契约假设」逐条复现脚本（只读）

抽象出来的判据（与 register.js 的 game.world.getFlag 幂等闸同类）：
    插件里每一处**依赖宿主契约的判断**——存在性判据、幂等闸、生命周期时机、
    注册表遍历、i18n 键、钩子里改参数是否生效——都必须能在钉死的上游源码里
    找到支撑。找不到支撑而写法又不抛异常时，它就**静默失效**。

本脚本把人工核实过的每一条固化成一个 assertion，跑一遍即可复现结论。
每条 check 输出 PASS（契约成立，插件写法正确）或 FAIL（契约不成立 = 缺陷）。

上游钉死版本：
    Foundry VTT 14.365.0   C:/Program Files/Foundry Virtual Tabletop/resources/app
    crucible   0.10.1      Data/systems/crucible/crucible-compiled.mjs
    ember      0.6.0       Data/modules/ember/scripts/ember.mjs
    babele     2.9.1       Data/modules/babele/script/**

假阳性说明：本脚本做的是**文本证据核对**，不是运行时验证。每条 check 的
`why` 字段写明了它凭什么这么判断；推翻结论只需推翻 why 里引的那几行源码。
"""
import os, re, sys, json

FOUNDRY = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
FDATA   = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
PROJ    = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

def read(p):
    return open(p, encoding="utf-8", errors="replace").read()

def has(path, pat, flags=0):
    return re.search(pat, read(path), flags) is not None

def count(path, pat, flags=0):
    return len(re.findall(pat, read(path), flags))

EMBER_MJS = os.path.join(FDATA, "modules", "ember", "scripts", "ember.mjs")
CRUCIBLE  = os.path.join(FDATA, "systems", "crucible", "crucible-compiled.mjs")
BACKEND   = os.path.join(FOUNDRY, "client", "data", "client-backend.mjs")
GAME      = os.path.join(FOUNDRY, "client", "game.mjs")
UIMJS     = os.path.join(FOUNDRY, "client", "ui.mjs")
APPV1     = os.path.join(FOUNDRY, "client", "appv1", "api", "application-v1.mjs")
APPV2     = os.path.join(FOUNDRY, "client", "applications", "api", "application.mjs")
CORE_EN   = os.path.join(FOUNDRY, "public", "lang", "en.json")

checks = []
def check(name, ok, why):
    checks.append((name, bool(ok), why))

# ---------------------------------------------------------------- C1
# 契约：ui.windows 里能找到 Ember 的日历条，从而可以强制重画。
v1_only  = has(UIMJS, r"@type \{Record<string, appv1\.api\.Application>\}[\s\S]{0,80}export const windows")
v1_write = has(APPV1, r"if \( this\.popOut \) ui\.windows\[this\.appId\] = this")
v2_write = has(APPV2, r"foundry\.applications\.instances\.set")
cal_is_v2 = has(EMBER_MJS, r"class EmberCalendarNavigation extends HandlebarsApplicationMixin\$?\d* ?\(ApplicationV2")
cal_frameless = has(EMBER_MJS, r'id: "ember-calendar"[\s\S]{0,220}frame: false')
check("C1 ui.windows 能拿到 Ember 日历条",
      not (v1_only and v1_write and v2_write and cal_is_v2),
      f"ui.mjs 标注 windows 只装 appv1={v1_only}; appv1 是唯一写入方={v1_write}; "
      f"V2 走 foundry.applications.instances={v2_write}; EmberCalendarNavigation 是 ApplicationV2={cal_is_v2}; "
      f"且 window.frame=false（连 popOut 都不是）={cal_frameless}")

# ---------------------------------------------------------------- C2
# 契约：i18n 顶层存在 Sort / sort 键，值得覆写。
core = json.load(open(CORE_EN, encoding="utf-8"))
core_has = any(k.lower().startswith("sort") for k in core)
cru_lang = json.load(open(os.path.join(FDATA, "systems", "crucible", "lang", "en.json"), encoding="utf-8"))
cru_has = any(k.lower().startswith("sort") for k in cru_lang)
cn = json.load(open(os.path.join(PROJ, "2-Crucible汉化插件", "lang", "cn.json"), encoding="utf-8"))
cn_has = any(k.lower().startswith("sort") for k in cn)
readers = count(CRUCIBLE, r'localize\(\s*["\']sort["\']\s*\)', re.I) + count(EMBER_MJS, r'localize\(\s*["\']sort["\']\s*\)', re.I)
check("C2 顶层 i18n 键 Sort/sort 真的存在且被读",
      core_has or cru_has or cn_has or readers > 0,
      f"foundry en.json 顶层 {len(core)} 键中 sort* = {core_has}; crucible lang en.json {len(cru_lang)} 键 = {cru_has}; "
      f"crucible-cn cn.json {len(cn)} 键 = {cn_has}; localize(\"sort\") 调用点 = {readers}")

# ---------------------------------------------------------------- C3
# 契约：在 preCreateX 钩子里就地改 data，能影响真正建出来的文档。
clone_before = has(BACKEND, r"doc = new documentClass\(foundry\.utils\.deepClone\(createData\)")
hook_after   = has(BACKEND, r"Hooks\.call\(`preCreate\$\{type\}`, doc, createData")
overwritten  = has(BACKEND, r"else operation\.data = documents;")
check("C3 preCreateItem 里改 data 生效",
      not (clone_before and hook_after and overwritten),
      f"文档在钩子前用 deepClone(createData) 构造={clone_before}; 钩子在其后触发={hook_after}; "
      f"最终 operation.data 被改写为 documents（丢弃 toCreate）={overwritten}")

# 对照组：preUpdate 侧就地改 changes 是**有效**的，正因如此上面那条才隐形
upd_hook = has(BACKEND, r"Hooks\.call\(`preUpdate\$\{type\}`, doc, changes")
upd_use  = has(BACKEND, r"doc\.updateSource\(changes, \{[\s\S]{0,200}data may have changed in preUpdate")
check("C3b（对照）preUpdate 里改 changes 生效", upd_hook and upd_use,
      f"钩子传 changes={upd_hook}; 钩子后 updateSource(changes) 且核心注释承认可被 preUpdate 改={upd_use}")

# ---------------------------------------------------------------- C4
# 契约：Foundry 的 setup 钩子早于世界文档准备（register.js:457 注释这么写的）。
g = read(GAME)
i_init = g.find("this.initializeDocuments();")
i_setup = g.find('Hooks.callAll("setup")')
prepares = has(GAME, r"for \( const document of collection \) \{?\s*document\._safePrepareData\(\);")
snapshot = has(CRUCIBLE, r'Object\.defineProperty\(this, "hooks", \{\s*value: CrucibleAction\.#prepareHooks')
frozen   = has(CRUCIBLE, r"return Object\.freeze\(hooks\);")
prep_now = has(CRUCIBLE, r"if \( !options\.lazy \) this\.prepare\(\);")
check("C4 setup 早于世界文档准备",
      not (0 < i_init < i_setup and prepares and snapshot and frozen and prep_now),
      f"game.mjs initializeDocuments@{i_init} < callAll('setup')@{i_setup}; "
      f"initializeDocuments 内确实逐个 _safePrepareData={prepares}; "
      f"CrucibleAction._initialize 冻结 hooks 快照={snapshot and frozen}; 并当场 prepare()={prep_now}")

# ---------------------------------------------------------------- C5
# 契约：ember 塞进 crucible.CONFIG 的硬编码英文只在 languages / knowledge 两张表里。
lc = re.findall(r'crucible\.CONFIG\.languageCategories\.(\w+) = \{label: "([^"]+)"\}', read(EMBER_MJS))
patched_tables = re.findall(r'\[\["languages", LANGUAGES\], \["knowledge", KNOWLEDGE\]\]',
                            read(os.path.join(PROJ, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")))
rendered = has(CRUCIBLE, r"toAdd\.group = crucible\.CONFIG\.languageCategories\[category\]\?\.label;")
localized = has(CRUCIBLE, r"localizeConfigObject\(crucible\.CONFIG\.languageCategories\);")
check("C5 patchCrucibleConfig 的表清单覆盖了 ember 的全部硬编码英文",
      not (lc and patched_tables and rendered),
      f"ember 往 languageCategories 塞的裸英文 = {lc}; 插件只遍历 languages/knowledge = {bool(patched_tables)}; "
      f"这些 label 被当作下拉分组名渲染 = {rendered}; crucible 会 localize 它们但键不存在故原样返回 = {localized}")

# ---------------------------------------------------------------- 输出
sys.stdout.reconfigure(encoding="utf-8")
bad = 0
for name, ok, why in checks:
    tag = "PASS" if ok else "FAIL"
    if not ok: bad += 1
    print(f"[{tag}] {name}\n       {why}\n")
print(f"# {len(checks)} checks, {bad} FAIL（FAIL = 契约不成立 = 插件那处静默失效）")
