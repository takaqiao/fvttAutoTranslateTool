# -*- coding: utf-8 -*-
"""骨架自检：按 Foundry v14 的 module.json 语义逐条核对。"""
import json, os, re, sys

P = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project"
DIRS = ["1-系统汉化插件", "2-新手包汉化插件", "3-核心书汉化插件"]
ALLOWED_ID = re.compile(r"^[A-Za-z0-9\-_]+$")          # base-package.mjs:592
PROHIBITED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.I)  # :596
REL_KEYS = {"systems", "requires", "recommends", "conflicts", "flags"}          # :48-52
fails, warns = [], []


def chk(cond, msg):
    if not cond:
        fails.append(msg)


for d in DIRS:
    root = os.path.join(P, d)
    mj = json.load(open(os.path.join(root, "module.json"), encoding="utf-8"))
    mid = mj["id"]
    tag = "v" + mj["version"]

    chk(ALLOWED_ID.match(mid) and not PROHIBITED.match(mid), f"{d}: 非法模块 id {mid}")
    chk(mj["download"].endswith(f"/releases/download/{tag}/module.zip"),
        f"{d}: download 未指向 {tag} -> {mj['download']}")
    chk(mj["changelog"].endswith(f"/releases/tag/{tag}"),
        f"{d}: changelog 未指向 {tag} -> {mj['changelog']}")
    chk(mj["manifest"].endswith("/releases/latest/download/module.json"),
        f"{d}: manifest 不是 latest 形态")
    for u in ("url", "manifest", "download", "bugs", "changelog"):
        chk(f"/{mid}/" in mj[u] or mj[u].endswith("/" + mid), f"{d}: {u} 里的仓库名与 id 不一致 -> {mj[u]}")

    comp = mj["compatibility"]
    chk(comp == {"minimum": "13", "verified": "14", "maximum": "14.999"},
        f"{d}: compatibility 不是约定值 -> {comp}")

    rel = mj.get("relationships", {})
    chk(set(rel) <= REL_KEYS, f"{d}: relationships 含 Foundry 不认识的字段 {set(rel) - REL_KEYS}")
    chk(any(r["id"] == "alienrpg" for r in rel.get("systems", [])), f"{d}: relationships.systems 缺 alienrpg")
    chk(any(r["id"] == "babele" and r["compatibility"]["minimum"] == "2.9.1"
            for r in rel.get("requires", [])), f"{d}: requires 缺 babele>=2.9.1")

    # 声明的文件必须存在
    for f in mj.get("esmodules", []):
        chk(os.path.isfile(os.path.join(root, f)), f"{d}: esmodule 不存在 {f}")
    for s in mj.get("styles", []):
        chk(os.path.isfile(os.path.join(root, s["src"])), f"{d}: style 不存在 {s['src']}")
    for l in mj.get("languages", []):
        chk(os.path.isfile(os.path.join(root, l["path"])), f"{d}: lang 文件不存在 {l['path']}")
        chk(l["lang"] == "cn", f"{d}: lang 不是 cn -> {l['lang']}")

    # 骨架文件齐备
    for f in ["README.md", "LICENSE", ".gitignore",
              ".github/workflows/release.yml", ".github/release-body-template.md",
              "register.js", "compendium/cn", "compendium/en", "compendium/en/_source.json"]:
        chk(os.path.exists(os.path.join(root, f)), f"{d}: 缺少 {f}")

    gi = open(os.path.join(root, ".gitignore"), encoding="utf-8").read()
    for line in gi.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            chk("json" not in s.lower(), f"{d}: .gitignore 出现涉及 json 的规则 -> {s}")

    # 中枢 vs 内容模块的分工
    reg = open(os.path.join(root, "register.js"), encoding="utf-8").read()
    # 先剥注释，否则文件头里那段「为什么没有 registerMapping」的说明会被算成调用
    code = re.sub(r"/\*.*?\*/", "", reg, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    calls = len(re.findall(r"babele\.registerMapping\(", code))
    if d == "1-系统汉化插件":
        chk(calls == 1, f"{d}: 中枢应恰好调用一次 registerMapping，实际 {calls}")
        chk(len(mj["languages"]) == 5, f"{d}: languages 应为 5 条，实际 {len(mj['languages'])}")
        gated = [l for l in mj["languages"] if "module" in l]
        chk({l["module"] for l in gated} ==
            {"alien-mu-th-ur", "motion_tracker", "token-action-hud-alien", "babele"},
            f"{d}: 插件门控集合不对 -> {[l.get('module') for l in mj['languages']]}")
        chk(sum(1 for l in mj["languages"] if "module" not in l) == 1,
            f"{d}: 无门控的系统覆盖条目应恰好 1 条")
    else:
        chk(calls == 0, f"{d}: 内容模块**不得**调用 registerMapping，实际 {calls}")
        chk("languages" not in mj, f"{d}: 内容模块不应声明 languages")
        chk(any(r["id"] == "alienrpg-cn" for r in rel.get("requires", [])), f"{d}: requires 缺 alienrpg-cn")
        tp = rel.get("flags", {}).get("alienrpg-cn", {}).get("targetPack")
        chk(bool(tp), f"{d}: relationships.flags 未记录 targetPack")

    # ── Babele 的两个保留文件名不得出现在 compendium/cn ────────────────────────
    #
    # register.js 里 `babele.register({dir:'compendium/cn'})` 把**同一个目录**
    # 同时交给 Babele 当 translationDirectories 和 mappingDirectories
    # （modules/babele/script/translation/translation-source-registry.js:61-62）。
    # 而 Babele 分辨一份 .json 是译文还是 mapping **只看文件名**：
    #   translation/translation-source.js:14-15
    #       static MAPPING_FILENAME        = "mappings.json";
    #       static LEGACY_MAPPING_FILENAME = "mapping.json";
    #   同文件 :115-118 #isMappingFile 用 endsWith 判这两个名字，
    #   :120-126 #matchesKind 据此把目录里的 .json 二选一分流。
    # 两种失效形态都**不报错、无日志**：
    #   · 叫这两个名字的**译文**被当成 mapping 读走，一个字都不翻；
    #   · 叫这两个名字的**mapping** 进 loadedMappings，而
    #     mapping/document-mappings.js:267-278 的 #rebuild 顺序是
    #     builtIn -> registered -> loaded，于是它**压过**中枢 registerMapping
    #     注册的那唯一一层全局 mapping。
    # 本项目的 mapping 一律走 alienrpg-cn/babele-mappings.js，这条通道必须封死。
    #
    # 运行时真正被扫的只有目录**第一层**（translation-source.js:83-95 的 #discover
    # 调 FilePicker.browse，不递归），但这里**递归**查：子目录里的同名文件是给
    # 未来某次「顺手把子目录也 register 进去」准备的地雷，现在就挡掉成本为零。
    cn_dir = os.path.join(root, "compendium", "cn")
    for dirpath, _dirnames, filenames in os.walk(cn_dir):
        for fn in filenames:
            if fn in ("mappings.json", "mapping.json"):
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
                chk(False, f"{d}: compendium/cn 里出现 Babele 保留文件名 -> {rel}"
                           f"（会被当 mapping 读走，压过中枢的全局层）")

    wf = open(os.path.join(root, ".github/workflows/release.yml"), encoding="utf-8").read()
    rm_i, zip_i = wf.find("rm -f module.zip"), wf.find("zip -r module.zip")
    chk(rm_i != -1 and zip_i != -1 and rm_i < zip_i, f"{d}: release.yml 里 rm -f 未排在 zip -r 之前")
    chk('-x "compendium/en/*"' in wf, f"{d}: release.yml 未排除 compendium/en/*")

    # 三仓的发布闸门必须齐平 —— 缺任何一道就等于那个仓可以把空壳打成正式发布。
    chk("name: Verify there is actually something to ship" in wf,
        f"{d}: release.yml 缺「有东西可发」闸门")
    chk("name: Assert compendium/cn holds no Babele mapping filename" in wf,
        f"{d}: release.yml 缺 compendium/cn 保留文件名断言")
    # 空正文判据不能退化回 `jq '[paths(scalars)]|length'` 那种写法：jq 对空输入
    # 退 0 且无输出，`[ "" -eq 0 ]` 返回 2，而 set -e 对 if 条件不生效 —— 0 字节
    # 与纯空白的 cn.json 会被静默放行（已实测）。判据必须含这三段。
    chk("""jq -e 'type == "object"'""" in wf, f"{d}: release.yml 的正文闸门缺 `jq -e type==object` 一段")
    chk('select(test("[^[:space:]]"))' in wf, f"{d}: release.yml 的正文闸门缺「非空白字符串叶子」一段")
    chk("[ ! -s " in wf, f"{d}: release.yml 的正文闸门缺 0 字节检查")
    if d == "1-系统汉化插件":
        for x in ['-x "lang/en.json"', '-x "lang/lang_keep_english.json"', '-x ".github/*"', '-x "*.zip"']:
            chk(x in wf, f"{d}: release.yml 缺排除项 {x}")
        chk('-x "scripts/' not in wf, f"{d}: release.yml 不应排除 scripts/（那是 esmodule）")
        # 只看真正的断言那一行（grep -E "..."），注释里提到 ^scripts/ 是刻意的说明
        assertion = [ln for ln in wf.splitlines() if "unzip -Z1 module.zip | grep -E" in ln]
        chk(len(assertion) == 1, f"{d}: release.yml 的排除断言不是恰好一条")
        chk(all("^scripts/" not in ln for ln in assertion),
            f"{d}: release.yml 的断言不应包含 ^scripts/")
        chk(any("compendium/en/" in ln for ln in assertion),
            f"{d}: release.yml 的断言未覆盖 compendium/en/")

print("=" * 70)
if fails:
    for f in fails:
        print("FAIL", f)
else:
    print("全部检查通过")
for w in warns:
    print("WARN", w)
print("=" * 70)
sys.exit(1 if fails else 0)
