# -*- coding: utf-8 -*-
"""给 ember 那 3326 条未覆盖串定性：**谁会看到它**。

判据不是猜的，是回上游源码看它住在什么结构里：
  vista资产 —— 命中点附近有 `category:` / `grounded:` / `placements:` / `spriteOptions`
               ⇒ 这是 Vista 场景合成器的贴图/图层名，只在 `vista-config-assets.hbs` 渲染，
                 而那个窗口由**场景控件**打开（`#onConfigureVista`）＝ GM 建图工具
  区域行为 —— 声明块名以 RegionBehavior 结尾 ⇒ GM 配置面板的字段标签
  编辑器块 —— registerProseMirrorBlocks ⇒ GM 写文档时的块名
  其余     —— 逐条列出来人看（玩家可见的都在这里）
"""
import io, re, sys, json, collections, bisect
sys.stdout.reconfigure(encoding='utf-8')
BS = chr(92)
STR = '"((?:[^"' + BS + BS + ']|' + BS + BS + '.)*)"'
FIELDS = ["label", "title", "hint", "tooltip", "placeholder", "content", "text", "message",
          "legend", "caption", "summary"]
RE_FIELD = re.compile(r'\b(' + '|'.join(FIELDS) + r')' + r'\s*:\s*' + STR)
RE_TOP = re.compile(r'^(?:export\s+)?(?:const|let|var|class|function|async function)\s+([A-Za-z_$][\w$]*)', re.M)
VISTA_NEAR = re.compile(r'category\s*:|grounded\s*:|placements\s*:|spriteOptions|parallax\s*:')

txt = io.open(r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs", encoding='utf-8').read()
tops = sorted((m.start(), m.group(1)) for m in RE_TOP.finditer(txt))
starts = [t[0] for t in tops]
owner = lambda p: tops[bisect.bisect_right(starts, p) - 1][1] if starts and bisect.bisect_right(starts, p) else "(顶层外)"

missing = set(json.load(io.open('coverage.json', encoding='utf-8'))["ember JS·展示字段"]["missing"])
cls = {}
where = {}
for m in RE_FIELD.finditer(txt):
    v = m.group(2)
    if v not in missing or v in cls: continue
    o = owner(m.start())
    ctx = txt[max(0, m.start() - 220): m.end() + 220]
    if VISTA_NEAR.search(ctx) or o.startswith(("SPRITES", "NIGHT_TINT", "VISTA_")) or o == "EmberVistaConfiguration":
        c = "vista资产/建图器"
    elif o.endswith("RegionBehavior"):
        c = "区域行为配置(GM)"
    elif o == "registerProseMirrorBlocks":
        c = "编辑器块名(GM)"
    else:
        c = "其余"
    cls[v] = c
    where[v] = o

cnt = collections.Counter(cls.values())
print(f"未覆盖 {len(missing)} 条，定性如下：\n")
for k, c in cnt.most_common():
    print(f"  {c:>5}  {k}")
rest = sorted(v for v, c in cls.items() if c == "其余")
byown = collections.Counter(where[v] for v in rest)
print(f"\n=== 「其余」{len(rest)} 条，按声明块 ===")
for o, c in byown.most_common(24):
    ex = [v for v in rest if where[v] == o][:5]
    print(f"  {c:>4}  {o:<32}{' · '.join(x[:22] for x in ex)[:74]}")
io.open('classify.json', 'w', encoding='utf-8').write(json.dumps(
    {"counts": dict(cnt), "rest": {v: where[v] for v in rest}}, ensure_ascii=False, indent=1))
