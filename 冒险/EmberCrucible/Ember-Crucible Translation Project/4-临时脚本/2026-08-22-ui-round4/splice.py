# -*- coding: utf-8 -*-
"""把 611 条装备族部件名接进 TOKEN_MAKER_PART_IDS。"""
import io, sys
sys.stdout.reconfigure(encoding='utf-8')
P = '1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs'
s = io.open(P, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'
tail = '  "Youthful": "少年感", "Youthful1": "少年感 1", "Youthful2": "少年感 2"' + nl + '};'
assert s.count(tail) == 1, s.count(tail)

block = io.open('4-临时脚本/2026-08-22-ui-round4/emit.txt', encoding='utf-8').read().rstrip('\n')
block = nl.join(block.split('\n'))

HEAD = nl.join([
'  "Youthful": "少年感", "Youthful1": "少年感 1", "Youthful2": "少年感 2",',
'',
'  // ══ 2026-08-22 第四轮｜装备族 611 条 ══════════════════════════════════════════',
'  // 上一轮把这块记成「仍缺 687，基本在装备族」。本轮把**分母重算了一遍**，因为那个数是',
'  // 拿图集当全集算的：图集是贴图，`templateLayer.parts` 才决定「这个部件会不会进选择器」。',
'  // 重算走的是**切片求值**（`4-临时脚本/2026-08-22-ui-round4/recon/slice_templates.mjs`）——',
'  // 整份 ember.mjs 装不进 Node（六层 stub 之后卡在 `HEXES[...].terrain`，与部件毫无关系），',
'  // 于是只把 53648~61260 行那一段（CHARACTER_ATLAS → cloneLayer → LAYERS$3 → 22 个模板',
'  // → `var templates=Object.freeze({...})`）切出来，用**真身** foundry.utils 求值。',
'  //   自证 A：切片首尾行逐字符等于写死的那两行（上游改行号会当场报错，不会静默切错段）',
'  //   自证 B：模板 id 集合 = 写死的 22 个，逐个相等',
'  //   自证 C：2579 个 id 全是 `<ns>/<layer>/<Part>` 三段，且逐个回**随包图集**（另一份文件、',
'  //           5649 帧、逐份对上已知真值）交叉核 —— 两个独立上游来源互校，不是自己验自己',
'  // ⇒ 真显示名全集 **1458** 条（末段去重），不是上一轮的 1512，也不是图集的 1535。',
'  //   本轮之前盖住 767（53%），缺 691；其中 **80 条是纯数字**（symbol 族的 10..82，',
'  //   上屏就是「10」，本来就不用翻）⇒ 真正要翻的 **611** 条，本块全部补齐。',
'  //   ⇒ 补完 1458/1458，装备族不再有英文。',
'  //',
'  // 译法怎么定的（三层，从严到宽，逐条核过语境）：',
'  //   ① **系统定译最高**：品质四档直接取 crucible-cn 的 `ITEM.Quality*` ——',
'  //      Shoddy 粗糙 / Standard 标准 / Fine 精良 / Superior 卓越。这四个词在本块出现 87 次，',
'  //      是最不该自己拍脑袋的一处（上游 `crucible/lang/en.json:1834` 那一族）。',
'  //   ② 其次是本表里已定过的同名段（48 个形态素）。',
'  //   ③ 再次是 glossary_ec（94 个命中）—— 但**六条套了就错**，逐条改判：',
'  //      `Shield`词表给「护盾术」那是**法术**（这里是盾牌）· `Point`给「岬」那是**地名裁决**',
'  //      R-point-cape（这里是帽子的尖顶）· `Split`给「分裂」（这里是开衩）·',
'  //      `Sticks`给「斯蒂克斯」专名（这里是一捆木棍）· `Water`给「水域」（这里是碗里的水）·',
'  //      `Alchemist`给「阿克图里安」（一眼串行的脏数据，这里是炼金术士）。',
'  //      ⇒ **「词表里有」不等于「这条能用」**，同一个英文词在不同语域本来就该分裂。',
'  //',
'  // ⚠ 拼串只出初稿，611 条**逐行读过一遍**才定稿（改动记在',
'  //   `4-临时脚本/2026-08-22-ui-round4/overrides.py`，连理由一起）。三类是规则必错的：',
'  //   两个中心词叠在一起（`ShieldKiteSteelStandard` 拼出「…盾牌鸢形盾」）·',
'  //   族名省了中心词（sleeve/pants/pauldron 三族靠图层兜底补「袖/裤/肩甲」，',
'  //   但末尾已是护甲名的 8 条不能再补，否则成「板甲肩甲」）·',
'  //   同一个词在这一族另有形制义（`HammerPole`＝长柄锤不是旗杆、`RingMail`＝环甲不是戒指、',
'   '  '  `Kettle`＝壶盔不是水壶、`Low`＝低领不是低帮）。',
'  // ⚠ 上表前三道机器核（`recon/collide.mjs`）：新键显示名互撞 0 · 撞发布中的表 0 ·',
'  //   同图层两个部件拿到同一个中文 0（唯一一处跨图层撞名是同一件长袍的上下段，已分开写）。',
'  // ⚠ 611 个键**逐个**回当前安装的上游语料查过字面量（`recon/litcheck.py`，529 份 / 10.2M 字符，',
'  //   纯 ASCII 加词边界，与面板 D 档同口径）：**查不到的 0 个** ⇒ 面板 miss 侧一条不涨。',
])
new = HEAD + nl + block + nl + '};'
io.open(P, 'w', encoding='utf-8', newline='').write(s.replace(tail, new))
print('已接入')
