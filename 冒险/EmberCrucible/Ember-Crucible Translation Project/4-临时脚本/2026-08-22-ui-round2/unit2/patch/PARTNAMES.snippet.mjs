/* ============================================================================
 * 【预备补丁 · 本轮未上线，等主控拍板】指示物制作器的**部件显示名**表
 * ----------------------------------------------------------------------------
 * 落法：把下面这张表放进 `1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs`，
 *       ① 并进 `TOKEN_MAKER_WINDOW_UI`（作用域只在那两个窗口，实测收得住）；
 *       ② 在 `SELFCHECK_TABLES` 里按 **`{ table: TOKEN_MAKER_PARTS, kind: "composed" }`**
 *          登记 —— 键是 `getLayerChoicesV2()` 在运行时用
 *          `partId.split("/").at(-1).replace(/(?<!^)([A-Z1-9])/g, " $1")`
 *          拼出来的，上游文本里**没有**这个字面量。按 `literal` 登记会让面板 D 档的
 *          `missDistinct` 整片上涨、**主闸当场红**（这是实测过的形态，不是推断）。
 *
 * ⚠⚠ 上线的代价（必须与主控一起改，两份文件都不归 1-Ember汉化插件 的作者）：
 *     `kind: "composed"` 的键计入 `uncheckedRaw`，而 `max.uncheckedRaw` 现值 **186** 是**上限**。
 *     加 N 键 ⇒ uncheckedRaw = 186 + N ⇒ **主闸红在 max.uncheckedRaw**。
 *     本表 N = 110 ⇒ 需把 `max.uncheckedRaw` 186 → **296**、
 *     `max.uncheckedDistinct` 164 → **274**、`min.registeredRaw` 1708 → **1818**、
 *     `min.registeredDistinct` 1104 → **1214**（以上为算出的预期值，**上线前必须用
 *     `selfcheck_panel_runner.mjs` 现跑一遍复核**，别照抄）。
 *     两份文件：`5-其他内容/RESOLUTIONS.assertions.json` 与
 *     `3-常用脚本/qa/assert_resolutions.py` 的 `PAYLOAD_FLOORS`。
 *
 * 覆盖账（2026-08-22 实测）：全库部件 id **1356** 条 → 唯一显示名 **1356** 条。
 *   · 已被 TOKEN_MAKER_UI 顺带覆盖 **18** 条（与颜色槽 / 体格 / 图层名同串）；
 *   · 本表备好 **110** 条（面部头部一族 12 个图层：ears · tail · eyebrows · mouth ·
 *     teeth · beard · mane · marks · scales · fur · jaw · horns，该族 112 条里
 *     `Cheeks` 已覆盖、`Heavy` 故意不加）；
 *   · **仍缺 1228 条**（hair 97 / head 121 / eyes 61 / face 74 / 装备一族等）。
 *
 * ⚠ `Heavy` **故意不加**：既有**体格**键 `Heavy` = 壮硕，而 tail/arm/foot/head/torso
 *   都有叫 `Heavy` 的部件（正确译法是「粗重」）。同一张扁平表一个键只能有一个值 ——
 *   加进去等于把体格那一栏的「壮硕」改坏。全库 1356 条里与既有键同串的共 18 条，
 *   逐条核过，只有 `Heavy` 与 `Lithe`（体格「纤瘦」）是**真的异义**，其余 16 条同串同义。
 *   ⇒ 若将来要把 1356 条全上，正解不是继续堆扁平表，而是**按图层查表**：
 *      `templates/applications/token-maker/layers.hbs:27` 的
 *      `<div class="choice layer" data-layer-id="{{layer.layerId}}">` 上带着 layerId，
 *      渲染钩子里读得到 —— 那是一条**新增面**，要主控裁。
 * ========================================================================== */
const TOKEN_MAKER_PARTS = {
  "Alert": "警觉", "Balanced": "齐整", "Biter": "利齿", "Bladed": "刃状",
  "Braided": "编辫", "Braids Many": "多股辫", "Braids Thick": "粗辫", "Braids Two": "双辫",
  "Brash": "张扬", "Broad": "宽阔", "Center": "居中", "Centered": "对中",
  "Chops 1": "颊须 1", "Chops 2": "颊须 2", "Circlet": "环冠", "Clean": "整洁",
  "Combined": "复合", "Comforting": "温和", "Complex": "繁复", "Contour": "弧线",
  "Corners": "折角", "Crazy": "狂乱", "Crown": "冠状", "Determined": "坚毅",
  "Divergent": "分岔", "Dots": "点状", "Down": "向下", "Droopy": "耷拉",
  "Feral": "野性", "Friendly": "亲和", "Gentle": "柔和", "Gnarly": "虬结",
  "Hackles": "竖立", "Hanging": "垂落", "Horned": "有角", "Large": "大型",
  "Layered": "层叠", "Lines": "线纹", "Long Curled": "长卷", "Long Twirled": "长螺旋",
  "Long Whippy": "长鞭", "Lower": "下段", "Majestic": "威仪", "Medium": "中型",
  "Membrane": "膜翼", "Nubbins": "细突角", "Nubs": "短突角", "Nubs 1": "短突 1",
  "Nubs 2": "短突 2", "Open": "张口", "Ornate": "华丽", "Out": "外张",
  "Overgrowth": "蔓生", "Perky": "精神", "Plated": "甲片", "Ponytail": "马尾",
  "Raised": "隆起", "Ram": "盘羊角", "Reptilian": "爬行类", "Restrained": "收敛",
  "Scaled": "覆鳞", "Sharp": "锐利", "Short": "短", "Short Curved": "短弯",
  "Simple": "简约", "Small": "小型", "Smaller": "偏小", "Smooth": "光滑",
  "Smooth 1": "光滑 1", "Smooth 2": "光滑 2", "Smooth 3": "光滑 3",
  "Smooth 4": "光滑 4", "Smooth 5": "光滑 5", "Soft": "柔软", "Spiked": "尖刺",
  "Spikes Large 1": "大棘刺 1", "Spikes Large 2": "大棘刺 2",
  "Spikes Small 1": "小棘刺 1", "Spikes Small 2": "小棘刺 2",
  "Spikey": "尖刺状", "Spined": "带棘", "Striking": "醒目", "Stubs": "残角",
  "Styled": "修饰", "Swept": "后掠", "Swirled": "螺旋",
  "Tendrils Large 1": "大触须 1", "Tendrils Large 2": "大触须 2",
  "Tendrils Small 1": "小触须 1", "Tendrils Small 2": "小触须 2",
  "Thick": "浓密", "Tied Up": "束起", "Tiered": "分层", "Toothy": "满口牙",
  "Trim": "修短", "Trio": "三叉", "Tusked": "獠牙", "Twisted": "扭曲",
  "Unbound": "松散", "Up": "向上", "Vine": "藤蔓",
  "Whiskers Down 1": "下垂须 1", "Whiskers Down 2": "下垂须 2",
  "Whiskers Flat 1": "平展须 1", "Whiskers Flat 2": "平展须 2",
  "Whiskers Up 1": "上翘须 1", "Whiskers Up 2": "上翘须 2",
  "Wild": "狂野", "Woody": "木质", "Youngling": "幼体"
};
export { TOKEN_MAKER_PARTS };
