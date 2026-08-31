# Book 2 汉化管线

## 产物

| 文件 | 说明 |
|---|---|
| `Season of Ghosts GM Guide - Book 2_zh.docx` | 中文 docx，保留全部图片、格式、超链接、列表、底色 |
| `fvtt-JournalEntry-Season-of-Ghosts-Book2-zh.json` | Foundry journal，10 页富 HTML |
| `book2_images/` | 从 docx 抽出的 19 张图（已拷贝到 `FoundryVTT/Data/assets/sog-gmguide-book2/`）|
| `book2_terms.md` | 本次锁定的术语表，Book 3 直接沿用 |

## 导入 Foundry

JSON 直接导入即可。图片已就位于 `Data/assets/sog-gmguide-book2/`；若要换路径，改 `build_fvtt_book2.py` 顶部的 `IMG_PREFIX` 后重跑。

## 重跑

```
python extract_book2.py      # 英文 docx -> book2_segments.json + book2_images/
python translate_book2.py    # + book2_zh.json -> 中文 docx
python build_fvtt_book2.py   # + book2_zh.json -> journal JSON
```

`book2_zh.json` 是译文本体：键为段落序号，值为该段各 chunk 的译文，长度必须与
`book2_segments.json` 里该段的 chunk 数完全一致（`translate_book2.py` 会校验并在不匹配时报错退出）。

## 与 Book 1 管线的区别

Book 1 的 `translate_mm.py` 式写回把整段塞进 `runs[0]`，超链接是独立的 `w:hyperlink` 元素、
不在 `p.runs` 里，于是链接文字被甩到段尾 —— Book 1 中文 docx 第 8、13 段就是这么坏掉的。

本管线按 XML 文档顺序遍历 `w:r` 与 `w:hyperlink`，逐 chunk 写回，链接锚文本留在原位。
`build_fvtt_book1.py` 输出的是纯 `<p>` 转义文本；本管线输出
`<strong>` / `<em>` / `<s>` / `<span style="background-color">` / `<a>` / 嵌套 `<ul>` / `<img>`。

## 文档内交叉引用

原 docx 有 12 处内部锚点（「见 侧边栏 —— 作祟机制」这类）。它们的目标**全部**落在别的 H1 页上，
因此裸 `#anchor` 在 Foundry 里必然失效。改为发 Foundry 相对 UUID 链接
`@UUID[.<pageId>#<headingId>]{标签}`：导入后自动解析并跳到目标页。

Foundry 的 `_makeHeadingNode` 优先取 `heading.id` 作为 TOC slug，所以标题上写死的
`id="h-…"` 与链接里的 fragment 能精确对上，中文标题也不会被 `slugify({strict:true})` 洗成空串。

`_4kpk55o53n30`（侧边栏 —— 作祟机制，被引用 4 次）的书签在 Google Docs 导出时丢了，
在 `build_fvtt_book2.py` 的 `EXTRA_ANCHORS` 里手工指向「❗ 复杂危害」H1。

---

# 压制作祟 Suppress Haunt（物品 + 宏）

| 文件 | 说明 |
|---|---|
| `fvtt-Item-suppress-haunt-*.json` | action 物品，4 条 Note 规则挂判定结果文字 |
| `fvtt-Macro-suppress-haunt-*.json` | 宏，`command` 由下面的 .js 生成 |
| `suppress_haunt_macro.js` | 宏脚本本体（改这个） |
| `sync_macro.py` | 把 .js 注入宏 JSON 的 `command`，注入前跑 `node --check` |

改脚本后跑 `python sync_macro.py` 重新注入。

## 联动关系

宏发出的检定串带 `options:action:suppress-haunt`，物品那 4 条 Note 规则的
`predicate: ["action:suppress-haunt"]` 据此匹配。**物品必须在掷骰玩家的角色卡上**，
否则只会得到一次普通技能检定，不会附带大成功／成功／失败／大失败的结果文字。

`system.slug` 固定为 `suppress-haunt`。PF2e 只在 slug 为空时才从 name 反推，
所以文档名可以随便改中文，不会破坏 predicate。

## 不能动的字符串

- 物品：`system.slug`、`rules[].predicate` / `selector` / `outcome`、`traits.value`
- 宏：`value="religion"` 之类的技能 slug、表单元素 id/name、DialogV2 的 `action` 键、
  `traits:haunt,hazard`、`options:action:suppress-haunt`

## 一条容易踩的坑

聊天消息只给 `speaker: { alias }`，**绝不能给 `speaker.actor`**。
pf2e 的 `resolveActorAndItemFromHTML` 把 `message.actor` 当作内联检定掷骰者的回退，
一旦把危害绑成 speaker，玩家没选中自己 token 时点检定就会变成危害自己掷。
