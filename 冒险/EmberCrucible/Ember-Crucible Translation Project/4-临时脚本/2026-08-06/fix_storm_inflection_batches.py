# -*- coding: utf-8 -*-
"""生成「Storm 改名残留 + Inflection 误译为词缀 + 若干硬错误」的修订批次。

背景：
  - 上游把 `Rune: Lightning` 改名为 `Rune: Storm`，port_orphans.py 把译文搬到了新路径，
    但**译文的值**还写着「闪电 / Rune: Lightning」。决议记录已定 `Rune: Storm`→符文：风暴。
  - `Inflection`（屈折）在 talent 包里被译成「词缀」，与 `Affix`（词缀）撞名。
  - 顺带修三个硬错误：
      · rules「符文」页的 @Embed 指向已不存在的 `runeLightning000`（链接是坏的）
      · Surgeweaver 的 Shocked 持续时间英文是 3 Rounds，译文写成 1 轮
      · energize 的 six-foot radius 被译成「六码 / 六码尺」（码≠英尺）
      · Storm Proficiency 译文把 @Action[...] 链接改写成了 <strong>充能</strong>，链接丢了

每条替换都是 (路径, 旧片段, 新片段)，只允许命中一次；命中数不符即报错退出，
避免"看起来跑成功了其实没改到"。输出按 pack 分文件，交给 apply_translations.py --force 落盘。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CN = ROOT / "2-Crucible汉化插件" / "compendium" / "cn"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

T = "crucible.talent.json"
P = "crucible.playtest.json"
G = "crucible.pregens.json"
S = "crucible.summons.json"
X = "crucible.taxonomy.json"
R = "crucible.rules.json"

PT = "entries.Playtest 1 - The Ring of Valor.actors"

# (pack, path, old, new)
EDITS = [
    # ── Inflection → 屈折，并与 affixes 包的 adjective 用词统一 ──────────────
    (T, "entries.Inflection: React.name", "词缀：反应", "屈折：反应"),
    (T, "entries.Inflection: Reshape.name", "词缀：重塑", "屈折：重塑"),
    (T, "entries.Inflection: Compose.name", "词缀：作曲", "屈折：编构"),
    (T, "entries.Inflection: Determine.name", "词缀：判定", "屈折：限定"),
    (T, "entries.Inflection: Elude.name", "词缀：闪避", "屈折：遁避"),
    (T, "entries.Inflection: Extend.name", "词缀：延展", "屈折：延展"),
    (T, "entries.Inflection: Negate.name", "词缀：否定", "屈折：否定"),
    (T, "entries.Inflection: Pull.name", "词缀：拉拽", "屈折：拉拽"),
    (T, "entries.Inflection: Quicken.name", "词缀：迅捷化", "屈折：迅捷"),
    (T, "entries.Inflection: Push.name", "屈折：推开", "屈折：推挤"),
    (P, f"{PT}.Harbinger of Disease.items.Inflection: Pull.name", "词缀：拉拽", "屈折：拉拽"),

    # ── Rune: Storm 名称残留 ────────────────────────────────────────────────
    (T, "entries.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),
    (T, "entries.Storm Proficiency.name", "闪电熟练度 Lightning Proficiency", "风暴熟练度 Storm Proficiency"),
    (P, f"{PT}.Agnath.items.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),
    (P, f"{PT}.Eliorwen.items.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),
    (P, f"{PT}.Fizzit.items.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),
    (G, "entries.Agnath.items.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),
    (G, "entries.Eliorwen.items.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),
    (G, "entries.Fizzit.items.Rune: Storm.name", "符文：闪电 Rune: Lightning", "符文：风暴 Rune: Storm"),

    # ── Rune: Storm 描述：符文名 + Electricity 伤害类型统一为「电力」──────────
    (T, "entries.Rune: Storm.description", "闪电符文支配电荷", "风暴符文支配电荷"),
    (T, "entries.Rune: Storm.description", "闪电符文使用 <strong>智力</strong>", "风暴符文使用 <strong>智力</strong>"),
    (T, "entries.Rune: Storm.description", "造成 <strong>闪电</strong> 伤害", "造成 <strong>电力</strong> 伤害"),
    (T, "entries.Rune: Storm.actions.energize.description", "你对闪电符文的掌控", "你对风暴符文的掌控"),
    (T, "entries.Rune: Storm.actions.energize.description", "在你当前位置以六英尺半径", "在你当前位置以 6 英尺为半径"),
    (T, "entries.Storm Proficiency.description", "你熟练于运用闪电符文", "你熟练于运用风暴符文"),
    (T, "entries.Storm Proficiency.description", "此天赋最终还将为<strong>充能 </strong>动作提供一次强化",
        "此天赋最终还将为 @Action[Compendium.crucible.talent.Item.runeStorm0000000 energize] 动作提供一次强化"),
    (T, "entries.Surgeweaver.description", "你极其擅长编织闪电符文", "你极其擅长编织风暴符文"),
    (T, "entries.Surgeweaver.description", "电击效果会持续 <strong>1 轮</strong>", "电击效果会持续 <strong>3 轮</strong>"),
    (T, "entries.Surgeweaver.description", "一半的<strong>闪电</strong>伤害", "一半的<strong>电力</strong>伤害"),
    (T, "entries.Rune: Earth.description", "它与混乱的<strong>闪电</strong>符文相对",
        "它与混乱的<strong>风暴</strong>符文相对"),
    (T, "entries.Gesture: Sense.description", "<strong>闪电</strong>：构装体和风暴元素",
        "<strong>风暴</strong>：构装体和风暴元素"),
    # Sense：EN 没有给手势名加粗，Presence 应为「存在」；末尾那段「试玩测试说明」英文早已删除
    (T, "entries.Gesture: Sense.description", "<p><strong>感知</strong>手势会随<strong>灵识</strong>成长",
        "<p>感知手势会随<strong>存在</strong>成长"),
    (T, "entries.Gesture: Sense.description",
        "</ul><h3 class=\"divider\">试玩测试说明</h3><p>目前尚无针对该法术临时赋予的感知的自动化支持，"
        "但未来可通过一种侦测模式提供此类自动化：当法术被维持时，它会暂时向施法者揭示相应类型的生物。</p>", "</ul>"),

    (P, f"{PT}.Agnath.items.Rune: Storm.description", "符文：闪电支配电荷", "风暴符文支配电荷"),
    (P, f"{PT}.Agnath.items.Rune: Storm.description", "符文：闪电以 <strong>智力</strong>", "风暴符文以 <strong>智力</strong>"),
    (P, f"{PT}.Agnath.items.Rune: Storm.description", "造成 <strong>闪电</strong>伤害", "造成 <strong>电力</strong>伤害"),
    (P, f"{PT}.Agnath.items.Rune: Storm.actions.energize.description", "你对闪电符文的命令", "你对风暴符文的掌控"),
    (P, f"{PT}.Agnath.items.Rune: Storm.actions.energize.description", "在你当前位置周围六码半径内", "在你当前位置周围 6 英尺半径内"),
    (P, f"{PT}.Eliorwen.items.Rune: Storm.description", "闪电符文支配电荷", "风暴符文支配电荷"),
    (P, f"{PT}.Eliorwen.items.Rune: Storm.description", "闪电符文以 <strong>智力</strong>", "风暴符文以 <strong>智力</strong>"),
    (P, f"{PT}.Eliorwen.items.Rune: Storm.description", "造成 <strong>电击</strong>伤害", "造成 <strong>电力</strong>伤害"),
    (P, f"{PT}.Eliorwen.items.Rune: Storm.actions.energize.description", "你对闪电符文的掌控", "你对风暴符文的掌控"),
    (P, f"{PT}.Eliorwen.items.Surgeweaver.description", "你极其擅长编织闪电符文", "你极其擅长编织风暴符文"),
    (P, f"{PT}.Eliorwen.items.Surgeweaver.description", "电击效果持续<strong>1 轮</strong>", "电击效果持续<strong>3 轮</strong>"),
    (P, f"{PT}.Eliorwen.items.Surgeweaver.description", "一半的<strong>闪电</strong>伤害", "一半的<strong>电力</strong>伤害"),
    (P, f"{PT}.Fizzit.items.Rune: Storm.description", "闪电符文掌管电荷", "风暴符文掌管电荷"),
    (P, f"{PT}.Fizzit.items.Rune: Storm.description", "闪电符文以<strong>智力</strong>", "风暴符文以<strong>智力</strong>"),
    (P, f"{PT}.Fizzit.items.Rune: Storm.description", "造成<strong>电击</strong>伤害", "造成<strong>电力</strong>伤害"),
    (P, f"{PT}.Fizzit.items.Rune: Storm.actions.energize.description", "你对闪电符文的掌控", "你对风暴符文的掌控"),
    (P, f"{PT}.Fizzit.items.Rune: Storm.actions.energize.description", "在你当前位置周围六码半径内", "在你当前位置周围 6 英尺半径内"),
    (P, f"{PT}.Fizzit.items.Surgeweaver.description", "你极其擅长编织闪电符文", "你极其擅长编织风暴符文"),
    (P, f"{PT}.Fizzit.items.Surgeweaver.description", "电击效果持续<strong>1 轮</strong>", "电击效果持续<strong>3 轮</strong>"),
    (P, f"{PT}.Fizzit.items.Surgeweaver.description", "一半的<strong>闪电</strong>伤害", "一半的<strong>电力</strong>伤害"),
    (P, f"{PT}.Kagura.items.Rune: Earth.description", "它与混乱的<strong>闪电</strong>符文相对",
        "它与混乱的<strong>风暴</strong>符文相对"),

    (G, "entries.Agnath.items.Rune: Storm.description", "闪电符文掌管电荷", "风暴符文掌管电荷"),
    (G, "entries.Agnath.items.Rune: Storm.description", "闪电符文以 <strong>智力</strong>", "风暴符文以 <strong>智力</strong>"),
    # 译文比英文多包了一层 <strong>伤害</strong>，顺手改回英文的标记结构
    (G, "entries.Agnath.items.Rune: Storm.description",
        "造成 <strong>电击</strong> <strong>伤害</strong> 至 <strong>生命值</strong>",
        "对 <strong>生命值</strong> 造成 <strong>电力</strong> 伤害"),
    (G, "entries.Agnath.items.Rune: Storm.actions.energize.description", "你对闪电符文的命令", "你对风暴符文的掌控"),
    (G, "entries.Agnath.items.Rune: Storm.actions.energize.description", "在六码尺半径内", "在 6 英尺半径内"),
    (G, "entries.Eliorwen.items.Rune: Storm.description", "闪电符文掌管电荷", "风暴符文掌管电荷"),
    (G, "entries.Eliorwen.items.Rune: Storm.description", "闪电符文以<strong>智力</strong>", "风暴符文以<strong>智力</strong>"),
    (G, "entries.Eliorwen.items.Rune: Storm.description", "造成<strong>闪电</strong>伤害", "造成<strong>电力</strong>伤害"),
    (G, "entries.Eliorwen.items.Rune: Storm.actions.energize.description", "你对闪电符文的掌控", "你对风暴符文的掌控"),
    (G, "entries.Eliorwen.items.Surgeweaver.description", "你极其擅长编织闪电符文", "你极其擅长编织风暴符文"),
    (G, "entries.Eliorwen.items.Surgeweaver.description", "电击效果持续<strong>1 轮</strong>", "电击效果持续<strong>3 轮</strong>"),
    (G, "entries.Eliorwen.items.Surgeweaver.description", "一半数值的<strong>闪电</strong>伤害", "一半数值的<strong>电力</strong>伤害"),
    (G, "entries.Fizzit.items.Rune: Storm.description", "闪电符文掌管电荷", "风暴符文掌管电荷"),
    (G, "entries.Fizzit.items.Rune: Storm.description", "闪电符文使用<strong>智力</strong>", "风暴符文使用<strong>智力</strong>"),
    (G, "entries.Fizzit.items.Rune: Storm.description",
        "造成<strong>电击</strong><strong>伤害</strong>至<strong>生命值</strong>",
        "对<strong>生命值</strong>造成<strong>电力</strong>伤害"),
    (G, "entries.Fizzit.items.Rune: Storm.actions.energize.description", "你对闪电符文的掌控", "你对风暴符文的掌控"),
    (G, "entries.Kagura.items.Rune: Earth.description", "它与混沌的<strong>闪电</strong>符文相对",
        "它与混沌的<strong>风暴</strong>符文相对"),

    # ── 元素/分类法描述 ─────────────────────────────────────────────────────
    (S, "entries.Storm Sprite.taxonomy.description", "与闪电元素对应", "与风暴元素对应"),
    (S, "entries.Storm Visitor.taxonomy.description", "与闪电元素对应", "与风暴元素对应"),
    (S, "entries.Earth Visitor.items.Rune: Earth.description", "它与混乱的闪电符文相对立", "它与混乱的风暴符文相对立"),
    (X, "entries.Storm Elemental.description", "与闪电元素相对应", "与风暴元素相对应"),

    # ── rules：坏掉的 @Embed + 符文表 + Inflection 术语 + 机翻残留 ────────────
    (R, "entries.Spellcraft.pages.Runes.text", "runeLightning000", "runeStorm0000000"),
    (R, "entries.Spellcraft.pages.Runes.text", "<tr><td><p>闪电</p></td><td><p>混沌</p></td>", "<tr><td><p>风暴</p></td><td><p>混沌</p></td>"),
    (R, "entries.Spellcraft.pages.Runes.text", "<td><p>强酸</p></td><td><p>反射</p></td><td><p>闪电</p></td>",
        "<td><p>强酸</p></td><td><p>反射</p></td><td><p>风暴</p></td>"),
    (R, "entries.Spellcraft.pages.Runes.text", "<tr><td><p>闪电</p></td><td><p>构装体和风暴元素</p></td></tr>",
        "<tr><td><p>风暴</p></td><td><p>构装体和风暴元素</p></td></tr>"),
    (R, "entries.Spellcraft.pages.Runes.text", "<th><p>魅力符文</p></th>", "<th><p>存在符文</p></th>"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text", "<h2 class=\"divider\">精妙构图</h2>", "<h2 class=\"divider\">精妙编构</h2>"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text", "<li><p><strong>词缀</strong>代表进一步的变化或修饰",
        "<li><p><strong>屈折</strong>代表进一步的变化或修饰"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text", "可以选择包含一个词缀（形容词/副词）", "可以选择包含一个屈折（形容词/副词）"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text", "扎拉贾还习得了 <strong>加速</strong> 词缀",
        "扎拉贾还习得了 <strong>迅捷</strong> 屈折"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text", "和 <strong>闪电</strong> 符文", "和 <strong>风暴</strong> 符文"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text",
        "扎拉贾最喜欢的法术是 <strong>加速 箭头 of 火焰</strong>", "扎拉贾最喜欢的法术是 <strong>迅捷的火焰之箭头</strong>"),
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text",
        "向一只俯冲的蝎狮施放 <strong>加速 箭头 of 火焰</strong> 时", "向一只俯冲的蝎狮施放 <strong>迅捷的火焰之箭头</strong> 时"),
    # 英文这里的 Critical Hit 没有加粗，译文多加了一对 <strong>
    (R, "entries.Spellcraft.pages.Spellcraft Overview.text", "同时也是一次<strong>暴击</strong>", "同时也是一次暴击"),
    # 页面标题：Turn Order 被译成「先攻顺序」，与 Initiative 重复
    (R, "entries.Combat.pages.Initiative and Turn Order.name", "先攻与先攻顺序 Initiative and Turn Order",
        "先攻与回合顺序 Initiative and Turn Order"),
]


def get_at(node, dotted):
    for part in dotted.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def main():
    packs = {}
    for pack in {e[0] for e in EDITS}:
        packs[pack] = json.loads((CN / pack).read_text(encoding="utf-8"))

    working = defaultdict(dict)   # pack -> path -> current value
    errors = []
    for pack, path, old, new in EDITS:
        # rules 的页面标题存在 `<page>.name` 下，这里允许直接给页面路径
        cur = working[pack].get(path)
        if cur is None:
            cur = get_at(packs[pack], path)
            if cur is None:
                cur = get_at(packs[pack], path + ".name")
                if cur is not None:
                    path = path + ".name"
                    cur = working[pack].get(path, cur)
        if not isinstance(cur, str):
            errors.append(f"{pack} :: {path} —— 路径不存在或不是字符串")
            continue
        n = cur.count(old)
        if n != 1:
            errors.append(f"{pack} :: {path} —— 片段命中 {n} 次（应为 1）: {old[:40]}")
            continue
        working[pack][path] = cur.replace(old, new)

    if errors:
        print("以下替换未能唯一命中，已全部中止：")
        for e in errors:
            print("  " + e)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for pack, edits in working.items():
        # apply_translations.py 的路径是相对 entries/ 的（folders 用 "(folders)" 前缀）
        edits = {k[len("entries."):] if k.startswith("entries.") else k: v
                 for k, v in edits.items()}
        p = OUT / f"fix.{pack}"
        p.write_text(json.dumps(edits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {p}  ({len(edits)} 条)")
        total += len(edits)
    print(f"共 {total} 条待回写（{len(EDITS)} 处替换）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
