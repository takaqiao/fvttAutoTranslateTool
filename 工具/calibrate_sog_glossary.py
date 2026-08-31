"""
SoG (肆季鬼志) 术语表校准工具

读取 glossary_sog_candidates.json 中的详细提取结果，
对存在多源/多变体的术语进行自动校准和统一，支持：

1. 纯脚本校准：基于投票、规则和 glossary.json 对齐
2. AI 校准：将有争议的术语送 AI 裁决最佳译名
3. 淘汰回收：检查被 _is_bad_term 过滤掉的条目中是否有可回收的

用法:
  python calibrate_sog_glossary.py [--config glossary_extract_config_sog.json] [--ai]
  python calibrate_sog_glossary.py             # 仅纯脚本校准
  python calibrate_sog_glossary.py --ai        # 加 AI 裁决
"""

import argparse
import json
import os
import re
import sys
import io
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Windows 控制台编码修复
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gbk"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ===========================================================================
# 工具函数
# ===========================================================================

def _contains_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

def _is_sentence(zh: str) -> bool:
    markers = re.compile(
        r"可以|能够|进行|使用|获得|承受|必须|如果|若是|持续|造成|对抗|"
        r"需要|试图|触发|处于|具有|你的|你会|你是|它们|该生物|"
        r"可为|也能|即便|就会|还会|不会|无法|应该|能让|可能|"
        r"一个|一次|一种|并且|或者|以及|因此|然而|尽管|"
        r"包括|字均|望为|但是|因为|所以|然后|检定以|"
        r"在.*时|在.*中|对.*的|为.*的|从.*中"
    )
    if markers.search(zh):
        return True
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", zh))
    return zh_chars > 14

def _is_bad_variant(en: str, zh: str) -> bool:
    """检查一个变体是否是坏的（句子碎片等）"""
    if not zh or not _contains_zh(zh):
        return True
    if zh[0] in "的了着":
        return True
    if "检定以" in zh:
        return True
    bad = re.compile(
        r"字均|望为|包括|但是|因为|所以|然后|来自|而且|不过|"
        r"的语言是|不死生物的|的运动或|灵魂将会|没有离开|当时|板宿|"
        r"你在.{1,6}技能|落入了|正在寻找|透过反常|代表柳岸|表中日期|"
        r"大成功或大失败|食物储备|当时适逢|运动或航海|"
        r"该效果的|该区域的|该生物的|该角色的"
    )
    if bad.search(zh):
        return True
    if _is_sentence(zh):
        return True
    # 太长不像术语
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", zh))
    if zh_chars > 12:
        return True
    return False


# ===========================================================================
# Phase A: 纯脚本校准
# ===========================================================================

def script_calibrate(terms_detail: list[dict], ref_glossary: dict) -> dict:
    """
    纯脚本校准逻辑：
    1. ref glossary 中已有的术语，优先采用 ref 译名
    2. 多变体投票：按出现次数选最佳，但过滤掉坏变体
    3. 特征术语对齐：如 "幽灵" vs "幽灵（特征）"，非特征语境取无括号版
    4. 统一同义异名
    """
    calibrated = {}
    changes = []
    ref_lower = {k.lower().strip(): v for k, v in ref_glossary.items()}

    # 已知的需要统一的映射（从分析中发现）
    FORCED_UNIFY = {
        # 这些是 ref glossary 中带 (特征) 后缀但 SoG 里应该用无后缀版
        # 不强制，而是根据上下文处理
    }

    for term in terms_detail:
        en = term["english"]
        lk = en.lower().strip()
        best_zh = term["chinese"]
        variants = term.get("zh_variants", {})

        # 1. 如果 ref glossary 有，优先对齐
        if lk in ref_lower:
            ref_zh = ref_lower[lk]
            ref_zh_str = ref_zh if isinstance(ref_zh, str) else ref_zh[0] if isinstance(ref_zh, list) else str(ref_zh)
            ref_base = re.sub(r"[（(].*?[）)]$", "", ref_zh_str).strip()
            is_trait_suffix = ref_zh_str.endswith("（特征）") or ref_zh_str.endswith("(特征)")

            if ref_zh_str in variants:
                # ref 版本在变体中直接出现
                if ref_zh_str != best_zh:
                    changes.append({
                        "english": en,
                        "old_zh": best_zh,
                        "new_zh": ref_zh_str,
                        "reason": "align_ref_glossary",
                    })
                    best_zh = ref_zh_str
            elif is_trait_suffix and ref_base == best_zh:
                # ref 是 "幽灵（特征）" 而 SoG 有 "幽灵" → 不变，SoG 无后缀版更合适
                pass
            elif ref_base in variants and ref_base != best_zh:
                # ref 是 "幽灵（特征）" 但 SoG 变体有 "幽灵"
                changes.append({
                    "english": en,
                    "old_zh": best_zh,
                    "new_zh": ref_base,
                    "reason": "align_ref_base",
                })
                best_zh = ref_base
            elif not is_trait_suffix and ref_zh_str != best_zh:
                # ref 版本不在变体中，但不是（特征）后缀 → 强制对齐 ref
                # 这种情况说明提取到的译名可能有误（如 Darklands→幽魂之森 应为 幽暗地域）
                changes.append({
                    "english": en,
                    "old_zh": best_zh,
                    "new_zh": ref_zh_str,
                    "reason": "force_ref_glossary",
                })
                best_zh = ref_zh_str

        # 2. 清理 best_zh 中可能残留的问题
        if _is_bad_variant(en, best_zh):
            # 尝试从变体中找一个好的
            for alt_zh, alt_occ in sorted(variants.items(), key=lambda x: -x[1]):
                if not _is_bad_variant(en, alt_zh):
                    changes.append({
                        "english": en,
                        "old_zh": best_zh,
                        "new_zh": alt_zh,
                        "reason": "fix_bad_primary",
                    })
                    best_zh = alt_zh
                    break

        # 3. 构建多义值
        multi = [best_zh]
        if variants:
            main_occ = variants.get(best_zh, 1)
            threshold = max(2, int(main_occ * 0.25))
            for alt_zh, alt_occ in sorted(variants.items(), key=lambda x: -x[1]):
                if alt_zh == best_zh:
                    continue
                if _is_bad_variant(en, alt_zh):
                    continue
                if alt_occ >= threshold:
                    multi.append(alt_zh)

        val = multi[0] if len(multi) == 1 else multi
        calibrated[en] = val

    return calibrated, changes


# ===========================================================================
# Phase B: AI 裁决
# ===========================================================================

AI_SYSTEM_PROMPT = """\
你是 PF2E TRPG 术语校准专家。你会收到一批有争议的中英术语对，
每个术语可能有多个中文翻译变体及其出现次数。

你的任务：
1. 选出最准确、最自然的**首选翻译**（放第一个）
2. 如果确实存在多义/多语境（如同一个英文在不同上下文有不同含义），
   可以保留多个翻译，按优先级排序
3. 明显错误的翻译直接淘汰
4. 注意 PF2E 术语惯例和天夏（Tian Xia）设定背景

输出要求：
- 严格 JSON 对象，key 为英文术语，value 为字符串（单义）或字符串数组（多义）
- 只输出有需要调整的条目。如果首选翻译无需改变，不需要输出
- 如果原来是单义但实际有多义需求，也输出

示例：
{
  "Society": ["社群", "社会"],
  "Diplomacy": "交涉",
  "Cerulean Teahouse": "青天茶馆"
}
"""

def ai_calibrate(disputed: list[dict], cfg: dict, ref_glossary: dict) -> dict:
    """用 AI 裁决有争议的术语"""
    if OpenAI is None:
        print("❌ 需要 openai: pip install openai")
        return {}

    api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ 未设置 API key")
        return {}

    client = OpenAI(
        api_key=api_key,
        base_url=cfg.get("openai_base_url", "https://api.openai.com/v1"),
    )
    model = cfg.get("model", "gpt-5.4")

    # 分批（每批最多 40 个术语）
    BATCH_SIZE = 40
    all_corrections = {}

    for i in range(0, len(disputed), BATCH_SIZE):
        batch = disputed[i:i + BATCH_SIZE]
        user_parts = []

        # 添加相关 ref glossary 参考
        ref_hints = []
        for item in batch:
            lk = item["english"].lower()
            for rk, rv in ref_glossary.items():
                if rk.lower() == lk:
                    rv_str = rv if isinstance(rv, str) else " | ".join(rv)
                    ref_hints.append(f"  {rk}: {rv_str}")
        if ref_hints:
            user_parts.append("【参考术语表】")
            user_parts.extend(ref_hints[:30])
            user_parts.append("")

        user_parts.append("【待裁决术语】")
        for item in batch:
            variants_str = ", ".join(
                f"{zh}({occ}次)" for zh, occ in sorted(item["variants"].items(), key=lambda x: -x[1])
            )
            user_parts.append(f"  {item['english']}: {variants_str}")

        user_parts.append("")
        user_parts.append("请裁决每个术语的最佳翻译，返回 JSON 对象。")

        try:
            resp = client.responses.create(
                model=model,
                instructions=AI_SYSTEM_PROMPT,
                input="\n".join(user_parts),
            )
            raw = (resp.output_text or "").strip()

            # 解析 JSON
            fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.I)
            if fence:
                text = fence.group(1)
            else:
                start = raw.find("{")
                end = raw.rfind("}")
                text = raw[start:end + 1] if start != -1 and end > start else "{}"

            corrections = json.loads(text)
            if isinstance(corrections, dict):
                all_corrections.update(corrections)
                print(f"  AI 批次 {i // BATCH_SIZE + 1}: {len(corrections)} 个校正")

        except Exception as e:
            print(f"  ⚠ AI 批次 {i // BATCH_SIZE + 1} 失败: {e}")

    return all_corrections


# ===========================================================================
# Phase C: 淘汰回收
# ===========================================================================

def recover_discarded(terms_detail: list[dict], calibrated: dict) -> list[dict]:
    """检查被过滤掉的变体中是否有可回收的好术语"""
    recovered = []
    cal_lower = {k.lower(): True for k in calibrated}

    for term in terms_detail:
        en = term["english"]
        if en.lower() not in cal_lower:
            # 这个术语整体被淘汰了，检查是否有好变体
            variants = term.get("zh_variants", {term["chinese"]: term.get("occurrences", 1)})
            for zh, occ in sorted(variants.items(), key=lambda x: -x[1]):
                if not _is_bad_variant(en, zh) and occ >= 2:
                    recovered.append({
                        "english": en,
                        "chinese": zh,
                        "occurrences": occ,
                        "reason": "recovered_from_filtered",
                    })
                    break

    return recovered


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SoG 术语表校准工具")
    parser.add_argument("--config", default="glossary_extract_config_sog.json", help="配置文件")
    parser.add_argument("--ai", action="store_true", help="启用 AI 裁决")
    parser.add_argument("--output", default="", help="输出术语表路径（默认覆盖 glossary_sog.json）")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ 配置文件不存在: {cfg_path}")
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    print("🔧 SoG 术语表校准工具")
    print(f"   AI 裁决: {'是' if args.ai else '否'}")

    # 加载 candidates 详细报告
    candidates_path = Path(cfg.get("output_candidates_json", "glossary_sog_candidates.json"))
    if not candidates_path.exists():
        print(f"❌ 候选报告不存在: {candidates_path}（请先运行 extract_sog_glossary.py）")
        return

    cand = json.loads(candidates_path.read_text(encoding="utf-8"))
    terms_detail = cand.get("terms_detail", [])
    print(f"加载 {len(terms_detail)} 个术语详情")

    # 加载参考术语表
    ref_glossary = {}
    ref_path = cfg.get("reference_glossary", "glossary.json")
    if ref_path:
        p = Path(ref_path)
        if p.exists():
            ref_glossary = json.loads(p.read_text(encoding="utf-8-sig"))
            if not isinstance(ref_glossary, dict):
                ref_glossary = {}
            print(f"参考术语表: {len(ref_glossary)} 条")

    # ── Phase A: 纯脚本校准 ──
    print(f"\n{'='*55}")
    print("Phase A: 纯脚本校准")
    print(f"{'='*55}")

    calibrated, script_changes = script_calibrate(terms_detail, ref_glossary)
    print(f"校准后术语: {len(calibrated)}")
    print(f"脚本修正: {len(script_changes)} 处")

    if script_changes:
        print("\n脚本修正示例 (前 20):")
        for c in script_changes[:20]:
            print(f"  [{c['reason']}] {c['english']}: {c['old_zh']} → {c['new_zh']}")

    # ── Phase B: AI 裁决（可选）──
    ai_changes = {}
    if args.ai:
        print(f"\n{'='*55}")
        print("Phase B: AI 裁决争议术语")
        print(f"{'='*55}")

        # 找出有争议的术语（变体 >= 3 且至少两个变体都有 >= 2 次出现）
        disputed = []
        for term in terms_detail:
            variants = term.get("zh_variants", {})
            if not variants or len(variants) < 2:
                continue
            # 计算有意义变体数
            good_variants = {zh: occ for zh, occ in variants.items()
                            if not _is_bad_variant(term["english"], zh) and occ >= 2}
            if len(good_variants) >= 2:
                disputed.append({
                    "english": term["english"],
                    "current": term["chinese"],
                    "variants": good_variants,
                })

        print(f"争议术语: {len(disputed)} 个")

        if disputed:
            ai_changes = ai_calibrate(disputed, cfg, ref_glossary)
            print(f"AI 校正: {len(ai_changes)} 个")

            # 应用 AI 校正
            for en, new_val in ai_changes.items():
                if en in calibrated:
                    # 如果 AI 返回 list，过滤掉坏变体
                    if isinstance(new_val, list):
                        cleaned = [v for v in new_val if not _is_bad_variant(en, v)]
                        if cleaned:
                            calibrated[en] = cleaned[0] if len(cleaned) == 1 else cleaned
                        # 如果全部坏，不应用
                    elif isinstance(new_val, str) and not _is_bad_variant(en, new_val):
                        calibrated[en] = new_val

    # ── Phase C: 淘汰回收 ──
    print(f"\n{'='*55}")
    print("Phase C: 淘汰回收")
    print(f"{'='*55}")

    recovered = recover_discarded(terms_detail, calibrated)
    print(f"可回收术语: {len(recovered)}")
    if recovered:
        for r in recovered[:20]:
            print(f"  {r['english']}: {r['chinese']} (出现{r['occurrences']}次)")
        # 加入校准结果
        for r in recovered:
            lk = r["english"].lower()
            if lk not in {k.lower() for k in calibrated}:
                calibrated[r["english"]] = r["chinese"]

    # ── 统计 ──
    total = len(calibrated)
    multi_count = sum(1 for v in calibrated.values() if isinstance(v, list))
    single_count = total - multi_count

    print(f"\n{'='*55}")
    print(f"最终术语表: {total} 条")
    print(f"  单义: {single_count}")
    print(f"  多义: {multi_count}")
    print(f"{'='*55}")

    # 多义术语示例
    multi_items = [(k, v) for k, v in calibrated.items() if isinstance(v, list)]
    if multi_items:
        print(f"\n多义术语 (前 30):")
        for en, vals in sorted(multi_items, key=lambda x: x[0])[:30]:
            print(f"  {en}: {' | '.join(vals)}")

    # ── 输出 ──
    output_path = Path(args.output) if args.output else Path(cfg.get("output_json", "glossary_sog.json"))
    sorted_glossary = dict(sorted(calibrated.items(), key=lambda x: x[0].lower()))
    output_path.write_text(json.dumps(sorted_glossary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 术语表: {output_path} ({total} 条, 其中 {multi_count} 多义)")

    # 输出校准报告
    report_path = output_path.with_name(output_path.stem + "_calibration_report.json")
    report = {
        "meta": {
            "total_terms": total,
            "single_meaning": single_count,
            "multi_meaning": multi_count,
            "script_changes": len(script_changes),
            "ai_changes": len(ai_changes),
            "recovered": len(recovered),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "script_changes": script_changes,
        "ai_changes": {k: v for k, v in ai_changes.items()} if ai_changes else {},
        "recovered_terms": recovered,
        "multi_meaning_terms": {k: v for k, v in sorted_glossary.items() if isinstance(v, list)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 校准报告: {report_path}")


if __name__ == "__main__":
    main()
