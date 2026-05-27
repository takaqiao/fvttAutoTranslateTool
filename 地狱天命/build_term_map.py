"""Build the AP-specific term map (state/02_term_map.json + .md).

Sources:
- reference/prequel_term_pairs.json  (307 entries: SEED + mined from 1.0 PDF + c71c1ec7)

The SRD canonical TM (38K terms across pf2_cn / pf2e_compendium / wiki) lives at
翻译流程/tm_cache/tm_3source.json — too large to load into per-session context.
That file is queried on-demand via Bash grep / lookup_term.py during translation.

Output schema (state/02_term_map.json):
{
  "<English>": {
    "zh": "<Chinese>",
    "tier": 1|2|3|4|5,        // 1=pf2_cn, 2=pf2e_compendium, 3=prequel/seed/AP-specific,
                              // 4=wiki, 5=self-create
    "frozen": true|false,     // true after ≥2 occurrences or seed; false=tentative
    "sources": ["seed", "1.0初校版", "c71c1ec7", ...],
    "first_seen_page": null,  // filled during translation
    "occurrences": 0          // incremented during translation
  }
}

Frozen rules:
- SEED entries → frozen (curated by human/Claude during planning)
- 1.0初校版 + c71c1ec7 both confirm same Chinese → frozen
- Single-source mining → tentative (promoted after first translation occurrence)
"""
import json
import sys
import io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
PREQUEL = HERE / "reference" / "prequel_term_pairs.json"
STATE = HERE / "state"
OUT_JSON = STATE / "02_term_map.json"
OUT_MD = STATE / "02_term_map.md"


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(PREQUEL.read_text(encoding="utf-8"))
    print(f"[in] {len(pairs)} pairs from prequel mining")

    out = {}
    for en, v in pairs.items():
        sources = v["sources"]
        candidates = v["candidates"]
        zh = v["zh"]

        if "seed" in sources:
            frozen = True
            tier = 3
        elif "1.0初校版" in sources and "c71c1ec7" in sources:
            # Confirmed by both prequel translations
            frozen = True
            tier = 3
        elif sum(candidates.values()) >= 2:
            # Multiple occurrences in same source
            frozen = True
            tier = 3
        else:
            frozen = False
            tier = 3

        out[en] = {
            "zh": zh,
            "tier": tier,
            "frozen": frozen,
            "sources": sources,
            "first_seen_page": None,
            "occurrences": 0,
        }

    OUT_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[json] {OUT_JSON}: {len(out)} entries")

    # Human-readable MD, organized by frozen/tentative
    frozen_entries = [(k, v) for k, v in out.items() if v["frozen"]]
    tentative_entries = [(k, v) for k, v in out.items() if not v["frozen"]]

    lines = [
        "# 02 — 地狱天命 AP-Specific 术语表",
        "",
        f"_机读版：`02_term_map.json` （{len(out)} 条）_",
        "",
        f"_SRD 通用术语：`翻译流程/tm_cache/tm_3source.json` 用 grep 查询_",
        "",
        "## 协议",
        "",
        "- **Frozen 区** = 已锁定术语，per-page 翻译时直接采用",
        "- **Tentative 区** = 仅 1 处来源，待第 2 次出现验证后 promote 为 frozen",
        "- 新术语先入 Tentative，达 frozen 条件后移过来（也可手动 promote 当确认）",
        "- 冲突进 `04_uncertain_terms.md`",
        "",
        "## Tier 优先级",
        "",
        "1. pf2_cn （FVTT 系统 i18n） — 通过 grep 查询 tm_3source.json",
        "2. pf2e_compendium （Babele zh-CN SRD pack） — 同上",
        "3. **本表（AP-specific：种子 + 前传挖矿）** ← 本文档",
        "4. wiki （pf2.huijiwiki.com，在线优先，离线 `_wiki_full_v2/` 兜底）",
        "5. 自创（实在找不到才用）",
        "",
        "---",
        "",
        f"## Frozen 区（{len(frozen_entries)} 条）",
        "",
        "| English | 中文 | 来源 |",
        "|---|---|---|",
    ]
    for en, v in sorted(frozen_entries, key=lambda kv: kv[0].lower()):
        lines.append(f"| {en} | {v['zh']} | {','.join(v['sources'])} |")
    lines += [
        "",
        f"## Tentative 区（{len(tentative_entries)} 条）",
        "",
        "| English | 中文 | 来源 |",
        "|---|---|---|",
    ]
    for en, v in sorted(tentative_entries, key=lambda kv: kv[0].lower()):
        lines.append(f"| {en} | {v['zh']} | {','.join(v['sources'])} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[md] {OUT_MD}: {len(out)} entries ({len(frozen_entries)} frozen / {len(tentative_entries)} tentative)")
    print(f"[md] size: {OUT_MD.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
