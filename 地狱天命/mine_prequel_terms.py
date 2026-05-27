"""Mine bilingual term pairs from the Hellbreakers (prequel) materials.

Sources:
- 地狱破灭译文-1.0初校版.pdf (212 pages, Chinese-first manual translation) — HIGH confidence
- pdf_output/c71c1ec7-PZO15222E.translated.md (258 pages, machine translation with per-page
  Chinese + collapsible English Text Reference + Vision Extracted) — MEDIUM confidence

Outputs:
- reference/prequel_official_pages/page_NNN.md  (extracted Chinese pages from 1.0 PDF)
- reference/prequel_term_pairs.json             (merged term pairs)
- reference/prequel_term_pairs.md               (human-readable)

Strategy:
1. Extract 1.0 PDF page-by-page (PyMuPDF text layer is clean for the existing file)
2. Mine 中文（English）and English（中文）paren-pair patterns from 1.0
3. Mine the same patterns from c71c1ec7 markdown
4. Add a hard-coded SEED list of canonical AP-specific terms encountered during planning
5. Merge with 1.0 winning conflicts; output JSON + MD
"""
import json
import re
import sys
import io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import fitz

HERE = Path(__file__).resolve().parent
PREQUEL_PDF = HERE / "地狱破灭译文-1.0初校版.pdf"
C71_MD = HERE.parent / "pdf_output" / "c71c1ec7-e1ef-4045-8428-eb4876feee1d-PZO15222E.translated.md"
REF_DIR = HERE / "reference"
OFFICIAL_PAGES_DIR = REF_DIR / "prequel_official_pages"
PAIRS_JSON = REF_DIR / "prequel_term_pairs.json"
PAIRS_MD = REF_DIR / "prequel_term_pairs.md"

# Word-boundary lookbehind so Chinese term doesn't start mid-sentence.
# Allow start at: line begin, whitespace, opening paren, or Chinese punctuation.
ZH_WORD_START = r"(?:(?<=^)|(?<=[\s（(【「『、，。；：！？\n]))"

# 中文（English）— Chinese followed by English in parens (Chinese or English parens)
PAIR_ZH_EN = re.compile(
    ZH_WORD_START +
    r"([一-鿿][一-鿿··]{1,8}?[一-鿿])"
    r"\s*[（(]\s*"
    r"([A-Z][A-Za-z][A-Za-z\s'’\-\.]{1,50}?[A-Za-z])"
    r"\s*[)）]"
)
# English（中文）— English followed by Chinese in parens (rare but appears in glossaries)
PAIR_EN_ZH = re.compile(
    r"(?<![A-Za-z])"
    r"([A-Z][A-Za-z][A-Za-z\s'’\-\.]{1,50}?[A-Za-z])"
    r"\s*[（(]\s*"
    r"([一-鿿][一-鿿··]{1,8}?[一-鿿])"
    r"\s*[)）]"
)

# Filter: drop pairs where the English part contains obvious noise
BAD_EN_KEYWORDS = {
    "Inc", "Page", "Chapter", "Section",
    "Yes", "No", "OK", "Critical", "Success", "Failure",
}

# Filter: drop pairs where the Chinese part begins with these particles/grammar words
# (heavy indicator that the regex captured a sentence fragment, not a clean term)
ZH_BAD_PREFIX = (
    "的", "是", "在", "这", "那", "有", "会", "从", "到", "把",
    "被", "着", "与", "和", "给", "对", "了", "也", "就", "可", "要", "能",
    "则", "不", "他", "她", "它", "们", "你", "我", "并", "但", "如",
    "或", "且", "向", "为", "由", "以", "于", "比", "等", "又",
)


def clean_en(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_zh(s: str) -> str:
    return re.sub(r"\s+", "", s).strip()


def valid_pair(zh: str, en: str) -> bool:
    if not zh or not en:
        return False
    if len(en) < 3 or len(en) > 55:
        return False
    if len(zh) < 2 or len(zh) > 10:
        return False
    if any(bad in en.split() for bad in BAD_EN_KEYWORDS):
        return False
    if not any(c.isalpha() for c in en):
        return False
    if zh.startswith(ZH_BAD_PREFIX):
        return False
    return True


def extract_1_0_pdf():
    """Extract 地狱破灭译文-1.0 to per-page MD under reference/prequel_official_pages/."""
    OFFICIAL_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PREQUEL_PDF)
    total = doc.page_count
    print(f"[1.0] {total} pages")
    full = []
    for i in range(total):
        text = doc[i].get_text("text").strip()
        n = i + 1
        (OFFICIAL_PAGES_DIR / f"page_{n:03d}.md").write_text(
            f"# 地狱破灭译文-1.0 Page {n}\n\n{text}\n", encoding="utf-8"
        )
        full.append(text)
    doc.close()
    return "\n\n".join(full)


def mine_pairs_from_text(text: str, source_label: str):
    """Find both directions of paren pairs in the given text."""
    out = defaultdict(lambda: defaultdict(int))  # en -> zh -> count
    for m in PAIR_ZH_EN.finditer(text):
        zh = clean_zh(m.group(1))
        en = clean_en(m.group(2))
        if valid_pair(zh, en):
            out[en][zh] += 1
    for m in PAIR_EN_ZH.finditer(text):
        en = clean_en(m.group(1))
        zh = clean_zh(m.group(2))
        if valid_pair(zh, en):
            out[en][zh] += 1
    print(f"[mine] {source_label}: {len(out)} unique English keys")
    return out


# Seed list — terms I've already identified from exploring source materials.
# These are HIGH confidence (verified against wiki canonical + 1.0 PDF + c71c1ec7).
SEED = {
    "Hell's Destiny": "地狱天命",
    "Hellbreakers": "破狱者",
    "Hellbreakers League": "破狱者联盟",
    "Hellfire Crisis": "地狱烈火危机",
    "Cheliax": "切利亚斯",
    "Andoran": "安多安",
    "Isger": "依斯嘉",
    "House Thrune": "斯戎家族",
    "Queen Abrogail II": "阿波罗盖二世女王",
    "Abrogail II": "阿波罗盖二世",
    "Abrogail": "阿波罗盖",
    "Corentyn": "寇兰廷",
    "Inner Sea": "内海",
    "Eagle Knights": "雄鹰骑士",
    "Steel Falcons": "钢隼",
    "Godsrain": "神雨",
    "Gorum": "古拉姆",
    "warshard": "战神碎片",
    "Breachill": "布雷克屯",
    "Reginald Cormoth": "雷金纳德·科莫斯",
    "Hellknight": "地狱骑士",
    "Hellknight Hill": "地狱骑士山",
    "Order of the Rack": "刑架骑士团",
    "Order of the Godclaw": "神爪骑士团",
    "Order of the Nail": "尖钉骑士团",
    "Order of the Pike": "尖矛骑士团",
    "Order of the Manacle": "锁铐骑士团",
    "Godclaw": "神爪",
    "Godclaw pantheon": "神爪万神殿",
    "Sisters of the Erinys": "金怒魔姐妹会",
    "Viro Ahala": "维罗·阿哈拉",
    "Hedvend VI": "海德温六世",
    "Caeto Vulaunex": "凯托·武劳内克斯",
    "Ezio Gaeta": "埃齐奥·盖塔",
    "Illcayna Alonnor": "伊尔凯娜·阿洛诺尔",
    "Urgathoa": "厄加图娅",
    "Asmodeus": "阿斯摩蒂斯",
    "Milani": "密拉妮",
    "Aroden": "阿罗登",
    "Mammon": "玛门",
    "Elidir": "艾利迪尔",
    "Saringallow": "萨里伽洛",
    "Chitterwood": "尖叫森林",
    "Finder's Gulch": "发现者峡谷",
    "Gillamoor": "吉拉穆尔",
    "Sarini Manor": "萨里尼庄园",
    "Druman": "杜鲁玛",
    "Druma": "杜鲁玛",
    "Taldor": "塔尔多",
    "Goblinblood Wars": "地精血战",
    "Pathfinder": "探路者",
    "Adventure Path": "冒险之路",
    "Linetta Seacarver": "琳奈塔·海凿",  # tentative — verify on first occurrence
    "Citadel Gheradesca": "格拉戴斯卡堡垒",  # tentative
    "Ravounel": "拉乌内尔",
    "Egorian": "埃戈里安",
    "Talmandor": "塔尔曼多",
    "Shackles": "镣铐群岛",
    "Whisperwood": "低语森林",
    "Khari": "卡里",
    "The Eye of Khari": "卡里之眼",
    "Imperious": "傲世号",  # likely a ship name
    "Talmandor's battalion": "塔尔曼多军团",
    "Operation Broken Key": "破钥行动",
    "Hellcoast": "地狱海岸",
    "Hellcoast Heresies": "地狱海岸异端",
    "Gathering Allies": "招纳盟友",
    "Tip of the Spear": "矛尖",
    "War Is Hell": "战争即地狱",
    "Tailors and Traitors": "裁缝与叛徒",
    "Among the Thorns": "荆棘之中",
    "Midnight Burning": "午夜焚烧",
    "The Final Damnation": "终极天罚",
    "Beyond the Campaign": "战役之外",
    "Chelaxian Aristocracy": "切利亚斯贵族",
    "Adventure Toolbox": "冒险工具箱",
    "Campaign Overview": "战役概览",
}


def merge_pair_dicts(pdf_pairs, c71_pairs):
    """Merge with 1.0 PDF winning conflicts."""
    out = {}
    for en, zh_counts in c71_pairs.items():
        zh = max(zh_counts.items(), key=lambda kv: kv[1])[0]
        out[en] = {
            "zh": zh,
            "sources": ["c71c1ec7"],
            "confidence": "medium",
            "candidates": dict(zh_counts),
        }
    for en, zh_counts in pdf_pairs.items():
        zh = max(zh_counts.items(), key=lambda kv: kv[1])[0]
        if en in out:
            out[en]["sources"].append("1.0初校版")
            out[en]["zh"] = zh
            out[en]["confidence"] = "high"
            for z, c in zh_counts.items():
                out[en]["candidates"][z] = out[en]["candidates"].get(z, 0) + c
        else:
            out[en] = {
                "zh": zh,
                "sources": ["1.0初校版"],
                "confidence": "high",
                "candidates": dict(zh_counts),
            }
    for en, zh in SEED.items():
        if en in out:
            out[en]["sources"].append("seed")
            out[en]["zh"] = zh  # seed wins (curated)
            out[en]["confidence"] = "high"
        else:
            out[en] = {
                "zh": zh,
                "sources": ["seed"],
                "confidence": "high",
                "candidates": {zh: 1},
            }
    return out


def write_json(out):
    PAIRS_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[done] {PAIRS_JSON}: {len(out)} entries, {PAIRS_JSON.stat().st_size:,} bytes")


def write_md(out):
    lines = [
        "# 前传术语挖矿结果（地狱破灭译文-1.0 + c71c1ec7 + seed）",
        "",
        "| English | 中文 | 信度 | 来源 | 候选 |",
        "|---|---|---|---|---|",
    ]
    for en in sorted(out, key=lambda s: s.lower()):
        v = out[en]
        cand = ", ".join(f"{z}({c})" for z, c in sorted(v["candidates"].items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| {en} | {v['zh']} | {v['confidence']} | {','.join(v['sources'])} | {cand} |"
        )
    PAIRS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {PAIRS_MD}: {len(out)} rows")


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_text = extract_1_0_pdf()
    pdf_pairs = mine_pairs_from_text(pdf_text, "1.0初校版")
    c71_text = C71_MD.read_text(encoding="utf-8") if C71_MD.exists() else ""
    if not c71_text:
        print(f"[warn] c71c1ec7 file not found at {C71_MD}")
    c71_pairs = mine_pairs_from_text(c71_text, "c71c1ec7") if c71_text else {}
    merged = merge_pair_dicts(pdf_pairs, c71_pairs)
    write_json(merged)
    write_md(merged)


if __name__ == "__main__":
    main()
