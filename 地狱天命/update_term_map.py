"""Patch state/02_term_map.json (and regenerate .md) with new term decisions.

Usage:
    python update_term_map.py
        — patches with hard-coded UPDATES list

Or use as a library: from update_term_map import patch_terms; patch_terms({...})
"""
import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
JSON_FILE = STATE / "02_term_map.json"
MD_FILE = STATE / "02_term_map.md"


def patch_terms(updates: dict):
    """updates: {en: {"zh": str, "tier": int, "frozen": bool, "sources": [str], "note": str optional}}"""
    data = json.loads(JSON_FILE.read_text(encoding="utf-8"))
    added = 0
    updated = 0
    for en, v in updates.items():
        if en in data:
            data[en]["zh"] = v["zh"]
            data[en]["tier"] = v.get("tier", data[en].get("tier", 3))
            data[en]["frozen"] = v.get("frozen", True)
            existing_sources = set(data[en].get("sources", []))
            for s in v.get("sources", []):
                existing_sources.add(s)
            data[en]["sources"] = sorted(existing_sources)
            if "note" in v:
                data[en]["note"] = v["note"]
            updated += 1
        else:
            data[en] = {
                "zh": v["zh"],
                "tier": v.get("tier", 4),
                "frozen": v.get("frozen", True),
                "sources": v.get("sources", ["wiki"]),
                "first_seen_page": None,
                "occurrences": 0,
            }
            if "note" in v:
                data[en]["note"] = v["note"]
            added += 1
    JSON_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[patched] +{added} new, {updated} updated. total {len(data)} entries")
    rebuild_md(data)


def rebuild_md(data: dict):
    frozen_entries = [(k, v) for k, v in data.items() if v.get("frozen")]
    tentative_entries = [(k, v) for k, v in data.items() if not v.get("frozen")]
    lines = [
        "# 02 — 地狱天命 AP-Specific 术语表",
        "",
        f"_机读版：`02_term_map.json` （{len(data)} 条）_",
        "",
        f"_SRD 通用术语：`翻译流程/tm_cache/tm_3source.json` 通过 `lookup_term.py <term>` 查询_",
        f"_wiki 离线 grep：`lookup_wiki.py <term>` 查询 _wiki_full_v2/pages/_",
        "",
        "## 协议",
        "- **Frozen 区** = 已锁定术语，per-page 翻译时直接采用",
        "- **Tentative 区** = 仅 1 处来源，待第 2 次出现验证后 promote 为 frozen",
        "- 新术语先入 Tentative，达 frozen 条件后移过来",
        "- 冲突进 `04_uncertain_terms.md`",
        "",
        "## Tier 优先级",
        "1. pf2_cn (FVTT 系统 i18n) — `lookup_term.py` 查 tm_3source.json",
        "2. pf2e_compendium (Babele zh-CN SRD pack) — 同上",
        "3. **本表（AP-specific：种子 + 前传挖矿 + 翻译中新建）**",
        "4. wiki (pf2.huijiwiki.com，在线优先；离线 _wiki_full_v2/)",
        "5. 自创",
        "",
        "---",
        "",
        f"## Frozen 区（{len(frozen_entries)} 条）",
        "",
        "| English | 中文 | tier | 来源 | 备注 |",
        "|---|---|---|---|---|",
    ]
    for en, v in sorted(frozen_entries, key=lambda kv: kv[0].lower()):
        note = v.get("note", "")
        lines.append(f"| {en} | {v['zh']} | {v.get('tier', 3)} | {','.join(v.get('sources', []))} | {note} |")
    lines += [
        "",
        f"## Tentative 区（{len(tentative_entries)} 条）",
        "",
        "| English | 中文 | tier | 来源 | 备注 |",
        "|---|---|---|---|---|",
    ]
    for en, v in sorted(tentative_entries, key=lambda kv: kv[0].lower()):
        note = v.get("note", "")
        lines.append(f"| {en} | {v['zh']} | {v.get('tier', 3)} | {','.join(v.get('sources', []))} | {note} |")
    MD_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[md] {MD_FILE.stat().st_size:,} bytes ({len(frozen_entries)} frozen / {len(tentative_entries)} tentative)")


# Updates discovered during pp.4-15 translation.
# tier 4 = wiki canonical; tier 3 = SEED/curated; tier 5 = self-create
UPDATES = {
    # New AP NPCs/locations from pp.11-15
    "Zefiro Balinger": {"zh": "泽菲罗·巴林杰", "tier": 3, "sources": ["AP-confirmed"], "note": "Ch1 NPC, 寇兰廷历史博物馆馆长，7级 精英学者"},
    "Zefiro": {"zh": "泽菲罗", "tier": 3, "sources": ["AP-confirmed"]},
    "Ignaro Vollai": {"zh": "伊格纳罗·沃莱", "tier": 3, "sources": ["AP-confirmed"], "note": "Ch1 A1 反派，多塔里头领，恶魔教士 11级"},
    "Kettermaul Charthagnion": {"zh": "凯特毛尔·查塔尼翁", "tier": 3, "sources": ["AP-confirmed"], "note": "碎冠者头领，archheathen 8级"},
    "Silent Silhouette": {"zh": "无声剪影", "tier": 3, "sources": ["AP-confirmed"], "note": "铃花会盗贼 9级"},
    "Cathilia Mutol": {"zh": "卡蒂利娅·穆托尔", "tier": 3, "sources": ["AP-confirmed"], "note": "锁链骑士团地狱骑士 10级，潜在盟友"},
    "Thin Wisps": {"zh": "细缕者", "tier": 3, "sources": ["self"], "note": "AP 派系，丝巾飞檐表演者+革命者"},
    "Crowncrushers Gauntlet": {"zh": "碎冠者长廊", "tier": 3, "sources": ["self"], "note": "Ch1 地图名"},
    "Dhoffram Keep": {"zh": "德霍夫拉姆堡", "tier": 5, "sources": ["self"], "note": "Ch1 监狱"},
    "High Quarter": {"zh": "高贵区", "tier": 3, "sources": ["self"], "note": "寇兰廷富人区"},
    "Easttown": {"zh": "东城", "tier": 3, "sources": ["self"], "note": "寇兰廷区域"},
    "Warehouse District": {"zh": "仓库区", "tier": 3, "sources": ["self"], "note": "寇兰廷区域"},
    "Operation Broken Key": {"zh": "破钥行动", "tier": 3, "sources": ["seed"], "note": "Ch2 主线行动"},
    "Brewing Trouble": {"zh": "风暴酝酿", "tier": 3, "sources": ["self"], "note": "Ch1 开篇小节"},
    "Archheathen": {"zh": "异端魁首", "tier": 3, "sources": ["self"], "note": "切利亚斯城邦统治者头衔；AP 新 NPC 类型"},
    # Stat block / mechanics terms (PF2e zh-CN canonical)
    "vordine": {"zh": "军团魔", "tier": 1, "sources": ["pf2_cn"]},
    "phistophilus": {"zh": "契约魔", "tier": 4, "sources": ["wiki"]},
    "Erebus": {"zh": "厄瑞玻斯", "tier": 5, "sources": ["self"], "note": "九层地狱第七层，PF 经典"},
    "diabolist": {"zh": "恶魔教士", "tier": 4, "sources": ["wiki"], "note": "亦可用「通魔者」"},
    "staff of control": {"zh": "脑控法杖", "tier": 2, "sources": ["pf2e_compendium"]},
    "Heroism": {"zh": "英雄气概", "tier": 2, "sources": ["pf2e_compendium"]},
    "Paralyze": {"zh": "麻痹术", "tier": 2, "sources": ["pf2e_compendium"]},
    "Summon Fiend": {"zh": "召唤魔族", "tier": 2, "sources": ["pf2e_compendium"]},
    "Divine Wrath": {"zh": "神圣狂怒", "tier": 2, "sources": ["pf2e_compendium"]},
    "Translocate": {"zh": "瞬移术", "tier": 2, "sources": ["pf2e_compendium"]},
    "Ignite Heresy": {"zh": "点燃异端", "tier": 3, "sources": ["self"], "note": "Ignaro 招式"},
    "Hell's Arrest": {"zh": "地狱拘捕", "tier": 3, "sources": ["self"]},
    "Hellknight's Orders": {"zh": "地狱骑士号令", "tier": 3, "sources": ["self"], "note": "Cathilia 招式 (远程音波)"},
    "Timely Advice": {"zh": "及时建议", "tier": 3, "sources": ["self"], "note": "NPC 援助能力"},
    # Conditions (canonical PF2e zh-CN)
    "off-guard": {"zh": "失备", "tier": 1, "sources": ["pf2_cn"]},
    "dazzled": {"zh": "目眩", "tier": 1, "sources": ["pf2_cn"]},
    "deafened": {"zh": "失聪", "tier": 1, "sources": ["pf2_cn"]},
    "blinded": {"zh": "失明", "tier": 1, "sources": ["pf2_cn"]},
    "Stunned": {"zh": "震慑", "tier": 1, "sources": ["pf2_cn"]},
    "Sickened": {"zh": "恶心", "tier": 1, "sources": ["pf2_cn"]},
    "Slowed": {"zh": "缓慢", "tier": 1, "sources": ["pf2_cn"]},
    "grabbed": {"zh": "被擒抓", "tier": 1, "sources": ["pf2_cn"]},
    "restrained": {"zh": "被束缚", "tier": 1, "sources": ["pf2_cn"]},
    # Actions
    "Recall Knowledge": {"zh": "识破", "tier": 4, "sources": ["wiki"], "note": "PF2e zh-CN 通译，常作「检索知识」"},
    "Sense Motive": {"zh": "察觉动机", "tier": 4, "sources": ["wiki"]},
    "Force Open": {"zh": "强行开启", "tier": 4, "sources": ["wiki"]},
    "Disable a Device": {"zh": "拆除装置", "tier": 4, "sources": ["wiki"]},
    "Grapple": {"zh": "擒抱", "tier": 4, "sources": ["wiki"]},
    "Escape": {"zh": "脱离", "tier": 4, "sources": ["wiki"]},
    "Sneak": {"zh": "潜行", "tier": 4, "sources": ["wiki"]},
    "Make an Impression": {"zh": "留下印象", "tier": 4, "sources": ["wiki"]},
    # Subsystems
    "chase subsystem": {"zh": "追逐子系统", "tier": 4, "sources": ["wiki"]},
    "influence subsystem": {"zh": "影响力子系统", "tier": 4, "sources": ["wiki"]},
    "Victory Point": {"zh": "胜利点", "tier": 4, "sources": ["wiki"]},
    "skirmish combat": {"zh": "前哨战", "tier": 4, "sources": ["wiki"]},
    # Originally seed; verify here
    "Corentyn Museum of History": {"zh": "寇兰廷历史博物馆", "tier": 3, "sources": ["self"]},
    "Imperious": {"zh": "傲世号", "tier": 3, "sources": ["self"], "note": "Ch1 末尾袭击的旗舰; PDF p25 地图标签 IMPERVIOUS 系笔误"},
    "Impervious": {"zh": "傲世号", "tier": 3, "sources": ["self"], "note": "PDF p25 笔误，正确为 Imperious"},
    # NPCs from pp.16-29
    "Triumph": {"zh": "凯旋", "tier": 3, "sources": ["AP-confirmed"], "note": "碎冠者双胞胎之一，伪暮爪 8级"},
    "Eryr": {"zh": "艾里尔", "tier": 3, "sources": ["AP-confirmed"], "note": "碎冠者双胞胎之一，新晋叛逆者 8级"},
    "Pesze": {"zh": "佩瑟", "tier": 3, "sources": ["AP-confirmed"], "note": "细缕者演员 8级"},
    "Ubenor Enos": {"zh": "乌贝诺·埃诺斯", "tier": 3, "sources": ["AP-confirmed"], "note": "锁链骑士团 工匠 9级"},
    "Uro Adom": {"zh": "乌罗·阿登", "tier": 3, "sources": ["AP-confirmed"], "note": "前锁链骑士团执棒官，倒戈为镣铐骑士团领袖"},
    "Sebenio du Tollin": {"zh": "塞贝尼奥·杜·托林", "tier": 3, "sources": ["AP-confirmed"], "note": "锁链骑士团 signifer 10级"},
    "Emertine Oughn": {"zh": "埃默汀·欧恩", "tier": 3, "sources": ["AP-confirmed"], "note": "锁链骑士团 狂热赏金猎人 11级"},
    "Abrogail Croon": {"zh": "阿波罗盖·克鲁恩", "tier": 3, "sources": ["AP-confirmed"], "note": "细缕者领袖，aiuvarin 喜剧演员 7级，性别流动"},
    "Ernetza Farrunner": {"zh": "厄内萨·法兰纳", "tier": 3, "sources": ["AP-confirmed"], "note": "邪叉酒馆招待 + 无声剪影身份"},
    "Neld Havasavu": {"zh": "内尔德·哈瓦萨乌", "tier": 3, "sources": ["AP-confirmed"], "note": "邪叉酒馆主人"},
    "Tourinio Metzapho": {"zh": "图里尼奥·梅扎福", "tier": 3, "sources": ["AP-confirmed"], "note": "拱桥地牢首席恶魔教士 10级"},
    "Tessa Fairwind": {"zh": "泰莎·微风", "tier": 4, "sources": ["wiki"], "note": "镣铐群岛海盗女王，PF 经典角色"},
    # Locations from pp.20-29
    "Wicked Fork": {"zh": "邪叉酒馆", "tier": 3, "sources": ["self"]},
    "West Drenches": {"zh": "西泡区", "tier": 3, "sources": ["self"]},
    "East Drenches": {"zh": "东泡区", "tier": 3, "sources": ["self"]},
    "Harbormaster Island": {"zh": "港务岛", "tier": 3, "sources": ["self"]},
    "Noble Quarter": {"zh": "贵族区", "tier": 3, "sources": ["self"]},
    "Castle Belverio": {"zh": "贝尔维里奥城堡", "tier": 3, "sources": ["self"]},
    "Zelkie Park": {"zh": "泽尔基公园", "tier": 5, "sources": ["self"]},
    "Ship Yard": {"zh": "船坞区", "tier": 3, "sources": ["self"]},
    "Navy Yard": {"zh": "海军港务", "tier": 3, "sources": ["self"]},
    "Hespereth Strait": {"zh": "赫斯佩雷斯海峡", "tier": 5, "sources": ["self"]},
    # Game mechanics
    "Awareness Points": {"zh": "警觉点", "tier": 4, "sources": ["wiki"]},
    "Infiltration Points": {"zh": "渗透点", "tier": 4, "sources": ["wiki"]},
    "Edge Point": {"zh": "优势点", "tier": 4, "sources": ["wiki"]},
    "Influence Points": {"zh": "影响力点", "tier": 4, "sources": ["wiki"]},
    "Discovery": {"zh": "探知", "tier": 4, "sources": ["wiki"], "note": "影响力子系统的检定类型之一"},
    "Influence": {"zh": "影响", "tier": 4, "sources": ["wiki"], "note": "影响力子系统的检定类型之一"},
    "infiltration subsystem": {"zh": "渗透子系统", "tier": 4, "sources": ["wiki"]},
    "Create a Forgery": {"zh": "制作伪造品", "tier": 4, "sources": ["wiki"]},
    "Gather Information": {"zh": "搜集情报", "tier": 4, "sources": ["wiki"]},
    "Request": {"zh": "请求", "tier": 4, "sources": ["wiki"], "note": "交涉动作之一"},
    "Demoralize": {"zh": "沮丧", "tier": 1, "sources": ["pf2_cn"]},
    "Stride": {"zh": "大步", "tier": 1, "sources": ["pf2_cn"]},
    "Create a Diversion": {"zh": "制造分心", "tier": 1, "sources": ["pf2_cn"]},
    "Feint": {"zh": "佯攻", "tier": 1, "sources": ["pf2_cn"]},
    "Squeeze": {"zh": "挤入", "tier": 1, "sources": ["pf2_cn"]},
    "Climb": {"zh": "攀爬", "tier": 1, "sources": ["pf2_cn"]},
    "Search": {"zh": "搜寻", "tier": 1, "sources": ["pf2_cn"]},
    "Interact": {"zh": "互动", "tier": 1, "sources": ["pf2_cn"]},
    "Grab an Edge": {"zh": "抓住边缘", "tier": 1, "sources": ["pf2_cn"]},
    "Balance": {"zh": "保持平衡", "tier": 1, "sources": ["pf2_cn"]},
    "Push": {"zh": "推开", "tier": 1, "sources": ["pf2_cn"]},
    # Materials & races
    "dawnsilver": {"zh": "晨曦银", "tier": 1, "sources": ["pf2_cn"]},
    "adamantine": {"zh": "精金", "tier": 1, "sources": ["pf2_cn"]},
    "aiuvarin": {"zh": "半精灵", "tier": 1, "sources": ["pf2_cn"], "note": "Remaster：取代 half-elf"},
    "Hellknight signifer": {"zh": "地狱骑士圣记官", "tier": 4, "sources": ["wiki"]},
    "Hellknight plate": {"zh": "地狱骑士板甲", "tier": 4, "sources": ["wiki"]},
    "Hellknight breastplate": {"zh": "地狱骑士胸甲", "tier": 4, "sources": ["wiki"]},
    "lictor": {"zh": "执棒官", "tier": 4, "sources": ["wiki"], "note": "地狱骑士军衔"},
    "Measure and the Chain": {"zh": "准绳与锁链", "tier": 4, "sources": ["wiki"], "note": "地狱骑士哲学典籍"},
    "Ennead Star": {"zh": "九芒星", "tier": 4, "sources": ["wiki"], "note": "地狱骑士神爪万神殿标志"},
    "Reckoning": {"zh": "自审", "tier": 3, "sources": ["self"], "note": "地狱骑士悔过仪式"},
    # Spells
    "alarm": {"zh": "警报术", "tier": 2, "sources": ["pf2e_compendium"]},
    "dream message": {"zh": "梦讯术", "tier": 2, "sources": ["pf2e_compendium"]},
    "scrying": {"zh": "窥探术", "tier": 2, "sources": ["pf2e_compendium"]},
    "shape stone": {"zh": "塑石术", "tier": 2, "sources": ["pf2e_compendium"]},
    "detect magic": {"zh": "探测魔法", "tier": 2, "sources": ["pf2e_compendium"]},
    "dispel magic": {"zh": "解除魔法", "tier": 2, "sources": ["pf2e_compendium"]},
    # Items
    "extending claw blade": {"zh": "伸缩爪刃", "tier": 2, "sources": ["pf2e_compendium"]},
    "main-gauche": {"zh": "左手剑", "tier": 2, "sources": ["pf2e_compendium"]},
    "starknife": {"zh": "星刃", "tier": 2, "sources": ["pf2e_compendium"]},
    "charlatan's cape": {"zh": "江湖术士披风", "tier": 2, "sources": ["pf2e_compendium"]},
    "weapon potency rune": {"zh": "武器潜能符文", "tier": 2, "sources": ["pf2e_compendium"]},
    "shadow signet": {"zh": "阴影信戒", "tier": 2, "sources": ["pf2e_compendium"]},
    "choker of elocution": {"zh": "雄辩颈环", "tier": 2, "sources": ["pf2e_compendium"]},
    "maestro's harmonica": {"zh": "大师之口琴", "tier": 5, "sources": ["self"], "note": "Battlecry 物品"},
    "tricky liniment": {"zh": "巧术药油", "tier": 5, "sources": ["self"], "note": "Battlecry 物品"},
    "crafter's eyepiece": {"zh": "工匠目镜", "tier": 2, "sources": ["pf2e_compendium"]},
    "sequestering wig": {"zh": "藏物假发", "tier": 3, "sources": ["self"]},
    # Creatures
    "sarglagon": {"zh": "萨格拉贡魔", "tier": 4, "sources": ["wiki"], "note": "Monster Core 89"},
    "alghollthu": {"zh": "阿果尔苏", "tier": 4, "sources": ["wiki"], "note": "古代水栖种族"},
    "stone bulwark": {"zh": "石卫垒", "tier": 4, "sources": ["wiki"], "note": "Monster Core 324"},
    # Skirmish warfare / Battlecry
    "mercenary band troop": {"zh": "雇佣兵小队", "tier": 4, "sources": ["wiki"], "note": "Battlecry 部队"},
    "troop threshold": {"zh": "部队阈值", "tier": 4, "sources": ["wiki"]},
    "Let 'em Have It!": {"zh": "让他们尝尝厉害！", "tier": 5, "sources": ["self"]},
    "Ready... Fire!": {"zh": "准备……开火！", "tier": 5, "sources": ["self"]},
    "Live for the Thrill": {"zh": "为刺激而生", "tier": 3, "sources": ["self"]},
    "By Order and By Law": {"zh": "依令法之名", "tier": 3, "sources": ["self"]},
    "Razzle and Dazzle": {"zh": "眼花缭乱", "tier": 3, "sources": ["self"]},
    # PDF cross references
    "Pathfinder Monster Core": {"zh": "《怪物核心》", "tier": 4, "sources": ["wiki"]},
    "Pathfinder NPC Core": {"zh": "《NPC 核心》", "tier": 4, "sources": ["wiki"]},
    "Pathfinder Player Core": {"zh": "《玩家核心》", "tier": 4, "sources": ["wiki"]},
    "Pathfinder Player Core 2": {"zh": "《玩家核心 2》", "tier": 4, "sources": ["wiki"]},
    "Pathfinder Treasure Vault": {"zh": "《珍宝密库》", "tier": 4, "sources": ["wiki"]},
    "Empyrean": {"zh": "至上界语", "tier": 4, "sources": ["wiki"], "note": "天界生物语言"},
    # Chapter section names
    "None Are Innocent": {"zh": "无人无辜", "tier": 3, "sources": ["self"], "note": "Ch1 任务三"},
    "Under Arch and Key": {"zh": "拱与钥之下", "tier": 3, "sources": ["self"], "note": "Ch1 任务四"},
    "Artifact Recovery": {"zh": "文物回收", "tier": 3, "sources": ["self"], "note": "Ch1 任务一"},
    "Rebellion on the Winds": {"zh": "乘风起义", "tier": 3, "sources": ["self"], "note": "Ch1 高潮 section"},
    "A New Tenacity": {"zh": "新的坚韧", "tier": 3, "sources": ["self"]},
    # Corrections to existing SEED
    "Khari": {"zh": "卡利", "tier": 4, "sources": ["wiki", "seed-corrected"], "note": "wiki 用「卡利城」，单独 Khari 用「卡利」"},
    "The Eye of Khari": {"zh": "卡利之眼", "tier": 4, "sources": ["seed-corrected"], "note": "Chapter 4 标题"},
    "Whisperwood": {"zh": "默语森林", "tier": 4, "sources": ["wiki", "seed-corrected"], "note": "wiki canonical（原误为 低语森林）"},
    "Hellbreakers League": {"zh": "破狱者联盟", "tier": 3, "sources": ["1.0初校版", "seed"], "note": "前传翻译沿用"},
    # New PF2e locations
    "Rahadoum": {"zh": "拉哈多姆", "tier": 4, "sources": ["wiki"]},
    "Molthune": {"zh": "摩尔苏恩", "tier": 4, "sources": ["wiki"]},
    "Westcrown": {"zh": "西冠城", "tier": 4, "sources": ["wiki"]},
    "Kintargo": {"zh": "金塔格", "tier": 4, "sources": ["wiki"]},
    "Nidal": {"zh": "奈多", "tier": 4, "sources": ["wiki"]},
    "Vyre": {"zh": "维里", "tier": 4, "sources": ["wiki"]},
    "Ostenso": {"zh": "欧斯坦索", "tier": 4, "sources": ["wiki"]},
    "Pezzack": {"zh": "佩扎克", "tier": 5, "sources": ["self"], "note": "wiki 未明确给出，按音译"},
    "Brastlewark": {"zh": "布拉斯托瓦克", "tier": 4, "sources": ["wiki"]},
    "Aspodell Mountains": {"zh": "阿斯珀德尔山脉", "tier": 4, "sources": ["wiki"]},
    "Aspodean Wall": {"zh": "阿斯珀德尔长城", "tier": 4, "sources": ["wiki"], "note": "也作「阿斯珀德尔山口」/Aspodell Pass"},
    "Menador Mountains": {"zh": "梅纳朵山脉", "tier": 4, "sources": ["wiki"]},
    "Mindspin Mountains": {"zh": "心旋山脉", "tier": 5, "sources": ["self"], "note": "wiki 未明确，按字面意译"},
    "Arch of Aroden": {"zh": "奥罗登拱桥", "tier": 4, "sources": ["wiki"]},
    "Hespereth Strait": {"zh": "赫斯佩雷斯海峡", "tier": 5, "sources": ["self"], "note": "音译"},
    "Hellmouth Gulf": {"zh": "地狱口湾", "tier": 5, "sources": ["self"], "note": "字面意译"},
    "Inner Sea": {"zh": "内海", "tier": 4, "sources": ["wiki", "seed"]},
    "Arcadian Ocean": {"zh": "阿卡迪亚洋", "tier": 4, "sources": ["wiki"]},
    "Garund": {"zh": "伽伦德", "tier": 4, "sources": ["wiki"]},
    "Avistan": {"zh": "阿维斯坦", "tier": 4, "sources": ["wiki"]},
    "Golarion": {"zh": "格拉利昂", "tier": 4, "sources": ["wiki"]},
    "Augustana": {"zh": "奥古斯塔纳", "tier": 5, "sources": ["self"], "note": "音译"},
    "Iadara": {"zh": "伊亚达拉", "tier": 5, "sources": ["self"], "note": "音译"},
    "Isarn": {"zh": "伊萨恩", "tier": 5, "sources": ["self"], "note": "音译"},
    "Highhelm": {"zh": "高盔堡", "tier": 4, "sources": ["wiki"], "note": "wiki canonical（PF2e 重要矮人都市）"},
    "Kerse": {"zh": "克瑟", "tier": 5, "sources": ["self"], "note": "音译"},
    "Oppara": {"zh": "欧帕拉", "tier": 5, "sources": ["self"], "note": "音译"},
    "Canorate": {"zh": "卡诺拉特", "tier": 5, "sources": ["self"]},
    "Pangolais": {"zh": "潘戈莱斯", "tier": 5, "sources": ["self"], "note": "Nidal 首都"},
    "Pezzack": {"zh": "佩扎克", "tier": 5, "sources": ["self"]},
    "Senara": {"zh": "塞纳拉", "tier": 5, "sources": ["self"]},
    "Nisroch": {"zh": "尼斯洛克", "tier": 5, "sources": ["self"]},
    "Ridwan": {"zh": "瑞德万", "tier": 5, "sources": ["self"]},
    "Logas": {"zh": "罗加斯", "tier": 5, "sources": ["self"]},
    "Hinji": {"zh": "辛吉", "tier": 5, "sources": ["self"]},
    "Absalom": {"zh": "艾巴萨罗姆", "tier": 4, "sources": ["wiki"]},
    "Katapesh": {"zh": "卡塔佩什", "tier": 4, "sources": ["wiki"]},
    "Osirion": {"zh": "奥西利安", "tier": 4, "sources": ["wiki"]},
    "Ravounel": {"zh": "瑞瓦诺", "tier": 4, "sources": ["wiki", "seed-corrected"], "note": "wiki 用 瑞瓦诺（旧 SEED 拉乌内尔 错误）"},
    "Uskwood": {"zh": "乌斯克森林", "tier": 5, "sources": ["self"]},
    "Barrowood": {"zh": "巴若森林", "tier": 5, "sources": ["self"]},
    "Dismal Nitch": {"zh": "凄凉裂隙", "tier": 5, "sources": ["self"]},
    "Egorian": {"zh": "埃戈里安", "tier": 4, "sources": ["wiki", "seed"]},
    # Factions / Organizations
    "Bellflower Network": {"zh": "铃花会", "tier": 4, "sources": ["wiki"]},
    "Crowncrushers": {"zh": "碎冠者", "tier": 5, "sources": ["self"], "note": "AP 新派系，wiki 未收录，按字面"},
    "Aspis Consortium": {"zh": "蛇徽财团", "tier": 4, "sources": ["wiki"]},
    "Anaphexia": {"zh": "安娜费西亚", "tier": 4, "sources": ["wiki"]},
    "Red Mantis": {"zh": "红螳螂", "tier": 4, "sources": ["wiki"]},
    "Order of the Chain": {"zh": "锁链骑士团", "tier": 4, "sources": ["wiki"]},
    "Order of the Whip": {"zh": "笞鞭骑士团", "tier": 4, "sources": ["wiki"]},
    "Order of the Gate": {"zh": "界门骑士团", "tier": 4, "sources": ["wiki"]},
    "Council of Thieves": {"zh": "盗贼议会", "tier": 4, "sources": ["wiki"]},
    "Twilight Talons": {"zh": "暮爪", "tier": 4, "sources": ["wiki"]},
    "Steel Falcons": {"zh": "钢隼", "tier": 4, "sources": ["wiki", "seed"]},
    "Gray Corsairs": {"zh": "灰盗", "tier": 4, "sources": ["wiki"], "note": "钢隼海军分支"},
    "Knights of Lastwall": {"zh": "终焉之墙骑士", "tier": 4, "sources": ["wiki"]},
    # Months (Golarion calendar) — using wiki canonical descriptive Chinese names
    "Abadius": {"zh": "财商月", "tier": 4, "sources": ["wiki"], "note": "首月，1月，31日；阿巴达尔之月（财富之神）"},
    "Calistril": {"zh": "仇欲月", "tier": 4, "sources": ["wiki"], "note": "2月，28日；卡莉丝翠之月（复仇女神）"},
    "Pharast": {"zh": "生死月", "tier": 4, "sources": ["wiki"], "note": "3月，31日；法莱斯玛之月"},
    "Gozran": {"zh": "风海月", "tier": 4, "sources": ["wiki"], "note": "4月，30日；哥兹莱之月"},
    "Desnus": {"zh": "星梦月", "tier": 4, "sources": ["wiki"], "note": "5月，31日；黛丝娜之月"},
    "Sarenith": {"zh": "阳愈月", "tier": 4, "sources": ["wiki"], "note": "6月，30日；塞恩蕾之月"},
    "Erastus": {"zh": "农狩月", "tier": 4, "sources": ["wiki"], "note": "7月，31日；埃拉斯蒂尔之月"},
    "Arodus": {"zh": "人文月", "tier": 4, "sources": ["wiki"], "note": "8月，31日；奥罗登之月"},
    "Rova": {"zh": "怒灾月", "tier": 4, "sources": ["wiki"], "note": "9月，30日；洛瓦古之月"},
    "Lamashan": {"zh": "狂噩月", "tier": 4, "sources": ["wiki"], "note": "10月，30日；拉玛什图之月"},
    "Neth": {"zh": "魔法月", "tier": 4, "sources": ["wiki"], "note": "11月，30日；内塞斯之月"},
    "Kuthona": {"zh": "苦暗月", "tier": 4, "sources": ["wiki"], "note": "12月，31日；宗-库山之月"},
    "AR": {"zh": "AR", "tier": 4, "sources": ["wiki"], "note": "Absalom Reckoning，PF 历，AR 4725 = 第 4725 年"},
    # Deities
    "Asmodeus": {"zh": "阿斯摩蒂斯", "tier": 4, "sources": ["wiki", "seed"]},
    "Aroden": {"zh": "奥罗登", "tier": 4, "sources": ["wiki", "seed-corrected"], "note": "wiki canonical 奥罗登（旧 SEED 阿罗登 错）"},
    "Rovagug": {"zh": "洛瓦古", "tier": 4, "sources": ["wiki"]},
    "Gorum": {"zh": "古拉姆", "tier": 4, "sources": ["wiki", "seed"]},
    "Calistria": {"zh": "卡莉丝翠", "tier": 4, "sources": ["wiki"]},
    "Pharasma": {"zh": "法莱斯玛", "tier": 4, "sources": ["wiki"]},
    "Gozreh": {"zh": "哥兹莱", "tier": 4, "sources": ["wiki"]},
    "Desna": {"zh": "黛丝娜", "tier": 4, "sources": ["wiki"]},
    "Sarenrae": {"zh": "塞恩蕾", "tier": 4, "sources": ["wiki"]},
    "Erastil": {"zh": "埃拉斯蒂尔", "tier": 4, "sources": ["wiki"]},
    "Lamashtu": {"zh": "拉玛什图", "tier": 4, "sources": ["wiki"]},
    "Nethys": {"zh": "内塞斯", "tier": 4, "sources": ["wiki"]},
    "Zon-Kuthon": {"zh": "宗-库山", "tier": 4, "sources": ["wiki"]},
    "Talmandor": {"zh": "塔尔曼多", "tier": 4, "sources": ["wiki", "seed"], "note": "安多安守护圣灵"},
    "Urgathoa": {"zh": "厄加图娅", "tier": 4, "sources": ["wiki", "seed"]},
    "Mammon": {"zh": "玛门", "tier": 4, "sources": ["wiki", "seed"]},
    "Milani": {"zh": "密拉妮", "tier": 4, "sources": ["wiki", "seed"]},
    # Races/creatures
    "strix": {"zh": "枭鬼", "tier": 4, "sources": ["wiki"]},
    "azarketi": {"zh": "阿扎克缇", "tier": 4, "sources": ["wiki"]},
    "nessari": {"zh": "尼萨利", "tier": 5, "sources": ["self"], "note": "魔鬼亚种，音译"},
    # Misc
    "Player's Guide": {"zh": "玩家指南", "tier": 4, "sources": ["wiki"]},
    "Pathfinder Battlecry!": {"zh": "《战吼！》", "tier": 4, "sources": ["wiki"], "note": "规则补充本"},
    "GM Core": {"zh": "《主持人核心》", "tier": 4, "sources": ["wiki"]},
    "Player Core": {"zh": "《玩家核心》", "tier": 4, "sources": ["wiki"]},
    "Monster Core": {"zh": "《怪物核心》", "tier": 4, "sources": ["wiki"]},
    "Victory Point": {"zh": "胜利点", "tier": 4, "sources": ["wiki"]},
    "skirmish combat": {"zh": "前哨战", "tier": 4, "sources": ["wiki"], "note": "Pathfinder Battlecry! 新规则"},
    "Hellfire Crisis": {"zh": "地狱火危机", "tier": 4, "sources": ["wiki"], "note": "战争事件总称"},
    "Shackles": {"zh": "镣铐群岛", "tier": 4, "sources": ["wiki", "seed"]},
    "Hellbreakers Adventure Path": {"zh": "《地狱破灭》冒险之路", "tier": 4, "sources": ["wiki"], "note": "前传 AP"},
    "Hell's Destiny Adventure Path": {"zh": "《地狱天命》冒险之路", "tier": 4, "sources": ["wiki"]},
    # Important NPCs / Characters
    "Linetta Seacarver": {"zh": "琳奈塔·海凿", "tier": 3, "sources": ["seed"], "note": "AP 主要 NPC，暮爪间谍"},
    "Urdun Gravelhands": {"zh": "厄登·砾掌", "tier": 5, "sources": ["self"], "note": "Linetta 师父；Gravelhands 字面译"},
    "Gorthoklek": {"zh": "戈索克莱克", "tier": 5, "sources": ["self"], "note": "nessari 魔鬼，斯戎家族顾问"},
    "Reginald Cormoth": {"zh": "雷金纳德·科莫斯", "tier": 3, "sources": ["seed", "1.0初校版"], "note": "雄鹰骑士将军，前传被处决"},
    "Abrogail Thrune II": {"zh": "阿波罗盖·斯戎二世", "tier": 4, "sources": ["wiki"], "note": "切利亚斯女王"},
    "Abrogail II": {"zh": "阿波罗盖二世", "tier": 3, "sources": ["seed"]},
    "House Thrune": {"zh": "斯戎家族", "tier": 4, "sources": ["wiki", "seed"]},
    "Triborn Waits": {"zh": "特里博恩·韦茨", "tier": 4, "sources": ["wiki"], "note": "钢隼将军"},
    "Aspexia Rugatonn": {"zh": "阿斯佩西娅·鲁加托恩", "tier": 5, "sources": ["self"], "note": "锁铐骑士团领袖，待 wiki 验证"},
    "Citadel Gheradesca": {"zh": "格拉戴斯卡堡垒", "tier": 4, "sources": ["wiki"], "note": "锁链骑士团总部监狱"},
    # PCs & generic
    "PC": {"zh": "PC", "tier": 3, "sources": ["seed"], "note": "玩家角色，按惯例不译"},
    "PCs": {"zh": "PC 们", "tier": 3, "sources": ["seed"]},
    "GM": {"zh": "GM", "tier": 3, "sources": ["seed"], "note": "主持人，按惯例不译"},
    "dottari": {"zh": "多塔里", "tier": 5, "sources": ["self"], "note": "切利亚斯当地警察的称呼，音译"},
    "Hellknight": {"zh": "地狱骑士", "tier": 4, "sources": ["wiki", "seed"]},
    "Hellknight Hill": {"zh": "地狱骑士山", "tier": 4, "sources": ["wiki", "seed"]},
    "Battle of Hellknight Hill": {"zh": "地狱骑士山之战", "tier": 4, "sources": ["wiki", "1.0初校版"]},
    "Hellbreakers": {"zh": "破狱者", "tier": 3, "sources": ["seed", "1.0初校版"]},
    "Order of the Rack": {"zh": "刑架骑士团", "tier": 4, "sources": ["wiki", "seed"]},
    "Order of the Godclaw": {"zh": "神爪骑士团", "tier": 4, "sources": ["wiki", "seed"]},
    "Order of the Nail": {"zh": "尖钉骑士团", "tier": 4, "sources": ["wiki", "seed"]},
    "Order of the Pike": {"zh": "尖矛骑士团", "tier": 3, "sources": ["seed"], "note": "对应英文 Pike（长矛）；前传称 长枪帮"},
    "Order of the Manacle": {"zh": "镣铐骑士团", "tier": 3, "sources": ["seed-corrected"], "note": "原 SEED「锁铐」改为更准确的「镣铐」对应 Manacle"},
    "Hellknight paralictor": {"zh": "副地狱骑士", "tier": 4, "sources": ["wiki"], "note": "中级地狱骑士军衔"},
    "armiger": {"zh": "扈从", "tier": 4, "sources": ["wiki"], "note": "见习地狱骑士"},
    "warshard": {"zh": "战神碎片", "tier": 4, "sources": ["wiki", "seed"]},
    "Godsrain": {"zh": "神雨", "tier": 4, "sources": ["wiki", "seed"]},
    "Goblinblood Wars": {"zh": "地精血战", "tier": 4, "sources": ["wiki", "seed"]},
    "Eagle Knights": {"zh": "雄鹰骑士", "tier": 4, "sources": ["wiki", "seed"]},
    "Andoran": {"zh": "安多安", "tier": 4, "sources": ["wiki", "seed"]},
    "Cheliax": {"zh": "切利亚斯", "tier": 4, "sources": ["wiki", "seed"]},
    "Isger": {"zh": "依斯嘉", "tier": 4, "sources": ["wiki", "seed"]},
    "Cheliax's puppet state of Isger": {"zh": "切利亚斯的傀儡国依斯嘉", "tier": 3, "sources": ["seed"]},
    "martial law": {"zh": "戒严令", "tier": 4, "sources": ["wiki"]},
}


def main():
    patch_terms(UPDATES)


if __name__ == "__main__":
    main()
