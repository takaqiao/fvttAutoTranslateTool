# -*- coding: utf-8 -*-
"""G4：把确认的跨通道不一致做成批次。

只产出 scratchpad 里的批次文件，不碰 compendium/cn、不碰 lang/cn.json。

  ① compendium（ember 两个孪生包）：历法季节 `Steading` 的中文由「安居」改回
     第 8 节已裁的「耕耘」—— lang 与 .mjs 早已是耕耘，只有 compendium 没跟上。
     只改**英文侧确实出现 `Steading`** 的叶子；库里另有 3 处「不安居民 / 安居于」
     是普通散文，绝不能碰。
  ② lang：三个 ember 键 + 一个 crucible 键，compendium 侧一面倒。
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path("C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project")
OUT = Path("C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/"
           "e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches")
OUT.mkdir(parents=True, exist_ok=True)
SKIP = {"_id", "path", "_variants", "_when", "mapping", "_identity"}
STEADING = re.compile(r"(?<![A-Za-z])Steading")


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


def compendium_steading():
    repo = P / "1-Ember汉化插件"
    for pack in ("ember.adventure.json", "ember.crucible-adventure.json"):
        en = json.loads((repo / "compendium/en" / pack).read_text(encoding="utf-8-sig"))
        cn = json.loads((repo / "compendium/cn" / pack).read_text(encoding="utf-8-sig"))
        rows = []
        walk(en, cn, [], rows)
        batch = {}
        for path, e, c in rows:
            if not c or "安居" not in c or not STEADING.search(e):
                continue
            for m in re.finditer(r".{0,26}安居.{0,26}", c):
                print(f"    {path.split('.')[-1]:<16} …{m.group(0)}…")
            batch[path[len("entries."):] if path.startswith("entries.") else path] = \
                c.replace("安居", "耕耘")
        f = OUT / f"G4__ember__{pack[:-5]}.json"
        f.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {f.name}  {len(batch)} 条\n")


LANG_EMBER = {
    # key: (旧值, 新值, 依据)
    "EMBER.EventTagSocial": ("社交活动", "社交事件",
                             "compendium 社交事件 173 : 社交活动 0；同族 EventTag* "
                             "全是战斗事件/探索事件/全局事件/独特事件"),
    "TYPES.RegionBehavior.ember.footstepSurface": ("足迹地表", "脚步地表",
                                                   "同名 RegionBehavior 的 name 字段 130 处全是脚步地表，"
                                                   "足迹地表 0 处"),
    "EMBER.ATTUNEMENT.ProgressionTitle": ("同调进程", "同调进阶",
                                          "compendium 同调进阶 17 叶 : 同调进程 0"),
}
LANG_CRUCIBLE = {
    "TRAVEL_PACES.Reckless": ("莽撞", "鲁莽",
                              "ember lang 的 EMBER.CONST.TRAVEL.reckless 就是鲁莽；"
                              "compendium 旅行速度语境 15 叶全是鲁莽"),
}


def lang_batches():
    for repo, table, tag in (("1-Ember汉化插件", LANG_EMBER, "ember"),
                             ("2-Crucible汉化插件", LANG_CRUCIBLE, "crucible")):
        cur = json.loads((P / repo / "lang/cn.json").read_text(encoding="utf-8-sig"))
        batch = {}
        for k, (old, new, why) in table.items():
            have = cur.get(k)
            assert have == old, f"{repo} {k} 现值是 {have!r}，不是 {old!r}，先核对"
            batch[k] = new
            print(f"  {tag:9} {k:<48} {old} → {new}   （{why}）")
        f = OUT / f"G4__{tag}__lang.json"
        f.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {f.name}  {len(batch)} 条\n")


if __name__ == "__main__":
    print("== compendium：Steading 安居 → 耕耘")
    compendium_steading()
    print("== lang")
    lang_batches()
