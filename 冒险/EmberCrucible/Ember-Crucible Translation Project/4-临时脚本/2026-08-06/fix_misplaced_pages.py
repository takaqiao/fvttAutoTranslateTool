#!/usr/bin/env python3
"""Put two misplaced translations right (see detect_swapped_pages.py 的实测结论).

`Spellbreaker Tower`：该卷有两间储藏室，开头 readaloud 逐字相同。中文实际是
`jyEjb9CXfSzRRZCf`（水/酒/灯油那间，还带 Useful Items / Curious Damage 两节）的译文，
却坐在 `Storage`（床单/囚服那间）上。搬过去，`Storage` 退回待译。

`Mythspire Observatory`：中文写的是 `CecLJBaIh4oKCvR8`（方厅升降梯）。第 2 批译者已经
把它写进了正确路径，但 `Ancient Lift`（三十尺石环，通往通路）上那份重复的旧文本还在，
描述的是另一个房间。删掉，退回待译。

**删掉错的译文比留着强**：留着的话覆盖率算它已译、待译清单永远不列它，玩家读到的是
另一个房间的描述；删掉之后它变回英文，并且会重新出现在待译清单里被排进下一轮。

  python fix_misplaced_pages.py [--write]
"""
from __future__ import annotations
import argparse
import json
import os

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
PACK = "ember.crucible-adventure.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    cn_path = os.path.join(P, "1-Ember汉化插件", "compendium", "cn", PACK)
    cn = json.load(open(cn_path, encoding="utf-8"))
    CJ = cn["entries"]["Ember Early Access"]["journals"]

    # 1. Spellbreaker Tower：搬过去，原位退回待译
    sp = CJ["Spellbreaker Tower"]["pages"]
    text = sp["Storage"].get("text")
    if text:
        sp.setdefault("jyEjb9CXfSzRRZCf", {})["text"] = text
        sp["Storage"].pop("text", None)
        print(f"Spellbreaker Tower: Storage({len(text)} 字) -> jyEjb9CXfSzRRZCf；Storage 退回待译")
    else:
        print("Spellbreaker Tower: 已处理过，跳过")

    # 2. Mythspire：正确路径已由第 2 批写入，删掉留在旧路径上的重复
    mp = CJ["Mythspire Observatory"]["pages"]
    dup = mp.get("Ancient Lift", {}).get("text")
    moved = mp.get("CecLJBaIh4oKCvR8", {}).get("text")
    if dup and moved and dup.strip() == moved.strip():
        mp["Ancient Lift"].pop("text", None)
        print(f"Mythspire Observatory: 删掉 Ancient Lift 上的重复文本({len(dup)} 字)，退回待译")
    elif dup and not moved:
        print("!! CecLJBaIh4oKCvR8 还没有译文，先别删 Ancient Lift")
    else:
        print("Mythspire Observatory: 已处理过，跳过")

    if a.write:
        with open(cn_path, 'w', encoding='utf-8') as f:
            json.dump(cn, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print("\n已写盘")
    else:
        print("\n(未加 --write，什么都没写)")


main()
