# 名绑定一致性（scan_name_binding.py）

- 场景针脚 570 条：NOT_BOUND 250 / OK 190 / BY_DESIGN 120 / UNCERTAIN 10 / **BROKEN 0**
- 表结果 697 条：NOT_BOUND 474 / NO_LABEL 0 / OK 106 / BY_DESIGN 2 / OUT_OF_SCOPE 114 / UNCERTAIN 1 / **BROKEN 0**

## BROKEN（0）


## UNCERTAIN（11）

- `ember.adventure.json` `Ember Early Access.scenes.Yakoshta Mine.notes.Blue Track`  
  EN `Blue Track` → 目标 EN `Yakoshta Mine`  
  CN `蓝色轨道 Blue Track` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.adventure.json` `Ember Early Access.scenes.The World of Ember.notes.SALISKA`  
  EN `SALISKA` → 目标 EN `Geography`  
  CN `萨利斯卡 SALISKA` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.adventure.json` `Ember Early Access.scenes.The World of Ember.notes.The Silken Sea`  
  EN `The Silken Sea` → 目标 EN `Geography`  
  CN `丝绸之海 The Silken Sea` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.adventure.json` `Ember Early Access.scenes.The World of Ember.notes.Lumarin Homelands`  
  EN `Lumarin Homelands` → 目标 EN `Geography`  
  CN `卢马林家园 Lumarin Homelands` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.adventure.json` `Ember Early Access.scenes.The World of Ember.notes.The Crossing Sea`  
  EN `The Crossing Sea` → 目标 EN `Geography`  
  CN `横渡海 The Crossing Sea` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.crucible-adventure.json` `Ember Early Access.scenes.The World of Ember.notes.SALISKA`  
  EN `SALISKA` → 目标 EN `Geography`  
  CN `萨利斯卡 SALISKA` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.crucible-adventure.json` `Ember Early Access.scenes.The World of Ember.notes.The Silken Sea`  
  EN `The Silken Sea` → 目标 EN `Geography`  
  CN `丝绸之海 The Silken Sea` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.crucible-adventure.json` `Ember Early Access.scenes.The World of Ember.notes.Lumarin Homelands`  
  EN `Lumarin Homelands` → 目标 EN `Geography`  
  CN `卢马林家园 Lumarin Homelands` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.crucible-adventure.json` `Ember Early Access.scenes.The World of Ember.notes.The Crossing Sea`  
  EN `The Crossing Sea` → 目标 EN `Geography`  
  CN `横渡海 The Crossing Sea` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.crucible-adventure.json` `Ember Early Access.scenes.Yakoshta Mine.notes.Blue Track`  
  EN `Blue Track` → 目标 EN `Yakoshta Mine`  
  CN `蓝色轨道 Blue Track` → 目标 CN `{}`  
  针脚的 pageId 悬空（上游删了那一页），会落到条目首页
- `ember.crucible-adventure.json` `Ember Early Access.tables.Corpse Loot.results.35-35.name`  
  EN `Prybar` → 目标 EN `None`  
  CN `撬棒 Prybar` → 目标 CN `{}`  
  目标 id 解析不出来（目标包已导，仍找不到 → 上游悬空 id）

