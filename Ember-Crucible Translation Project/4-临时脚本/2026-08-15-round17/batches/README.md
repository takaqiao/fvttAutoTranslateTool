# 第十七轮 A3 产出的 compendium 批次（**已于 2026-08-15 全部落地**）

出自 `scan_cross_channel` B 段 `MJS_ORPHAN_CN` 的逐条裁决（详见
`1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs` 里 `WEATHER` 表上方的注释）。
产出时本段只出批次、不写 compendium；落盘由本轮的落盘人统一执行，
落前做过碰撞检测、落后逐字符回读比对。三个文件当时都已 `--force --dry` 过闸：
`applied 1 / REJECTED 0`（markup / no-EN / no-CJK / clobber 全 0）。

落地命令（主控决定是否执行；改的是既有译文，必须带 `--force`）：

```
python "3-常用脚本\qa\apply_translations.py" --repo "1-Ember汉化插件" ^
  --pack ember.adventure.json ^
  --batch "4-临时脚本\2026-08-15-round17\batches\r17-weather-tiers-ember.adventure.json" --force
python "3-常用脚本\qa\apply_translations.py" --repo "1-Ember汉化插件" ^
  --pack ember.crucible-adventure.json ^
  --batch "4-临时脚本\2026-08-15-round17\batches\r17-weather-tiers-ember.crucible-adventure.json" --force
python "3-常用脚本\qa\apply_translations.py" --repo "1-Ember汉化插件" ^
  --pack ember.crucible-character.json ^
  --batch "4-临时脚本\2026-08-15-round17\batches\r17-tempest-moon-ember.crucible-character.json" --force
```

---

## ① `r17-weather-tiers-*`（2 叶，孪生包各一份）—— 优先级高

`Players' Guide.pages.Weather` 是**唯一**一处在天气域里写到档位名的正文，而它现在与
玩家屏幕上的天气图例／「创建天气」下拉**对不上两处**（那批 label 由
`ember-hardcoded-cn.mjs` 的 `WEATHER` 表在数据侧改写，是玩家真正看到的字）：

| 英文原文 | 现译 | 改为 | 依据 |
|---|---|---|---|
| `the weakest strength is a Drizzle` | 毛毛雨 | **细雨** | `WEATHER["Drizzle"]` |
| `the greatest strength is a Tempest` | 风暴 | **狂风暴雨** | `WEATHER["Tempest"]`；「风暴」是本族**中间档** `Storm` 的名字，原译等于把「最强档」写成了中间档 |
| `such as Rain, Fog, Dust Storm, Wildfire` | 降雨、雾、沙尘暴、野火 | **雨**、雾、沙尘暴、野火 | `WEATHER["Rain"]`；这一句列的就是图例里的类型名，四个里只有第一个对不上 |

⚠ 这一处正是 B 段判 `WEATHER.Tempest` 为 `MJS_ORPHAN_CN` 的**唯一同域证据**，而它本身是错的。
判据给的多数写法「风暴 20/21」里，13 叶是玛伊斯的月亮尊号 `The Tempest Moon`、
5 叶是 `the great city of Tempest` 那座城市，都不是天气档位。**`.mjs` 侧不动。**

## ② `r17-tempest-moon-*`（1 叶）—— 与①同源，优先级中

`ember.crucible-character` 的 `Mayis Attunement.description` 把 `The Tempest Moon`
译成「狂澜之月」，全库其余 13 叶（含 `Cosmos.pages.Mayis.subtitle` 这个整叶等值的锚点、
以及同页 `Other Names` 列表）都是「**风暴之月**」。改这一叶归口。

## ⚠ 顺带查出、**本轮不出批次**的两条（留给主控定，见结尾「升级项」）

`ember.crucible-character` 的同调描述里，另有两个月亮尊号与 `Cosmos` 页的 subtitle 不一致：

| 月亮 | crucible-character | Cosmos.subtitle |
|---|---|---|
| `Aura` / The Hollow Moon | 空心之月 | 空洞之月 |
| `Ragen` / The Charred Moon | 焦炭之月 | 焦灼之月 |

`Aura` 那条是 **2026-08-15 第十六轮的批次 `ac-aura-moon-crucible-character.json` 亲手写进去的**，
所以它到底是有意分开还是漏了 Cosmos 那一边，本段无从判断，不擅自出批次。
`Ragen` 同理成组处理更稳。两条都不影响 A3 的任何一条裁决。
