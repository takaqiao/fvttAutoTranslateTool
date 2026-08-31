# A7 注入回测：`R-exclusions-closed` 搬家后仍然会变红

第十七轮 A7 把 `R-exclusions-closed` 读的那张 125 条豁免表从
`4-临时脚本/2026-08-13-round12/findings/EXCLUSIONS.json`（被 `.gitignore` 挡在仓外）
挪到了 `5-其他内容/EXCLUSIONS.same_en_split.json`。

这条断言本轮**刚**从「读旧报告快照」改成「现场重跑扫描器」，搬文件最容易把它搬瘸成
「路径还在、但再也不会红」。所以搬完做了三步实证，不只跑一次绿：

| # | 做法 | 结果 |
|---|---|---|
| ① 真库 | `python 3-常用脚本/qa/assert_resolutions.py` | **46 通过 / 0 失败**，`R-exclusions-closed` detail = 现场扫描：英文唯一串 14191 条 → 分叉 **15 组 / 140 叶**；归档 **125 条**（2 组靠松匹配过闸）。说明它确实读到了搬过去的那份表 |
| ② 注入新分叉 | 副本树里给一处中文加前缀，造一个原表里没有的分叉组 | **FAIL**，`分叉 16 组 / 142 叶`，违反 1 处 = `"Redfiend" Loris Tezran`「该分叉组不在已归档豁免表里」。**只多报注入的那一组，没有连带误报** |
| ③ 复原 + 抽走豁免表 | 把注入的那个文件复原后重跑 → 回到 15 组、绿；再把副本树里的豁免表挪走重跑 | 复原后**绿**（证明②的红确实由注入引起）；抽走表后 **FAIL**「找不到归档豁免表 —— 先确认它是不是又被挪回 4-临时脚本/ 被 .gitignore 挡掉了」。⚠ 这条走的是**非空 bad 列表**，计入**失败**而不是跳过 —— 表没了必须吵出来 |

## 复现

副本树（约 46 MB，全是 `*.json`，本目录被 `.gitignore` 挡着不进仓）**跑完已删**，重建：

```sh
P="C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
T="$P/4-临时脚本/2026-08-15-round17/backtest-A7"
mkdir -p "$T/5-其他内容" "$T/1-Ember汉化插件" "$T/2-Crucible汉化插件"
cp -r "$P/1-Ember汉化插件/compendium" "$T/1-Ember汉化插件/"
cp -r "$P/2-Crucible汉化插件/compendium" "$T/2-Crucible汉化插件/"
cp "$P/5-其他内容/EXCLUSIONS.same_en_split.json" "$T/5-其他内容/"

python "$P/3-常用脚本/qa/assert_resolutions.py" --root "$T"   # 基线：本条应为 ok / 15 组
python "$T/inject_new_split.py"                                # 注入
python "$P/3-常用脚本/qa/assert_resolutions.py" --root "$T"   # 应为 FAIL / 16 组
```

⚠ 副本树只放 `compendium/`，所以 `--root` 下另有 7 条断言会因为**副本里没有
`module.json` / `PROJECT.md` / `lang/`** 而失败（`R-version-matrix`、lang 通道那几条）。
那是回测环境的缺件，与本条无关 —— 判 `R-exclusions-closed` 一条即可，别把 `通过 38 / 失败 8`
当成库的结论。

⚠ 副本树是**旧库的一份死拷贝**。回测完就删，别留着：留着迟早有人对它改译文然后发现改丢了。
