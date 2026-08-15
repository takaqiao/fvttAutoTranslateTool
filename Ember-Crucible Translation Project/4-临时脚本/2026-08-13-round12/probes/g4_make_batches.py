#!/usr/bin/env python3
"""G4 出批次。

每条修订写成「叶子路径 + 精确子串替换 + 期望替换次数」，脚本读**当前中文原值**做
定点替换后整叶输出。这样有三个保证：

1. 未改动部分与库里逐字节一致（不靠手抄整叶）；
2. `<hN id="...">` 的 id 属性天然原样保留 —— 交付前仍会断言 id 计数不减；
3. 期望次数对不上就报错，不会静默漏改或多改。

孪生包（ember.adventure / ember.crucible-adventure）逐条自动各写一份。
"""
import json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
OUT = os.path.join(ROOT, '4-临时脚本', '2026-08-13-round12', 'batches')
EMBER, CRUC = '1-Ember汉化插件', '2-Crucible汉化插件'
TWINS = ('ember.adventure.json', 'ember.crucible-adventure.json')

# (repo, 包, 叶子路径, [(旧串, 新串, 期望次数)], 备注)
FIXES = [
    # ---- 阵营：上游把缩写改成全称，中文还停在缩写上 ----
    (EMBER, TWINS, 'Ember Early Access.journals.Smoldering Cinders.pages.A Conflagration of Lumé.text',
     [('伊斯卡（CN，', '伊斯卡（混乱中立，', 1),
      ('萨拉（CG，', '萨拉（混乱善良，', 1)],
     'EN 由 (CN/(CG 改为 (Chaotic Neutral/(Chaotic Good；玩家可见字段的阵营'),

    # ---- 链接显示名停在旧英文 ----
    (EMBER, TWINS, 'Ember Early Access.journals.Ooze Control.pages.Saving Jasper.text',
     [('#entrance-bridge]{入口桥}', '#entrance-bridge]{瀑桥}', 1)],
     'EN 标签 Entrance Bridge -> Waterfall Bridge；同一锚点在 Loading Zone 已作「瀑桥」'),

    # ---- 物品名与词条不符，玩家照名字找不到东西 ----
    (EMBER, TWINS, 'Ember Early Access.journals.Yakoshta Mine.pages.Supply Cache.text',
     [('贾斯珀钥匙环上的储藏室钥匙', '贾斯珀的钥匙圈上的储藏室钥匙', 1)],
     "物品 Jasper's Key Ring 的中文名是「贾斯珀的钥匙圈」"),

    # ---- 同句内数字体例自相矛盾 ----
    (EMBER, TWINS, 'Ember Early Access.journals.Lightless Halls.pages.The Chained Door.text',
     [('约为90吨', '约为九十吨', 1)],
     'EN 90 tons -> ninety tons；同句已作「二十英尺／五英尺／十五英尺」'),

    # ---- 主语被砍成通名，与同场景的「烽灯旅巡逻者」混淆 ----
    # 这条只在 crucible-adventure 里有（ember.adventure 该敌手没有 description 叶）
    (EMBER, ('ember.crucible-adventure.json',),
     'Ember Early Access.actors.Wandren Patroller.items.Multiattack.description',
     [('<p>巡逻者用其', '<p>万德伦巡逻者用其', 1)],
     '同库另两条 Multiattack（肯瑞斯／肯瑞斯空壳）都保留全名'),

    # ---- 词条离群：区域地图 998 : 地区地图 12 ----
    (EMBER, TWINS, "Ember Early Access.journals.A Brush With Death.pages.The Bard's Trail.text",
     [('发生在地区地图上', '发生在区域地图上', 1)],
     'Region Map 全库作「区域地图」998 次'),

    # ---- Side Quest / Main Quest 限定词被吞 ----
    (EMBER, TWINS, 'Ember Early Access.journals.Arctus Plateau Gazetteer.pages.Brevin.text',
     [('{棘手的困境}任务中', '{棘手的困境}支线任务中', 1)],
     'EN 新增 Side Quest 限定；全库「支线任务」238 次'),
    (EMBER, TWINS, 'Ember Early Access.journals.Writhing Grave.pages.Waterfall Nexus.text',
     [('<h4>月上之旅任务</h4>', '<h4>月上之旅支线任务</h4>', 2),
      ('{月上之旅}任务中的', '{月上之旅}支线任务中的', 2)],
     'EN Over the Moon Side Quest；该页 dnd5e/crucible 两套写了两遍，故各 2 处'),
    (EMBER, TWINS, 'Ember Early Access.journals.Ooze Control.pages.Alchemical Decisions.text',
     [('emberSignalInten] 任务中的', 'emberSignalInten] 支线任务中的', 1)],
     'EN ... Side Quest'),
    (EMBER, TWINS, 'Ember Early Access.journals.Flotsam Canal Market.pages.Cluttered Dock.text',
     [('localColorSketchy]] 任务事件之前', 'localColorSketchy]] 支线任务事件之前', 1),
      ('{地方色彩} 任务事件期间', '{地方色彩} 支线任务事件期间', 1)],
     'EN Local Color Side Quest（两处）'),
    (EMBER, TWINS, 'Ember Early Access.journals.Ooze Control.pages.Ooze Friends.text',
     [('<p>这个任务没有后续步骤了', '<p>这个支线任务没有后续步骤了', 1)],
     'EN there are no further steps in this Side Quest'),
    (EMBER, TWINS, 'Ember Early Access.journals.Local Color.pages.Commissioned Work.text',
     [('<p>随着这项任务完成', '<p>随着这项支线任务完成', 1)],
     'EN With the completion of this Side Quest'),
    (EMBER, TWINS, 'Ember Early Access.journals.An Old Friend.pages.Traveling with Lyla.text',
     [('在此次任务中最近遭遇', '在此次主线任务中最近遭遇', 1)],
     'EN while on this Main Quest；全库「主线任务」220 次'),

    # ---- crucible：construction 区块标题离群 ----
    (CRUC, ('crucible.rules.json',), 'Combat.pages.Movement.text',
     [('<h4>建设中</h4>', '<h4>施工中</h4>', 1)],
     'class="construction" 标题全库 268 次作「施工中」，仅此 1 处作「建设中」'),
]


def load(p):
    return json.loads(open(p, encoding='utf-8-sig').read())


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


def main():
    os.makedirs(OUT, exist_ok=True)
    cache, batches, errs, n = {}, {}, [], 0
    for repo, packs, path, reps, why in FIXES:
        for pack in packs:
            key = (repo, pack)
            if key not in cache:
                d = {}
                leaves(load(os.path.join(ROOT, repo, 'compendium', 'cn', pack)).get('entries', {}), [], d)
                cache[key] = d
            cur = cache[key].get(path)
            if cur is None:
                errs.append(f'缺叶子 {repo}/{pack} :: {path}')
                continue
            new = cur
            for old, rep, want in reps:
                got = new.count(old)
                if got != want:
                    errs.append(f'{pack} :: {path[-46:]} :: {old!r} 命中 {got} 期望 {want}')
                    continue
                new = new.replace(old, rep)
            if new == cur:
                errs.append(f'{pack} :: {path[-46:]} :: 无变化')
                continue
            if new.count('id="') < cur.count('id="'):
                errs.append(f'{pack} :: {path[-46:]} :: id="" 数量下降！')
                continue
            batches.setdefault((repo, pack), {})[path] = new
            n += 1
    for (repo, pack), data in sorted(batches.items()):
        tag = '1' if repo == EMBER else '2'
        fp = os.path.join(OUT, f'G4.{tag}.{pack}')
        json.dump(data, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'  {len(data):3d} 叶  -> {fp}')
    print(f'共写出 {n} 条')
    for e in errs:
        print('  !!', e)
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
