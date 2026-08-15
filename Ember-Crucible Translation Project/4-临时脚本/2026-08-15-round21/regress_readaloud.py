#!/usr/bin/env python3
r"""真库上的双向变异回测（第二十一轮）—— 不是合成样本，是 `ember.adventure.json` 的真叶。

背景：48 段英文朗读正文里**一个阿拉伯数字都没有**（probe_signal.py 实测），
所以「把 readaloud 纳入数字闸之后全库报 0」这句话本身**不构成任何证据**。
本脚本因此分两条判据各做一次双向回测：

  路径 N（数字闸）：往**临时拷贝**的英文 readaloud 里注入一个数字（中文侧不动）
      ①N 新口径必须**报**该叶  ②N 原样必须**不报**  ③N 旧口径对①N 必须**报不出来**
  路径 T（专名闸，--with-terms）：把**临时拷贝**里某叶的中文 readaloud 正文清空
      ①T 新口径必须**多报出**该段里的专名  ②T 原样不多报  ③T 旧口径必须**多报 0 个**

⚠ 真库一个字都不动：只在 tempfile 目录里拷贝 + 变异，finally 里删掉。
⚠ 探针自证：每一步都打印它实际改了几个字符 / 扫了几条叶；任何计数为 0 直接判 FAIL。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

P = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
QA = os.path.join(P, '3-常用脚本', 'qa')
sys.path.insert(0, QA)
import scan_content_coverage as S   # noqa: E402

REPO = os.path.join(P, '1-Ember汉化插件')
SCAN = os.path.join(QA, 'scan_content_coverage.py')
OUT = os.path.join(P, '4-临时脚本', '2026-08-15-round21')
PACK = 'ember.adventure.json'
RA = re.compile(r'(readaloud\s*=\s*")([^"]*)(")')

ok = True


def check(cond, msg):
    global ok
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    ok = ok and bool(cond)


def leaves(cn_data=None):
    o = []
    cn = cn_data if cn_data is not None else json.load(
        open(os.path.join(REPO, 'compendium', 'cn', PACK), encoding='utf-8'))
    S.walk(json.load(open(os.path.join(REPO, 'compendium', 'en', PACK), encoding='utf-8')).get('entries', {}),
           cn.get('entries', {}), [], o)
    return o


def set_leaf(entries, path, value):
    """按 walk() 的 '.' 连接路径写回一个叶。键名本身可能含 '.'，所以逐级贪婪匹配。"""
    cur, parts = entries, path.split('.')
    while parts:
        for i in range(len(parts), 0, -1):
            k = '.'.join(parts[:i])
            if isinstance(cur, dict) and k in cur:
                nxt = cur[k]
            elif isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
                nxt = cur[int(k)]
            else:
                continue
            if i == len(parts):
                if isinstance(cur, dict):
                    cur[k] = value
                else:
                    cur[int(k)] = value
                return True
            if isinstance(nxt, (dict, list)):
                cur, parts = nxt, parts[i:]
                break
        else:
            return False
    return False


def run(repo, extra):
    out = os.path.join(OUT, '_tmp_report.json')
    cmd = [sys.executable, SCAN, '--repo', repo, '--pack', PACK,
           '--top', '0', '--out', out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit('扫描器跑挂了')
    return json.load(open(out, encoding='utf-8'))


def mkcopy():
    tmp = tempfile.mkdtemp(prefix='cov_regress_')
    for side in ('en', 'cn'):
        d = os.path.join(tmp, 'compendium', side)
        os.makedirs(d)
        shutil.copy2(os.path.join(REPO, 'compendium', side, PACK), os.path.join(d, PACK))
    return tmp


def write_side(tmp, side, path, value):
    p = os.path.join(tmp, 'compendium', side, PACK)
    data = json.load(open(p, encoding='utf-8'))
    okw = set_leaf(data['entries'], path, value)
    json.dump(data, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    back = json.load(open(p, encoding='utf-8'))
    tgt = []
    S.walk(back.get('entries', {}), back.get('entries', {}), [], tgt)
    got = [e for pp, e, _ in tgt if pp == path]
    return okw and bool(got) and got[0] == value


def main():
    rows = leaves()
    # ⚠ 不能用 `'readaloud' in e` 挑靶子：库里 `readaloud` 绝大多数是 CSS 类名
    # （`<section class="block readaloud">`，全 pack 1883 次），那是普通 HTML，本来就在覆盖率里。
    # 真正的增强器参数只有 `readaloud="` 那 48 处。第一版就是这么挑错靶子、变异写了个空的。
    tgt = [(p, e, c) for p, e, c in rows if c and RA.search(e) and RA.search(c)]
    n_class = sum(1 for p, e, c in rows if 'readaloud' in e and not RA.search(e))
    print(f'靶子池：{PACK} 共 {len(rows)} 叶；含 readaloud= **参数**的 {len(tgt)} 叶，'
          f'另有 {n_class} 叶只是用了 readaloud CSS 类名（不算）')
    check(len(tgt) > 0, f'含 readaloud 的靶子叶 {len(tgt)} 条（0 条就没法回测）')
    if not tgt:
        return 1

    # ---------------- 路径 N：数字闸 ----------------
    print('\n路径 N（数字闸）：往英文 readaloud 注入一个数字，中文不动')
    path, e, c = tgt[0]
    print(f'  靶子叶 {path[-70:]}')
    en_digits_now = sum(len(re.findall(r'(?<!\d)\d+(?!\d)', m.group(2))) for m in RA.finditer(e))
    check(en_digits_now == 0,
          f'  前提复核：该叶英文 readaloud 里现有阿拉伯数字 {en_digits_now} 个'
          '（0 = 与 probe_signal 的全库结论一致）')

    tmp = mkcopy()
    try:
        base = run(tmp, [])
        base_hit = {r['path'] for r in base['items']}
        check(base['enricher_text']['leaves'] > 0,
              f'  ②N 原样 + 新口径：捞到可见文本的叶 {base["enricher_text"]["leaves"]} 条，'
              f'EN={base["enricher_text"]["en"]}')
        check(path not in base_hit,
              f'  ②N 原样 + 新口径：靶子叶不报（全 pack 共 {len(base["items"])} 条）')

        injected = [0]

        def inject(m):
            injected[0] += 1
            return m.group(1) + m.group(2) + ' The wardens count 7 lanterns here.' + m.group(3)
        mut_en = RA.sub(inject, e, count=1)
        check(injected[0] == 1 and mut_en != e and '7 lanterns' in mut_en,
              f'  变异写入：往 {injected[0]} 段英文 readaloud 注入了数字 7')
        check(write_side(tmp, 'en', path, mut_en), '  变异落盘并回读确认')

        mn = run(tmp, [])
        hit = [r for r in mn['items'] if r['path'] == path]
        check(bool(hit) and any(x == '7' for x in (hit[0]['missing_numbers'] if hit else [])),
              f'  ①N 变异 + 新口径：靶子叶被报出 {hit[0]["missing_numbers"] if hit else "（没报！）"}')

        mo = run(tmp, ['--no-enricher-text'])
        old_hit = [r for r in mo['items'] if r['path'] == path]
        check(not old_hit,
              f'  ③N 变异 + 旧口径：报不出来（全 pack 共 {len(mo["items"])} 条）'
              ' —— 旧口径在真数据上确实全盲')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---------------- 路径 T：定译专名闸 ----------------
    print('\n路径 T（--with-terms 专名闸）：清空中文 readaloud 正文')
    # 挑一条：中文 readaloud 里含至少一个词表定译（清掉才会有信号）
    gloss = {}
    for k, v in json.load(open(os.path.join(P, '5-其他内容', 'glossary', 'glossary_ec.json'),
                               encoding='utf-8')).items():
        zh = v.split(' ')[0].strip() if isinstance(v, str) else ''
        if len(k) >= 5 and len(zh) >= 2 and S.CJK.search(zh) and not re.search(r'[A-Za-z]', zh):
            gloss[k] = zh
    picked = None
    for p, e2, c2 in tgt:
        en_ra = ' '.join(m.group(2) for m in RA.finditer(e2))
        cn_ra = ' '.join(m.group(2) for m in RA.finditer(c2))
        ms = [k for k in gloss
              if re.search(r'\b' + re.escape(k) + r'\b', en_ra) and gloss[k] in cn_ra]
        # 该定译在**叶的其余部分**不能出现，否则清掉 readaloud 也不会缺
        rest = c2.replace(cn_ra, '')
        ms = [k for k in ms if gloss[k] not in rest]
        if ms:
            picked = (p, e2, c2, ms)
            break
    check(picked is not None,
          f'  找到「专名只出现在 readaloud 里」的叶：{"有" if picked else "没有"}')
    if picked:
        p, e2, c2, ms = picked
        print(f'  靶子叶 {p[-70:]}  专名 {[f"{k}->{gloss[k]}" for k in ms[:4]]}')
        # `--with-terms` 是 7601 锚点 × 每叶一次正则，整包跑要几分钟。
        # 所以这一路把**这一条真叶原文**单独装进一个只有一叶的临时包里跑 ——
        # 文本一个字没改，只是隔离出来，判据行为完全相同（路径名换成 T.leaf）。
        p = 'T.leaf'
        tmp = tempfile.mkdtemp(prefix='cov_regress_T_')
        for side, val in (('en', e2), ('cn', c2)):
            d = os.path.join(tmp, 'compendium', side)
            os.makedirs(d)
            json.dump({'entries': {'T': {'leaf': val}}},
                      open(os.path.join(d, PACK), 'w', encoding='utf-8'), ensure_ascii=False)
        try:
            b = run(tmp, ['--with-terms'])
            b_terms = {r['path']: set(r['missing_terms']) for r in b['items']}
            check(len(b['items']) >= 0 and b['checked'] > 0,
                  f'  ②T 原样 + 新口径 + 专名闸：查了 {b["checked"]} 叶，报 {len(b["items"])} 条')

            killed = [0]

            def kill(m):
                killed[0] += len(m.group(2))
                return m.group(1) + m.group(3)
            mut_cn = RA.sub(kill, c2)
            check(killed[0] > 0, f'  变异写入：清掉中文朗读正文 {killed[0]} 字')
            check(write_side(tmp, 'cn', p, mut_cn), '  变异落盘并回读确认')

            mt = run(tmp, ['--with-terms'])
            m_terms = {r['path']: set(r['missing_terms']) for r in mt['items']}
            new = m_terms.get(p, set()) - b_terms.get(p, set())
            check(bool(new), f'  ①T 变异 + 新口径：该叶**新增**缺失专名 {sorted(new)[:5]}')

            ot = run(tmp, ['--with-terms', '--no-enricher-text'])
            o_terms = {r['path']: set(r['missing_terms']) for r in ot['items']}
            ob = o_terms.get(p, set())
            # 旧口径下 readaloud 的英文压根不在 pe 里，所以这些专名根本不会被要求
            check(not (ob & new),
                  f'  ③T 变异 + 旧口径：同样的专名**报不出来**（旧口径该叶报 {sorted(ob)[:5]}）')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print('\n=> ' + ('全绿' if ok else '**有失败**'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
