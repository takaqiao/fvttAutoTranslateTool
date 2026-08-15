# -*- coding: utf-8 -*-
import io, re, os, json
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TAG = re.compile(r"<[^>]+>")

def flat(d):
    out = {}
    def w(ob, p):
        if isinstance(ob, dict):
            for k, v in ob.items(): w(v, p + [str(k)])
        elif isinstance(ob, list):
            for i, v in enumerate(ob): w(v, p + ["[%d]" % i])
        elif isinstance(ob, str): out[".".join(p)] = ob
    w(d, [])
    return out

o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep15.txt"), "w", encoding="utf-8")

def show(repo, fn, key, pats):
    E = flat(json.load(io.open(os.path.join(R, repo, "compendium", "en", fn), encoding="utf-8")))
    C = flat(json.load(io.open(os.path.join(R, repo, "compendium", "cn", fn), encoding="utf-8")))
    e, c = E.get(key, ""), C.get(key, "")
    o.write("\n##### %s :: %s\n" % (fn, key))
    for pe, pc in pats:
        for m in re.finditer(pe, TAG.sub(" ", e)):
            o.write("  EN … %s …\n" % m.group(0))
        for m in re.finditer(pc, TAG.sub(" ", c)):
            o.write("  CN … %s …\n" % m.group(0))

show("2-Crucible汉化插件", "crucible.rules.json", "entries.Character Creation.pages.Finishing Touches.text",
     [(r".{0,50}(Age|Pronouns|Public Biography|Private Biography|Scaled Price).{0,60}",
       r".{0,50}(年代|代称|公开（Public）传记|私密（Private）传记|缩放价格|年龄|代词|按比例定价).{0,60}")])
show("2-Crucible汉化插件", "crucible.rules.json", "entries.Equipment.pages.Armor.text",
     [(r".{0,40}(Heavy Armor|Mundane).{0,60}", r".{0,40}(重型护甲|重甲|凡俗|凡品).{0,60}")])
show("2-Crucible汉化插件", "crucible.rules.json", "entries.Adversaries.pages.Swarms.text",
     [(r".{0,40}Minion.{0,60}", r".{0,40}(爪牙|仆从).{0,60}")])
show("2-Crucible汉化插件", "crucible.rules.json", "entries.Character Creation.pages.Overview.text",
     [(r".{0,40}\bGroup\b.{0,60}", r".{0,40}(群组|团队).{0,60}")])
show("1-Ember汉化插件", "ember.crucible-adventure.json",
     "entries.Ember Early Access.items.Moon Ring.effects.Lunar Shield.description",
     [(r".{0,200}", r".{0,200}")])

o.write("\n##### lang 对应键\n")
for repo in ["2-Crucible汉化插件", "1-Ember汉化插件"]:
    LE = flat(json.load(io.open(os.path.join(R, repo, "lang", "en.json"), encoding="utf-8")))
    LC = flat(json.load(io.open(os.path.join(R, repo, "lang", "cn.json"), encoding="utf-8")))
    for k in sorted(LE):
        if LE[k].strip() in ("Age", "Pronouns", "Public Biography", "Private Biography", "Scaled Price",
                             "Heavy Armor", "Mundane", "Minion", "Group", "Waxing", "Waning", "Full",
                             "Token", "Wall", "Restoration"):
            o.write("  [%s] %-50s EN=%-20s CN=%s\n" % (repo[:2], k[-50:], LE[k], LC.get(k)))

o.write("\n##### 令牌 vs 指示物 计数\n")
import collections
cnt = collections.Counter()
for repo in ["1-Ember汉化插件", "2-Crucible汉化插件"]:
    d = os.path.join(R, repo, "compendium", "cn")
    for fn in os.listdir(d):
        s = io.open(os.path.join(d, fn), encoding="utf-8").read()
        cnt["compendium 令牌"] += s.count("令牌")
        cnt["compendium 指示物"] += s.count("指示物")
    s = io.open(os.path.join(R, repo, "lang", "cn.json"), encoding="utf-8").read()
    cnt["lang 令牌 " + repo[:2]] = s.count("令牌")
    cnt["lang 指示物 " + repo[:2]] = s.count("指示物")
for k, v in cnt.items():
    o.write("  %s = %d\n" % (k, v))
o.close()
print("ok")
