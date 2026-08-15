# -*- coding: utf-8 -*-
import io, re, os, json
R = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
en = json.load(io.open(os.path.join(R, "2-Crucible汉化插件", "compendium", "en", "crucible.rules.json"), encoding="utf-8"))
cn = json.load(io.open(os.path.join(R, "2-Crucible汉化插件", "compendium", "cn", "crucible.rules.json"), encoding="utf-8"))
o = io.open(os.path.join(os.environ.get("OUTDIR", "."), "rep14.txt"), "w", encoding="utf-8")
e = en["entries"]["Combat"]["pages"]["Actions"]["text"]
c = cn["entries"]["Combat"]["pages"]["Actions"]["text"]
for tag in ["Target Scope", "target scope"]:
    for m in re.finditer(r".{0,120}" + tag + r".{0,420}", e):
        o.write("EN: %s\n\n" % re.sub(r"<[^>]+>", " ", m.group(0)))
for m in re.finditer(r".{0,60}目标范围.{0,260}", re.sub(r"<[^>]+>", " ", c)):
    o.write("CN: %s\n\n" % m.group(0))
o.write("---- Wall heading ----\n")
for m in re.finditer(r"<h4>Wall</h4>.{0,240}", e):
    o.write("EN: %s\n" % re.sub(r"<[^>]+>", " ", m.group(0)))
for m in re.finditer(r"<h4>墙</h4>.{0,240}", c):
    o.write("CN: %s\n" % re.sub(r"<[^>]+>", " ", m.group(0)))
o.write("\n---- all h4 in Actions page ----\nEN: %s\nCN: %s\n" % (re.findall(r"<h4>([^<]+)</h4>", e), re.findall(r"<h4>([^<]+)</h4>", c)))
o.close()
print("ok")
