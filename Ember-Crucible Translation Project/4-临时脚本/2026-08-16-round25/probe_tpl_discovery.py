import re, io, os, glob

os.chdir(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")

PAT = r"modules/ember/templates/[A-Za-z0-9_./-]*\.(?:hbs|html)"

refs = set()
for f in glob.glob("scripts/*.mjs"):
    t = io.open(f, encoding="utf-8", errors="replace").read()
    refs |= set(re.findall(PAT, t))

allf = set()
for root, d, fs in os.walk("templates"):
    for f in fs:
        allf.add("modules/ember/" + os.path.join(root, f).replace(os.sep, "/"))

print("total template files:", len(allf))
print("referenced by scripts:", len(allf & refs))
missing = allf - refs
print("missing from script refs:", len(missing))
for p in sorted(missing):
    print("   ", p)

hrefs = set()
for p in sorted(allf):
    fp = p.replace("modules/ember/", "")
    t = io.open(fp, encoding="utf-8", errors="replace").read()
    hrefs |= set(re.findall(PAT, t))
print("of those, covered by hbs partial refs:", len(missing & hrefs))
print("still missing:", sorted(missing - hrefs))
