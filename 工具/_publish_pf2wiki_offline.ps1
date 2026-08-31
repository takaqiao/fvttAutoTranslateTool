# Stage source-only files for github push to a new repo `pf2-wiki-offline`.
# Excludes 1.84+GB built corpus + 3+GB scraper output + Rust target/.
$src = 'C:\Users\Taka\Desktop\fvtt'
$dest = 'C:\Users\Taka\pf2-wiki-offline'

if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory $dest | Out-Null

# Helper: copy with rel path preserved
function CopyRel($relPath) {
    $s = Join-Path $src $relPath
    $d = Join-Path $dest $relPath
    if (-not (Test-Path $s)) { Write-Output "SKIP (missing): $relPath"; return }
    $dParent = Split-Path $d -Parent
    if (-not (Test-Path $dParent)) { New-Item -ItemType Directory $dParent -Force | Out-Null }
    if ((Get-Item $s).PSIsContainer) {
        Copy-Item $s $d -Recurse -Force
    } else {
        Copy-Item $s $d -Force
    }
    Write-Output "OK: $relPath"
}

# _wiki_full_v2 — source only (no built pages, no images dir, no index/shards)
CopyRel '_wiki_full_v2\build_v2.py'
CopyRel '_wiki_full_v2\build_browse_v2.py'
CopyRel '_wiki_full_v2\build_class_hubs_v2.py'
CopyRel '_wiki_full_v2\build_search_v2.py'
CopyRel '_wiki_full_v2\serve.py'
CopyRel '_wiki_full_v2\index.html'
CopyRel '_wiki_full_v2\search.html'
CopyRel '_wiki_full_v2\_snippets'

# assets: CSS + JS + favicon (not large 4k images)
New-Item -ItemType Directory "$dest\_wiki_full_v2\assets" -Force | Out-Null
Get-ChildItem "$src\_wiki_full_v2\assets" -File -Filter '*.css' -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName "$dest\_wiki_full_v2\assets\" -Force
}
Get-ChildItem "$src\_wiki_full_v2\assets" -File -Filter '*.js' -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName "$dest\_wiki_full_v2\assets\" -Force
}
foreach ($f in @('favicon.ico', 'favicon.png', 'site_avatar_s.webp', 'site_avatar_m.webp', 'site_avatar_l.webp')) {
    $sf = "$src\_wiki_full_v2\assets\$f"
    if (Test-Path $sf) { Copy-Item $sf "$dest\_wiki_full_v2\assets\" -Force }
}
# native CSS-referenced assets (~17 small files, fonts + sprites)
if (Test-Path "$src\_wiki_full_v2\assets\native") {
    Copy-Item "$src\_wiki_full_v2\assets\native" "$dest\_wiki_full_v2\assets\native" -Recurse -Force
}

# pf2wiki-scraper — Python source only (no .venv, no out_v2, no .browser-profile)
$scraperSrc = "$src\pf2wiki-scraper"
$scraperDst = "$dest\pf2wiki-scraper"
New-Item -ItemType Directory $scraperDst -Force | Out-Null
Get-ChildItem $scraperSrc -File -Filter '*.py' -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName "$scraperDst\" -Force
}
foreach ($f in @('README.md', 'requirements.txt', '.gitignore', 'run_v2_pipeline.ps1')) {
    $sf = "$scraperSrc\$f"
    if (Test-Path $sf) { Copy-Item $sf "$scraperDst\" -Force }
}

# src-tauri (full source)
CopyRel 'src-tauri\Cargo.toml'
CopyRel 'src-tauri\Cargo.lock'
CopyRel 'src-tauri\build.rs'
CopyRel 'src-tauri\tauri.conf.json'
CopyRel 'src-tauri\README.md'
CopyRel 'src-tauri\src'
CopyRel 'src-tauri\icons'

# placeholders + docs
CopyRel '_tauri_placeholder'
CopyRel 'agent_outputs_v2'
CopyRel 'LICENSE'

Write-Output ''
Write-Output '=== final size of pf2-wiki-offline source mirror ==='
$totalSize = (Get-ChildItem $dest -Recurse -File | Measure-Object -Sum Length).Sum
$totalCount = (Get-ChildItem $dest -Recurse -File | Measure-Object).Count
Write-Output ("files: $totalCount, size: " + [math]::Round($totalSize/1MB, 2) + ' MB')
Write-Output ''
Write-Output '=== top-level layout ==='
Get-ChildItem $dest | Select-Object Name, @{N='Type';E={if($_.PSIsContainer){'DIR'}else{'FILE'}}}, @{N='KB';E={if($_.PSIsContainer){[math]::Round((Get-ChildItem $_.FullName -Recurse -EA 0 | Measure-Object -Sum Length).Sum/1KB,1)}else{[math]::Round($_.Length/1KB,1)}}} | Format-Table -AutoSize
