$ErrorActionPreference = 'Stop'

$baseDir = 'c:\Users\Taka\Desktop\fvtt\sf2adventure'
$cnName = 'sf2e-murder-in-metal-city.adventures.json'
$origPath = (
  Get-ChildItem -LiteralPath $baseDir -File |
    Where-Object {
      $_.Name -like '*.adventures*.json' -and
      $_.Name -ne $cnName
    } |
    Select-Object -First 1
).FullName
$cnPath = (Get-ChildItem -LiteralPath $baseDir -File | Where-Object { $_.Name -eq $cnName } | Select-Object -First 1).FullName
$outPath = Join-Path $baseDir 'compare_html_report.txt'

if (-not $origPath -or -not $cnPath) {
  throw 'Cannot locate original or translated file in sf2adventure.'
}

$orig = Get-Content -LiteralPath $origPath -Raw | ConvertFrom-Json
$cn = Get-Content -LiteralPath $cnPath -Raw | ConvertFrom-Json

function Get-TextNodes {
  param($obj, [string]$path = 'root')

  if ($null -eq $obj) { return }
  if ($obj -is [string]) { return }

  if ($obj -is [System.Collections.DictionaryEntry]) {
    $entryPath = "$path/$($obj.Key)"
    if ($obj.Key -eq 'text' -and $obj.Value -is [string]) {
      [pscustomobject]@{ Path = $path; Text = $obj.Value }
    }
    Get-TextNodes -obj $obj.Value -path $entryPath
    return
  }

  if ($obj -is [System.Collections.IDictionary]) {
    foreach ($k in $obj.Keys) {
      $v = $obj[$k]
      $childPath = "$path/$k"
      if ($k -eq 'text' -and $v -is [string]) {
        [pscustomobject]@{ Path = $path; Text = $v }
      }
      Get-TextNodes -obj $v -path $childPath
    }
    return
  }

  if ($obj -is [pscustomobject]) {
    foreach ($p in $obj.PSObject.Properties) {
      $childPath = "$path/$($p.Name)"
      if ($p.Name -eq 'text' -and $p.Value -is [string]) {
        [pscustomobject]@{ Path = $path; Text = $p.Value }
      }
      Get-TextNodes -obj $p.Value -path $childPath
    }
    return
  }

  if ($obj -is [System.Collections.IEnumerable]) {
    $i = 0
    foreach ($item in $obj) {
      Get-TextNodes -obj $item -path ($path + '[' + $i + ']')
      $i++
    }
    return
  }
}

$origNodes = @(Get-TextNodes -obj $orig)
$cnNodes = @(Get-TextNodes -obj $cn)

$origMap = @{}
foreach ($n in $origNodes) { $origMap[$n.Path] = $n.Text }
$cnMap = @{}
foreach ($n in $cnNodes) { $cnMap[$n.Path] = $n.Text }

$common = @($origMap.Keys | Where-Object { $cnMap.ContainsKey($_) })
$tags = @('p','div','section','span','aside','ul','li','h1','h2','h3','h4','h5','h6')
$rows = @()

foreach ($k in $common) {
  $ot = $origMap[$k]
  $ct = $cnMap[$k]
  $lenRatio = [math]::Round(($ct.Length / [math]::Max($ot.Length,1)), 2)

  $newImbalance = 0
  foreach ($t in $tags) {
    $oo = [regex]::Matches($ot, "<$t\\b").Count
    $oc = [regex]::Matches($ot, "</$t>").Count
    $co = [regex]::Matches($ct, "<$t\\b").Count
    $cc = [regex]::Matches($ct, "</$t>").Count
    if (($oo -eq $oc) -and ($co -ne $cc)) { $newImbalance++ }
  }

  $emptyAction = [regex]::Matches($ct, '<section class="action"[^>]*>\s*(?:<h2[^>]*>.*?</h2>\s*)?<hr \/>\s*</section>', 'Singleline').Count
  $orphanDot = [regex]::Matches($ct, '<p>\.</p>').Count
  $extraImg = [regex]::Matches($ct, '<img [^>]*>').Count - [regex]::Matches($ot, '<img [^>]*>').Count
  $extraUuidPara = [regex]::Matches($ct, '<p>\s*@UUID\[[^\]]+\]\{[^\}]+\}\s*</p>', 'Singleline').Count - [regex]::Matches($ot, '<p>\s*@UUID\[[^\]]+\]\{[^\}]+\}\s*</p>', 'Singleline').Count
  $doubleActionGlyph = [regex]::Matches($ct, '<span class="action-glyph">2</span>\s*</p>\s*<hr \/>', 'Singleline').Count

  if ($newImbalance -gt 0 -or $lenRatio -ge 1.8 -or $emptyAction -gt 0 -or $orphanDot -gt 0 -or $extraImg -ge 2 -or $extraUuidPara -ge 2 -or $doubleActionGlyph -gt 0) {
    $rows += [pscustomobject]@{
      Path = $k
      LenRatio = $lenRatio
      NewImbalance = $newImbalance
      EmptyAction = $emptyAction
      OrphanDot = $orphanDot
      ExtraImg = $extraImg
      ExtraUuidPara = $extraUuidPara
      DoubleActionGlyph = $doubleActionGlyph
    }
  }
}

$globalOrig = Get-Content -LiteralPath $origPath -Raw
$globalCn = Get-Content -LiteralPath $cnPath -Raw

$summary = @()
$summary += "orig_file=$origPath"
$summary += "cn_file=$cnPath"
$summary += "orig_size=$((Get-Item -LiteralPath $origPath).Length)"
$summary += "cn_size=$((Get-Item -LiteralPath $cnPath).Length)"
$summary += "orig_text_nodes=$($origNodes.Count)"
$summary += "cn_text_nodes=$($cnNodes.Count)"
$summary += "common_text_nodes=$($common.Count)"
$summary += "flagged_nodes=$($rows.Count)"
$summary += "global_orphan_dot_cn=$([regex]::Matches($globalCn,'<p>\.</p>').Count)"
$summary += "global_orphan_dot_orig=$([regex]::Matches($globalOrig,'<p>\.</p>').Count)"
$summary += "global_two_semicolon_cn=$([regex]::Matches($globalCn,'2;</p>').Count)"
$summary += "global_three_semicolon_cn=$([regex]::Matches($globalCn,'3;</p>').Count)"
$summary += ""

$top = $rows | Sort-Object @{Expression='NewImbalance';Descending=$true}, @{Expression='LenRatio';Descending=$true} | Select-Object -First 80
foreach ($r in $top) {
  $summary += ("PATH=" + $r.Path)
  $summary += ("  ratio=" + $r.LenRatio + " newImbalance=" + $r.NewImbalance + " emptyAction=" + $r.EmptyAction + " orphanDot=" + $r.OrphanDot + " extraImg=" + $r.ExtraImg + " extraUuidPara=" + $r.ExtraUuidPara + " doubleActionGlyph=" + $r.DoubleActionGlyph)
}

Set-Content -LiteralPath $outPath -Value $summary -Encoding UTF8
Write-Output "report_written=$outPath"
