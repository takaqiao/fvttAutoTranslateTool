# 把所有子文件夹的文件移动到当前目录，并删除空白文件夹
# 然后检查包含“night”的 .mp4 文件，记录其“night”之前的同名文件是否存在

$ErrorActionPreference = "Stop"

$root = Get-Location

# 1) 移动所有子文件夹中的文件到根目录
Get-ChildItem -Path $root -Recurse -File | ForEach-Object {
    $src = $_.FullName
    if ($_.Directory.FullName -ieq $root.Path) {
        return
    }

    $destName = $_.Name
    $dest = Join-Path $root $destName

    if (Test-Path $dest) {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($destName)
        $ext = [System.IO.Path]::GetExtension($destName)
        $i = 1
        do {
            $destName = "{0} ({1}){2}" -f $base, $i, $ext
            $dest = Join-Path $root $destName
            $i++
        } while (Test-Path $dest)
    }

    Move-Item -LiteralPath $src -Destination $dest
}

# 删除空白文件夹（自内向外）
Get-ChildItem -Path $root -Recurse -Directory |
    Sort-Object FullName -Descending |
    ForEach-Object {
        if (-not (Get-ChildItem -LiteralPath $_.FullName -Force)) {
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }

# 2) 查找以“night”结尾的 mp4 文件（仅匹配字母/数字，忽略空格与特殊符号）
$mp4Files = Get-ChildItem -Path $root -File -Filter "*.mp4"

function Normalize-Name([string]$name) {
    return ([regex]::Replace($name, "[^\p{L}\p{Nd}]", "")).ToLowerInvariant()
}

$normalizedMap = @{}
foreach ($f in $mp4Files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $norm = Normalize-Name $name
    if (-not $normalizedMap.ContainsKey($norm)) {
        $normalizedMap[$norm] = New-Object System.Collections.Generic.List[string]
    }
    $normalizedMap[$norm].Add($f.FullName) | Out-Null
}

# 删除重复项：规范化后名字相同的只保留一个
$duplicateRemoved = New-Object System.Collections.Generic.List[string]
foreach ($key in $normalizedMap.Keys) {
    $paths = $normalizedMap[$key]
    if ($paths.Count -gt 1) {
        $keep = $paths | Sort-Object | Select-Object -First 1
        $toDelete = $paths | Where-Object { $_ -ne $keep }
        foreach ($p in $toDelete) {
            Remove-Item -LiteralPath $p -Force
            $duplicateRemoved.Add([System.IO.Path]::GetFileNameWithoutExtension($p)) | Out-Null
        }
    }
}

# 重新获取文件列表与规范化集合
$mp4Files = Get-ChildItem -Path $root -File -Filter "*.mp4"
$normalizedSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$nightBaseSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($f in $mp4Files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $norm = Normalize-Name $name
    $normalizedSet.Add($norm) | Out-Null
    if ($norm.EndsWith("night")) {
        $baseNorm = $norm.Substring(0, $norm.Length - 5)
        if ($baseNorm.Length -gt 0) {
            $nightBaseSet.Add($baseNorm) | Out-Null
        }
    }
}

$missing = New-Object System.Collections.Generic.List[string]
$noNightEnd = New-Object System.Collections.Generic.List[string]
$reverseCutoff = [datetime]::ParseExact("2022-09-25", "yyyy-MM-dd", $null)

foreach ($f in $mp4Files) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $norm = Normalize-Name $name
    if ($norm.EndsWith("night")) {
        $baseNorm = $norm.Substring(0, $norm.Length - 5)
        if ($baseNorm.Length -gt 0 -and -not $normalizedSet.Contains($baseNorm)) {
            $missing.Add($name) | Out-Null
        }
    } elseif ($norm.EndsWith("short")) {
        # 反向检查时忽略 short 结尾
        continue
    } else {
        if ($f.LastWriteTime -lt $reverseCutoff) {
            continue
        }
        if (-not $nightBaseSet.Contains($norm)) {
            $noNightEnd.Add($name) | Out-Null
        }
    }
}

$reportPath = Join-Path $root "night_missing_report.txt"
$missing | Sort-Object | Set-Content -LiteralPath $reportPath -Encoding UTF8

$reverseReportPath = Join-Path $root "night_not_ending_report.txt"
$noNightEnd | Sort-Object | Set-Content -LiteralPath $reverseReportPath -Encoding UTF8

$dupeReportPath = Join-Path $root "night_duplicates_removed.txt"
$duplicateRemoved | Sort-Object | Set-Content -LiteralPath $dupeReportPath -Encoding UTF8

Write-Host "完成。报告已写入: $reportPath"
Write-Host "反向报告已写入: $reverseReportPath"
Write-Host "重复项删除记录已写入: $dupeReportPath"
