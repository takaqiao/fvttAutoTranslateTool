<#
  Deploys the localized Terminal module from the working copy into the Foundry data directory.

  Ships the runtime payload only: tests/, tools/, package.json and the untouched upstream
  audio/ and background/ asset folders stay behind. Refuses to run while Foundry is open,
  prunes artefacts from earlier builds, and verifies every copied file by SHA256.
#>
[CmdletBinding()]
param(
  [string]$Source,
  [string]$Target = "$env:LOCALAPPDATA\FoundryVTT\Data\modules\terminal"
)

$ErrorActionPreference = "Stop"

# Resolved in the body rather than as a parameter default: $PSScriptRoot is not reliably
# populated during param-block evaluation under Windows PowerShell 5.1.
if (-not $Source) {
  $Source = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

if (Get-Process -Name "FoundryVTT" -ErrorAction SilentlyContinue) {
  throw "Foundry Virtual Tabletop is running. Close it before deploying."
}
if (-not (Test-Path -LiteralPath $Target)) {
  throw "Target module directory not found: $Target"
}

$files = @("module.json", "macro.js", "global.css")
$dirs = @("scripts", "lang", "templates", "packs")

# Artefacts from the build that registered the catalog as "zh-CN". Foundry only reads what
# module.json declares, but leaving them behind makes the installed module confusing to inspect.
$stale = @("lang\zh-CN.json", "templates\zh-CN")
foreach ($path in $stale) {
  $full = Join-Path $Target $path
  if (Test-Path -LiteralPath $full) {
    Remove-Item -LiteralPath $full -Recurse -Force
    Write-Output "pruned stale $path"
  }
}

foreach ($file in $files) {
  $from = Join-Path $Source $file
  if (-not (Test-Path -LiteralPath $from)) { throw "Missing source file: $from" }
  Copy-Item -LiteralPath $from -Destination (Join-Path $Target $file) -Force
}
foreach ($dir in $dirs) {
  $from = Join-Path $Source $dir
  if (-not (Test-Path -LiteralPath $from)) { throw "Missing source directory: $from" }
  Copy-Item -LiteralPath $from -Destination $Target -Recurse -Force
}

# Verify the intended payload — and only the intended payload — file by file.
$expected = [System.Collections.ArrayList]::new()
foreach ($file in $files) { [void]$expected.Add($file) }
foreach ($dir in $dirs) {
  $base = Join-Path $Source $dir
  foreach ($item in (Get-ChildItem -LiteralPath $base -Recurse -File)) {
    [void]$expected.Add($item.FullName.Substring($Source.Length).TrimStart('\'))
  }
}

$mismatch = @()
foreach ($relative in $expected) {
  $from = Join-Path $Source $relative
  $to = Join-Path $Target $relative
  if (-not (Test-Path -LiteralPath $to)) { $mismatch += "missing: $relative"; continue }
  $a = (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash
  $b = (Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash
  if ($a -ne $b) { $mismatch += "differs: $relative" }
}

if ($mismatch.Count -gt 0) {
  $mismatch | ForEach-Object { Write-Error $_ }
  throw "Deployment verification failed for $($mismatch.Count) file(s)."
}

Write-Output "Deployed and SHA256-verified $($expected.Count) file(s) to $Target"
