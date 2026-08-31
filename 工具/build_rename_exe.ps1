$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw "Python not found. Please install Python and ensure python or py is in PATH."
}

& $python.Source -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller not installed. Installing..."
    & $python.Source -m pip install pyinstaller
}

& $python.Source -m PyInstaller --onefile --name rename_from_csv rename_from_csv.py
Write-Host "Done. exe is at dist\\rename_from_csv.exe"
