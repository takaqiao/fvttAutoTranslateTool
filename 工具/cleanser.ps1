# 进入目标文件夹后运行此脚本
Get-ChildItem -Recurse -Directory | ForEach-Object {
    $sig = Join-Path $_.FullName "signature.json"
    if (Test-Path $sig) { Remove-Item $sig -Force }

    $mod = Join-Path $_.FullName "module.json"
    if (Test-Path $mod) {
        (Get-Content $mod -Raw) -replace '"protected"\s*:\s*true', '"protected": false' |
            Set-Content $mod -NoNewline
    }
}