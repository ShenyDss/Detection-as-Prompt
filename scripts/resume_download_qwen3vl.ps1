$ProjectRoot = Resolve-Path -LiteralPath "$PSScriptRoot\.."
$Python = Join-Path $ProjectRoot ".conda\dap\python.exe"

Write-Host "Resuming Qwen/Qwen3-VL-2B-Instruct download..."
& $Python scripts\download_modelscope_model.py

Write-Host ""
Write-Host "Checking model integrity..."
& $Python scripts\check_model.py

