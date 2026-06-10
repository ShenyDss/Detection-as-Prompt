$ProjectRoot = Resolve-Path -LiteralPath "$PSScriptRoot\.."
$CondaRoot = "C:\ProgramData\miniconda3"
$EnvPath = Join-Path $ProjectRoot ".conda\dap"

$env:CONDARC = Join-Path $ProjectRoot ".condarc"
$env:CONDA_NO_PLUGINS = "true"
$env:CONDA_NOTICES = "false"
$env:PATH = "$EnvPath;$EnvPath\Scripts;$CondaRoot;$CondaRoot\Scripts;$CondaRoot\Library\bin;$env:PATH"

Write-Host "Detection-as-Prompt environment is ready:"
Write-Host "  Python: $EnvPath\python.exe"
Write-Host "  CONDARC: $env:CONDARC"
