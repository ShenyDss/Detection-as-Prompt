$ProjectRoot = Resolve-Path -LiteralPath "$PSScriptRoot\.."
$Python = Join-Path $ProjectRoot ".conda\dap\python.exe"

& $Python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope ms-swift transformers peft datasets accelerate
& $Python scripts\check_project.py

