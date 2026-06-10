# Detection-as-Prompt

> English version follows the Chinese version.

## 中文说明

Detection-as-Prompt 是一个面向织物/工业缺陷决策的研究型代码框架。项目将检测器输出转换为可审查的缺陷假设，再结合不确定性路由、视觉语言模型审查、知识图谱约束和结构化评估，形成如下流程：

```text
Detector -> Hypotheses -> Detection Prompt -> Routing -> VLM Review -> KG Constraint -> Structured Decision -> Metrics
```

本仓库适合上传源码、配置、文档和可复现实验脚本。真实数据集、训练日志、训练记录、模型权重、本地环境和推理/训练输出不应提交到 GitHub。

### 项目结构

```text
configs/    实验、模型、数据、训练和消融配置
docs/       数据格式、标注规范、知识图谱和实验协议文档
scripts/    数据处理、检测、提示构建、VLM 审查、评估和训练入口
src/dap/    Detection-as-Prompt 核心 Python 包
data/       本地数据目录，默认不上传真实/生成数据
models/     本地模型权重目录，默认不上传
outputs/    训练、推理、评估和可视化输出目录，默认不上传
```

### 环境安装

建议使用 Python 3.10 或更新版本，并在仓库外或本地虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

当前基础依赖很轻量：

```text
PyYAML>=6.0.1
```

如果需要运行 YOLO、Qwen3-VL 或 LoRA 相关流程，还需要按实际硬件和模型版本安装 `ultralytics`、`torch`、`transformers`、`peft`、`datasets`、`accelerate` 等额外依赖。

### 快速检查

```powershell
python scripts\print_config.py --config-dir configs
python scripts\check_project.py
```

### 最小演示流程

以下命令可生成 mock 检测结果、构建提示、执行 mock VLM 审查并导出评估结果：

```powershell
python scripts\02_mock_predictions_to_hypotheses.py --write-demo-input
python scripts\05_build_prompts.py
python scripts\06_route_hypotheses.py
python scripts\07_run_vlm_review.py --mode mock
python scripts\08_evaluate.py
```

生成文件会写入 `data/` 或 `outputs/` 下，这些路径已在 `.gitignore` 中排除。

### 真实数据准备

真实数据集不随仓库发布。请在本地准备如下目录：

```text
SM_norm/
NZB_dataset/
data/images/
data/yolo/
data/imported/
```

Labelme 标注导入示例：

```powershell
python scripts\12_import_labelme.py `
  --labelme-dir SM_norm/norm/images `
  --output-dir data/imported/sm_norm `
  --class-map configs/class_map.sm_nzb.json `
  --include-empty

python scripts\12_import_labelme.py `
  --json-dir NZB_dataset/json `
  --image-dir NZB_dataset/images `
  --output-dir data/imported/nzb_dataset `
  --class-map configs/class_map.sm_nzb.json `
  --include-empty
```

构建 YOLO 数据集：

```powershell
python scripts\13_build_yolo_dataset.py `
  --class-map configs/class_map.sm_nzb.json `
  --output-dir data/yolo/sm_nzb `
  --include-unlabeled-images `
  --val-ratio 0.2 `
  --seed 42
```

### 模型与权重

模型权重不随仓库发布。请将下载后的模型放到本地 `models/`，例如：

```text
models/Qwen3-VL-2B-Instruct
```

YOLO 初始权重、训练得到的 `best.pt`/`last.pt`、LoRA adapter 和 `.safetensors` 文件都已被 `.gitignore` 排除。

### 训练与评估

YOLO dry run：

```powershell
python scripts\20_train_yolo26n.py --dry-run
```

YOLO 训练：

```powershell
python scripts\20_train_yolo26n.py
```

Qwen3-VL LoRA dry run：

```powershell
python scripts\10_train_lora.py --dry-run
```

真实 LoRA 训练仍需要补充与具体 Qwen-VL checkpoint 匹配的训练循环和依赖环境。

### GitHub 上传前检查

上传前建议检查是否还有大文件或敏感实验产物：

```powershell
git status --short
git ls-files --others --ignored --exclude-standard
git ls-files | Select-String -Pattern "\.pt$|\.pth$|\.safetensors$|\.ckpt$|data/|models/|outputs/"
```

如果某些大文件已经被 Git 跟踪，仅添加 `.gitignore` 不会自动移除。需要执行：

```powershell
git rm --cached <path>
```

然后再提交。

## English

Detection-as-Prompt is a research codebase for defect decision making in textile/industrial inspection. It converts detector outputs into reviewable defect hypotheses, then combines uncertainty-aware routing, vision-language model review, knowledge-graph constraints, and structured evaluation:

```text
Detector -> Hypotheses -> Detection Prompt -> Routing -> VLM Review -> KG Constraint -> Structured Decision -> Metrics
```

This repository is intended to publish source code, configurations, documentation, and reproducible scripts. Real datasets, training logs, experiment records, model weights, local environments, and generated training/inference outputs should not be committed to GitHub.

### Repository Layout

```text
configs/    Experiment, model, dataset, training, and ablation configs
docs/       Data format, annotation, knowledge graph, and experiment docs
scripts/    Entry points for data processing, detection, prompt building, VLM review, evaluation, and training
src/dap/    Core Detection-as-Prompt Python package
data/       Local data directory; real/generated data is ignored by default
models/     Local model checkpoint directory; ignored by default
outputs/    Training, inference, evaluation, and visualization outputs; ignored by default
```

### Installation

Python 3.10 or newer is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The base dependency list is intentionally small:

```text
PyYAML>=6.0.1
```

YOLO, Qwen3-VL, and LoRA workflows require additional packages such as `ultralytics`, `torch`, `transformers`, `peft`, `datasets`, and `accelerate`, depending on your checkpoint and hardware.

### Quick Check

```powershell
python scripts\print_config.py --config-dir configs
python scripts\check_project.py
```

### Minimal Demo Pipeline

The following commands create mock detector predictions, build prompts, route hypotheses, run mock VLM review, and export metrics:

```powershell
python scripts\02_mock_predictions_to_hypotheses.py --write-demo-input
python scripts\05_build_prompts.py
python scripts\06_route_hypotheses.py
python scripts\07_run_vlm_review.py --mode mock
python scripts\08_evaluate.py
```

Generated files are written under `data/` or `outputs/`, both of which are ignored for generated or private artifacts.

### Real Data

Real datasets are not distributed with this repository. Prepare them locally, for example:

```text
SM_norm/
NZB_dataset/
data/images/
data/yolo/
data/imported/
```

Import Labelme annotations:

```powershell
python scripts\12_import_labelme.py `
  --labelme-dir SM_norm/norm/images `
  --output-dir data/imported/sm_norm `
  --class-map configs/class_map.sm_nzb.json `
  --include-empty

python scripts\12_import_labelme.py `
  --json-dir NZB_dataset/json `
  --image-dir NZB_dataset/images `
  --output-dir data/imported/nzb_dataset `
  --class-map configs/class_map.sm_nzb.json `
  --include-empty
```

Build a YOLO dataset:

```powershell
python scripts\13_build_yolo_dataset.py `
  --class-map configs/class_map.sm_nzb.json `
  --output-dir data/yolo/sm_nzb `
  --include-unlabeled-images `
  --val-ratio 0.2 `
  --seed 42
```

### Models and Weights

Model files are not distributed with this repository. Put local checkpoints under `models/`, for example:

```text
models/Qwen3-VL-2B-Instruct
```

YOLO seed weights, trained `best.pt`/`last.pt`, LoRA adapters, and `.safetensors` files are ignored by default.

### Training and Evaluation

YOLO dry run:

```powershell
python scripts\20_train_yolo26n.py --dry-run
```

YOLO training:

```powershell
python scripts\20_train_yolo26n.py
```

Qwen3-VL LoRA dry run:

```powershell
python scripts\10_train_lora.py --dry-run
```

Full LoRA training still requires a checkpoint-specific training loop and matching dependencies.

### Before Uploading to GitHub

Check ignored and tracked files before pushing:

```powershell
git status --short
git ls-files --others --ignored --exclude-standard
git ls-files | Select-String -Pattern "\.pt$|\.pth$|\.safetensors$|\.ckpt$|data/|models/|outputs/"
```

If a large or private file is already tracked, `.gitignore` alone will not remove it. Untrack it with:

```powershell
git rm --cached <path>
```

then commit the cleanup.
