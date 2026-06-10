# Detection-as-Prompt

Python scaffold for the paper idea:

**Detection-as-Prompt: Hypothesis-Guided Vision-Language Reasoning for Knowledge-Constrained Defect Decision Making**

This repository is organized around the paper pipeline:

```text
Detector -> Hypotheses -> Detection Prompt -> Routing -> VLM Review -> KG Constraint -> Structured Decision -> Metrics
```

The current step only builds the project skeleton, configuration system, and core dataclass schemas. It does not connect real detectors, VLMs, or training code yet.

## Project Layout

```text
configs/                 YAML configuration files
data/                    Local datasets, annotations, predictions, and KG files
src/dap/                 Python package for Detection-as-Prompt
scripts/                 Command-line entry scripts
outputs/                 Predictions, metrics, tables, and visualizations
```

## Quick Start

```powershell
C:\ProgramData\miniconda3\Scripts\conda.exe activate .\.conda\dap
python scripts\print_config.py --config-dir configs
```

Expected result: the script prints a merged experiment configuration containing dataset, detector, VLM, KG, and experiment settings.

The project-local Conda environment is stored at:

```text
.conda/dap
```

VSCode is configured to use this interpreter by default.

## Core Objects

- `DefectHypothesis`: detector output represented as a reviewable hypothesis.
- `EvidenceRecord`: visual, uncertainty, and graph-path evidence.
- `ExpertDecision`: graph-grounded causes, actions, and risk notes.
- `VLMDecision`: structured VLM review output with `verify`, `reject`, or `revise` status.

These objects correspond to the paper formulation:

```text
h_i = (c_i, b_i, s_i, C_i)
y_i = (z_i, c_hat_i, b_hat_i, e_i, r_i)
```

## Next Steps

1. Add a mock detector and YOLO adapter.
2. Implement detection-to-prompt construction.
3. Add uncertainty-aware routing.
4. Add mock VLM inference and structured output parsing.
5. Implement evaluation metrics and LaTeX table export.

## Step 2: Mock Predictions to Hypotheses

Create a small demo detector-prediction file and convert it into paper-style hypotheses:

```powershell
.\.conda\dap\python.exe scripts\02_mock_predictions_to_hypotheses.py --write-demo-input
```

The script writes:

```text
data/detector_predictions/mock_detector_predictions.jsonl
data/detector_predictions/hypotheses.jsonl
```

## Step 3: Detector Inference

Run the detector interface in mock mode:

```powershell
.\.conda\dap\python.exe scripts\01_run_detector.py --mode mock
```

This writes raw detector-like predictions and unified hypotheses:

```text
data/detector_predictions/raw_detector_predictions.jsonl
data/detector_predictions/hypotheses.jsonl
```

To use a real Ultralytics YOLO checkpoint later:

```powershell
.\.conda\dap\python.exe scripts\01_run_detector.py `
  --mode ultralytics `
  --checkpoint path\to\best.pt `
  --image-dir data\images
```

Both modes produce the same Detection-as-Prompt hypothesis schema:

```json
{
  "image_id": "mock_image_0000",
  "hypothesis_id": "mock_image_0000_000",
  "pred_class": "loose_weft",
  "bbox": [96.0, 128.0, 260.0, 288.0],
  "score": 0.82,
  "confusion_candidates": ["broken_weft", "coarse_weft", "yarn_irregularity"],
  "detector_meta": {}
}
```

## Step 4: Knowledge Graph

The demo defect knowledge graph is stored at:

```text
data/knowledge_graph/defect_kg.json
```

Inspect category-specific knowledge context and graph consistency:

```powershell
.\.conda\dap\python.exe scripts\04_check_kg.py `
  --class-name broken_weft `
  --cause "yarn breakage" `
  --action "stop and repair broken yarn"
```

The KG module supports:

- loading a JSON-backed defect graph;
- retrieving visual attributes, confusion classes, causes, actions, risks, and cause-action paths for a defect class;
- checking whether a VLM decision is graph-consistent for a corrected class.

## Step 5: Detection-to-Prompt Constructor

Build structured VLM prompts from detector hypotheses and KG context:

```powershell
.\.conda\dap\python.exe scripts\05_build_prompts.py
```

This reads:

```text
data/detector_predictions/hypotheses.jsonl
data/detector_predictions/raw_detector_predictions.jsonl
data/knowledge_graph/defect_kg.json
```

and writes:

```text
outputs/predictions/detection_prompts.jsonl
outputs/regions/
```

If a source image exists, the candidate region is cropped to `outputs/regions/`.
For mock image paths, prompts are still generated and `region_path` is set to `null`.

## Step 6: Uncertainty-Aware Routing

Route each detector hypothesis into a review mode:

```powershell
.\.conda\dap\python.exe scripts\06_route_hypotheses.py
```

This writes:

```text
outputs/predictions/routes.jsonl
```

Supported review modes:

- `light_verification`: high-confidence hypotheses with clear detector evidence;
- `class_comparison`: ambiguous or medium-confidence hypotheses;
- `false_alarm_checking`: low-confidence hypotheses;
- `box_refinement`: low-quality or suspicious bounding boxes.

When `routes.jsonl` exists, `scripts/05_build_prompts.py` automatically injects the route hint and uncertainty signals into each structured prompt.

## Step 7: VLM Review and Output Parsing

Run the VLM review interface in mock mode:

```powershell
.\.conda\dap\python.exe scripts\07_run_vlm_review.py --mode mock
```

This reads:

```text
outputs/predictions/detection_prompts.jsonl
```

and writes:

```text
outputs/predictions/vlm_decisions.jsonl
```

The mock reviewer emits structured decisions with:

- `hypothesis_status`: `verify`, `reject`, or `revise`;
- visual evidence;
- uncertainty evidence;
- graph path;
- corrected class;
- refined bbox;
- graph-grounded causes, actions, and risks.

Run a one-sample Qwen3-VL smoke test after downloading the local checkpoint:

```powershell
.\.conda\dap\python.exe scripts\07_qwen_smoke_test.py
```

Run the real Qwen3-VL reviewer on the current prompt file without overwriting mock decisions:

```powershell
.\.conda\dap\python.exe scripts\07_run_vlm_review.py `
  --mode qwen3_vl `
  --output outputs/predictions/vlm_decisions_qwen.jsonl
```

The Qwen adapter reads `configs/vlm.yaml` and expects the downloaded checkpoint at:

```text
models/Qwen3-VL-2B-Instruct
```

## Step 8: Evaluation and Table Export

Evaluate current predictions:

```powershell
.\.conda\dap\python.exe scripts\08_evaluate.py
```

This writes:

```text
outputs/metrics/metrics.json
outputs/metrics/summary.csv
outputs/tables/evaluation_tables.tex
```

If `data/annotations/test_decisions.jsonl` does not exist, supervised metrics are exported as placeholders and graph/evidence metrics are still computed.
For a full smoke test with demo labels copied from the current mock VLM output:

```powershell
.\.conda\dap\python.exe scripts\08_write_demo_ground_truth.py
.\.conda\dap\python.exe scripts\08_evaluate.py
```

The real experiment should replace `data/annotations/test_decisions.jsonl` with human-corrected status labels, corrected classes, refined boxes, causes, and actions.

## Step 9: Ablation Runner

Run ablation variants:

```powershell
.\.conda\dap\python.exe scripts\09_run_ablation.py
```

This writes:

```text
outputs/ablations/
outputs/metrics/ablation_results.csv
outputs/tables/ablation_results.tex
```

Current variants:

- `Full model`
- `w/o detection prompt`
- `w/o self-verification`
- `w/o uncertainty routing`
- `w/o confusion candidates`
- `w/o KG constraint`
- `w/o instruction tuning`

Each variant changes module switches and reuses the same detector, prompt, routing, VLM, and evaluation components.

## Step 10: Instruction Data and LoRA Skeleton

Build correction-oriented instruction data:

```powershell
.\.conda\dap\python.exe scripts\10_build_instruction_data.py
```

This writes:

```text
data/instruction_data/dap_instruction_train.jsonl
```

Each sample contains:

- image and hypothesis identifiers;
- image path and optional region crop path;
- structured Detection-as-Prompt input prompt;
- target JSON with verify/reject/revise, corrected class, refined bbox, evidence, causes, actions, and risks.

Validate the LoRA training plan without starting real model training:

```powershell
.\.conda\dap\python.exe scripts\10_train_lora.py --dry-run
```

The dry run writes:

```text
outputs/checkpoints/dap_lora/training_plan.json
```

Real LoRA training is intentionally left as a checkpoint-specific skeleton. It requires installing `transformers`, `peft`, `datasets`, and `accelerate`, then selecting the exact Qwen-VL/Qwen2-VL/Qwen3-VL checkpoint and GPU memory budget.

## Data-Free Preparation Steps

The following utilities can run before real data is ready:

```powershell
.\.conda\dap\python.exe scripts\check_project.py
.\.conda\dap\python.exe scripts\13_generate_demo_cases.py
.\.conda\dap\python.exe scripts\run_pipeline.py
.\.conda\dap\python.exe scripts\14_visualize_cases.py
```

Important docs for later manual data preparation:

```text
docs/data_format.md
docs/annotation_guideline.md
docs/kg_schema.md
docs/experiment_protocol.md
```

## Step 11: Real Data Validation

After adding real images, detector hypotheses, annotations, and a curated KG, validate the dataset before running formal experiments:

```powershell
.\.conda\dap\python.exe scripts\11_validate_real_data.py
```

This writes:

```text
outputs/metrics/data_validation_report.json
```

Use this annotation template when preparing human labels:

```text
data/annotations/annotation_template.jsonl
```

The validator checks duplicate hypothesis ids, unknown KG classes, invalid scores, invalid boxes, missing annotation coverage, unresolved images, and cause/action labels that are not supported by the KG.

## Step 12: Labelme Import and YOLO Dataset

Import Labelme annotations as Detection-as-Prompt pseudo hypotheses:

```powershell
.\.conda\dap\python.exe scripts\12_import_labelme.py `
  --labelme-dir SM_norm/norm/images `
  --output-dir data/imported/sm_norm `
  --class-map configs/class_map.sm_nzb.json `
  --include-empty

.\.conda\dap\python.exe scripts\12_import_labelme.py `
  --json-dir NZB_dataset/json `
  --image-dir NZB_dataset/images `
  --output-dir data/imported/nzb_dataset `
  --class-map configs/class_map.sm_nzb.json `
  --include-empty
```

Build the merged YOLO train/val dataset:

```powershell
.\.conda\dap\python.exe scripts\13_build_yolo_dataset.py `
  --class-map configs/class_map.sm_nzb.json `
  --output-dir data/yolo/sm_nzb `
  --include-unlabeled-images `
  --val-ratio 0.2 `
  --seed 42
```

The generated YOLO config is:

```text
data/yolo/sm_nzb/dataset.yaml
```

The dataset intentionally uses `test: val/images` because only train/val splits are prepared at this stage.

## Step 13: YOLO26n Training

Check the YOLO26n training plan without starting training:

```powershell
.\.conda\dap\python.exe scripts\20_train_yolo26n.py --dry-run
```

Start training with the default 8GB-GPU-friendly parameters:

```powershell
.\.conda\dap\python.exe scripts\20_train_yolo26n.py
```

Default parameters are stored in:

```text
configs/yolo26n_train.yaml
```

The initial training configuration uses `yolo26n.pt`, `imgsz=640`, `batch=8`, `epochs=300`, `patience=25`, `workers=0`, `amp=true`, and `test: val/images`. `workers=0` avoids Windows multiprocessing permission issues.

## Step 14: Match YOLO Predictions to Ground Truth

After YOLO inference produces hypothesis JSONL files, automatically create hypothesis-level supervision:

```powershell
.\.conda\dap\python.exe scripts\15_match_yolo_gt_to_decisions.py `
  --hypotheses outputs/imported/yolo26n_val/hypotheses.jsonl `
  --dataset-yaml data/yolo/sm_nzb/dataset.yaml `
  --label-dir data/yolo/sm_nzb/val/labels `
  --image-dir data/yolo/sm_nzb/val/images `
  --output outputs/imported/yolo26n_val/test_decisions.jsonl `
  --report outputs/imported/yolo26n_val/match_report.json `
  --iou-threshold 0.5
```

Matching rules:

- `verify`: IoU passes threshold and predicted class matches the ground truth class.
- `revise`: IoU passes threshold but predicted class differs.
- `reject`: no ground-truth box reaches the IoU threshold.

Detector misses are reported separately because they do not have a detector hypothesis for the VLM to review.

## Step 15: Build Qwen3-VL Reviewer Instruction Data

Build correction-oriented instruction samples from real YOLO hypotheses, Detection-as-Prompt prompts, and matched supervision:

```powershell
.\.conda\dap\python.exe scripts\10_build_instruction_data.py `
  --prompts outputs/imported/yolo26n_val/detection_prompts.jsonl `
  --hypotheses outputs/imported/yolo26n_val/hypotheses.jsonl `
  --ground-truth outputs/imported/yolo26n_val/test_decisions.jsonl `
  --output data/instruction_data/yolo26n_val_instruction.jsonl
```

Each instruction sample contains the candidate image region, the structured Detection-as-Prompt input, and a JSON target with `verify`, `reject`, or `revise` supervision.

## Step 16: Qwen3-VL LoRA Dry Run

Validate Qwen3-VL reviewer tuning inputs and LoRA parameters:

```powershell
.\.conda\dap\python.exe scripts\10_train_lora.py --dry-run
```

The default config is:

```text
configs/training.yaml
```

It currently points to:

```text
data/instruction_data/yolo26n_val_instruction.jsonl
models/Qwen3-VL-2B-Instruct
outputs/checkpoints/dap_lora
```

The dry run checks sample counts, status distribution, image and crop paths, model directory, LoRA rank/alpha/dropout, and optimization settings. Real Qwen3-VL LoRA training still requires implementing the checkpoint-specific training loop.

## ModelScope Model Download

The intended small VLM checkpoint is:

```text
Qwen/Qwen3-VL-2B-Instruct
```

Download script:

```powershell
.\scripts\install_model_tools.ps1
.\scripts\resume_download_qwen3vl.ps1
```

This writes to:

```text
models/Qwen3-VL-2B-Instruct
```

The `models/` directory is ignored by git because model files are large.

The model is usable only when `scripts/check_model.py` reports root weight files, for example `model.safetensors` or sharded `.safetensors` files. If the only weight is under `._____temp/`, the download is incomplete.

Check the local model and run a real VLM smoke test:

```powershell
.\.conda\dap\python.exe scripts\check_model.py
.\.conda\dap\python.exe scripts\07_qwen_smoke_test.py
```

Evaluate Qwen3-VL decisions separately from mock decisions:

```powershell
.\.conda\dap\python.exe scripts\08_evaluate.py `
  --decisions outputs/predictions/vlm_decisions_qwen.jsonl `
  --metrics-json outputs/metrics/metrics_qwen.json `
  --metrics-csv outputs/metrics/summary_qwen.csv `
  --latex-table outputs/tables/evaluation_tables_qwen.tex `
  --method-name "Qwen3-VL"
```

## Step 17: Qwen3-VL LoRA Review

After LoRA training, `configs/vlm.yaml` points to:

```text
outputs/checkpoints/dap_lora/final
```

Run a one-sample smoke test on the real YOLO validation prompts:

```powershell
.\.conda\dap\python.exe scripts\07_qwen_smoke_test.py `
  --prompts outputs/imported/yolo26n_val/detection_prompts.jsonl `
  --index 0 `
  --max-new-tokens 256 `
  --use-lora
```

Run a short LoRA-backed review batch:

```powershell
.\.conda\dap\python.exe scripts\07_run_vlm_review.py `
  --mode qwen3_vl `
  --prompts outputs/imported/yolo26n_val/detection_prompts.jsonl `
  --output outputs/imported/yolo26n_val/vlm_decisions_qwen_lora_sample.jsonl `
  --limit 8 `
  --use-lora `
  --continue-on-error
```

Run the full validation review after the smoke batch looks reasonable:

```powershell
.\.conda\dap\python.exe scripts\07_run_vlm_review.py `
  --mode qwen3_vl `
  --prompts outputs/imported/yolo26n_val/detection_prompts.jsonl `
  --output outputs/imported/yolo26n_val/vlm_decisions_qwen_lora.jsonl `
  --use-lora `
  --continue-on-error
```

Evaluate the LoRA reviewer against the automatically matched validation labels:

```powershell
.\.conda\dap\python.exe scripts\08_evaluate.py `
  --hypotheses outputs/imported/yolo26n_val/hypotheses.jsonl `
  --decisions outputs/imported/yolo26n_val/vlm_decisions_qwen_lora.jsonl `
  --ground-truth outputs/imported/yolo26n_val/test_decisions.jsonl `
  --kg-path outputs/imported/yolo26n_val/defect_kg.generated.json `
  --metrics-json outputs/metrics/metrics_qwen_lora.json `
  --metrics-csv outputs/metrics/summary_qwen_lora.csv `
  --latex-table outputs/tables/evaluation_tables_qwen_lora.tex `
  --method-name "Qwen3-VL LoRA"
```
