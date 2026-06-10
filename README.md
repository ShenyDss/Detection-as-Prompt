# Detection-as-Prompt

Detection-as-Prompt is a research framework for defect decision making with detector-guided vision-language reasoning. It converts detector outputs into structured defect hypotheses, builds detection-aware prompts, routes uncertain cases, reviews them with a vision-language model, applies knowledge-graph constraints, and exports evaluation metrics.


## Pipeline
Detector -> Hypotheses -> Detection Prompt -> Routing -> VLM Review -> KG Constraint -> Structured Decision -> Metrics
Project Structure
configs/    Experiment, model, dataset, training, and ablation configs
docs/       Data format, annotation guideline, KG schema, and experiment protocol
scripts/    Command-line scripts for data processing, review, training, and evaluation
src/dap/    Core Detection-as-Prompt Python package
data/       Local datasets and generated intermediate files
models/     Local model checkpoints
outputs/    Training, inference, evaluation, and visualization outputs
Note: data/, models/, and outputs/ are intended for local files. Large datasets, training logs, checkpoints, and generated results should not be committed to GitHub.

## Installation
Python 3.10 or newer is recommended.

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
For YOLO, Qwen-VL, or LoRA workflows, install the required deep learning packages according to your hardware environment.

## Quick Start
Check the project configuration:

python scripts\print_config.py --config-dir configs
python scripts\check_project.py
Run a minimal mock pipeline:

python scripts\02_mock_predictions_to_hypotheses.py --write-demo-input
python scripts\05_build_prompts.py
python scripts\06_route_hypotheses.py
python scripts\07_run_vlm_review.py --mode mock
python scripts\08_evaluate.py
The generated files will be saved under data/ and outputs/.

Real Data and Models
Prepare real datasets locally, for example:

SM_norm/
NZB_dataset/
data/images/
data/yolo/
data/imported/
Place downloaded model checkpoints under:

models/
These local assets are excluded from Git by default.
