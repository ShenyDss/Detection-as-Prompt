# Experiment Protocol

## Pipeline

1. Run detector and export hypotheses.
2. Route hypotheses by uncertainty.
3. Build Detection-as-Prompt prompts with KG context.
4. Run VLM review.
5. Evaluate correction quality and decision quality.
6. Run ablations.
7. Export LaTeX tables.

## Required Splits

- train
- validation
- test

The test split should contain verified, rejected, and revised detector hypotheses.

## Main Metrics

- Defect-F1
- Corr-Cls
- MCR
- FPRed
- Ref-IoU
- HSA
- FRA
- Rev-P
- Rev-R
- CMA
- ARA
- GPC
- EC
- HR
