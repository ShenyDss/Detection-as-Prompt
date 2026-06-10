# Annotation Guideline

Annotate each detector hypothesis, not only each image.

## Required Fields

- `hypothesis_status`
- `corrected_class`
- `bbox`
- `causes`
- `actions`

## Decision Rules

Use `verify` when the detector correctly localizes and classifies a defect.

Use `reject` when the candidate region is normal texture, noise, illumination artifact, or another non-defect pattern.

Use `revise` when the defect is real but the predicted category or bounding box is wrong.

## Evidence Notes

When possible, record visual cues such as missing yarn, local density change, coarse stripe, or irregular texture. These notes can later become `visual_evidence` targets for instruction tuning.
