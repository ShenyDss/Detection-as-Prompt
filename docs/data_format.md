# Data Format

## Detector Hypotheses

`data/detector_predictions/hypotheses.jsonl`

```json
{
  "image_id": "mock_image_0000",
  "hypothesis_id": "mock_image_0000_000",
  "pred_class": "loose_weft",
  "bbox": [96.0, 128.0, 260.0, 288.0],
  "score": 0.82,
  "confusion_candidates": ["broken_weft", "coarse_weft"],
  "detector_meta": {}
}
```

## Human-Corrected Decisions

`data/annotations/test_decisions.jsonl`

```json
{
  "image_id": "mock_image_0000",
  "hypothesis_id": "mock_image_0000_000",
  "hypothesis_status": "verify",
  "corrected_class": "loose_weft",
  "bbox": [96.0, 128.0, 260.0, 288.0],
  "causes": ["unstable weft tension"],
  "actions": ["adjust weft tension"]
}
```

Allowed `hypothesis_status` values:

- `verify`: detector category and localization are acceptable.
- `reject`: detector alarm is a false positive.
- `revise`: defect exists, but category or box should be corrected.

Use this template when preparing real annotations:

```text
data/annotations/annotation_template.jsonl
```

Field rules:

- `image_id`: must match the detector hypothesis image id.
- `hypothesis_id`: must match one row in `data/detector_predictions/hypotheses.jsonl`.
- `hypothesis_status`: one of `verify`, `reject`, or `revise`.
- `corrected_class`: required for `verify` and `revise`; use `null` for false alarms.
- `bbox`: refined `xyxy` box for `verify` or `revise`; use `null` for rejected false alarms.
- `causes` and `actions`: should use labels listed in `data/knowledge_graph/defect_kg.json`.

## Real Data Preparation Checklist

Before running experiments with real data, prepare:

```text
data/images/
data/detector_predictions/raw_detector_predictions.jsonl
data/detector_predictions/hypotheses.jsonl
data/annotations/test_decisions.jsonl
data/knowledge_graph/defect_kg.json
```

Then validate the dataset:

```powershell
.\.conda\dap\python.exe scripts\11_validate_real_data.py
```

The script writes:

```text
outputs/metrics/data_validation_report.json
```

Validation blocks formal experiments when:

- a hypothesis JSONL file cannot be parsed;
- a `hypothesis_id` is duplicated;
- a predicted or corrected class is absent from the KG;
- a score is outside `[0, 1]`;
- a bbox has invalid order, e.g. `x2 <= x1`.

Warnings indicate issues that should be fixed before paper-grade experiments:

- missing human annotation for a detector hypothesis;
- image path cannot be resolved;
- a confusion class is absent from the KG;
- a cause/action is not listed under the corrected class in the KG;
- a verified or revised annotation is missing a refined bbox.
