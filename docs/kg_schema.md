# Knowledge Graph Schema

The demo graph lives at:

```text
data/knowledge_graph/defect_kg.json
```

Each defect class contains:

- `name`
- `visual_attributes`
- `confusion_classes`
- `causes`
- `actions`
- `risk_notes`

Global `cause_action_edges` define valid production paths:

```json
{
  "yarn breakage": ["stop and repair broken yarn"]
}
```

The evaluator uses this graph to compute graph-path consistency and hallucination rate.
