from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from dap.schemas.core import DefectHypothesis, HypothesisStatus, VLMDecision


STATUS_COLORS = {
    HypothesisStatus.VERIFY: "#2f9e44",
    HypothesisStatus.REJECT: "#e03131",
    HypothesisStatus.REVISE: "#f08c00",
}


def render_case_visualizations(
    *,
    hypotheses: list[DefectHypothesis],
    decisions: list[VLMDecision],
    image_lookup: dict[str, str | None],
    output_dir: str | Path,
) -> int:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    decision_lookup = {item.hypothesis_id: item for item in decisions}
    count = 0
    for hypothesis in hypotheses:
        image_path = image_lookup.get(hypothesis.image_id)
        if not image_path or not Path(image_path).exists():
            continue
        decision = decision_lookup.get(hypothesis.hypothesis_id)
        if decision is None:
            continue
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            color = STATUS_COLORS.get(decision.hypothesis_status, "#1971c2")
            box = hypothesis.bbox.to_list()
            draw.rectangle(tuple(box), outline=color, width=4)
            label = f"{decision.hypothesis_status.value}: {decision.corrected_class or hypothesis.pred_class}"
            draw.rectangle((box[0], max(0, box[1] - 22), box[0] + 260, box[1]), fill=color)
            draw.text((box[0] + 4, max(0, box[1] - 19)), label, fill="white")
            image.save(output_root / f"{hypothesis.hypothesis_id}.png")
            count += 1
    return count
