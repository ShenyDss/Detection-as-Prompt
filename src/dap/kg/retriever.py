from __future__ import annotations

from dap.kg.schema import DefectKnowledgeGraph, KGContext


def retrieve_category_context(
    graph: DefectKnowledgeGraph,
    class_name: str,
    *,
    max_visual_attributes: int = 6,
    max_confusion_classes: int = 3,
    max_causes: int = 5,
    max_actions: int = 5,
) -> KGContext:
    node = graph.get_class(class_name)
    causes = node.causes[:max_causes]
    actions = node.actions[:max_actions]
    paths = _build_cause_action_paths(graph, causes, actions)

    return KGContext(
        class_name=node.name,
        visual_attributes=node.visual_attributes[:max_visual_attributes],
        confusion_classes=node.confusion_classes[:max_confusion_classes],
        causes=causes,
        actions=actions,
        risk_notes=node.risk_notes,
        cause_action_paths=paths,
    )


def _build_cause_action_paths(
    graph: DefectKnowledgeGraph,
    causes: list[str],
    actions: list[str],
) -> list[list[str]]:
    action_set = set(actions)
    paths: list[list[str]] = []
    for cause in causes:
        valid_actions = graph.cause_action_edges.get(cause, [])
        for action in valid_actions:
            if action in action_set:
                paths.append([cause, action])
    return paths
