from __future__ import annotations

from dataclasses import dataclass, field

from dap.kg.schema import DefectKnowledgeGraph


@dataclass
class GraphConsistencyResult:
    is_consistent: bool
    class_valid: bool
    causes_valid: bool
    actions_valid: bool
    cause_action_paths_valid: bool
    valid_paths: list[list[str]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def score(self) -> float:
        checks = [
            self.class_valid,
            self.causes_valid,
            self.actions_valid,
            self.cause_action_paths_valid,
        ]
        return sum(1 for item in checks if item) / len(checks)


def check_graph_consistency(
    graph: DefectKnowledgeGraph,
    *,
    corrected_class: str,
    causes: list[str],
    actions: list[str],
    require_valid_class: bool = True,
    require_valid_cause_action_path: bool = True,
) -> GraphConsistencyResult:
    issues: list[str] = []
    class_valid = graph.has_class(corrected_class)
    if require_valid_class and not class_valid:
        issues.append(f"Unknown corrected class: {corrected_class}")
        return GraphConsistencyResult(
            is_consistent=False,
            class_valid=False,
            causes_valid=False,
            actions_valid=False,
            cause_action_paths_valid=False,
            issues=issues,
        )

    node = graph.get_class(corrected_class) if class_valid else None
    allowed_causes = set(node.causes if node else [])
    allowed_actions = set(node.actions if node else [])
    causes_valid = all(cause in allowed_causes for cause in causes)
    actions_valid = all(action in allowed_actions for action in actions)

    if not causes_valid:
        invalid = [cause for cause in causes if cause not in allowed_causes]
        issues.append(f"Unsupported causes for {corrected_class}: {invalid}")
    if not actions_valid:
        invalid = [action for action in actions if action not in allowed_actions]
        issues.append(f"Unsupported actions for {corrected_class}: {invalid}")

    valid_paths = []
    for cause in causes:
        for action in actions:
            if cause in allowed_causes and action in allowed_actions and action in graph.cause_action_edges.get(cause, []):
                valid_paths.append([cause, action])

    if causes and actions:
        cause_action_paths_valid = bool(valid_paths)
    else:
        cause_action_paths_valid = not require_valid_cause_action_path

    if require_valid_cause_action_path and not cause_action_paths_valid:
        issues.append("No valid cause-action path found.")

    is_consistent = class_valid and causes_valid and actions_valid
    if require_valid_cause_action_path:
        is_consistent = is_consistent and cause_action_paths_valid

    return GraphConsistencyResult(
        is_consistent=is_consistent,
        class_valid=class_valid,
        causes_valid=causes_valid,
        actions_valid=actions_valid,
        cause_action_paths_valid=cause_action_paths_valid,
        valid_paths=valid_paths,
        issues=issues,
    )
