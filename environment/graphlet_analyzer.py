"""Graphlet analyzer for detecting costly control-flow patterns in Python code.

Parses Python source into a simplified Control Flow Graph (CFG) and identifies
expensive subgraph patterns (graphlets of 2-3 nodes) that indicate energy-
inefficient code structures.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class GraphletPattern:
    """A detected costly pattern in the control flow graph."""

    name: str
    description: str
    cost_weight: float
    location: str = ""


# ── Pattern cost weights ──────────────────────────────────────────────────────
PATTERN_COSTS: Dict[str, float] = {
    "NestedLoop": 3.0,              # For→For or While→While (very expensive)
    "LoopWithCall": 1.5,            # For/While→Call inside body (moderate)
    "DeepBranch": 2.0,              # If→If→If 3+ levels deep
    "RepeatedComprehension": 1.0,   # Multiple ListComp in same function
}


def _detect_nested_loops(tree: ast.AST) -> List[GraphletPattern]:
    """Detect For→For or While→While nested loop patterns."""
    patterns: List[GraphletPattern] = []
    loop_types = (ast.For, ast.While)

    for node in ast.walk(tree):
        if isinstance(node, loop_types):
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, loop_types):
                    loc = f"line {getattr(node, 'lineno', '?')}"
                    patterns.append(GraphletPattern(
                        name="NestedLoop",
                        description=f"Nested loop at {loc}",
                        cost_weight=PATTERN_COSTS["NestedLoop"],
                        location=loc,
                    ))
                    break  # one match per outer loop is enough

    return patterns


def _detect_loop_with_call(tree: ast.AST) -> List[GraphletPattern]:
    """Detect function calls directly inside loop bodies."""
    patterns: List[GraphletPattern] = []
    loop_types = (ast.For, ast.While)

    for node in ast.walk(tree):
        if isinstance(node, loop_types):
            for child in ast.iter_child_nodes(node):
                for subnode in ast.walk(child):
                    if isinstance(subnode, ast.Call):
                        loc = f"line {getattr(node, 'lineno', '?')}"
                        patterns.append(GraphletPattern(
                            name="LoopWithCall",
                            description=f"Call inside loop at {loc}",
                            cost_weight=PATTERN_COSTS["LoopWithCall"],
                            location=loc,
                        ))
                        break  # one per loop body
                break  # only check first body-level children

    return patterns


def _detect_deep_branches(tree: ast.AST, depth: int = 0) -> List[GraphletPattern]:
    """Detect If→If→If chains that are 3+ levels deep."""
    patterns: List[GraphletPattern] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            max_depth = _measure_if_depth(node)
            if max_depth >= 3:
                loc = f"line {getattr(node, 'lineno', '?')}"
                patterns.append(GraphletPattern(
                    name="DeepBranch",
                    description=f"If chain depth {max_depth} at {loc}",
                    cost_weight=PATTERN_COSTS["DeepBranch"],
                    location=loc,
                ))

    return patterns


def _measure_if_depth(node: ast.If, current: int = 1) -> int:
    """Recursively measure the deepest If nesting from a given If node."""
    max_d = current
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If):
            max_d = max(max_d, _measure_if_depth(child, current + 1))
    return max_d


def _detect_repeated_comprehensions(tree: ast.AST) -> List[GraphletPattern]:
    """Detect multiple ListComp nodes within the same function."""
    patterns: List[GraphletPattern] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            comps = [n for n in ast.walk(node) if isinstance(n, ast.ListComp)]
            if len(comps) >= 2:
                loc = f"function '{node.name}' line {getattr(node, 'lineno', '?')}"
                patterns.append(GraphletPattern(
                    name="RepeatedComprehension",
                    description=f"{len(comps)} list comprehensions in {loc}",
                    cost_weight=PATTERN_COSTS["RepeatedComprehension"],
                    location=loc,
                ))

    return patterns


def analyze_graphlets(code: str) -> Dict[str, Any]:
    """Analyze Python source code for costly graphlet patterns.

    Args:
        code: Python source code string.

    Returns:
        Dictionary with keys:
            - patterns_found: list of detected pattern dicts
            - total_cost: sum of all pattern cost weights
            - score: energy-efficiency score (0.0 to 1.0)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"patterns_found": [], "total_cost": 0.0, "score": 1.0}

    all_patterns: List[GraphletPattern] = []
    all_patterns.extend(_detect_nested_loops(tree))
    all_patterns.extend(_detect_loop_with_call(tree))
    all_patterns.extend(_detect_deep_branches(tree))
    all_patterns.extend(_detect_repeated_comprehensions(tree))

    total_cost = sum(p.cost_weight for p in all_patterns)
    score = max(0.0, 1.0 - total_cost / 10.0)

    patterns_dicts = [
        {"name": p.name, "description": p.description,
         "cost_weight": p.cost_weight, "location": p.location}
        for p in all_patterns
    ]

    return {
        "patterns_found": patterns_dicts,
        "total_cost": round(total_cost, 3),
        "score": round(score, 4),
    }
