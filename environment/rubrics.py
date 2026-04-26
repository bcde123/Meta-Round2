"""Composable Rubric system for the Green-Code Optimizer environment.

Mirrors `openenv-core`'s Rubric API (RFC 004) so this environment integrates
cleanly with OpenEnv's training infrastructure. When `openenv-core` is
installed, we extend its base classes directly; otherwise we provide a
zero-dependency fallback shim with the same surface area.

The reward signal is decomposed into named child rubrics so:
  • Each component is independently logged via `named_rubrics()`.
  • Components are easy to reweight or swap (compositional design).
  • A `Gate` enforces hard correctness gates without polluting downstream rubrics.
  • Reward-hacking is structurally hard: the gate kills the gradient if code
    doesn't parse, no matter how cleverly other rubrics are gamed.

Reward tree:

    GreenCodeRubric (root)
    └─ Sequential
       ├─ syntax_gate           : binary {0, 1}  — must parse
       ├─ hack_gate              : binary {0, 1}  — must not tamper with tests
       └─ WeightedSum            : weighted soft signals
          ├─ green     (0.70)    : graphlet + cpu + memory composite
          │  ├─ graphlet  (0.40)
          │  ├─ cpu       (0.35)
          │  └─ memory    (0.25)
          └─ compliance (0.30)   : engineering rules satisfied

Usage in training:
    rubric = build_green_rubric(active_rules=ep["rules_active"])
    reward = rubric(action=Action(updated_files=...), observation=Observation(orig_files=...))
    for name, r in rubric.named_rubrics():
        print(f"{name}: {r.last_score}")
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# ── Try to import openenv-core; fall back to a local shim if unavailable ─────
try:
    from openenv.core.rubrics.base import Rubric as _OERubric
    from openenv.core.rubrics.containers import (
        Gate as _OEGate,
        Sequential as _OESequential,
        WeightedSum as _OEWeightedSum,
        RubricDict as _OERubricDict,
    )
    OPENENV_AVAILABLE = True
    _BaseRubric = _OERubric
    _BaseGate = _OEGate
    _BaseSequential = _OESequential
    _BaseWeightedSum = _OEWeightedSum
    _BaseRubricDict = _OERubricDict
except ImportError:
    OPENENV_AVAILABLE = False

    class _BaseRubric:
        """Local nn.Module-style Rubric (fallback when openenv-core not installed).

        Mirrors openenv.core.rubrics.base.Rubric: implement `forward`,
        children auto-register on attribute assignment, hooks supported.
        """
        def __init__(self):
            self._children: Dict[str, "_BaseRubric"] = {}
            self._forward_hooks: List[Callable] = []
            self.last_score: Optional[float] = None

        def __setattr__(self, name: str, value: Any):
            if isinstance(value, _BaseRubric) and not name.startswith("_"):
                self._children[name] = value
            object.__setattr__(self, name, value)

        def forward(self, action: Any, observation: Any) -> float:
            raise NotImplementedError

        def __call__(self, action: Any, observation: Any) -> float:
            score = float(self.forward(action, observation))
            self.last_score = score
            for h in self._forward_hooks:
                try:
                    h(self, action, observation, score)
                except Exception:
                    pass
            return score

        def register_forward_hook(self, hook: Callable) -> None:
            self._forward_hooks.append(hook)

        def named_children(self) -> Iterator[Tuple[str, "_BaseRubric"]]:
            yield from self._children.items()

        def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, "_BaseRubric"]]:
            for name, child in self._children.items():
                full = f"{prefix}.{name}" if prefix else name
                yield full, child
                yield from child.named_rubrics(full)

        def get_rubric(self, path: str) -> "_BaseRubric":
            cur = self
            for part in path.split("."):
                cur = cur._children[part]
            return cur

        def reset(self) -> None:
            self.last_score = None
            for c in self._children.values():
                c.reset()

    class _BaseSequential(_BaseRubric):
        """Run children in order; the LAST child's score is returned. If any
        earlier child returns 0.0, short-circuit (gate semantics)."""
        def __init__(self, *rubrics: _BaseRubric):
            super().__init__()
            for i, r in enumerate(rubrics):
                setattr(self, f"r{i}", r)
            self._ordered = list(rubrics)

        def forward(self, action, observation) -> float:
            score = 1.0
            for r in self._ordered:
                score = r(action, observation)
                if score <= 0.0:
                    return 0.0
            return score

    class _BaseGate(_BaseRubric):
        """Pass-through if rubric ≥ threshold, else return 0.0."""
        def __init__(self, rubric: _BaseRubric, threshold: float = 1.0):
            super().__init__()
            self.gated = rubric
            self.threshold = threshold

        def forward(self, action, observation) -> float:
            s = self.gated(action, observation)
            return 1.0 if s >= self.threshold else 0.0

    class _BaseWeightedSum(_BaseRubric):
        """Σ wᵢ·rᵢ(a,o), weights normalised to sum to 1."""
        def __init__(self, rubrics: List[_BaseRubric], weights: List[float]):
            super().__init__()
            assert len(rubrics) == len(weights), "rubrics/weights length mismatch"
            for i, r in enumerate(rubrics):
                setattr(self, f"r{i}", r)
            total = sum(weights) or 1.0
            self._w = [w / total for w in weights]
            self._ordered = list(rubrics)

        def forward(self, action, observation) -> float:
            return sum(w * r(action, observation) for w, r in zip(self._w, self._ordered))

    class _BaseRubricDict(_BaseRubric):
        """Keyed access to rubrics — analogous to nn.ModuleDict."""
        def __init__(self, rubrics: Optional[Dict[str, _BaseRubric]] = None):
            super().__init__()
            for k, v in (rubrics or {}).items():
                setattr(self, k, v)


# ── Action / Observation types ────────────────────────────────────────────────
@dataclass
class CodeAction:
    """An agent's action: a (possibly partial) edit of the codebase."""
    updated_files: Dict[str, str]
    diff: str = ""


@dataclass
class CodeObservation:
    """The episode observation given to the rubric."""
    orig_files: Dict[str, str]
    active_rules: List[int] = field(default_factory=list)
    standards_path: str = ""


# ── Leaf rubrics ──────────────────────────────────────────────────────────────
class SyntaxGateRubric(_BaseRubric):
    """Binary gate: 1.0 if every refactored file parses, 0.0 otherwise.

    This is the structural anti-hack: any reward signal is multiplied by 0
    if the agent produces code that doesn't parse. No "clean" code that
    doesn't compile gets credit.
    """
    def forward(self, action: CodeAction, observation: CodeObservation) -> float:
        if not action.updated_files:
            return 0.0
        for content in action.updated_files.values():
            try:
                ast.parse(content)
            except SyntaxError:
                return 0.0
        return 1.0


class HackGateRubric(_BaseRubric):
    """Binary gate: 0.0 if the agent edited a protected test-infrastructure
    file (conftest.py, test_*.py, pytest.ini, setup.cfg). Otherwise 1.0."""
    PROTECTED = {"conftest.py", "pytest.ini", "setup.cfg"}

    def forward(self, action: CodeAction, observation: CodeObservation) -> float:
        for fname in action.updated_files:
            base = os.path.basename(fname)
            if base in self.PROTECTED or base.startswith("test_"):
                if observation.orig_files.get(fname) != action.updated_files[fname]:
                    return 0.0
        return 1.0


class GraphletRubric(_BaseRubric):
    """Average graphlet score across all refactored files (range [0, 1])."""
    def forward(self, action: CodeAction, observation: CodeObservation) -> float:
        from .track_c import GreenCodeEvaluator
        return GreenCodeEvaluator().evaluate_graphlets(action.updated_files)


class CPURubric(_BaseRubric):
    """Relative CPU-time reduction vs. original code (range [0, 1])."""
    def forward(self, action: CodeAction, observation: CodeObservation) -> float:
        from .track_c import GreenCodeEvaluator
        ev = GreenCodeEvaluator()
        orig_t = ev.measure_execution(observation.orig_files)["cpu_time_ms"]
        new_t = ev.measure_execution(action.updated_files)["cpu_time_ms"]
        if orig_t <= 0:
            return 0.5
        return max(0.0, min(1.0, (orig_t - new_t) / orig_t))


class MemoryRubric(_BaseRubric):
    """Relative peak-memory reduction vs. original code (range [0, 1])."""
    def forward(self, action: CodeAction, observation: CodeObservation) -> float:
        from .track_c import GreenCodeEvaluator
        ev = GreenCodeEvaluator()
        orig_m = ev.measure_execution(observation.orig_files)["peak_memory_mb"]
        new_m = ev.measure_execution(action.updated_files)["peak_memory_mb"]
        if orig_m <= 0:
            return 0.5
        return max(0.0, min(1.0, (orig_m - new_m) / orig_m))


class ComplianceRubric(_BaseRubric):
    """Fraction of active engineering rules satisfied (range [0, 1])."""
    def forward(self, action: CodeAction, observation: CodeObservation) -> float:
        from .track_b import ComplianceChecker
        if not observation.standards_path or not observation.active_rules:
            return 0.0
        checker = ComplianceChecker(observation.standards_path)
        checker.reset(observation.orig_files, observation.active_rules)
        for fname in action.updated_files:
            if (fname in observation.orig_files
                    and action.updated_files[fname] != observation.orig_files[fname]):
                checker.step({"tool": "edit_file", "args": {"filename": fname}},
                             f"Edited {fname}")
        return float(checker.get_score())


# ── Composite green rubric ────────────────────────────────────────────────────
class GreenScoreRubric(_BaseRubric):
    """Composite green score: 0.40·graphlet + 0.35·cpu + 0.25·memory.

    Decomposed so each component is individually inspectable via
    `rubric.named_rubrics()` for logging in training infrastructure.
    """
    def __init__(self):
        super().__init__()
        self.graphlet = GraphletRubric()
        self.cpu = CPURubric()
        self.memory = MemoryRubric()

    def forward(self, action, observation) -> float:
        g = self.graphlet(action, observation)
        c = self.cpu(action, observation)
        m = self.memory(action, observation)
        return 0.40 * g + 0.35 * c + 0.25 * m


# ── Top-level rubric builder ──────────────────────────────────────────────────
def build_green_rubric() -> _BaseRubric:
    """Build the full Green-Code Optimizer reward tree.

    Returns a Rubric that, when called, evaluates:

        R = (syntax_gate ∧ hack_gate) × (0.70 · green + 0.30 · compliance)

    The two binary gates are short-circuited via `Sequential`: if either is
    0, downstream rubrics aren't even evaluated (saves compute).
    """

    class GreenCodeRubric(_BaseRubric):
        def __init__(self):
            super().__init__()
            self.syntax_gate = SyntaxGateRubric()
            self.hack_gate = HackGateRubric()
            self.green = GreenScoreRubric()
            self.compliance = ComplianceRubric()

        def forward(self, action, observation) -> float:
            if self.syntax_gate(action, observation) <= 0.0:
                return 0.0
            if self.hack_gate(action, observation) <= 0.0:
                return -1.0  # P_hack penalty
            g = self.green(action, observation)
            c = self.compliance(action, observation)
            return 0.70 * g + 0.30 * c

    return GreenCodeRubric()


__all__ = [
    "Rubric",
    "Sequential",
    "Gate",
    "WeightedSum",
    "RubricDict",
    "CodeAction",
    "CodeObservation",
    "SyntaxGateRubric",
    "HackGateRubric",
    "GraphletRubric",
    "CPURubric",
    "MemoryRubric",
    "ComplianceRubric",
    "GreenScoreRubric",
    "build_green_rubric",
    "OPENENV_AVAILABLE",
]

# Re-export names so `from environment.rubrics import Rubric` works in either mode
Rubric = _BaseRubric
Sequential = _BaseSequential
Gate = _BaseGate
WeightedSum = _BaseWeightedSum
RubricDict = _BaseRubricDict
