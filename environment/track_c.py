"""Track C: Green-Code Evaluator for energy-efficiency scoring.

Measures execution cost (CPU time + peak memory) and structural
efficiency (graphlet analysis) of refactored Python code, producing
a composite green score for the RL reward signal.
"""

import os
import sys
import tempfile
import subprocess
import json
from dataclasses import dataclass
from typing import Dict

from .graphlet_analyzer import analyze_graphlets


@dataclass
class GreenScore:
    """Composite energy-efficiency score for refactored code."""

    graphlet_score: float
    cpu_improvement: float
    memory_improvement: float
    total: float


class GreenCodeEvaluator:
    """Evaluates energy efficiency of Python code via graphlet
    analysis and runtime measurement."""

    def measure_execution(self, files: Dict[str, str]) -> Dict[str, float]:
        """Measure CPU time and peak memory for each file.

        We first try to execute top-level, pure-looking functions with synthetic
        sample inputs inside a subprocess (timeout-bounded). Files that cannot be
        safely imported/executed fall back to compile-time profiling, which still
        catches AST-size and memory-bloat regressions without hanging training.

        Args:
            files: Mapping of filename → source code.

        Returns:
            Dict with 'cpu_time_ms' and 'peak_memory_mb'.
        """
        total_cpu_ms = 0.0
        peak_mem_bytes = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, content in files.items():
                fpath = os.path.join(tmpdir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w") as f:
                    f.write(content)

            for fname in files:
                fpath = os.path.join(tmpdir, fname)
                profile = self._profile_file(fpath)
                total_cpu_ms += profile["cpu_time_ms"]
                mem_bytes = profile["peak_memory_bytes"]
                peak_mem_bytes = max(peak_mem_bytes, mem_bytes)

        return {
            "cpu_time_ms": round(total_cpu_ms, 3),
            "peak_memory_mb": round(peak_mem_bytes / (1024 * 1024), 4),
        }

    def evaluate_graphlets(self, files: Dict[str, str]) -> float:
        """Compute average graphlet score across all files.

        Args:
            files: Mapping of filename → source code.

        Returns:
            Average graphlet score (0.0 to 1.0).
        """
        if not files:
            return 1.0

        scores = []
        for content in files.values():
            result = analyze_graphlets(content)
            scores.append(result["score"])

        return sum(scores) / len(scores)

    def evaluate(
        self,
        orig_files: Dict[str, str],
        updated_files: Dict[str, str],
    ) -> GreenScore:
        """Evaluate green-code improvements between original and updated code.

        Args:
            orig_files: Original codebase files.
            updated_files: Refactored codebase files.

        Returns:
            GreenScore dataclass with all component scores.
        """
        graphlet_score = self.evaluate_graphlets(updated_files)

        orig_exec = self.measure_execution(orig_files)
        new_exec = self.measure_execution(updated_files)

        cpu_improvement = self._relative_improvement(
            orig_exec["cpu_time_ms"], new_exec["cpu_time_ms"]
        )
        memory_improvement = self._relative_improvement(
            orig_exec["peak_memory_mb"], new_exec["peak_memory_mb"]
        )

        total = (
            0.40 * graphlet_score
            + 0.35 * cpu_improvement
            + 0.25 * memory_improvement
        )
        total = max(0.0, min(1.0, total))

        return GreenScore(
            graphlet_score=round(graphlet_score, 4),
            cpu_improvement=round(cpu_improvement, 4),
            memory_improvement=round(memory_improvement, 4),
            total=round(total, 4),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _profile_file(self, filepath: str) -> Dict[str, float]:
        """Return runtime profile, falling back to compile profile on failure."""
        runtime = self._profile_runtime(filepath)
        if runtime is not None:
            return runtime
        return self._profile_compile(filepath)

    @staticmethod
    def _profile_runtime(filepath: str) -> Dict[str, float] | None:
        """Profile top-level function calls in a subprocess.

        Returns None when a file cannot be safely executed (e.g. package-relative
        imports in the synthetic codebase); caller then uses compile fallback.
        """
        harness = r'''
import ast, datetime as _dt, inspect, json, sys, time, tracemalloc

path = sys.argv[1]
code = open(path).read()
tree = ast.parse(code)
func_names = [
    node.name for node in tree.body
    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
]

def sample_for(param):
    name = param.name.lower()
    if "date_str" in name or name.endswith("_str"):
        return "2026-01-01T00:00:00"
    if "date" in name:
        return _dt.datetime(2026, 1, 1)
    if "items" in name or "rows" in name or "users" in name or "tasks" in name:
        return list(range(200))
    if "data" in name or "dict" in name:
        return {"title": "sample task", "due_date": "2026-01-01T00:00:00"}
    if "page" in name or "size" in name or "count" in name or "n" == name:
        return 10
    if "input" in name or "title" in name or "name" in name or "text" in name:
        return "Sample Input"
    return 10

ns = {"__name__": "__green_profile__"}
exec(compile(code, path, "exec"), ns)
calls = []
for fname in func_names:
    fn = ns.get(fname)
    if not callable(fn) or inspect.iscoroutinefunction(fn):
        continue
    sig = inspect.signature(fn)
    args = []
    skip = False
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            skip = True
            break
        if p.default is not p.empty:
            continue
        args.append(sample_for(p))
    if not skip:
        calls.append((fn, args))

if not calls:
    # Still execute module top-level under tracing; compile-only fallback handles
    # the fully non-callable case.
    calls = []

tracemalloc.start()
t0 = time.perf_counter()
for _ in range(5):
    for fn, args in calls:
        try:
            fn(*args)
        except Exception:
            pass
elapsed_ms = (time.perf_counter() - t0) * 1000.0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({"cpu_time_ms": elapsed_ms, "peak_memory_bytes": peak}))
'''
        try:
            result = subprocess.run(
                [sys.executable, "-c", harness, filepath],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout.strip())
            return {
                "cpu_time_ms": float(data["cpu_time_ms"]),
                "peak_memory_bytes": float(data["peak_memory_bytes"]),
            }
        except Exception:
            return None

    @staticmethod
    def _profile_compile(filepath: str) -> Dict[str, float]:
        """Profile compile cost using a subprocess fallback."""
        harness = r'''
import json, sys, time, tracemalloc
path = sys.argv[1]
code = open(path).read()
tracemalloc.start()
t0 = time.perf_counter()
compile(code, path, "exec")
elapsed_ms = (time.perf_counter() - t0) * 1000.0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({"cpu_time_ms": elapsed_ms, "peak_memory_bytes": peak}))
'''
        try:
            result = subprocess.run(
                [sys.executable, "-c", harness, filepath],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                return {
                    "cpu_time_ms": float(data["cpu_time_ms"]),
                    "peak_memory_bytes": float(data["peak_memory_bytes"]),
                }
        except Exception:
            pass
        return {"cpu_time_ms": 1.0, "peak_memory_bytes": 0.0}

    @staticmethod
    def _relative_improvement(orig: float, new: float) -> float:
        """Compute clamped relative improvement ratio.

        Args:
            orig: Original metric value.
            new: New metric value (lower is better).

        Returns:
            Improvement ratio clamped to [0.0, 1.0].
        """
        if orig <= 0:
            return 0.5  # no baseline → neutral
        improvement = (orig - new) / orig
        return max(0.0, min(1.0, improvement))
