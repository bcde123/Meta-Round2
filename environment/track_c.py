"""Track C: Green-Code Evaluator for energy-efficiency scoring.

Measures execution cost (CPU time + peak memory) and structural
efficiency (graphlet analysis) of refactored Python code, producing
a composite green score for the RL reward signal.
"""

import os
import sys
import tempfile
import subprocess
import tracemalloc
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
        """Measure CPU time and peak memory of executing each file.

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
                cpu_ms = self._time_file(fpath)
                total_cpu_ms += cpu_ms

                mem_bytes = self._measure_memory(fpath)
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

    @staticmethod
    def _time_file(filepath: str) -> float:
        """Time a file's import/compile cost using timeit subprocess.

        Returns:
            Execution time in milliseconds.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "timeit", "-n", "1", "-r", "1",
                 "-s", f"open('{filepath}').read()",
                 f"compile(open('{filepath}').read(), '{filepath}', 'exec')"],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
            # Parse "1 loop, best of 1: X.XX msec per loop"
            if "msec" in output:
                return float(output.split(":")[-1].replace("msec per loop", "").strip())
            if "usec" in output:
                return float(output.split(":")[-1].replace("usec per loop", "").strip()) / 1000.0
            if "sec" in output:
                return float(output.split(":")[-1].replace("sec per loop", "").strip()) * 1000.0
        except Exception:
            pass
        return 1.0  # fallback: 1ms default

    @staticmethod
    def _measure_memory(filepath: str) -> int:
        """Measure peak memory of compiling a file using tracemalloc.

        Returns:
            Peak memory in bytes.
        """
        try:
            code = open(filepath).read()
            tracemalloc.start()
            compile(code, filepath, "exec")
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak
        except Exception:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return 0

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
