"""CO2 savings calculator and dashboard data generator.

Converts CPU time savings into real-world CO2 impact estimates
using industry-standard carbon intensity factors and provides
a structured dashboard payload for the /dashboard/co2 endpoint.
"""

from typing import Dict, Any


# ── Carbon constants (real-world data) ────────────────────────────────────────
CARBON_INTENSITY_G_PER_KWH: float = 475.0   # global average grid, gCO2/kWh
CPU_TDP_WATTS: float = 15.0                 # typical server CPU core TDP

# Equivalence factors
TREE_ABSORPTION_KG_PER_YEAR: float = 21.0   # 1 tree absorbs ~21 kg CO2/year
CAR_EMISSION_G_PER_KM: float = 120.0        # average car emits ~120 g CO2/km


def estimate_co2_saved(
    cpu_time_saved_ms: float,
    runs_per_day: int = 10_000,
) -> Dict[str, float]:
    """Estimate CO2 savings from reduced CPU execution time.

    Args:
        cpu_time_saved_ms: CPU time reduction per execution in milliseconds.
        runs_per_day: Estimated number of daily executions.

    Returns:
        Dictionary with CO2 savings metrics:
            - grams_per_day: daily CO2 savings in grams
            - kg_per_year: annual CO2 savings in kilograms
            - equivalent_trees: number of trees absorbing equivalent CO2/year
            - equivalent_car_km: equivalent car travel distance in km
    """
    # CPU time saved per day (seconds)
    cpu_seconds_per_day = (cpu_time_saved_ms / 1000.0) * runs_per_day

    # Energy saved per day (kWh): power(W) × time(h)
    energy_kwh_per_day = (CPU_TDP_WATTS * cpu_seconds_per_day) / 3600.0

    # CO2 saved per day (grams)
    grams_per_day = energy_kwh_per_day * CARBON_INTENSITY_G_PER_KWH

    # Annual projection
    kg_per_year = (grams_per_day * 365.0) / 1000.0

    # Real-world equivalences
    equivalent_trees = kg_per_year / TREE_ABSORPTION_KG_PER_YEAR if kg_per_year > 0 else 0.0
    equivalent_car_km = (kg_per_year * 1000.0) / CAR_EMISSION_G_PER_KM if kg_per_year > 0 else 0.0

    return {
        "grams_per_day": round(grams_per_day, 4),
        "kg_per_year": round(kg_per_year, 4),
        "equivalent_trees": round(equivalent_trees, 2),
        "equivalent_car_km": round(equivalent_car_km, 2),
    }


def generate_dashboard_data(
    green_score: Any,
    orig_files: Dict[str, str],
    updated_files: Dict[str, str],
) -> Dict[str, Any]:
    """Generate full dashboard payload with green metrics and CO2 estimates.

    Args:
        green_score: GreenScore dataclass from Track C evaluation.
        orig_files: Original codebase files.
        updated_files: Refactored codebase files.

    Returns:
        Complete dashboard payload dictionary.
    """
    from .track_c import GreenCodeEvaluator

    evaluator = GreenCodeEvaluator()
    orig_exec = evaluator.measure_execution(orig_files)
    new_exec = evaluator.measure_execution(updated_files)

    cpu_saved_ms = max(0.0, orig_exec["cpu_time_ms"] - new_exec["cpu_time_ms"])
    memory_saved_mb = max(0.0, orig_exec["peak_memory_mb"] - new_exec["peak_memory_mb"])

    co2_data = estimate_co2_saved(cpu_saved_ms)

    return {
        "green_score": {
            "graphlet_score": green_score.graphlet_score,
            "cpu_improvement": green_score.cpu_improvement,
            "memory_improvement": green_score.memory_improvement,
            "total": green_score.total,
        },
        "execution_metrics": {
            "original": orig_exec,
            "refactored": new_exec,
            "cpu_saved_ms": round(cpu_saved_ms, 3),
            "memory_saved_mb": round(memory_saved_mb, 4),
        },
        "co2_savings": co2_data,
        "assumptions": {
            "carbon_intensity_g_per_kwh": CARBON_INTENSITY_G_PER_KWH,
            "cpu_tdp_watts": CPU_TDP_WATTS,
            "runs_per_day": 10_000,
        },
        "file_count": {
            "original": len(orig_files),
            "refactored": len(updated_files),
        },
    }
