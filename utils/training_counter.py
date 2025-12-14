import json
from pathlib import Path

COUNTERS_PATH = Path("models") / "training_counters.json"


def _load_all():
    if COUNTERS_PATH.exists():
        with open(COUNTERS_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return {}

        if not isinstance(data, dict):
            return {}

        legacy_like = False
        for _, v in data.items():
            if isinstance(v, (int, float)):
                legacy_like = True
                break
        if legacy_like:
            return {}

        return data
    return {}


def _save_all(data):
    COUNTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COUNTERS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def list_runs(algorithm: str):
    data = _load_all()
    algo_data = data.get(algorithm, {})
    if not isinstance(algo_data, dict):
        return []
    return list(algo_data.keys())


def load_training_stats(algorithm: str, run_name: str):
    data = _load_all()
    algo_data = data.get(algorithm, {})
    run_data = algo_data.get(run_name, {})
    iterations = run_data.get("iterations", 0)
    total_timesteps = run_data.get("total_timesteps", 0)
    best_score = run_data.get("best_score")
    return iterations, total_timesteps, best_score


def save_training_stats(algorithm: str, run_name: str, iterations: int, total_timesteps: int, best_score=None):
    data = _load_all()
    if algorithm not in data or not isinstance(data[algorithm], dict):
        data[algorithm] = {}
    data[algorithm][run_name] = {
        "iterations": int(iterations),
        "total_timesteps": int(total_timesteps),
        "best_score": best_score,
    }
    _save_all(data)
