"""Utilities to track training progress and best rewards across runs/models."""

import json
from pathlib import Path

COUNTERS_PATH = Path("models") / "training_counters.json"


def _load_all():
    """Load counters file into memory, handling legacy/invalid formats."""
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
    """Persist counters to disk, ensuring parent directory exists."""
    COUNTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COUNTERS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _list_run_dirs(algorithm: str):
    """Return existing run directories for a given algorithm."""
    algo_dir = COUNTERS_PATH.parent / algorithm
    if not algo_dir.exists():
        return []
    return [p.name for p in algo_dir.iterdir() if p.is_dir()]


def list_runs(algorithm: str):
    """List known runs combining counters file and actual directories."""
    data = _load_all()
    algo_data = data.get(algorithm, {})
    counter_runs = list(algo_data.keys()) if isinstance(algo_data, dict) else []
    dir_runs = _list_run_dirs(algorithm)
    # Remover entradas sem diret¢rio
    counter_runs = [name for name in counter_runs if name in dir_runs]
    # Unique, preserve insertion with counter first then dirs
    seen = set()
    result = []
    for name in counter_runs + dir_runs:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def load_training_stats(algorithm: str, run_name: str):
    """Load iteration and timestep counters for a run."""
    data = _load_all()
    algo_data = data.get(algorithm, {})
    run_data = algo_data.get(run_name, {})
    iterations = run_data.get("iterations", 0)
    total_timesteps = run_data.get("total_timesteps", 0)
    return iterations, total_timesteps


def load_best_reward(algorithm: str, run_name: str):
    """Return best_reward stored for a run, or None."""
    data = _load_all()
    algo_data = data.get(algorithm, {})
    run_data = algo_data.get(run_name, {})
    return run_data.get("best_reward")


def save_training_stats(algorithm: str, run_name: str, iterations: int, total_timesteps: int):
    """Persist iteration and timestep counters, keeping existing best_reward if present."""
    data = _load_all()
    if algorithm not in data or not isinstance(data[algorithm], dict):
        data[algorithm] = {}
    existing = data[algorithm].get(run_name, {})
    data[algorithm][run_name] = {
        "iterations": int(iterations),
        "total_timesteps": int(total_timesteps),
        "best_reward": existing.get("best_reward"),
    }
    _save_all(data)


def save_best_reward(algorithm: str, run_name: str, best_reward):
    """Persist best_reward for a run, creating the entry if needed."""
    data = _load_all()
    if algorithm not in data or not isinstance(data[algorithm], dict):
        data[algorithm] = {}
    if run_name not in data[algorithm] or not isinstance(data[algorithm][run_name], dict):
        data[algorithm][run_name] = {"iterations": 0, "total_timesteps": 0}
    data[algorithm][run_name]["best_reward"] = best_reward
    _save_all(data)


def prune_missing_dirs():
    """Remove counter entries whose run directories no longer exist."""
    data = _load_all()
    changed = False
    for algo, runs in list(data.items()):
        if not isinstance(runs, dict):
            continue
        valid_dirs = set(_list_run_dirs(algo))
        for run in list(runs.keys()):
            if run not in valid_dirs:
                runs.pop(run, None)
                changed = True
    if changed:
        _save_all(data)
