import json
import os
from utils.config import MODELS_DIR, TOTAL_TIMESTEPS


def get_counter_path():
    return os.path.join(MODELS_DIR, "training_counters.json")


def _load_all():
    path = get_counter_path()

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}


def _save_all(data):
    path = get_counter_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================
# TREINO PADRÃO
# ============================

def load_training_count(algorithm):
    data = _load_all()
    return data.get(algorithm, {}).get("count", 0)


def load_training_timesteps(algorithm):
    data = _load_all()
    return data.get(algorithm, {}).get("timesteps", 0)


def save_training_progress(algorithm, new_count):
    data = _load_all()

    if algorithm not in data:
        data[algorithm] = {}

    data[algorithm]["count"] = new_count
    data[algorithm]["timesteps"] = new_count * TOTAL_TIMESTEPS

    _save_all(data)


# ============================
# ✅ BEST SCORE (NOVO)
# ============================

def load_best_score(algorithm, env_type):
    data = _load_all()
    return data.get(algorithm, {}).get(f"{env_type}_best_score", float("-inf"))


def save_best_score(algorithm, env_type, best_score):
    data = _load_all()

    if algorithm not in data:
        data[algorithm] = {}

    data[algorithm][f"{env_type}_best_score"] = best_score
    _save_all(data)
