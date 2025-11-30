import json
import os
from utils.config import MODELS_DIR, TOTAL_TIMESTEPS


def get_counter_path():
    """Unique JSON file for all algorithms."""
    return os.path.join(MODELS_DIR, "training_counters.json")


def _load_all():
    """Load the full JSON file with counters."""
    path = get_counter_path()

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}  # corrupted file → reset


def _save_all(data):
    """Save all counters back to the JSON file."""
    path = get_counter_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ----- PUBLIC FUNCTIONS -----

def load_training_count(algorithm):
    """Return iteration count for this algorithm."""
    data = _load_all()
    return data.get(algorithm, {}).get("count", 0)


def load_training_timesteps(algorithm):
    """Return accumulated timesteps for this algorithm."""
    data = _load_all()
    return data.get(algorithm, {}).get("timesteps", 0)


def save_training_progress(algorithm, new_count):
    """
    Save:
      - iteration count
      - total accumulated timesteps
    """
    data = _load_all()

    if algorithm not in data:
        data[algorithm] = {"count": 0, "timesteps": 0}

    data[algorithm]["count"] = new_count
    data[algorithm]["timesteps"] = new_count * TOTAL_TIMESTEPS

    _save_all(data)
