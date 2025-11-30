import json
import os
from utils.config import MODELS_DIR

def get_counter_path(algorithm):
    """Returns the file where the counter for this algorithm is stored."""
    return os.path.join(MODELS_DIR, algorithm, f"training_counter_{algorithm}.json")


def load_training_count(algorithm):
    """Loads training iteration count for this specific algorithm."""
    path = get_counter_path(algorithm)

    if not os.path.exists(path):
        return 0  # No training done yet for this algorithm

    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("count", 0)
    except:
        return 0


def save_training_count(algorithm, count):
    """Saves the updated counter for this algorithm."""
    path = get_counter_path(algorithm)
    folder = os.path.dirname(path)

    os.makedirs(folder, exist_ok=True)

    with open(path, "w") as f:
        json.dump({"count": count}, f, indent=2)
