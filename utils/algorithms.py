"""Stable-Baselines3 algorithm registry used by the CLI menus."""

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG

# Map algorithms to constructors
ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}


def get_algorithm_list():
    """Return list of algorithm names in the registry."""
    return list(ALGORITHMS.keys())
