from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG

# Map algorithms to constructors
ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}

HUMAN_ALGO_NAME = "Human"


def get_algorithm_list(include_human: bool = False):
    """Return list of algorithm names."""
    algos = list(ALGORITHMS.keys())
    if include_human:
        algos.append(HUMAN_ALGO_NAME)
    return algos
