"""Centralized configuration for environments, training loop and algorithm hyperparameters."""

import torch

# ENVIRONMENT SETTINGS
ENV_NAME = "CarRacing-v3"

# TRAINING SETTINGS
TOTAL_TIMESTEPS = 50_000    # steps per iteration
ITERATIONS = 50             # training loops
EVAL_EPISODES = 3           # evaluation episodes

# ALGORITHM HYPERPARAMETERS
# Tweak values here to change default setups project-wide.
HYPERPARAMS = {
    "PPO": {
        # Stable-Baselines3 defaults for PPO (kept explicit here for clarity).
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    # Off-policy algos need smaller replay buffers for pixel inputs to avoid OOM.
    "SAC": {
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        # optimize_memory_usage is not supported for Dict obs; keep it False.
        "optimize_memory_usage": False,
    },
    "TD3": {
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "optimize_memory_usage": False,
    },
    "DDPG": {
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "optimize_memory_usage": False,
    },
}

# PATHS
MODELS_DIR = "models"
LOGS_DIR = "logs"
LATEST_MODEL_NAME = "latest_model.zip"
BEST_MODEL_NAME = "best_model.zip"

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
