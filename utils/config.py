import torch

# ENVIRONMENT SETTINGS
ENV_NAME = "CarRacing-v3"

# TRAINING SETTINGS
TOTAL_TIMESTEPS = 50_000    # steps per iteration
ITERATIONS = 50             # training loops
EVAL_EPISODES = 3           # evaluation episodes

# PATHS
MODELS_DIR = "models"
LOGS_DIR = "logs"

# MODEL FILENAMES (EXPLÍCITOS)
ORIGINAL_LATEST_MODEL_NAME = "original_latest_model.zip"
ORIGINAL_BEST_MODEL_NAME   = "original_best_model.zip"

COINS_LATEST_MODEL_NAME = "coins_latest_model.zip"
COINS_BEST_MODEL_NAME   = "coins_best_model.zip"

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
