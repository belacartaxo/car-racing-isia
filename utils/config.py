import torch

# ENVIRONMENT SETTINGS
ENV_NAME = "CarRacing-v3"

# TRAINING SETTINGS
TOTAL_TIMESTEPS = 50_000    # steps per iteration (legacy, kept for reference)
ITERATIONS = 50              # training loops (legacy, kept for reference)

TRAIN_BUDGET_TIMESTEPS = 300_000
CHECKPOINT_TIMESTEPS = 50_000

EVAL_EPISODES = 10           # evaluation episodes

# PATHS
MODELS_DIR = "models"
LOGS_DIR = "logs"
LATEST_MODEL_NAME = "latest_model.zip"

#  DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
