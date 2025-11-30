import torch

# ENVIRONMENT SETTINGS
ENV_NAME = "CarRacing-v3"

# TRAINING SETTINGS
TOTAL_TIMESTEPS = 10_000    # steps per iteration
ITERATIONS = 30              # training loops
EVAL_EPISODES = 3           # evaluation episodes

# PATHS
MODELS_DIR = "models"
LOGS_DIR = "logs"
LATEST_MODEL_NAME = "latest_model.zip"

#  DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
