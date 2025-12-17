# Car Racing Reinforcement Learning

## Project Overview

This project explores Reinforcement Learning (RL) applied to the CarRacing Box2D environment from Gymnasium.
The original environment was extended with additional mechanics and reward shaping to study how different RL algorithms behave.

Three RL algorithms were implemented and evaluated:

- PPO (Proximal Policy Optimization)

- A2C (Advantage Actor-Critic)

- DDPG (Deep Deterministic Policy Gradient)

The goal is to compare their performance across the original and modified environments.

## Environments
- Original Environment
    - Based on CarRacing-v3 (Gymnasium)

    - Observation: 96×96 RGB image

    - Default reward function

- Custom Environments

    - CustomCarRacingEnv

        - Reward shaping (off-track penalties, speed bonus)

        - Explicit on/off-track detection

    - CoinsCarRacingEnv

        - Collectible coins as intermediate goals

        - Additional rewards for coin collection

        - Optional multimodal observations (image + structured coin features)

These extensions make the task more constrained and realistic, encouraging stable driving behavior.

## How to Run
1. Create a Virtual Environment

```bash
python3.10 -m venv venv
```

Activate the environment:

Linux / macOS
```bash
source venv/bin/activate
```

Windows
```bash
venv\Scripts\activate
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Project
```bash
python -m main
```

This command launches training and evaluation according to the configuration defined in main.py.
