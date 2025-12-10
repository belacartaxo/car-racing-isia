import time
import sys
from pathlib import Path

import gymnasium as gym
from loguru import logger

from environment.custom_car_racing_env import CustomCarRacingEnv
from environment.coins_car_racing_env import CoinsCarRacingEnv
from utils.algorithms import ALGORITHMS, get_algorithm_list
from utils.config import (
    ITERATIONS,
    TOTAL_TIMESTEPS,
    EVAL_EPISODES,
    MODELS_DIR,
    LOGS_DIR,
    LATEST_MODEL_NAME,
    DEVICE,
)
from utils.training_counter import (
    list_runs,
    load_training_stats,
    save_training_stats,
)


# Inspect environment
def see_space(env):
    logger.info("Inspecting environment spaces")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Sample action: {env.action_space.sample()}")
    logger.info(f"Sample observation shape: {env.observation_space.sample().shape}")


# Prepare folders
def prepare_algo_folder(algorithm):
    algo_dir = Path(MODELS_DIR) / algorithm
    algo_dir.mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    return algo_dir


def select_environment(render_mode: str):
    print("\nSelect environment type:")
    print("1 - Original CarRacing (no custom reward, no coins)")
    print("2 - Custom CarRacing (off-track penalty only)")
    print("3 - Custom CarRacing with coins (off-track penalty + coin reward)")

    choice = input("\nYour choice: ").strip()

    if choice == "1":
        env = gym.make("CarRacing-v3", render_mode=render_mode)
    elif choice == "2":
        env = CustomCarRacingEnv(render_mode=render_mode)
    elif choice == "3":
        env = CoinsCarRacingEnv(render_mode=render_mode)
    else:
        print("Invalid choice, defaulting to Original CarRacing.")
        env = gym.make("CarRacing-v3", render_mode=render_mode)

    return env


def select_action_for_run(algorithm: str):
    print("\nChoose action:")
    print("1 - New training run")
    print("2 - Continue training (select existing run)")
    print("3 - Evaluate model (select existing run)")
    print("4 - Override existing run (reset model and counters)")

    choice = input("\nYour choice: ").strip()
    return choice


def select_run_name(algorithm: str):
    runs = list_runs(algorithm)
    if not runs:
        print(f"\nNo existing runs for algorithm {algorithm}.")
        return None
    print("\nAvailable runs:")
    for i, name in enumerate(runs, start=1):
        print(f"{i} - {name}")
    sel = input("\nSelect run number: ").strip()
    try:
        idx = int(sel) - 1
        if 0 <= idx < len(runs):
            return runs[idx]
    except ValueError:
        pass
    print("Invalid selection.")
    return None


# Train model
def train_agent(env, model, algorithm, run_dir, run_name, start_iterations=0, start_total_timesteps=0):
    iterations = start_iterations
    total_timesteps = start_total_timesteps

    for i in range(1, ITERATIONS + 1):
        iterations += 1

        logger.info(f"Training run '{run_name}' for {algorithm}: global iteration {iterations}")
        logger.info(f"Local iteration {i}/{ITERATIONS} with {TOTAL_TIMESTEPS} timesteps")

        iteration_start = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
        iteration_end = time.time()

        duration = iteration_end - iteration_start
        logger.info(f"Iteration time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

        latest_path = Path(run_dir) / LATEST_MODEL_NAME
        model.save(str(latest_path))
        logger.info(f"Saved updated model to: {latest_path}")

        total_timesteps += TOTAL_TIMESTEPS
        save_training_stats(algorithm, run_name, iterations, total_timesteps)
        logger.info(f"Saved training progress for {algorithm} run '{run_name}': iteration {iterations}")

    return model, iterations, total_timesteps


# Evaluate model
def evaluate_agent(env, model):
    logger.info("Starting evaluation")

    for ep in range(EVAL_EPISODES):
        logger.info(f"Episode {ep + 1}/{EVAL_EPISODES}")

        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            env.render()

        logger.info(f"Episode reward: {total_reward}")


# MAIN
if __name__ == "__main__":
    logger.add("logs/training.log", rotation="1 MB")

    print("\nDo you want to see the episodes?")
    print("1 - Yes, show the simulation")
    print("2 - No, run without rendering (faster)")

    view_choice = input("\nYour choice: ").strip()
    render_mode = "human" if view_choice == "1" else None

    env = select_environment(render_mode)
    see_space(env)

    # SELECT ALGORITHM
    algo_list = get_algorithm_list()

    print("\nSelect training algorithm:")
    for i, name in enumerate(algo_list, start=1):
        print(f"{i} - {name}")

    algo_choice = int(input("\nAlgorithm number: "))
    algorithm = algo_list[algo_choice - 1]

    logger.info(f"Selected algorithm: {algorithm}")

    algo_dir = prepare_algo_folder(algorithm)
    action = select_action_for_run(algorithm)
    AlgoClass = ALGORITHMS[algorithm]

    model = None
    run_name = None
    iterations = 0
    total_timesteps = 0

    if action == "1":
        run_name = input("\nEnter a name for this run: ").strip()
        run_dir = Path(algo_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        model = AlgoClass(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=str(Path(LOGS_DIR) / f"{algorithm}_{run_name}"),
            device=DEVICE,
        )

    elif action in {"2", "3", "4"}:
        run_name = select_run_name(algorithm)
        if run_name is None:
            print("No valid run selected. Exiting.")
            sys.exit(0)

        run_dir = Path(algo_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / LATEST_MODEL_NAME

        if action != "4" and not model_path.exists():
            print(f"No saved model found for run '{run_name}' at {model_path}.")
            print("You may need to start a new training run first.")
            sys.exit(0)

        if action == "4":
            confirm = input(f"Type YES to override run '{run_name}': ").strip()
            if confirm != "YES":
                print("Override cancelled. Exiting.")
                sys.exit(0)
            logger.warning(f"Overriding existing run '{run_name}' for {algorithm}. Resetting model and counters.")
            model = AlgoClass(
                "CnnPolicy",
                env,
                verbose=1,
                tensorboard_log=str(Path(LOGS_DIR) / f"{algorithm}_{run_name}"),
                device=DEVICE,
            )
            iterations = 0
            total_timesteps = 0
        else:
            logger.info(f"Loading existing {algorithm} model from {model_path}")
            model = AlgoClass.load(str(model_path), env=env, device=DEVICE)
            iterations, total_timesteps = load_training_stats(algorithm, run_name)
            logger.info(f"Loaded training progress: iterations={iterations}, total_timesteps={total_timesteps}")
    else:
        print("Invalid action. Exiting.")
        sys.exit(0)

    if action in {"1", "2", "4"}:
        model, iterations, total_timesteps = train_agent(
            env,
            model,
            algorithm,
            run_dir,
            run_name,
            start_iterations=iterations,
            start_total_timesteps=total_timesteps,
        )
        evaluate_agent(env, model)
    elif action == "3":
        evaluate_agent(env, model)

    env.close()
