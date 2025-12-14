import time
import sys
from pathlib import Path

import gymnasium as gym
from loguru import logger

from environment.custom_car_racing_env import CustomCarRacingEnv
from environment.coins_car_racing_env import CoinsCarRacingEnv
from environment.wrappers import CoinVisionChannelWrapper
from utils.algorithms import ALGORITHMS, HUMAN_ALGO_NAME, get_algorithm_list
from utils.config import (
    EVAL_EPISODES,
    TRAIN_BUDGET_TIMESTEPS,
    CHECKPOINT_TIMESTEPS,
    MODELS_DIR,
    LOGS_DIR,
    LAST_MODEL_NAME,
    BEST_MODEL_NAME,
    DEVICE,
)
from utils.csv_logger import attach_csv_logger, read_last_metric
from utils.training_counter import (
    list_runs,
    load_training_stats,
    save_training_stats,
)
from utils.human_control import run_human_control


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


def build_csv_log_path(algorithm: str, run_name: str, env):
    env_label = env.unwrapped.__class__.__name__
    safe_run = run_name.replace(" ", "_")
    filename = f"{algorithm}_{safe_run}_{env_label}_metrics.csv"
    return Path(LOGS_DIR) / filename, env_label


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


def ensure_human_render_env(env):
    """
    Recreate the environment with render_mode='human' if needed so we can capture keyboard input.
    """
    base_env = env.unwrapped
    if getattr(base_env, "render_mode", None) == "human":
        return env

    logger.info("Render was disabled; recreating environment with render_mode='human' for manual control.")
    env.close()

    if isinstance(base_env, CoinsCarRacingEnv):
        new_env = CoinsCarRacingEnv(render_mode="human")
    elif isinstance(base_env, CustomCarRacingEnv):
        new_env = CustomCarRacingEnv(render_mode="human")
    else:
        new_env = gym.make("CarRacing-v3", render_mode="human")

    return gym.wrappers.TimeLimit(new_env, max_episode_steps=1000)


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
def train_agent(
    env,
    model,
    algorithm,
    run_dir,
    run_name,
    csv_path,
    best_score=None,
    start_iterations=0,
    start_total_timesteps=0,
):
    iterations = start_iterations  # counts chunks
    total_timesteps = start_total_timesteps
    best_path = Path(run_dir) / BEST_MODEL_NAME

    remainder = total_timesteps % TRAIN_BUDGET_TIMESTEPS
    remaining_budget = TRAIN_BUDGET_TIMESTEPS if remainder == 0 else TRAIN_BUDGET_TIMESTEPS - remainder
    logger.info(
        f"Starting budgeted training: {remaining_budget} remaining out of {TRAIN_BUDGET_TIMESTEPS} for this cycle "
        f"({algorithm} / {run_name})"
    )

    if not best_path.exists():
        model.save(str(best_path))
        logger.info("Initialized best model at start (no existing best).")

    chunk_idx = 0
    while remaining_budget > 0:
        chunk_idx += 1
        chunk_steps = min(CHECKPOINT_TIMESTEPS, remaining_budget)
        iterations += 1

        logger.info(
            f"Training chunk {chunk_idx}: {chunk_steps} timesteps "
            f"(global chunk {iterations}, target total {total_timesteps + chunk_steps})"
        )

        iteration_start = time.time()
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        iteration_end = time.time()

        duration = iteration_end - iteration_start
        logger.info(f"Chunk time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

        last_path = Path(run_dir) / LAST_MODEL_NAME
        model.save(str(last_path))
        checkpoint_path = Path(run_dir) / f"checkpoint_{total_timesteps + chunk_steps}.zip"
        model.save(str(checkpoint_path))
        logger.info(f"Saved models to: last={last_path.name}, checkpoint={checkpoint_path.name}")

        # Get latest ep_rew_mean from CSV log (if available)
        latest_score = read_last_metric(csv_path, "rollout/ep_rew_mean")

        # If no best exists yet, initialize best from last immediately
        # Promote best if improved
        if latest_score is not None:
            if best_score is None or latest_score > best_score:
                best_score = latest_score
                model.save(str(best_path))
                logger.info(f"New best model (ep_rew_mean={latest_score:.3f}) saved to: {best_path.name}")

        total_timesteps += chunk_steps
        save_training_stats(algorithm, run_name, iterations, total_timesteps, best_score)
        logger.info(
            f"Saved training progress for {algorithm} run '{run_name}': chunks={iterations}, total_timesteps={total_timesteps}, best_score={best_score}"
        )

        remaining_budget -= chunk_steps

    return model, iterations, total_timesteps, best_score


# Evaluate model
def evaluate_agent(env, model):
    logger.info("Starting evaluation")

    render_is_human = getattr(env.unwrapped, "render_mode", None) == "human"
    if render_is_human:
        logger.warning("Render enabled during evaluation: timing will differ. Use headless for consistent metrics.")
    else:
        logger.info("Evaluation running headless for consistent timing.")

    for ep in range(EVAL_EPISODES):
        logger.info(f"Episode {ep + 1}/{EVAL_EPISODES}")

        obs, info = env.reset()
        done = False
        total_reward = 0.0
        total_coin_bonus = 0.0
        steps = 0

        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            if steps % 100 == 0:
                print("step", steps, "| off_track:", info.get("off_track"), "| total_reward:", total_reward)
                print("action:", action)  # action = [steer, gas, brake]


            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_coin_bonus += float(info.get("coin_bonus", 0.0))
            done = terminated or truncated

            if render_is_human:
                env.render()

            steps += 1

        base_reward = total_reward - total_coin_bonus
        logger.info(
            f"Episode reward: {total_reward:.2f} | base(no coins): {base_reward:.2f} "
            f"| coin_bonus: {total_coin_bonus:.2f} | coins_episode: {info.get('coins_collected_episode', 0)} "
            f"| steps: {steps}"
        )

# MAIN
if __name__ == "__main__":
    logger.add("logs/training.log", rotation="1 MB")

    print("\nDo you want to see the episodes?")
    print("1 - Yes, show the simulation")
    print("2 - No, run without rendering (faster)")

    view_choice = input("\nYour choice: ").strip()
    render_mode = "human" if view_choice == "1" else None

    env = select_environment(render_mode)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    if isinstance(env.unwrapped, CoinsCarRacingEnv):
        env = CoinVisionChannelWrapper(env)
    see_space(env)

    print(
        f"\nThis run will train for {TRAIN_BUDGET_TIMESTEPS} timesteps in chunks of {CHECKPOINT_TIMESTEPS}."
    )

    # SELECT ALGORITHM
    algo_list = get_algorithm_list(include_human=True)

    print("\nSelect training algorithm:")
    for i, name in enumerate(algo_list, start=1):
        print(f"{i} - {name}")

    algo_choice = int(input("\nAlgorithm number: "))
    algorithm = algo_list[algo_choice - 1]

    logger.info(f"Selected algorithm: {algorithm}")

    if algorithm == HUMAN_ALGO_NAME:
        env = ensure_human_render_env(env)
        run_human_control(env)
        env.close()
        sys.exit(0)

    algo_dir = prepare_algo_folder(algorithm)
    action = select_action_for_run(algorithm)
    AlgoClass = ALGORITHMS[algorithm]

    model = None
    run_name = None
    iterations = 0
    total_timesteps = 0
    best_score = None

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
        best_score = None

    elif action in {"2", "3", "4"}:
        run_name = select_run_name(algorithm)
        if run_name is None:
            print("No valid run selected. Exiting.")
            sys.exit(0)

        run_dir = Path(algo_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / BEST_MODEL_NAME
        last_path = run_dir / LAST_MODEL_NAME
        legacy_path = run_dir / "latest_model.zip"
        load_path = None

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
            best_score = None
        else:
            if best_path.exists():
                load_path = best_path
            elif last_path.exists():
                load_path = last_path
            elif legacy_path.exists():
                load_path = legacy_path

            if load_path is None:
                print(f"No saved model found for run '{run_name}'.")
                print("You may need to start a new training run first.")
                sys.exit(0)

            logger.info(f"Loading existing {algorithm} model from {load_path}")
            model = AlgoClass.load(str(load_path), env=env, device=DEVICE)
            iterations, total_timesteps, best_score = load_training_stats(algorithm, run_name)
            logger.info(
                f"Loaded training progress: iterations={iterations}, total_timesteps={total_timesteps}, best_score={best_score}"
            )
    else:
        print("Invalid action. Exiting.")
        sys.exit(0)

    if action in {"1", "2", "4"}:
        csv_path, env_label = build_csv_log_path(algorithm, run_name, env)
        attach_csv_logger(
            model,
            csv_path,
            meta={
                "meta/algorithm": algorithm,
                "meta/run": run_name,
                "meta/env": env_label,
            },
        )
        logger.info(f"Metrics CSV (append mode): {csv_path}")

        model, iterations, total_timesteps, best_score = train_agent(
            env,
            model,
            algorithm,
            run_dir,
            run_name,
            csv_path,
            best_score=best_score,
            start_iterations=iterations,
            start_total_timesteps=total_timesteps,
        )
        evaluate_agent(env, model)
    elif action == "3":
        evaluate_agent(env, model)

    env.close()
