"""CLI to train, evaluate, resume and play CarRacing agents with custom environments."""

import time
import sys
from pathlib import Path
import csv
from typing import Any
from numbers import Real

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from loguru import logger
from stable_baselines3.common.logger import (
    Logger,
    HumanOutputFormat,
    TensorBoardOutputFormat,
    KVWriter,
    filter_excluded_keys,
    Video,
    Figure,
    Image,
    HParam,
)
from stable_baselines3.common.save_util import load_from_zip_file

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
    BEST_MODEL_NAME,
    DEVICE,
    HYPERPARAMS,
)
from utils.training_counter import (
    list_runs,
    load_training_stats,
    save_training_stats,
    prune_missing_dirs,
    load_best_reward,
    save_best_reward,
)


# Inspect environment
def see_space(env):
    """Log the action/observation spaces and a sample observation for debugging."""
    logger.info("Inspecting environment spaces")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Sample action: {env.action_space.sample()}")
    sample_obs = env.observation_space.sample()
    if isinstance(env.observation_space, spaces.Dict):
        logger.info(f"Sample observation keys: {list(sample_obs.keys())}")
        for key, value in sample_obs.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            logger.info(f"Sample '{key}' shape: {shape}, dtype: {dtype}")
    else:
        logger.info(f"Sample observation shape: {sample_obs.shape}")


# Prepare folders
def prepare_algo_folder(algorithm):
    """Create base folders for models/logs of the chosen algorithm."""
    algo_dir = Path(MODELS_DIR) / algorithm
    algo_dir.mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    return algo_dir


class SemicolonCSVOutputFormat(KVWriter):
    """
    Igual ao CSVOutputFormat, mas usando ';' para facilitar abertura em locale pt-BR/Excel.
    """

    def __init__(self, filename: str):
        self.separator = ';'
        self.quotechar = '"'
        self.keys: list[str] = []
        self.file_path = Path(filename)
        mode = 'r+' if self.file_path.exists() else 'w+'
        self.file = open(self.file_path, mode)
        self._load_existing_keys()
        # Garante que novos logs sejam acrescentados ao final do arquivo existente
        self.file.seek(0, 2)

    def _load_existing_keys(self) -> None:
        """Recarrega cabecalho se o CSV ja existir para nao sobrescrever dados anteriores."""
        self.file.seek(0)
        header = self.file.readline()
        if header:
            self.keys = header.strip().split(self.separator)

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        key_values = filter_excluded_keys(key_values, key_excluded, 'csv')
        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for i, key in enumerate(self.keys):
                if i > 0:
                    self.file.write(self.separator)
                self.file.write(key)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line.rstrip("\n"))
                self.file.write(self.separator * len(extra_keys))
                self.file.write("\n")
            # remove qualquer resquicio de escrita anterior caso o arquivo fosse maior
            self.file.truncate()
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(self.separator)
            value = key_values.get(key)

            if isinstance(value, Video):
                raise RuntimeError('Formato video nao suportado no CSV.')
            elif isinstance(value, Figure):
                raise RuntimeError('Formato figure nao suportado no CSV.')
            elif isinstance(value, Image):
                raise RuntimeError('Formato image nao suportado no CSV.')
            elif isinstance(value, HParam):
                raise RuntimeError('Formato hparam nao suportado no CSV.')
            elif isinstance(value, Real) and not isinstance(value, bool):
                fval = float(value)
                if fval.is_integer():
                    self.file.write(str(int(fval)))
                else:
                    txt = format(fval, '.3g')
                    self.file.write(txt)
            elif isinstance(value, str):
                value = value.replace(self.quotechar, self.quotechar + self.quotechar)
                self.file.write(self.quotechar + value + self.quotechar)
            elif value is not None:
                self.file.write(str(value))
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()



def _set_sb3_logger(model, algorithm, run_name, env_label=None):
    """Attach Human, TensorBoard and semicolon-separated CSV loggers to the model."""
    if not algorithm or not run_name:
        return model
    log_dir = Path(LOGS_DIR) / f"{algorithm}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = (env_label or "env").lower()
    csv_name = f"progress_{label}_{algorithm}_{run_name}.csv"
    output_formats = [
        HumanOutputFormat(sys.stdout),
        TensorBoardOutputFormat(str(log_dir)),
        SemicolonCSVOutputFormat(str(log_dir / csv_name)),
    ]
    model.set_logger(Logger(str(log_dir), output_formats))
    return model


def _log_eval_metrics(model, iteration, total_timesteps, episode, reward):
    """Log evaluation metrics to the model logger if available."""
    if not getattr(model, "logger", None):
        return
    if iteration is not None:
        model.logger.record("eval/iteration", iteration)
    if total_timesteps is not None:
        model.logger.record("eval/total_timesteps", total_timesteps)
    model.logger.record("eval/episode", episode)
    model.logger.record("eval/episode_reward", reward)
    model.logger.dump(total_timesteps if total_timesteps is not None else 0)


def _log_run_config(model, env, algorithm, run_name, policy, env_label):
    """Record config values (algo, env, hyperparams) into the logger for traceability."""
    if not getattr(model, "logger", None):
        return

    def _rec(name, val):
        if val is None:
            return
        model.logger.record(name, val)

    _rec("config/algorithm", algorithm)
    _rec("config/run_name", run_name)
    _rec("config/policy", policy)
    _rec("config/env_label", env_label)
    _rec("config/env_class", env.__class__.__name__)
    if hasattr(env, "observation_mode"):
        _rec("config/env_observation_mode", getattr(env, "observation_mode"))
    if hasattr(env, "include_mask"):
        _rec("config/env_include_mask", getattr(env, "include_mask"))

    lr = getattr(model, "learning_rate", None)
    _rec("config/learning_rate", lr if isinstance(lr, Real) else str(lr))
    for key in ["n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"]:
        val = getattr(model, key, None)
        if val is not None:
            if isinstance(val, Real):
                _rec(f"config/{key}", val)
            else:
                _rec(f"config/{key}", str(val))

    # Environment-specific settings
    for key in [
        "off_track_penalty",
        "off_track_border",
        "max_off_track_steps",
        "idle_min_speed",
        "idle_patience_steps",
        "idle_penalty",
        "speed_reward_scale",
        "coin_reward",
        "n_coins",
    ]:
        if hasattr(env, key):
            val = getattr(env, key)
            _rec(f"config/env_{key}", val)

    model.logger.dump(0)


def select_environment_choice():
    """Prompt user for base environment selection (original/custom/coins)."""
    print("\nSelect environment type:")
    print("1 - Original CarRacing (no custom reward, no coins)")
    print("2 - Custom CarRacing (off-track penalty only)")
    print("3 - Custom CarRacing with coins")
    return input("\nYour choice: ").strip()


COINS_VARIANTS = {
    "cnn": {
        "observation_mode": "image_only",
        "include_mask": False,
        "policy": "CnnPolicy",
        "label": "coins-cnn",
        "desc": "Coins CNN (image only)",
    },
    "multi_coins": {
        "observation_mode": "coins_only",
        "include_mask": False,
        "policy": "MultiInputPolicy",
        "label": "coins-coins",
        "desc": "MultiInput (coins only)",
    },
    "multi_full": {
        "observation_mode": "image_and_coins",
        "include_mask": True,
        "policy": "MultiInputPolicy",
        "label": "coins-full",
        "desc": "MultiInput (coins + track)",
    },
}


def select_coins_variant_for_new():
    """Prompt for coins observation/policy variant when starting a new run."""
    print("\nSelect coins observation/policy type:")
    options = [
        ("1", COINS_VARIANTS["cnn"]["desc"]),
        ("2", COINS_VARIANTS["multi_coins"]["desc"]),
        ("3", COINS_VARIANTS["multi_full"]["desc"]),
    ]
    for k, label in options:
        print(f"{k} - {label}")
    choice = input("\nYour choice: ").strip()
    if choice == "1":
        return "cnn"
    if choice == "2":
        return "multi_coins"
    return "multi_full"


def infer_coins_variant_from_model(model_path):
    """Inspect a saved model to guess which coins observation variant it expects."""
    try:
        loaded = load_from_zip_file(model_path)
    except Exception as e:
        logger.warning(f"Could not inspect model at {model_path}: {e}. Falling back to coins-full with mask.")
        return "multi_full", True

    if isinstance(loaded, tuple):
        data = loaded[0]
    else:
        data = loaded
    obs_space = data.get("observation_space")
    if isinstance(obs_space, spaces.Dict):
        keys = set(obs_space.spaces.keys())
        if keys == {"coins"}:
            return "multi_coins", False
        if "image" in keys:
            img_space = obs_space.spaces["image"]
            shape = getattr(img_space, "shape", ())
            channels = shape[0] if len(shape) == 3 else shape[-1] if len(shape) >= 1 else 3
            include_mask = channels >= 4
            return "multi_full", include_mask
    if isinstance(obs_space, spaces.Box):
        shape = getattr(obs_space, "shape", ())
        if len(shape) == 3:
            channels = shape[0]
            include_mask = channels >= 4
            return "cnn", include_mask
        if len(shape) == 1:
            return "multi_coins", False
    return "multi_full", True


def build_environment(env_choice, render_mode, coins_variant=None, include_mask_override=None):
    """Instantiate the selected environment and return it along with label and policy hint."""
    if env_choice == "1":
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        return env, "original", "CnnPolicy"
    if env_choice == "2":
        env = CustomCarRacingEnv(render_mode=render_mode, include_mask=True)
        return env, "custom", "CnnPolicy"
    if env_choice == "3":
        variant_key = coins_variant or "multi_full"
        variant = COINS_VARIANTS[variant_key]
        include_mask = variant["include_mask"] if include_mask_override is None else include_mask_override
        env = CoinsCarRacingEnv(
            render_mode=render_mode,
            observation_mode=variant["observation_mode"],
            include_mask=include_mask,
        )
        label = variant["label"]
        # Ajusta label se usamos mask desligada
        if variant["observation_mode"] == "image_and_coins" and not include_mask:
            label = "coins-full-3c"
        if variant["observation_mode"] == "image_only" and include_mask:
            label = "coins-cnn-4c"
        return env, label, variant["policy"]
    # fallback
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    return env, "original", "CnnPolicy"


def play_human(env):
    """Play the environment with keyboard arrows, supporting pyglet (original) and pygame renders."""
    # Detect backend: pyglet (original) vs pygame (coins/custom render override)
    viewer = getattr(env.unwrapped, "viewer", None)
    use_pyglet = viewer is not None and getattr(viewer, "window", None) is not None
    keys = None
    pygame = None

    if use_pyglet:
        try:
            from pyglet.window import key
            keys = key.KeyStateHandler()
            viewer.window.push_handlers(keys)
        except Exception as e:
            print(f"Cannot start human play (pyglet unavailable): {e}")
            return
    else:
        try:
            import pygame as _pg
            pygame = _pg
        except Exception as e:
            print(f"Cannot start human play (pygame unavailable): {e}")
            return

    obs, info = env.reset()
    print("Playing as human. Use arrow keys to drive, ESC to quit.")

    while True:
        env.render()

        steer = 0.0
        gas = 0.0
        brake = 0.0

        if use_pyglet:
            if keys[key.LEFT]:
                steer = -1.0
            if keys[key.RIGHT]:
                steer = 1.0
            if keys[key.UP]:
                gas = 1.0
            if keys[key.DOWN]:
                brake = 1.0
            if keys[key.ESCAPE]:
                break
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_LEFT]:
                steer = -1.0
            if pressed[pygame.K_RIGHT]:
                steer = 1.0
            if pressed[pygame.K_UP]:
                gas = 1.0
            if pressed[pygame.K_DOWN]:
                brake = 1.0
            if pressed[pygame.K_ESCAPE]:
                break

        action = np.array([steer, gas, brake], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


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
    start_iterations=0,
    start_total_timesteps=0,
    best_model_path=None,
    best_reward=None,
):
    """Run the training loop for ITERATIONS, saving checkpoints and updating best model."""
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

        # Evaluate after each iteration and update best if improved
        avg_reward = evaluate_agent(env, model, algorithm, run_name, iterations, total_timesteps)
        if avg_reward is not None and best_model_path is not None:
            current_best = best_reward
            if current_best is None or avg_reward > current_best:
                save_best_reward(algorithm, run_name, avg_reward)
                model.save(str(best_model_path))
                logger.info(f"New best model saved to: {best_model_path} (avg_reward={avg_reward})")
                best_reward = avg_reward
            else:
                logger.info(f"Best model kept (current best {current_best} >= new {avg_reward})")

    return model, iterations, total_timesteps, best_reward


# Evaluate model
def evaluate_agent(env, model, algorithm=None, run_name=None, iteration=None, total_timesteps=None):
    """Run deterministic evaluation episodes and log rewards."""
    logger.info("Starting evaluation")

    rewards = []
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

            if getattr(env, "render_mode", None) == "human":
                env.render()

        logger.info(f"Episode reward: {total_reward}")
        _log_eval_metrics(model, iteration, total_timesteps, ep + 1, total_reward)
        rewards.append(total_reward)

    avg_reward = sum(rewards) / len(rewards) if rewards else None
    if avg_reward is not None:
        logger.info(f"Average eval reward: {avg_reward}")

    return avg_reward


# MAIN
if __name__ == "__main__":
    logger.add("logs/training.log", rotation="1 MB")

    print("\nDo you want to see the episodes?")
    print("1 - Yes, show the simulation")
    print("2 - No, run without rendering (faster)")

    view_choice = input("\nYour choice: ").strip()
    render_mode = "human" if view_choice == "1" else None

    env_choice = select_environment_choice()

    # SELECT ALGORITHM
    algo_list = get_algorithm_list()
    algo_list.append("Human")

    print("\nSelect training algorithm:")
    for i, name in enumerate(algo_list, start=1):
        print(f"{i} - {name}")

    algo_choice = int(input("\nAlgorithm number: "))
    algorithm = algo_list[algo_choice - 1]

    logger.info(f"Selected algorithm: {algorithm}")
    algo_kwargs = HYPERPARAMS.get(algorithm, {})

    algo_dir = prepare_algo_folder(algorithm)
    # Human mode: play with arrows, no training
    if algorithm == "Human":
        # Force human render mode
        render_mode = "human"
        env, env_label, policy = build_environment(env_choice, render_mode)
        play_human(env)
        sys.exit(0)

    action = select_action_for_run(algorithm)
    prune_missing_dirs()
    AlgoClass = ALGORITHMS[algorithm]

    model = None
    run_name = None
    iterations = 0
    total_timesteps = 0
    best_reward = None
    run_dir = None
    best_model_path = None
    latest_model_path = None
    coins_variant = None
    include_mask_override = None

    # If we already know the run (continue/evaluate/override), discover model and adjust env/policy
    if action in {"2", "3", "4"}:
        run_name = select_run_name(algorithm)
        if run_name is None:
            print("No valid run selected. Exiting.")
            sys.exit(0)
        run_dir = Path(algo_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        latest_model_path = run_dir / LATEST_MODEL_NAME
        best_model_path = run_dir / BEST_MODEL_NAME
        has_model = latest_model_path.exists() or best_model_path.exists()
        best_reward = load_best_reward(algorithm, run_name)
        chosen_model_path = best_model_path if best_model_path and best_model_path.exists() else latest_model_path
        if env_choice == "3":
            if has_model and chosen_model_path and chosen_model_path.exists():
                coins_variant, include_mask_override = infer_coins_variant_from_model(str(chosen_model_path))
            else:
                coins_variant = select_coins_variant_for_new()
        elif not has_model and action == "3":
            print("No saved model found for evaluation. Exiting.")
            sys.exit(0)
    else:
        has_model = False

    # Build environment and select policy based on choices
    if env_choice == "3":
        if coins_variant is None:
            coins_variant = select_coins_variant_for_new()
        env, env_label, policy = build_environment(env_choice, render_mode, coins_variant, include_mask_override)
    else:
        env, env_label, policy = build_environment(env_choice, render_mode)

    see_space(env)
    logger.info(f"Selected policy: {policy}")

    if action == "1":
        run_name = input("\nEnter a name for this run: ").strip()
        run_dir = Path(algo_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = Path(run_dir) / BEST_MODEL_NAME
        save_training_stats(algorithm, run_name, iterations, total_timesteps)
        best_reward = None

        model = AlgoClass(
            policy,
            env,
            verbose=1,
            tensorboard_log=str(Path(LOGS_DIR) / f"{algorithm}_{run_name}"),
            device=DEVICE,
            **algo_kwargs,
        )
        _set_sb3_logger(model, algorithm, run_name, env_label)
        _log_run_config(model, env, algorithm, run_name, policy, env_label)

    elif action in {"2", "3", "4"}:
        if action == "4":
            confirm = input(f"Type YES to override run '{run_name}': ").strip()
            if confirm != "YES":
                print("Override cancelled. Exiting.")
                sys.exit(0)
            logger.warning(f"Overriding existing run '{run_name}' for {algorithm}. Resetting model and counters.")
            save_training_stats(algorithm, run_name, iterations, total_timesteps)
            model = AlgoClass(
                policy,
                env,
                verbose=1,
                tensorboard_log=str(Path(LOGS_DIR) / f"{algorithm}_{run_name}"),
                device=DEVICE,
                **algo_kwargs,
            )
            _set_sb3_logger(model, algorithm, run_name, env_label)
            _log_run_config(model, env, algorithm, run_name, policy, env_label)
            iterations = 0
            total_timesteps = 0
        elif not has_model:
            logger.warning(f"No saved model found for run '{run_name}'. Starting from scratch.")
            iterations = 0
            total_timesteps = 0
            best_reward = None
            save_training_stats(algorithm, run_name, iterations, total_timesteps)
            model = AlgoClass(
                policy,
                env,
                verbose=1,
                tensorboard_log=str(Path(LOGS_DIR) / f"{algorithm}_{run_name}"),
                device=DEVICE,
                **algo_kwargs,
            )
            _set_sb3_logger(model, algorithm, run_name, env_label)
            _log_run_config(model, env, algorithm, run_name, policy, env_label)
            # if action was evaluate-only, fall into training flow
            if action == "3":
                action = "2"
        else:
            chosen_model_path = best_model_path if best_model_path.exists() else latest_model_path
            logger.info(f"Loading existing {algorithm} model from {chosen_model_path}")
            model = AlgoClass.load(str(chosen_model_path), env=env, device=DEVICE)
            _set_sb3_logger(model, algorithm, run_name, env_label)
            _log_run_config(model, env, algorithm, run_name, policy, env_label)
            iterations, total_timesteps = load_training_stats(algorithm, run_name)
            logger.info(f"Loaded training progress: iterations={iterations}, total_timesteps={total_timesteps}")
    else:
        print("Invalid action. Exiting.")
        sys.exit(0)

    if action in {"1", "2", "4"}:
        model, iterations, total_timesteps, best_reward = train_agent(
            env,
            model,
            algorithm,
            run_dir,
            run_name,
            start_iterations=iterations,
            start_total_timesteps=total_timesteps,
            best_model_path=best_model_path,
            best_reward=best_reward,
        )
    elif action == "3":
        avg_reward = evaluate_agent(env, model, algorithm, run_name, iterations, total_timesteps)
        if avg_reward is not None and "best_model_path" in locals():
            best_reward = load_best_reward(algorithm, run_name)
            if best_reward is None or avg_reward > best_reward:
                save_best_reward(algorithm, run_name, avg_reward)
                model.save(str(best_model_path))
                logger.info(f"New best model saved to: {best_model_path} (avg_reward={avg_reward})")

    env.close()
