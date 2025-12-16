import time
import sys
from pathlib import Path
import csv
from typing import Any
from numbers import Real

import gymnasium as gym
from gymnasium import spaces
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

    lr = getattr(model, "learning_rate", None)
    _rec("config/learning_rate", lr if isinstance(lr, Real) else str(lr))
    for key in ["n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"]:
        val = getattr(model, key, None)
        if val is not None:
            if isinstance(val, Real):
                _rec(f"config/{key}", val)
            else:
                _rec(f"config/{key}", str(val))

    # Config especÇ¦fica do ambiente
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


def select_environment(render_mode: str):
    print("\nSelect environment type:")
    print("1 - Original CarRacing (no custom reward, no coins)")
    print("2 - Custom CarRacing (off-track penalty only)")
    print("3 - Custom CarRacing with coins (off-track penalty + coin reward)")

    choice = input("\nYour choice: ").strip()

    if choice == "1":
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env_label = "original"
    elif choice == "2":
        env = CustomCarRacingEnv(render_mode=render_mode)
        env_label = "custom"
    elif choice == "3":
        env = CoinsCarRacingEnv(render_mode=render_mode)
        env_label = "coins"
    else:
        print("Invalid choice, defaulting to Original CarRacing.")
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env_label = "original"

    return env, env_label


def select_policy(env):
    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Dict):
        return "MultiInputPolicy"
    shape = getattr(obs_space, "shape", ())
    if len(shape) == 3:
        return "CnnPolicy"
    return "MlpPolicy"


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

        # Avalia a cada iteraÇõÇœ e atualiza best se melhorar
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

    env, env_label = select_environment(render_mode)
    see_space(env)
    policy = select_policy(env)
    logger.info(f"Selected policy: {policy}")

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
    prune_missing_dirs()
    AlgoClass = ALGORITHMS[algorithm]

    model = None
    run_name = None
    iterations = 0
    total_timesteps = 0
    best_reward = None

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
        )
        _set_sb3_logger(model, algorithm, run_name, env_label)
        _log_run_config(model, env, algorithm, run_name, policy, env_label)

    elif action in {"2", "3", "4"}:
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
            )
            _set_sb3_logger(model, algorithm, run_name, env_label)
            _log_run_config(model, env, algorithm, run_name, policy, env_label)
            # se aÇ¦o era apenas avaliar, cairÇ­ no fluxo de treino
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
