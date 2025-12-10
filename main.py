import os
import time
import gymnasium as gym
import numpy as np
import pygame
import csv
from datetime import datetime
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_car_racing_env import CustomCarRacingEnv
from environment.coins_car_racing_env import CoinsCarRacingEnv

from utils.config import (
    ENV_NAME, TOTAL_TIMESTEPS, ITERATIONS, EVAL_EPISODES,
    MODELS_DIR, LOGS_DIR,
    ORIGINAL_LATEST_MODEL_NAME, ORIGINAL_BEST_MODEL_NAME,
    COINS_LATEST_MODEL_NAME, COINS_BEST_MODEL_NAME,
    DEVICE
)

from utils.algorithms import ALGORITHMS, get_algorithm_list
from utils.training_counter import (
    load_training_count,
    load_training_timesteps,
    save_training_progress,
    load_best_score,
    save_best_score
)

# ===========================
# UTILIDADES
# ===========================

def see_space(env):
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")


def prepare_algo_folder(algorithm, env_type):
    algo_dir = os.path.join(MODELS_DIR, env_type, algorithm)
    os.makedirs(algo_dir, exist_ok=True)

    logs_dir = os.path.join(LOGS_DIR, env_type)
    os.makedirs(logs_dir, exist_ok=True)

    return algo_dir, logs_dir


# ===========================
# AVALIAÇÃO AUTOMÁTICA
# ===========================

def evaluate_agent(env, model):
    total_rewards = []
    total_coins = []

    for ep in range(EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        coins_ep = 0
        steps = 0
        ep_start = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            coins_ep = info.get("coins_collected_episode", 0)
            done = terminated or truncated
            steps += 1

        total_rewards.append(episode_reward)
        total_coins.append(coins_ep)
        ep_end = time.time()
        logger.info(
            f"[EVAL-ep {ep + 1}/{EVAL_EPISODES}] reward={episode_reward:.2f} | "
            f"moedas(ep)={coins_ep} | steps={steps} | tempo={ep_end - ep_start:.1f}s"
        )

    mean_reward = float(np.mean(total_rewards))
    mean_coins_ep = float(np.mean(total_coins))

    return mean_reward, mean_coins_ep


class TrainingLoggerCallback(BaseCallback):
    def __init__(
        self,
        run_id,
        algo,
        env_type,
        iteration,
        csv_path,
        log_every_episodes=1,
        log_every_steps=5000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.run_id = run_id
        self.algo = algo
        self.env_type = env_type
        self.iteration = iteration
        self.csv_path = csv_path
        self.log_every_episodes = log_every_episodes
        self.log_every_steps = log_every_steps
        self.episode_counter = 0
        self.ep_start_time = time.time()
        self.last_step_log = 0
        self.last_rollout_eps = 0

    def _write_csv(self, row):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "run_id",
                    "algo",
                    "env_type",
                    "iteration",
                    "type",
                    "episode",
                    "reward",
                    "len",
                    "coins_ep",
                    "steps_total",
                    "ep_time_sec",
                ],
                delimiter=";",  # Excel pt-BR usa ; como separador
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            info = self.locals.get("info")
            infos = [info] if info is not None else []

        # Log periódico por step, mesmo sem fechar episódio
        if self.num_timesteps - self.last_step_log >= self.log_every_steps:
            coins_ep = 0
            for info in infos:
                if info:
                    coins_ep = info.get("coins_collected_episode", coins_ep)
            logger.info(
                f"[TRAIN step] t={self.num_timesteps} | ep={self.episode_counter} | moedas(ep)={coins_ep}"
            )
            self.last_step_log = self.num_timesteps

        for info in infos:
            if not info:
                continue
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            self.episode_counter += 1
            if self.episode_counter % self.log_every_episodes == 0:
                coins_ep = info.get("coins_collected_episode", 0)
                now = time.time()
                ep_time = now - self.ep_start_time
                self.ep_start_time = now
                rew = ep_info.get("r", 0)
                length = ep_info.get("l", 0)
                logger.info(
                    "[TRAIN ep] "
                    f"ep={self.episode_counter} | "
                    f"rew={rew:.2f} | "
                    f"len={length} | "
                    f"moedas={coins_ep} | "
                    f"steps_totais={self.num_timesteps} | "
                    f"tempo={ep_time:.1f}s"
                )
                self._write_csv(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "run_id": self.run_id,
                        "algo": self.algo,
                        "env_type": self.env_type,
                        "iteration": self.iteration,
                        "type": "episode",
                        "episode": self.episode_counter,
                        "reward": rew,
                        "len": length,
                        "coins_ep": coins_ep,
                        "steps_total": self.num_timesteps,
                        "ep_time_sec": round(ep_time, 2),
                    }
                )
        return True

    def _on_rollout_end(self) -> None:
        # Garantir log ao fim de cada rollout, mesmo sem episódio fechado no intervalo
        buffer = list(self.model.ep_info_buffer)
        new_eps = len(buffer) - self.last_rollout_eps
        if new_eps > 0:
            recent = buffer[-new_eps:]
        else:
            recent = buffer

        if recent:
            rews = [ep.get("r", 0) for ep in recent]
            lens = [ep.get("l", 0) for ep in recent]
            logger.info(
                f"[TRAIN rollout] eps={len(buffer)} (+{max(new_eps,0)}) | "
                f"rew_mean={np.mean(rews):.2f} | len_mean={np.mean(lens):.0f} | "
                f"steps_totais={self.num_timesteps}"
            )
            self._write_csv(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "run_id": self.run_id,
                    "algo": self.algo,
                    "env_type": self.env_type,
                    "iteration": self.iteration,
                    "type": "rollout",
                    "episode": len(buffer),
                    "reward": float(np.mean(rews)),
                    "len": float(np.mean(lens)),
                    "coins_ep": None,
                    "steps_total": self.num_timesteps,
                    "ep_time_sec": None,
                }
            )
        else:
            logger.info(f"[TRAIN rollout] steps_totais={self.num_timesteps}")

        self.last_rollout_eps = len(buffer)


# ===========================
# TREINO COM LAST x BEST
# ===========================

def train_agent(env, model, algorithm, algo_dir, env_type,
                latest_name, best_name):

    training_count = load_training_count(algorithm)
    base_timesteps = load_training_timesteps(algorithm)
    best_score = load_best_score(algorithm, env_type)

    for i in range(ITERATIONS):
        training_count += 1
        prev_ts = base_timesteps + (i * TOTAL_TIMESTEPS)
        after_ts = prev_ts + TOTAL_TIMESTEPS
        iter_start = time.time()
        logger.info(
            f"[{algorithm}] Iteração {i + 1}/{ITERATIONS} | steps {prev_ts} -> {after_ts}"
        )

        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        cb = TrainingLoggerCallback(
            run_id=run_id,
            algo=algorithm,
            env_type=env_type,
            iteration=i + 1,
            csv_path=os.path.join(logs_dir, "training_metrics.csv"),
            log_every_episodes=1,
            log_every_steps=1000,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, callback=cb)
        iter_end = time.time()
        logger.info(f"[{algorithm}] Iteração {i + 1} concluída em {iter_end - iter_start:.1f}s")

        latest_path = os.path.join(algo_dir, latest_name)
        best_path = os.path.join(algo_dir, best_name)

        model.save(latest_path)
        save_training_progress(algorithm, training_count)

        mean_reward, mean_coins_ep = evaluate_agent(env, model)

        # CRITÉRIO C → reward + moedas do episódio (peso 3)
        score_final = mean_reward + mean_coins_ep * 3.0

        logger.info(
            f"[EVAL] reward={mean_reward:.2f} | moedas(ep)={mean_coins_ep:.1f} | score={score_final:.2f}"
        )

        if score_final > best_score:
            best_score = score_final
            save_best_score(algorithm, env_type, best_score)
            model.save(best_path)
            logger.info(f"[MODEL] ✅ MODELO MELHOR SALVO: {best_name}")
        else:
            logger.info(
                f"[MODEL] ❌ MODELO DESCARTADO | atual={score_final:.2f} < best={best_score:.2f}"
            )

    return model


# ===========================
# MODO MANUAL (TECLADO)
# ===========================

def run_manual(env):
    print("Modo MANUAL ativo.")
    print("Setas ← ↑ → para dirigir | ESC para sair")

    obs, info = env.reset()
    pygame.init()

    clock = pygame.time.Clock()

    while True:
        action = np.array([0.0, 0.0, 0.0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        if keys[pygame.K_RIGHT]:
            action[0] = 1.0
        if keys[pygame.K_UP]:
            action[1] = 1.0
        if keys[pygame.K_DOWN]:
            action[2] = 0.8

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

        clock.tick(60)


# ===========================
# ========== MAIN ===========
# ===========================

if __name__ == "__main__":
    logger.add("logs/training.log", rotation="1 MB")

    print("\nDo you want to see the episodes?")
    print("1 - Yes")
    print("2 - No")

    render_mode = "human" if input("Choice: ") == "1" else None

    print("\nChoose the environment:")
    print("1 - Original")
    print("2 - Coins")
    env_choice = input("Choice: ")

    if env_choice == "2":
        env_type = "PPO_coins"
        env_ctor = CoinsCarRacingEnv
        latest_name = COINS_LATEST_MODEL_NAME
        best_name = COINS_BEST_MODEL_NAME
    else:
        env_type = "PPO_original"
        env_ctor = CustomCarRacingEnv
        latest_name = ORIGINAL_LATEST_MODEL_NAME
        best_name = ORIGINAL_BEST_MODEL_NAME

    env = env_ctor(render_mode=render_mode)
    see_space(env)

    print("\nChoose the mode:")
    print("1 - Train agent (RL)")
    print("2 - Run trained model")
    print("3 - Manual mode (keyboard)")

    mode = input("Choice: ")

    # =======================
    # MODO MANUAL
    # =======================

    if mode == "3":
        run_manual(env)
        env.close()
        exit()

    # Wrap em Monitor para ter métricas de episódio nos callbacks
    env = Monitor(env)

    # =======================
    # PPO
    # =======================

    algo_list = get_algorithm_list()
    for i, name in enumerate(algo_list, start=1):
        print(f"{i} - {name}")

    algorithm = algo_list[int(input("Algorithm: ")) - 1]
    algo_dir, logs_dir = prepare_algo_folder(algorithm, env_type)

    latest_path = os.path.join(algo_dir, latest_name)
    best_path = os.path.join(algo_dir, best_name)

    if os.path.exists(latest_path):
        logger.info("Modelo encontrado. Continuando treino.")
        model = ALGORITHMS[algorithm].load(latest_path, env=env, device=DEVICE)
    else:
        logger.info("Criando modelo novo.")
        model = ALGORITHMS[algorithm](
            "CnnPolicy", env, verbose=1, tensorboard_log=logs_dir, device=DEVICE
        )

    # =======================
    # TREINAR OU RODAR
    # =======================

    if mode == "1":
        train_agent(env, model, algorithm, algo_dir, env_type, latest_name, best_name)

    elif mode == "2":
        print("Rodando modelo treinado...")
        obs, info = env.reset()

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "human":
                env.render()

            if terminated or truncated:
                obs, info = env.reset()

    env.close()
