import os
import gymnasium as gym
from loguru import logger
import time

from utils.config import (
    ENV_NAME, TOTAL_TIMESTEPS, ITERATIONS, EVAL_EPISODES,
    MODELS_DIR, LOGS_DIR, LATEST_MODEL_NAME, DEVICE
)

from utils.algorithms import ALGORITHMS, get_algorithm_list
from utils.training_counter import load_training_count, save_training_count


#   Inspect environment
def see_space(env):
    logger.info("Inspecting environment spaces")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Sample action: {env.action_space.sample()}")
    logger.info(f"Sample observation shape: {env.observation_space.sample().shape}")


#   Prepare folders
def prepare_algo_folder(algorithm):
    algo_dir = os.path.join(MODELS_DIR, algorithm)
    os.makedirs(algo_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    return algo_dir


#   Load or create model
def load_or_create_model(env, algorithm, algo_dir):
    algo_class = ALGORITHMS[algorithm]
    model_path = os.path.join(algo_dir, LATEST_MODEL_NAME)

    if os.path.exists(model_path):
        logger.info(f"Loading existing {algorithm} model from {model_path}")
        model = algo_class.load(model_path, env=env, device=DEVICE)
        return model, True

    logger.info(f"No saved model found for {algorithm}. Creating new model")
    model = algo_class(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        device=DEVICE,
    )
    return model, False


#   Train model
def train_agent(env, model, algorithm, algo_dir):
    training_count = load_training_count(algorithm)

    for i in range(1, ITERATIONS + 1):
        training_count += 1

        logger.info(f"Global training iteration for {algorithm}: {training_count}")
        logger.info(f"Running local iteration {i}/{ITERATIONS} with {TOTAL_TIMESTEPS} timesteps")

        iteration_start = time.time()  # start timing

        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)

        iteration_end = time.time()  # end timing
        duration = iteration_end - iteration_start

        minutes = duration / 60
        logger.info(f"Iteration time: {duration:.2f} seconds ({minutes:.2f} minutes)")

        latest_path = os.path.join(algo_dir, LATEST_MODEL_NAME)
        model.save(latest_path)

        logger.info(f"Saved updated model to: {latest_path}")

    save_training_count(algorithm, training_count)
    return model



#   Evaluate model
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


#   MAIN
if __name__ == "__main__":
    logger.add("logs/training.log", rotation="1 MB")  # Optional: file logging

    print("\nDo you want to see the episodes?")
    print("1 - Yes, show the simulation")
    print("2 - No, run without rendering (faster)")

    view_choice = input("\nYour choice: ")
    render_mode = "human" if view_choice == "1" else None

    env = gym.make(ENV_NAME, render_mode=render_mode)
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
    model_path = os.path.join(algo_dir, LATEST_MODEL_NAME)
    exists = os.path.exists(model_path)

    current_count = load_training_count(algorithm)
    logger.info(f"Training iterations completed so far for {algorithm}: {current_count}")


    # MODEL EXISTS
    if exists:
        logger.info(f"A saved {algorithm} model already exists.")

        print("\nChoose action:")
        print("1 - Continue training")
        print("2 - Evaluate model")
        print("3 - Start from zero (overwrite model)")

        action = input("\nYour choice: ")


        if action == "1":
            model, _ = load_or_create_model(env, algorithm, algo_dir)
            model = train_agent(env, model, algorithm, algo_dir)
            evaluate_agent(env, model)

        elif action == "2":
            model, _ = load_or_create_model(env, algorithm, algo_dir)
            evaluate_agent(env, model)

        elif action == "3":
            logger.warning("Overwriting old model and starting from zero")
            AlgoClass = ALGORITHMS[algorithm]
            model = AlgoClass(
                "CnnPolicy",
                env,
                verbose=1,
                tensorboard_log=LOGS_DIR,
                device=DEVICE,
            )
            model = train_agent(env, model, algorithm, algo_dir)
            evaluate_agent(env, model)

        else:
            logger.error("Invalid option. Exiting.")

    # NO MODEL EXISTS
    else:
        logger.info(f"No existing {algorithm} model found. Training new model.")

        AlgoClass = ALGORITHMS[algorithm]
        model = AlgoClass(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            device=DEVICE,
        )
        model = train_agent(env, model, algorithm, algo_dir)
        evaluate_agent(env, model)

    env.close()