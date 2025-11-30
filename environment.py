import gymnasium as gym
from gymnasium.envs.registration import register

# Registrar o ambiente customizado
try:
    register(
        id='CustomCarRacing-v0',
        entry_point='environment.custom_car_racing_env:CustomCarRacingEnv',
        max_episode_steps=1000,
        reward_threshold=900,
    )
except:
    pass 


def see_space(env):
    print("Environment Spaces:")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Sample action: {env.action_space.sample()}")
    print(f"Sample observation: {env.observation_space.sample()}")


def start_environment(env):
    # Reinicia o ambiente
    observation, info = env.reset()
    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    # Loop principal — AÇÕES ALEATÓRIAS
    while not episode_over:
        action = env.action_space.sample()   # << AQUI É O RANDOM
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

        print(f"Step reward: {reward}, Total reward: {total_reward}")

    print(f"Episode finished with total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    # Criar o ambiente
    env = gym.make("CustomCarRacing-v0", render_mode="human")

    see_space(env)
    start_environment(env)
