import numpy as np
from loguru import logger

try:
    import pygame
except Exception:  # pragma: no cover - fallback when pygame is missing
    pygame = None

try:
    from pyglet.window import key as pyglet_key
except Exception:  # pragma: no cover - fallback when pyglet is missing
    pyglet_key = None

from environment.coins_car_racing_env import CoinsCarRacingEnv

# Constants for manual control
STEER_SCALE = 1.0
GAS_SCALE = 1.0
BRAKE_SCALE = 0.8


def _build_action(left: bool, right: bool, gas: bool, brake: bool) -> np.ndarray:
    steer = 0.0
    if left:
        steer -= STEER_SCALE
    if right:
        steer += STEER_SCALE

    # action = [steer, gas, brake]
    return np.array(
        [
            np.clip(steer, -1.0, 1.0),
            GAS_SCALE if gas else 0.0,
            BRAKE_SCALE if brake else 0.0,
        ],
        dtype=np.float32,
    )


def _control_with_pygame(env) -> None:
    if pygame is None:
        logger.error("pygame não está disponível para capturar o teclado.")
        return

    logger.info("Modo humano (pygame): setas para controlar, R reseta, ESC fecha.")
    obs, info = env.reset()
    env.render()

    running = True
    while running:
        reset_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_requested = True

        keys = pygame.key.get_pressed()
        action = _build_action(
            left=keys[pygame.K_LEFT],
            right=keys[pygame.K_RIGHT],
            gas=keys[pygame.K_UP],
            brake=keys[pygame.K_DOWN],
        )

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated or reset_requested:
            obs, info = env.reset()
            env.render()

    logger.info("Encerrando controle humano (pygame).")


def _control_with_pyglet(env) -> None:
    if pyglet_key is None:
        logger.error("pyglet não está disponível para capturar o teclado. Instale com `pip install pyglet`.")
        return

    logger.info("Modo humano (pyglet): setas para controlar, R reseta, ESC fecha.")

    control_state = {
        "left": False,
        "right": False,
        "gas": False,
        "brake": False,
        "reset": False,
        "exit": False,
    }

    def key_press(k, mod):
        if k == pyglet_key.LEFT:
            control_state["left"] = True
        elif k == pyglet_key.RIGHT:
            control_state["right"] = True
        elif k == pyglet_key.UP:
            control_state["gas"] = True
        elif k == pyglet_key.DOWN:
            control_state["brake"] = True
        elif k == pyglet_key.R:
            control_state["reset"] = True
        elif k in (pyglet_key.ESCAPE, pyglet_key.Q):
            control_state["exit"] = True

    def key_release(k, mod):
        if k == pyglet_key.LEFT:
            control_state["left"] = False
        elif k == pyglet_key.RIGHT:
            control_state["right"] = False
        elif k == pyglet_key.UP:
            control_state["gas"] = False
        elif k == pyglet_key.DOWN:
            control_state["brake"] = False

    obs, info = env.reset()
    env.render()

    base_env = env.unwrapped
    viewer = getattr(base_env, "viewer", None)
    if viewer is None or getattr(viewer, "window", None) is None:
        env.render()
        viewer = getattr(base_env, "viewer", None)

    if viewer is None or getattr(viewer, "window", None) is None:
        logger.error("Não foi possível inicializar a janela do ambiente para controle humano.")
        return

    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release

    while not control_state["exit"]:
        action = _build_action(
            left=control_state["left"],
            right=control_state["right"],
            gas=control_state["gas"],
            brake=control_state["brake"],
        )

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated or control_state["reset"]:
            control_state["reset"] = False
            obs, info = env.reset()
            env.render()

    logger.info("Encerrando controle humano (pyglet).")


def run_human_control(env) -> None:
    """
    Run an interactive control loop allowing the user to drive the car with the arrow keys.
    Prefers pygame (CarRacing usa pygame por padrão); cai para pyglet apenas se pygame não estiver disponível.
    """
    base_env = env.unwrapped
    if getattr(base_env, "render_mode", None) != "human":
        logger.error("Habilite render_mode='human' para usar o controle humano.")
        return

    if pygame is not None:
        _control_with_pygame(env)
    else:
        _control_with_pyglet(env)
