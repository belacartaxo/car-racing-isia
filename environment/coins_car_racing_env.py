import math
import numpy as np
import pygame
import pygame.gfxdraw

from gymnasium.envs.box2d.car_racing import SCALE, WINDOW_W, WINDOW_H, ZOOM
from environment.custom_car_racing_env import CustomCarRacingEnv


class CoinsCarRacingEnv(CustomCarRacingEnv):
    def __init__(self, n_coins=10, coin_reward=10.0, **kwargs):
        super().__init__(**kwargs)

        self.n_coins = n_coins
        self.coin_reward = coin_reward

        self.coins = []
        self.coin_track_indices = []
        self.coins_collected = []
        self.coins_removed = []  # used to hide after being passed
        self.coins_collected_total = 0  # accumulated across the session/laps
        self.coins_collected_episode = 0  # resets each episode

        self.track_points = []
        self.max_track_progress = 0
        self.progress_buffer = 2          # margin to remove coins already passed
        self.progress_activation_dist = 10.0  # only update progress when near the track
        self.prev_track_idx = 0

        self.coin_radius = 1.2          # visual size of the coin
        self.coin_pickup_radius = 3.5   # pickup radius (more lenient)

    # -------------------------------
    # RESET DO AMBIENTE
    # -------------------------------
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._reset_coins()
        return obs, info

    # -------------------------------
    # POSICIONAMENTO DAS MOEDAS
    # -------------------------------
    def _reset_coins(self):
        self.coins = []
        self.coin_track_indices = []
        self.coins_collected = []
        self.coins_removed = []
        self.coins_collected_episode = 0

        self.track_points = []
        self.max_track_progress = 0
        self.prev_track_idx = 0

        if hasattr(self, "track") and self.track is not None and len(self.track) > 0:
            # keep full track for progress estimation
            self.track_points = [(x, y) for _, _, x, y in self.track]

            track_len = len(self.track_points)
            num_coins = max(1, min(self.n_coins, track_len))

            # spread limited coins across the track
            coin_indices = np.linspace(0, track_len - 1, num_coins, dtype=int)

            for idx in coin_indices:
                x, y = self.track_points[idx]
                self.coins.append((x, y))
                self.coin_track_indices.append(idx)
                self.coins_collected.append(False)
                self.coins_removed.append(False)

    def _update_progress(self, car_x, car_y):
        if not self.track_points:
            return

        pts = np.array(self.track_points)
        dists = np.hypot(pts[:, 0] - car_x, pts[:, 1] - car_y)
        idx = int(np.argmin(dists))

        # Only count progress when the car is close to the track
        if dists[idx] > self.progress_activation_dist:
            return

        # Detect lap completion: if we were far along and jump back to the start, respawn coins
        track_len = len(self.track_points)
        crossed_lap = (
            track_len > 0
            and self.prev_track_idx > track_len * 0.6
            and idx < track_len * 0.1
        )
        if crossed_lap:
            self.coins_collected = [False] * len(self.coins)
            self.coins_removed = [False] * len(self.coins)
            self.max_track_progress = idx

        if idx > self.max_track_progress:
            self.max_track_progress = idx

        self.prev_track_idx = idx
        self.current_track_idx = idx

    # -------------------------------
    # STEP COM RECOMPENSA EXTRA
    # -------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.car is None or not self.coins:
            return obs, reward, terminated, truncated, info

        car_x, car_y = self.car.hull.position

        self._update_progress(car_x, car_y)

        bonus = 0.0

        for i, (cx, cy) in enumerate(self.coins):
            if self.coins_collected[i] or self.coins_removed[i]:
                continue

            dist = np.sqrt((car_x - cx) ** 2 + (car_y - cy) ** 2)

            if dist < self.coin_pickup_radius:
                self.coins_collected[i] = True
                bonus += self.coin_reward
                self.coins_collected_total += 1
                self.coins_collected_episode += 1
            else:
                coin_track_idx = self.coin_track_indices[i]
                if coin_track_idx < self.max_track_progress - self.progress_buffer:
                    self.coins_removed[i] = True

        reward += bonus

        info["coin_bonus"] = bonus
        info["coins_collected"] = sum(self.coins_collected)
        info["coins_collected_total"] = self.coins_collected_total
        info["coins_collected_episode"] = self.coins_collected_episode
        info["coins_removed"] = sum(self.coins_removed)

        return obs, reward, terminated, truncated, info

    # -------------------------------
    # DESENHAR MOEDAS NA TELA
    # -------------------------------
    def _render_coins(self, zoom, translation, angle):
        for i, (cx, cy) in enumerate(self.coins):
            if self.coins_collected[i] or self.coins_removed[i]:
                continue

            pos = pygame.math.Vector2((cx, cy)).rotate_rad(angle)

            px = pos[0] * zoom + translation[0]
            py = pos[1] * zoom + translation[1]

            # The surface was already flipped in the base render; adjust Y to anchor to the track
            py = WINDOW_H - py

            pulse = 1.0 + 0.05 * math.sin(self.t * 4.0)
            radius_px = max(2, int(self.coin_radius * zoom * pulse))

            if px < -50 or px > WINDOW_W + 50 or py < -50 or py > WINDOW_H + 50:
                continue

            hud_height = (5 * WINDOW_H) / 40.0  # same HUD band as base game
            hud_y = WINDOW_H - hud_height
            if py >= hud_y:
                continue

            # Gold palette and layers
            shadow_color = (40, 40, 40)
            outer_color = (200, 160, 20)
            mid_color = (238, 196, 60)
            inner_color = (255, 215, 0)
            highlight_color = (255, 248, 180)

            px_i = int(px)
            py_i = int(py)

            # Light shadow (no alpha to keep surface compatible)
            pygame.gfxdraw.filled_circle(self.surf, px_i + 2, py_i + 3, radius_px + 2, shadow_color)

            # Body with simple gradient (concentric circles)
            pygame.gfxdraw.filled_circle(self.surf, px_i, py_i, radius_px, outer_color)
            pygame.gfxdraw.aacircle(self.surf, px_i, py_i, radius_px, outer_color)

            mid_r = max(1, int(radius_px * 0.75))
            pygame.gfxdraw.filled_circle(self.surf, px_i, py_i, mid_r, mid_color)

            inner_r = max(1, int(radius_px * 0.45))
            pygame.gfxdraw.filled_circle(self.surf, px_i, py_i, inner_r, inner_color)

            # Highlight on upper-left corner
            hl_r = max(1, int(radius_px * 0.3))
            pygame.gfxdraw.filled_circle(self.surf, px_i - radius_px // 3, py_i - radius_px // 3, hl_r, highlight_color)

    def _render_coin_counter(self):
        total = len(self.coins)
        collected = sum(self.coins_collected)
        removed = sum(self.coins_removed)
        visible = total - collected - removed
        collected_total = getattr(self, "coins_collected_total", collected)
        collected_episode = getattr(self, "coins_collected_episode", collected)

        pygame.font.init()
        font = pygame.font.Font(pygame.font.get_default_font(), 24)
        small = pygame.font.Font(pygame.font.get_default_font(), 16)

        title_text = f"Coins total: {collected_total}"
        sub_text = f"Episode: {collected_episode}/{total} | On track: {visible}"

        title_surf = font.render(title_text, True, (255, 215, 0))
        sub_surf = small.render(sub_text, True, (220, 220, 220))

        padding = 12
        spacing = 4
        width = max(title_surf.get_width(), sub_surf.get_width()) + padding * 3
        height = title_surf.get_height() + sub_surf.get_height() + spacing + padding * 3

        hud_height = (5 * WINDOW_H) / 40.0
        bg_rect = pygame.Rect(0, 0, width, height)
        bg_rect.bottom = WINDOW_H - padding
        bg_rect.right = WINDOW_W - padding

        # box with border and slight glow
        pygame.draw.rect(self.surf, (25, 25, 25), bg_rect, border_radius=10)
        pygame.draw.rect(self.surf, (90, 90, 90), bg_rect, width=1, border_radius=10)

        # simple coin icon next to the title
        icon_r = 8
        icon_x = bg_rect.left + padding
        icon_y = bg_rect.top + padding + title_surf.get_height() // 2
        pygame.gfxdraw.filled_circle(self.surf, icon_x, icon_y, icon_r + 1, (120, 90, 10))
        pygame.gfxdraw.filled_circle(self.surf, icon_x, icon_y, icon_r, (255, 215, 0))
        pygame.gfxdraw.aacircle(self.surf, icon_x, icon_y, icon_r, (255, 230, 120))

        title_pos = (icon_x + icon_r + 8, bg_rect.top + padding)
        sub_pos = (bg_rect.left + padding, bg_rect.top + padding + title_surf.get_height() + spacing)

        self.surf.blit(title_surf, title_pos)
        self.surf.blit(sub_surf, sub_pos)

    # -------------------------------
    # RENDER CORRETO (ESTE É O PONTO-CHAVE)
    # -------------------------------
    def _render(self, mode):
        if mode != "human":
            return super()._render(mode)

        # Replicando o _render base para inserir moedas/HUD antes do flip
        pygame.font.init()
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        angle = -self.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        translation = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, translation, angle)
        self.car.draw(
            self.surf,
            zoom,
            translation,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # HUD base do jogo
        self._render_indicators(WINDOW_W, WINDOW_H)
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        # Overlay de moedas
        self._render_coins(zoom, translation, angle)
        self._render_coin_counter()

        # Flip final (único) para evitar piscadas
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        self.screen.fill(0)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()

        return self.isopen
