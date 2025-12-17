"""CarRacing variant with collectible coins and optional observation modes."""

import math
import numpy as np
import pygame
import pygame.gfxdraw

from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import SCALE, WINDOW_W, WINDOW_H, ZOOM, TRACK_RAD
from environment.custom_car_racing_env import CustomCarRacingEnv


class CoinsCarRacingEnv(CustomCarRacingEnv):
    """CustomCarRacingEnv with coin placement, rewards and multi-modal observations."""

    def __init__(self, n_coins=10, coin_reward=100.0, observation_mode: str = "image_and_coins", include_mask: bool = True, **kwargs):
        """
        observation_mode:
            - "image_only": image only (CNN)
            - "coins_only": coins feature vector only (Dict with key 'coins')
            - "image_and_coins": Dict with image + coins vector (default)
        """
        super().__init__(include_mask=include_mask, **kwargs)

        self.n_coins = n_coins
        self.coin_reward = coin_reward
        self.observation_mode = observation_mode

        self.coins = []
        self.coins_collected = []
        self.coins_removed = []  # hide coins already passed
        self.coins_collected_total = 0  # accumulated across laps
        self.coins_collected_episode = 0  # reset each episode

        self.track_points = []
        self.max_track_progress = 0
        self.progress_buffer = 2          # margin before removing coins already passed
        self.progress_activation_dist = 10.0  # only update progress when near the track
        self.prev_track_idx = 0

        self.coin_radius = 1.2          # visual size of the coin
        self.coin_pickup_radius = 3.0   # pickup radius (tolerant)

        # Structured coin features (k nearest + counts)
        self.coin_feature_k = 5
        self.coin_feature_dim = self.coin_feature_k * 3 + 3
        base_image_space = self.observation_space
        coin_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.coin_feature_dim,),
            dtype=np.float32,
        )
        if self.observation_mode == "image_only":
            self.observation_space = base_image_space
        elif self.observation_mode == "coins_only":
            self.observation_space = spaces.Dict({"coins": coin_space})
        else:  # "image_and_coins"
            self.observation_mode = "image_and_coins"
            self.observation_space = spaces.Dict(
                {
                    "image": base_image_space,
                    "coins": coin_space,
                }
            )
        self._track_norm = TRACK_RAD

    # -------------------------------
    # ENV RESET
    # -------------------------------
    def reset(self, **kwargs):
        """Reset environment state and compose initial observation with coins features."""
        obs, info = super().reset(**kwargs)
        self._reset_coins()
        return self._compose_obs(obs), info

    # -------------------------------
    # COIN PLACEMENT
    # -------------------------------
    def _reset_coins(self):
        """Place coins along the track and reset counters."""
        self.coins = []
        self.coins_collected = []
        self.coins_removed = []
        self.coins_collected_episode = 0

        self.track_points = []
        self.max_track_progress = 0
        self.prev_track_idx = 0

        if hasattr(self, "track") and self.track is not None and len(self.track) > 0:
            # dense placement: one coin per track tile
            for _, _, x, y in self.track:
                self.coins.append((x, y))
                self.coins_collected.append(False)
                self.coins_removed.append(False)
                self.track_points.append((x, y))

    def _update_progress(self, car_x, car_y):
        """Track how far along the track the car is to respawn or remove coins."""
        if not self.track_points:
            return

        pts = np.array(self.track_points)
        dists = np.hypot(pts[:, 0] - car_x, pts[:, 1] - car_y)
        idx = int(np.argmin(dists))

        # Only update progress if car is close to the track
        if dists[idx] > self.progress_activation_dist:
            return

        # Detect lap: if we wrapped after sufficient progress, respawn all coins
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

    def _build_coin_features(self, car_x, car_y):
        """Return fixed-size feature vector with nearest k coins and normalized counts."""
        entries = []
        for i, (cx, cy) in enumerate(self.coins):
            if self.coins_collected[i] or self.coins_removed[i]:
                continue
            dx = (cx - car_x) / self._track_norm
            dy = (cy - car_y) / self._track_norm
            dist_sq = dx * dx + dy * dy
            entries.append((dist_sq, dx, dy))

        entries.sort(key=lambda x: x[0])

        features = []
        for _, dx, dy in entries[: self.coin_feature_k]:
            features.extend([dx, dy, 1.0])

        while len(features) < self.coin_feature_k * 3:
            features.extend([0.0, 0.0, 0.0])

        total = max(len(self.coins), 1)
        remaining = total - sum(self.coins_collected) - sum(self.coins_removed)
        collected_episode = self.coins_collected_episode
        collected_total = self.coins_collected_total

        features.extend(
            [
                remaining / total,
                collected_episode / total,
                min(collected_total / total, 1.0),
            ]
        )

        return np.array(features, dtype=np.float32)

    def _compose_obs(self, image_obs):
        """Compose observation according to observation_mode."""
        if self.car is None:
            coin_features = np.zeros(self.coin_feature_dim, dtype=np.float32)
        else:
            car_x, car_y = self.car.hull.position
            coin_features = self._build_coin_features(car_x, car_y)

        if self.observation_mode == "image_only":
            return image_obs
        if self.observation_mode == "coins_only":
            return {"coins": coin_features}
        return {"image": image_obs, "coins": coin_features}

    # -------------------------------
    # STEP WITH EXTRA COIN REWARD
    # -------------------------------
    def step(self, action):
        """Apply coin bonuses and return composed observation with info diagnostics."""
        # During reset CarRacing calls step(None); avoid composing obs twice
        if action is None:
            return super().step(action)

        obs, reward, terminated, truncated, info = super().step(action)

        if self.car is None or not self.coins:
            return self._compose_obs(obs), reward, terminated, truncated, info

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
                # If we've passed this track index already, remove without bonus
                if i < self.max_track_progress - self.progress_buffer:
                    self.coins_removed[i] = True

        reward += bonus
        if bonus != 0.0:
            # Keep self.reward aligned with HUD to avoid jump on next step_reward
            self.reward += bonus
            self.prev_reward = self.reward

        info["coin_bonus"] = bonus
        info["coins_collected"] = sum(self.coins_collected)
        info["coins_collected_total"] = self.coins_collected_total
        info["coins_collected_episode"] = self.coins_collected_episode
        info["coins_removed"] = sum(self.coins_removed)

        return self._compose_obs(obs), reward, terminated, truncated, info

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

            # Surface already flipped in base render; adjust Y to stick to track
            py = WINDOW_H - py

            pulse = 1.0 + 0.05 * math.sin(self.t * 4.0)
            radius_px = max(2, int(self.coin_radius * zoom * pulse))

            if px < -50 or px > WINDOW_W + 50 or py < -50 or py > WINDOW_H + 50:
                continue

            hud_height = (5 * WINDOW_H) / 40.0  # same black HUD band as base
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

            # Soft shadow (no alpha to keep surface compatibility)
            pygame.gfxdraw.filled_circle(self.surf, px_i + 2, py_i + 3, radius_px + 2, shadow_color)

            # Simple gradient using concentric circles
            pygame.gfxdraw.filled_circle(self.surf, px_i, py_i, radius_px, outer_color)
            pygame.gfxdraw.aacircle(self.surf, px_i, py_i, radius_px, outer_color)

            mid_r = max(1, int(radius_px * 0.75))
            pygame.gfxdraw.filled_circle(self.surf, px_i, py_i, mid_r, mid_color)

            inner_r = max(1, int(radius_px * 0.45))
            pygame.gfxdraw.filled_circle(self.surf, px_i, py_i, inner_r, inner_color)

            # Highlight on top-left
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

        # Small box with border and soft glow
        pygame.draw.rect(self.surf, (25, 25, 25), bg_rect, border_radius=10)
        pygame.draw.rect(self.surf, (90, 90, 90), bg_rect, width=1, border_radius=10)

        # Simple coin icon next to the title
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
    # Render path used to inject coins/HUD before flipping
    # -------------------------------
    def _render(self, mode):
        if mode != "human":
            return super()._render(mode)

        # Mirror base _render to insert coins/HUD before the flip
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

        # Base game HUD
        self._render_indicators(WINDOW_W, WINDOW_H)
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        # Coins overlay
        self._render_coins(zoom, translation, angle)
        self._render_coin_counter()

        # Final flip to avoid flicker
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        self.screen.fill(0)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()

        return self.isopen
