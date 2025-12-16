"""Extended CarRacing with off-track penalties, idle handling, speed bonus and optional mask channel."""

from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import CarRacing
import numpy as np


class CustomCarRacingEnv(CarRacing):
    """CarRacing variant with reward shaping and an optional on-track mask on observations."""

    def __init__(self, include_mask: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_mask = include_mask
        self.off_track_border = 0.2
        self.off_track_penalty = 60.0
        self.off_track_penalty_growth = 0.4
        self.max_off_track_steps = 20
        self._off_track_steps = 0
        # Anti-idle: terminate if too slow for too long
        self.idle_min_speed = 1.2
        self.idle_patience_steps = 20
        self.idle_penalty = 70.0
        self._idle_steps = 0
        # Bonus proportional to speed to encourage acceleration
        self.speed_reward_scale = 0.3
        # Bonus for staying on track
        self.on_track_bonus = 0.1
        # Ajusta observation_space se for incluir mask
        if self.include_mask:
            h, w, _ = self.observation_space.shape
            self.observation_space = spaces.Box(
                low=0.0,
                high=255.0,
                shape=(h, w, 4),
                dtype=self.observation_space.dtype,
            )

    def reset(self, *, seed=None, options=None):
        """Reset counters and return the first observation with optional mask appended."""
        self._off_track_steps = 0
        self._idle_steps = 0
        obs, info = super().reset(seed=seed, options=options)
        on_track = self.car_on_track(border_width=self.off_track_border)
        obs = self._augment_obs(obs, on_track)
        return obs, info

    def _augment_obs(self, obs, on_track):
        """Normalize to 3-channel RGB and optionally append a binary on-track mask."""
        if obs.ndim == 2:  # raro, mas garante HxW -> HxWxc
            obs = np.expand_dims(obs, axis=2)
        if obs.shape[2] > 3:
            obs = obs[:, :, :3]
        elif obs.shape[2] < 3:
            obs = np.repeat(obs, 3 // max(obs.shape[2], 1), axis=2) if obs.shape[2] > 0 else np.zeros((*obs.shape[:2], 3), dtype=obs.dtype)
            obs = obs[:, :, :3]

        if not self.include_mask:
            return obs

        mask = np.full((obs.shape[0], obs.shape[1], 1), 1.0 if on_track else 0.0, dtype=obs.dtype)
        return np.concatenate([obs, mask], axis=2)

    def _point_in_poly(self, x, y, poly):
        """Return True if point (x, y) lies inside polygon poly."""
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cond = ((y1 > y) != (y2 > y))
            if cond:
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-9)
                if x_intersect > x:
                    inside = not inside
        return inside

    def car_on_track(self, border_width=0.0):
        """Check whether the car hull is inside any road polygon (with optional widened border)."""
        if not hasattr(self, "road_poly") or self.car is None:
            return True

        x, y = self.car.hull.position

        for poly, color in self.road_poly:
            vertices = poly

            if border_width != 0:
                xs = [vx for vx, vy in vertices]
                ys = [vy for vx, vy in vertices]
                cx, cy = np.mean(xs), np.mean(ys)
                scale = 1.0 + border_width
                scaled = []
                for vx, vy in vertices:
                    sx = cx + (vx - cx) * scale
                    sy = cy + (vy - cy) * scale
                    scaled.append((sx, sy))
                vertices_check = scaled
            else:
                vertices_check = vertices

            if self._point_in_poly(x, y, vertices_check):
                return True

        return False

    def step(self, action):
        """Apply reward shaping, penalties and termination logic, then augment observation."""
        obs, reward, terminated, truncated, info = super().step(action)

        on_track = self.car_on_track(border_width=self.off_track_border)

        if not on_track:
            self._off_track_steps += 1
            penalty_multiplier = 1.0 + self.off_track_penalty_growth * max(self._off_track_steps - 1, 0)
            reward -= self.off_track_penalty * penalty_multiplier
            info["off_track"] = True
            info["off_track_steps"] = self._off_track_steps
        else:
            self._off_track_steps = 0
            info["off_track"] = False
            reward += self.on_track_bonus

        if self._off_track_steps > self.max_off_track_steps:
            reward -= self.off_track_penalty * 2  # extra penalty and terminate if exceeding off-track steps
            terminated = True

        # Anti-ociosidade: se ficar muito tempo devagar, termina com penalidade
        lin_vel = getattr(self.car.hull, "linearVelocity", None)
        speed = 0.0
        if lin_vel is not None:
            speed = (lin_vel[0] ** 2 + lin_vel[1] ** 2) ** 0.5

        # Bonus leve por se mover (aplica somente na pista)
        if on_track:
            reward += self.speed_reward_scale * speed

        if lin_vel is not None:
            if speed < self.idle_min_speed:
                self._idle_steps += 1
            else:
                self._idle_steps = 0

            if self._idle_steps > self.idle_patience_steps:
                reward -= self.idle_penalty
                terminated = True
                info["idle_termination"] = True

        obs = self._augment_obs(obs, on_track)
        return obs, reward, terminated, truncated, info
