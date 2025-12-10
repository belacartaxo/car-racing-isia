from gymnasium.envs.box2d.car_racing import CarRacing
import numpy as np


class CustomCarRacingEnv(CarRacing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.off_track_border = 0.2
        self.off_track_penalty = 5.0
        self.max_off_track_steps = 50
        self._off_track_steps = 0

    def reset(self, *, seed=None, options=None):
        self._off_track_steps = 0
        return super().reset(seed=seed, options=options)

    def _point_in_poly(self, x, y, poly):
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
        obs, reward, terminated, truncated, info = super().step(action)

        on_track = self.car_on_track(border_width=self.off_track_border)

        if not on_track:
            self._off_track_steps += 1
            reward -= self.off_track_penalty
            info["off_track"] = True
        else:
            self._off_track_steps = 0
            info["off_track"] = False

        if self._off_track_steps > self.max_off_track_steps:
            terminated = True

        return obs, reward, terminated, truncated, info
