import numpy as np
import gymnasium as gym


class CoinVisionChannelWrapper(gym.Wrapper):
    """
    Adds a 4th channel to the observation containing a binary mask of coin positions.
    - Uses info["coin_vision"] if present (CoinsCarRacingEnv); otherwise leaves the channel zeroed.
    - Keeps the base observation untouched; only concatenates the mask along the channel dimension.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_space = env.observation_space
        assert len(obs_space.shape) == 3, "Expected HxWxC observation."

        h, w, c = obs_space.shape
        self._h = h
        self._w = w
        self._dtype = obs_space.dtype

        self.observation_space = gym.spaces.Box(
            low=obs_space.low.min(),
            high=obs_space.high.max(),
            shape=(h, w, c + 1),
            dtype=self._dtype,
        )

    def _build_coin_mask(self, info):
        grid = info.get("coin_vision")
        if grid is None:
            return np.zeros((self._h, self._w, 1), dtype=self._dtype)

        grid = np.asarray(grid, dtype=np.float32)
        gh, gw = grid.shape
        # upscale grid to match image resolution (e.g., 32 -> 96)
        scale_h = max(1, self._h // gh)
        scale_w = max(1, self._w // gw)
        mask = np.kron(grid, np.ones((scale_h, scale_w), dtype=np.float32))
        mask = mask[: self._h, : self._w]  # crop in case of mismatch
        mask = (mask > 0).astype(self._dtype)

        # Match pixel range; assume uint8 (0-255) or float [0,1]
        if np.issubdtype(self._dtype, np.integer):
            mask = mask * np.array(self.observation_space.high.max(), dtype=self._dtype)
        return mask.reshape(self._h, self._w, 1)

    def _augment(self, obs, info):
        mask = self._build_coin_mask(info)
        return np.concatenate([obs, mask], axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, info
