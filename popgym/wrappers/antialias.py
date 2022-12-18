from typing import Tuple

from gym import Env, Wrapper, spaces
from gym.core import ActType, ObsType

from popgym.core.observability import OBS, STATE

IS_T0 = "is_t0"


class Antialias(Wrapper):
    """Wrapper that undoes aliasing produces by the PreviousAction wrapper

    Outputs a boolean flag denoting whether the observation was taken
    at the first timestep.

    Args:
        env: The environment

    Returns:
        A gym environment with a Discrete(2) appended to the observation space
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = Antialias.antialias_obs_space(
            self.env.observation_space
        )

    @staticmethod
    def antialias_obs_space(observation_space: spaces.Space) -> spaces.Space:
        flag_space = spaces.Discrete(2)
        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            observation_space = spaces.Tuple((observation_space, flag_space))
        elif isinstance(observation_space, spaces.Tuple):
            observation_space = spaces.Tuple(
                tuple(observation_space.spaces) + (flag_space,)
            )
        elif isinstance(observation_space, spaces.Dict):
            observation_space = observation_space.spaces.copy()
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper
                # with observability level FULL_AND_PARTIAL
                observation_space[OBS] = Antialias.antialias_obs_space(
                    observation_space[OBS]
                )
            else:
                observation_space[IS_T0] = flag_space
            observation_space = spaces.Dict(observation_space)
        else:
            raise NotImplementedError("Unknown observation space")
        return observation_space

    @staticmethod
    def antialias_obs(
        observation_space: spaces.Space, obs: ObsType, timestep: int
    ) -> ObsType:
        is_t0 = int(timestep == 0)
        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            obs = (obs, is_t0)
        elif isinstance(observation_space, spaces.Tuple):
            obs = tuple(obs) + (is_t0,)
        elif isinstance(observation_space, spaces.Dict):
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper
                # with observability level FULL_AND_PARTIAL
                obs[OBS] = Antialias.antialias_obs(
                    observation_space[OBS], obs[OBS], is_t0
                )
            else:
                obs[IS_T0] = is_t0
        else:
            raise NotImplementedError("Unknown observation space")
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self._tstep += 1
        obs = Antialias.antialias_obs(self.env.observation_space, obs, self._tstep)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self._tstep = 0
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            obs = self.antialias_obs(self.env.observation_space, obs, self._tstep)
            return obs, info
        else:
            obs = self.env.reset(**kwargs)
            obs = self.antialias_obs(self.env.observation_space, obs, self._tstep)
            return obs
