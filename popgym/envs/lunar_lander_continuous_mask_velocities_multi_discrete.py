from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.box2d.lunar_lander import LunarLander

from popgym.core.env import POPGymEnv
import time


class LunarLanderContinuousMaskVelocitiesMultiDiscrete(POPGymEnv):
    
    def __init__(self, *args, **kwargs):
        self._env = gym.make("LunarLander-v2", continuous = True, render_mode = "rgb_array")#, render_mode = render_mode)
        self.max_episode_length = kwargs.pop("max_episode_length", 1000)
        self._obs_mask =  np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float32)
        # self._obs_mask =  np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([11,11])
        self.observation_space = self._env.observation_space
        self.mask_start = 10 
        self.mask_len = kwargs.get("mask_len", 10) 

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[gym.core.ObsType, dict]:
        self._rewards = []
        self.num_steps = 0
        init_obs, info = self._env.reset(seed=seed, options=options)
        return init_obs * self._obs_mask, info

    def step(self, action):
        continuous_action = np.zeros(2)
        continuous_action[0] = np.linspace(-1,1,self.action_space.nvec[0])[action[0]]
        continuous_action[1] = np.linspace(-1,1,self.action_space.nvec[1])[action[1]]
        obs, reward, done, truncated, info = self._env.step(continuous_action)
        self.num_steps += 1
        if self.num_steps >= self.max_episode_length:
            truncated = True
        self._rewards.append(reward)
        if done or truncated:
            if info == None:
                info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
            else:
                info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = {}
        if self.mask_start >= self.num_steps >= self.mask_start + self.mask_len:
            obs = obs * 0
        return obs * self._obs_mask, reward, done, truncated , info

    def render(self):
        self._env.render_mode = "human"
        self._env.render()
        self._env.render_mode = "rgb_array"

class LunarLanderContinuousMaskVelocitiesMultiDiscreteEasy(LunarLanderContinuousMaskVelocitiesMultiDiscrete):
    def __init__(self, *args, **kwargs):
        super().__init__(mask_len=0)

class LunarLanderContinuousMaskVelocitiesMultiDiscreteMedium(LunarLanderContinuousMaskVelocitiesMultiDiscrete):
    def __init__(self, *args, **kwargs):
        super().__init__(mask_len=40)

class LunarLanderContinuousMaskVelocitiesMultiDiscreteHard(LunarLanderContinuousMaskVelocitiesMultiDiscrete):
    def __init__(self, *args, **kwargs):
        super().__init__(mask_len=60)

# Short example script to create and run the environment with
# constant action for 1 simulation second.
import time
if __name__ == "__main__":
    from time import sleep
    env = LunarLanderContinuousMaskVelocitiesMultiDiscrete()
    #env = JSBSimReachEnv()
    env.reset()
    '''while True:
        env.render()
        env._get_simulation_state()'''
    #env.render()
    start = time.time()
    for i in range(2500):
        #env.step(np.array([0.05, -0.2, 0, .5]))
        obs,_,_,_ = env.step(env.action_space.sample())
        print(obs)
        #print("forward "+str(np.asarray([obs[1],obs[2],obs[3]])))
        #print("right "+str(np.asarray([obs[4],obs[5],obs[6]])))
        #print("down "+str(np.asarray([obs[7],obs[8],obs[9]])))
        #print("\n")
        env.render()
        #sleep(0.001)
        #sleep(1/100)
        #sleep(1/200)
        #sleep(1)
        #sleep(0.1*5)
    end = time.time()
    print(end-start)
    env.close()
    



'''class LunarLanderContinuousMaskVelocities:
    def __init__(self, mask_velocity = True, realtime_mode = False):
        render_mode = "human" if realtime_mode else None
        self._env = gym.make("LunarLanderContinuous-v2")#, render_mode = render_mode)
        # Whether to make CartPole partial observable by masking out the velocity.
        if False:#not mask_velocity:
            self._obs_mask = np.ones(4, dtype=np.float32)
        else:
            self._obs_mask =  np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float32)

    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        #obs, _ = self._env.reset()
        obs = self._env.reset()
        return obs * self._obs_mask

    def step(self, action):
        #obs, reward, done, truncation, info = self._env.step(action[0])
        #self._rewards.append(reward)
        #if done or truncation:
        #    info = {"reward": sum(self._rewards),
        #            "length": len(self._rewards)}
        #else:
        #    info = None
        #return obs * self._obs_mask, reward / 100.0, done or truncation, info
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs * self._obs_mask, reward, done , info

    def render(self):
        self._env.render()
        #time.sleep(0.033)

    def close(self):
        self._env.close()'''
