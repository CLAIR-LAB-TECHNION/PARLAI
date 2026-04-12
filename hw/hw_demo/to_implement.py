from typing import Any

import gymnasium as gym


class AGymEnv(gym.Env):
    def __init__(self, a_string: str):
        self.a_string = a_string

        #TODO initialize the observation space
        self.observation_space = ...

        #TODO initialize the action space
        self.action_space = ...

    def reset(self, *, seed, options):
        super().reset(seed, options)

        #TODO implement the reset method
        ...

    def step(self, action: int):
        #TODO implement the step method
        ...
