from typing import Any

import gymnasium as gym


class AGymEnv(gym.Env):
    def __init__(self, a_string: str):
        self.a_string = a_string

        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(low=0, high=4, shape=(2,), dtype=int),
                "energy": gym.spaces.Discrete(21),
                "sampled_here": gym.spaces.Discrete(2),
            }
        )

        #TODO initialize the action space
        self.action_space = ...

    def reset(self, *, seed, options):
        super().reset(seed, options)

        #TODO implement the reset method
        ...

    def step(self, action: int):
        #TODO implement the step method
        ...
