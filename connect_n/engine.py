from abc import ABC, abstractmethod
import numpy as np
from gymnasium import Env

class _Engine(ABC):
    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        raise NotImplementedError



class RandomEngine(_Engine):
    def __init__(self, env: Env):
        self.action_space = env.action_space

    def get_action(self, state) -> int:
        return self.action_space.sample()