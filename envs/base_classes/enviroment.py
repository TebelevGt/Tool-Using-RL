from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from .data_holder import Data


class ToolEnv(ABC):
    """
    Multi-step tool-using environment.
    Text in / text out.
    """

    def __init__(self, name: str):
        self.name = name

        @abstractmethod
        def reset(self, data: Data) -> str:
            """
            Initialize an episode from Data and return initial
            observation (text).
            """
            raise NotImplementedError

        @abstractmethod
        def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
            """
            Apply one agent action and return:
            observation (text), reward (float), done (bool), info
            (dict)
            """
            raise NotImplementedError

        @abstractmethod
        def generate(
            self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: Optional[int] = 1, **kwargs
        ) -> list[Data]:
            """
            Procedurally generate episodes. Must support both difficulty
            and direct hyperparams via kwargs.
            """
            raise NotImplementedError
