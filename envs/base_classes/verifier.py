from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from .data_holder import Data


class TrajectoryVerifier(ABC):
    """
    Verifier that evaluates a *given* action trajectory in a multistep ToolEnv.
    It does NOT call the agent and does NOT generate actions.
    """

    @abstractmethod
    def verify_trajectory(
        self,
        env,
        data: Data,
        actions: List[str],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs env on the provided actions and returns metrics.
        Required keys in the returned dict:
        - success: bool
        - total_reward: float
        - steps: int
        - tool_calls: int
        - policy_violations: int
        Recommended keys:
        - terminated_early: bool
        - invalid_actions: int
        - info_trace: List[dict] (per-step env info)
        """
        raise NotImplementedError
