"""
Tabular Q-learning agent.
"""

import numpy as np
from typing import Dict, Tuple
import pickle
from collections import defaultdict


class QTableAgent:
    """
    Q-learning agent with Q-value table.

    Suitable for discretized state space (features).
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        """
        Args:
            n_actions: number of actions
            learning_rate: learning rate (alpha)
            discount_factor: discount factor (gamma)
            epsilon_start: initial epsilon value
            epsilon_end: minimum epsilon value
            epsilon_decay: epsilon decay rate
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table as defaultdict
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )

        # Statistics
        self.training_steps = 0

    def discretize_state(self, observation: np.ndarray) -> Tuple:
        """
        Converts continuous observation to discrete key.

        For feature observation (18 values 0-1):
        - Binarize values > 0.5
        """
        discrete = tuple((observation > 0.5).astype(int))
        return discrete

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Selects action using epsilon-greedy strategy.

        Args:
            observation: observation
            training: whether in training mode

        Returns:
            Action index
        """
        state = self.discretize_state(observation)

        # epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Greedy
        q_values = self.q_table[state]

        # If all Q-values are equal, choose randomly
        if np.allclose(q_values, q_values[0]):
            return np.random.randint(self.n_actions)

        return np.argmax(q_values)

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> float:
        """
        Updates Q-table.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        Returns:
            TD error
        """
        state = self.discretize_state(observation)
        next_state = self.discretize_state(next_observation)

        # Current Q-value
        current_q = self.q_table[state][action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # TD error
        td_error = target_q - current_q

        # Update
        self.q_table[state][action] += self.lr * td_error

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

        self.training_steps += 1

        return td_error

    def save(self, path: str):
        """Saves agent."""
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "params": {
                "n_actions": self.n_actions,
                "lr": self.lr,
                "gamma": self.gamma,
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Loads agent."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            data["q_table"]
        )
        self.epsilon = data["epsilon"]
        self.training_steps = data["training_steps"]

    def get_stats(self) -> Dict:
        """Returns statistics."""
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
