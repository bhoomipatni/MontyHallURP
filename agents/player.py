import random
from typing import List

class PlayerAgent:
    """Q-Learning agent for player strategy (stay vs switch)"""
    
    ACTION_STAY = 0
    ACTION_SWITCH = 1
    
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.99):
        """
        Args:
            alpha: Learning rate
            epsilon: Exploration rate
            gamma: Discount factor
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = [0.0, 0.0]  # Q[0]=stay, Q[1]=switch
        self.episode_count = 0
    
    def select_action(self, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            explore: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            0 (stay) or 1 (switch)
        """
        if explore and random.random() < self.epsilon:
            return random.choice([self.ACTION_STAY, self.ACTION_SWITCH])
        return self.ACTION_SWITCH if self.Q[self.ACTION_SWITCH] > self.Q[self.ACTION_STAY] else self.ACTION_STAY
    
    def update(self, action: int, reward: int, next_state_value: float = 0):
        """
        Q-learning update rule.
        
        Args:
            action: Action taken (0 or 1)
            reward: Reward received (0 or 1)
            next_state_value: Value of next state (0 in episodic task)
        """
        td_error = reward + self.gamma * next_state_value - self.Q[action]
        self.Q[action] += self.alpha * td_error
        self.episode_count += 1
    
    def reset(self):
        """Reset Q-values and episode count"""
        self.Q = [0.0, 0.0]
        self.episode_count = 0
    
    def get_q_values(self) -> List[float]:
        """Get current Q-values"""
        return self.Q.copy()