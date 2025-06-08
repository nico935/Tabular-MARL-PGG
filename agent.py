import numpy as np
from typing import Dict, Tuple
import pickle
import os

class TabularQLearningAgent:
    """
    Tabular Q-learning agent for the Public Goods Game.
    """
    
    def __init__(self, 
                 agent_id: str,
                 observation_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize the tabular Q-learning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            observation_space_size: Size of the observation space
            action_space_size: Size of the action space
            learning_rate: Learning rate for Q-table updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.agent_id = agent_id
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((observation_space_size, action_space_size))
        
        # Track learning statistics
        self.total_reward = 0.0
        self.episode_count = 0
        self.step_count = 0
    
    def select_action(self, state: int) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether agent is in training mode
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(0, self.action_space_size)
        else:
            # Exploit: best known action
            action = np.argmax(self.q_table[state])
        
        return action
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Q-learning update rule
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        self.step_count += 1
        
        # Decay epsilon after each learning step (for continuous learning without resets)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def act(self, observation: int) -> int:
        """
        Select an action for the current observation.
        
        Args:
            observation: Current observation
            training: Whether agent is in training mode
            
        Returns:
            Selected action
        """
        action = self.select_action(observation)
        
        return action
    
    def learn(self, state: int, action: int, reward: float, next_observation: int, done: bool):
        """
        Learn from the previous experience.
        
        Args:
            state: State from which the action was taken
            action: Action taken
            reward: Reward received for the action
            next_observation: Next state observation
            done: Whether episode is done
        """
        # Learn from previous experience
        self.update_q_table(state, action, reward, next_observation, done)
        
        # Update statistics
        self.total_reward += reward
    
    def end_episode(self, final_reward: float = None):
        """
        Handle end of episode cleanup and statistics.
        
        Args:
            final_reward: Final reward of the episode
        """
        if final_reward is not None:
            self.total_reward += final_reward
            
        self.episode_count += 1
    
    def get_average_reward(self) -> float:
        """Get average reward per episode."""
        if self.episode_count == 0:
            return 0.0
        return self.total_reward / self.step_count
    
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics."""
        return {
            "total_reward": self.total_reward,
            "average_reward": self.get_average_reward(),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "epsilon": self.epsilon
        }
    
    def save_q_table(self, filepath: str):
        """Save Q-table to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "agent_id": self.agent_id,
            "stats": self.get_stats(),
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_q_table(self, filepath: str):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data["q_table"]
        # Optionally restore other attributes
        if "stats" in data:
            stats = data["stats"]
            self.total_reward = stats.get("total_reward", 0.0)
            self.episode_count = stats.get("episode_count", 0)
            self.step_count = stats.get("step_count", 0)
            self.epsilon = stats.get("epsilon", self.epsilon)
    
    def reset_stats(self):
        """Reset learning statistics."""
        self.total_reward = 0.0
        self.episode_count = 0
        self.step_count = 0
