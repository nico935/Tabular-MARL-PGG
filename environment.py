import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import gymnasium as gym
from pettingzoo import ParallelEnv
from gymnasium import spaces

class TabularPublicGoodsGame(ParallelEnv):
    """
    Public Goods Game environment for tabular Q-learning.
    
    Each agent can either contribute (1) or not contribute (0) a fixed amount.
    The observation space is the history of past round contributions of all agents.

    """
    
    metadata = {"render_modes": ["human"], "name": "tabular_public_goods_v1"}
    
    def __init__(self, 
                 n_agents: int = 4, 
                 n_rounds: int = 10, 
                 n_history: int = 3,
                 pgg_multiplier: float = 1.6,
                 initial_endowment: float = 10.0,
                 render_mode: Optional[str] = None):
        """
        Initialize the Tabular Public Goods Game with binary actions.
        
        Args:
            n_agents: Number of agents in the game
            n_rounds: Number of rounds per episode
            n_history: Number of historical rounds to include in observation
            pgg_multiplier: Multiplier for the public goods
            initial_endowment: Initial endowment for each agent (contributed when action=1)
            render_mode: Render mode for the environment
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_history = n_history
        self.pgg_multiplier = pgg_multiplier
        self.initial_endowment = initial_endowment
        self.render_mode = render_mode
        
        # PettingZoo required attributes
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        
        # Action space: 0 (don't contribute) or 1 (contribute)
        self._action_space = spaces.Discrete(2)
        
        # Observation space: binary contribution history
        # State is contribution history: n_history rounds of n_agents binary contributions
        # Each contribution is 0 or 1, so state space = 2^(n_history * n_agents)
        self._observation_space = spaces.Discrete(2 ** (n_history * n_agents))
        
        # State variables
        self.current_round = 0
        self.history = deque(maxlen=n_history)
        self.total_payoffs = {agent: 0.0 for agent in self.agents}
    @property
    def action_space(self):
        """Action space for each agent."""
        return self._action_space
    
    @property 
    def observation_space(self):
        """Observation space for each agent."""
        return self._observation_space
    
    def action_spaces(self, agent):
        """Action space for a specific agent."""
        return self._action_space
    
    def observation_spaces(self, agent):
        """Observation space for a specific agent."""
        return self._observation_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, int], Dict[str, dict]]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_round = 0
        self.history.clear()
        
        # Initialize history with random binary values (0s and 1s)
        for _ in range(self.n_history):
            random_contributions = np.random.randint(0, 2, size=self.n_agents, dtype=int)
            self.history.append(random_contributions)
        
        self.total_payoffs = {agent: 0.0 for agent in self.agents}
        
        # Ensure agents list is reset
        self.agents = self.possible_agents[:]
        
        # Return initial observations and info
        observations = {}
        observation = self._get_observation()
        for agent in self.agents:
            observations[agent] = observation
            
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def _get_observation(self) -> int:
        """
        Get the current observation state.
        
        Returns:
            An integer representing the state based on past contribution history
        """
        # Flatten the contribution history into a single state index
        state_components = []
        for round_contributions in self.history:
            for agent_contrib in round_contributions:
                state_components.append(int(agent_contrib))
        
        # Convert to single state index using binary encoding
        state = 0
        for i, component in enumerate(state_components):
            state += component * (2 ** i)
            
        return state
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent names to their actions (contribution amounts)
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        if self.current_round >= self.n_rounds:
            # Episode is already done - return empty dicts for terminated environment
            return {}, {}, {}, {}, {}
        
        # Extract actions for active agents
        active_agents = self.agents[:]
        contributions = np.array([actions.get(agent, 0) for agent in active_agents], dtype=int)
        
        # Add current contributions to history
        self.history.append(contributions.copy())
        
        # Calculate actual contribution amounts (full endowment or nothing)
        actual_contributions = contributions * self.initial_endowment
        total_contribution = np.sum(actual_contributions)
        public_pool = total_contribution * self.pgg_multiplier
        equal_share = public_pool / len(active_agents)
        
        # Calculate rewards for each active agent
        rewards = {}
        for i, agent in enumerate(active_agents):
            money_kept = self.initial_endowment - actual_contributions[i]
            agent_reward = money_kept + equal_share
            rewards[agent] = agent_reward
            self.total_payoffs[agent] += agent_reward
        
        # Get next observations for active agents
        observation = self._get_observation()
        observations = {agent: observation for agent in active_agents}
        
        # Check if episode is done
        self.current_round += 1
        done = self.current_round >= self.n_rounds
        
        # Remove agents when episode ends
        if done:
            self.agents = []
            
        terminations = {agent: done for agent in active_agents}
        truncations = {agent: done for agent in active_agents}  
        # Info for active agents
        infos = {
            agent: {
                "actual_contribution": actual_contributions[i],
                "total_contribution": total_contribution,
            } for i, agent in enumerate(active_agents)
        }
        
        return observations, rewards, terminations, truncations, infos
    
    def get_observation_space_size(self) -> int:
        """Return the size of the observation space (for backward compatibility)."""
        return self._observation_space.n
    
    def get_action_space_size(self) -> int:
        """Return the size of the action space (for backward compatibility)."""
        return self._action_space.n
    
    def render(self):
        """Render the current state of the game."""
        if self.render_mode == "human":
            print(f"Round: {self.current_round}/{self.n_rounds}")
            if self.history:
                print("Contribution History:")
                for i, round_contributions in enumerate(self.history):
                    round_num = self.current_round - len(self.history) + i + 1
                    if round_num > 0:
                        print(f"  Round {round_num}: {round_contributions}")
            print(f"Total Payoffs: {self.total_payoffs}")
            print("-" * 50)
    
    def close(self):
        """Close the environment."""
        pass
