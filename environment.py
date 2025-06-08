import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque

class TabularPublicGoodsGame:
    """
    Binary Public Goods Game environment for tabular Q-learning.
    
    Each agent can either contribute (1) or not contribute (0) a fixed amount.
    The observation space is the binary history of past round contributions.
    """
    
    def __init__(self, 
                 n_agents: int = 4, 
                 n_rounds: int = 10, 
                 n_history: int = 3,
                 pgg_multiplier: float = 1.6,
                 initial_endowment: float = 10.0):
        """
        Initialize the Tabular Public Goods Game with binary actions.
        
        Args:
            n_agents: Number of agents in the game
            n_rounds: Number of rounds per episode
            n_history: Number of historical rounds to include in observation
            pgg_multiplier: Multiplier for the public goods
            initial_endowment: Initial endowment for each agent (contributed when action=1)
        """
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_history = n_history
        self.pgg_multiplier = pgg_multiplier
        self.initial_endowment = initial_endowment
        
        # State variables
        self.current_round = 0
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.history = deque(maxlen=n_history)
        self.total_payoffs = {agent: 0.0 for agent in self.agents}
        
        # Action space: 0 (don't contribute) or 1 (contribute)
        self.action_space_size = 2
        
        # Observation space: binary contribution history
        # State is contribution history: n_history rounds of n_agents binary contributions
        # Each contribution is 0 or 1, so state space = 2^(n_history * n_agents)
        self.observation_space_size = 2 ** (n_history * n_agents)
        
    def reset(self) -> Dict[str, int]:
        """Reset the environment and return initial observations."""
        self.current_round = 0
        self.history.clear()
        
        # Initialize history with random binary values (0s and 1s)
        for _ in range(self.n_history):
            random_contributions = np.random.randint(0, 2, size=self.n_agents, dtype=int)
            self.history.append(random_contributions)
        
        self.total_payoffs = {agent: 0.0 for agent in self.agents}
        
        # Return initial observations
        observations = {}
        observation = self._get_observation()
        for agent in self.agents:
            observations[agent] = observation
        return observations
    
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
            observations, rewards, terminations, truncations, info
        """
        if self.current_round >= self.n_rounds:
            # Episode is already done
            return {}, {}, {}, {}, {}
        
        # Extract binary actions (already 0 or 1)
        contributions = np.array([actions.get(agent, 0) for agent in self.agents], dtype=int)
        
        # Add current contributions to history
        self.history.append(contributions.copy())
        
        # Calculate actual contribution amounts (full endowment or nothing)
        actual_contributions = contributions * self.initial_endowment
        total_contribution = np.sum(actual_contributions)
        public_pool = total_contribution * self.pgg_multiplier
        equal_share = public_pool / self.n_agents
        
        # Calculate rewards for each agent
        rewards = {}
        for i, agent in enumerate(self.agents):
            money_kept = self.initial_endowment - actual_contributions[i]
            agent_reward = money_kept + equal_share
            rewards[agent] = agent_reward
            self.total_payoffs[agent] += agent_reward
        
        # Get next observations
        # Since the observation is the same for all agents, compute once and reuse
        observation = self._get_observation()
        observations = {agent: observation for agent in self.agents}
        
        # Check if episode is done
        self.current_round += 1
        done = self.current_round >= self.n_rounds
        
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}  # No truncation in this simple version
        
        # Info
        info = {
            agent: {
                "actual_contribution": actual_contributions[i],
                "total_contribution": total_contribution,
            } for i, agent in enumerate(self.agents)
        }
        
        return observations, rewards, terminations, truncations, info
    
    def get_observation_space_size(self) -> int:
        """Return the size of the observation space."""
        return self.observation_space_size
    
    def get_action_space_size(self) -> int:
        """Return the size of the action space."""
        return self.action_space_size
    
    def render(self):
        """Print current state of the game."""
        print(f"Round: {self.current_round}/{self.n_rounds}")
        if self.history:
            print("Contribution History:")
            for i, round_contributions in enumerate(self.history):
                round_num = self.current_round - len(self.history) + i + 1
                if round_num > 0:
                    print(f"  Round {round_num}: {round_contributions}")
        print(f"Total Payoffs: {self.total_payoffs}")
        print("-" * 50)
