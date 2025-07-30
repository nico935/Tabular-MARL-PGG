# Tabular Q-Learning Public Goods Game

Implementation of tabular Q-learning agents in a Public Goods Game using PettingZoo's ParallelEnv interface.

## Features

- **PettingZoo Compatible**: Uses PettingZoo's ParallelEnv interface for standardized multi-agent RL
- **Tabular Q-Learning**: Discrete state/action spaces with Q-table learning
- **Modular Design**: Separate modules for environment, agents, and plotting
- **Comprehensive Visualization**: Automated plotting of training metrics

## Quick Start

1. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure experiment** in `config.py`:
   ```python
   PGG_MULTIPLIER = 1.6  #[1,n_agents] is mixed motive, [0,1] competitive, [n_agents,inf) cooperative   
   INITIAL_ENDOWMENT = 1.0  
   N_AGENTS = 4
   N_EPISODES = 160000
   ```

3. **Run experiment**:
   ```bash
   python main.py
   ```

## Files

- `config.py` - Main configuration (edit this to change experiments)
- `main.py` - Training and evaluation script
- `environment.py` - PGG environment (PettingZoo ParallelEnv)
- `agent.py` -  Q-learning agent with centralized epsilon calculation
- `plotting.py` - Visualization utilities
- `requirements.txt` - Dependencies (includes PettingZoo and Gymnasium)

## PettingZoo Interface

The environment implements PettingZoo's ParallelEnv interface:

```python
from environment import TabularPublicGoodsGame

env = TabularPublicGoodsGame(n_agents=3, n_rounds=100)
observations, infos = env.reset()

while env.agents:  # Continue while agents are active
    actions = {agent: env.action_space.sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

