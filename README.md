# Tabular Q-Learning Public Goods Game

Simple implementation of tabular Q-learning agents in a Public Goods Game.

## Quick Start

1. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure experiment** in `config.py`:
   ```python
   #The multiplier dictates the nature of the game
   PGG_MULTIPLIER = 1.6  
   # Between: [1,n_agents] game is mixed motive, [0,1] is competitive and [n_agents,inf) is cooperative   

   INITIAL_ENDOWMENT = 1.0  
   N_AGENTS = 4
   N_ROUNDS=160000
   ```

3. **Run experiment**:
   ```bash
   python main.py
   ```
## Summary
N_AGENTS play the PGG for N_ROUNDS repeatedly. After each round they update their Q-table. The state consists of the actions of all the agents in the past N_HISTORY rounds, i.e., if history is 2, all agents see the actions of the previous two rounds as the state. 
## Files

- `config.py` - Main configuration (edit this to change experiments)
- `main.py` - Training and evaluation script
- `environment.py` - PGG environment 
- `agent.py` -  Q-learning agent 
- `plotting.py` - Visualization utilities
- `requirements.txt` - Dependencies

