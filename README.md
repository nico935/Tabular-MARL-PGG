# Tabular Q-Learning Public Goods Game

Implementation of tabular Q-learning agents in a Public Goods Game.

## Quick Start

1. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure experiment** in `config.py`:
   ```python
   PGG_MULTIPLIER = 1.6  #[1,n_agents] is mixed motive, [0,1] is competitive and [n_agents,inf) is cooperative   
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
- `environment.py` - PGG environment 
- `agent.py` -  Q-learning agent 
- `plotting.py` - Visualization utilities
- `requirements.txt` - Dependencies
