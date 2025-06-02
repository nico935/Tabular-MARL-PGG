# Tabular Q-Learning Public Goods Game

Simple, clean implementation of tabular Q-learning agents in a Public Goods Game.

## Quick Start

1. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure experiment** in `config.py`:
   ```python
   PGG_MULTIPLIER = 1.6      # Key parameter!
   INITIAL_ENDOWMENT = 10.0  # Key parameter!
   N_AGENTS = 4
   N_EPISODES = 2000
   ```

3. **Run experiment**:
   ```bash
   python main.py
   ```

## Key Parameters

- **`PGG_MULTIPLIER`**: >2.0 encourages cooperation, <1.5 discourages
- **`INITIAL_ENDOWMENT`**: Starting money for each agent  
- **`N_AGENTS`**: Number of players
- **`N_EPISODES`**: Training episodes

## Files

- `config.py` - **Main configuration** (edit this to change experiments)
- `main.py` - Training and evaluation script
- `environment.py` - Discrete PGG environment
- `agent.py` - Tabular Q-learning agent
- `requirements.txt` - Dependencies (numpy, matplotlib)

## Quick Experiments

Edit `config.py` and uncomment presets:

```python
# High cooperation
PGG_MULTIPLIER = 2.5

# Low cooperation  
PGG_MULTIPLIER = 1.2

# Large group
N_AGENTS = 8
```

That's it! Simple and DRY.
