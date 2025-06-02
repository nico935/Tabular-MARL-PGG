# Tabular PGG Configuration
# Modify these parameters to run different experiments

# === CURRENT EXPERIMENT SETUP ===
# Change these values to run different experiments
N_AGENTS = 5
N_ROUNDS = 30000
N_HISTORY = 1
PGG_MULTIPLIER =0.9           # Key parameter: >2.0 encourages cooperation, <1.5 discourages
INITIAL_ENDOWMENT = 1       # Key parameter: starting money for each agent (contribute all or nothing)

# Learning Parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# Training Parameters
N_EPISODES = 1                # Single long episode (no resets)
PRINT_EVERY = 100             # Print progress every N rounds
MOVING_AVERAGE_WINDOW = 100   # Window size for moving average plots

# === QUICK EXPERIMENT PRESETS ===
# Uncomment one of these to quickly switch experiments:

# High cooperation (multiplier = 2.5)
# PGG_MULTIPLIER = 2.5

# Low cooperation (multiplier = 1.2) 
# PGG_MULTIPLIER = 1.2

# Large group (8 agents)
# N_AGENTS = 8
# N_EPISODES = 3000

# Small endowment
# INITIAL_ENDOWMENT = 5.0

# Large endowment  
# INITIAL_ENDOWMENT = 20.0
