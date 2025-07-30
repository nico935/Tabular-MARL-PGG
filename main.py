import numpy as np
from collections import defaultdict

from environment import TabularPublicGoodsGame
from agent import TabularQLearningAgent
from plotting import create_training_plots
from config import *



def main():
    """Main function to run tabular PGG."""
    print("=" * 60)
    print("TABULAR Q-LEARNING PUBLIC GOODS GAME")
    print("=" * 60)
    print(f"Agents: {N_AGENTS}, Rounds: {N_ROUNDS}, Multiplier: {PGG_MULTIPLIER}")
    print(f"Endowment: {INITIAL_ENDOWMENT}, Episodes: {N_EPISODES}")
    print("-" * 60)
    
    # Initialize environment
    env = TabularPublicGoodsGame(
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
        n_history=N_HISTORY,
        pgg_multiplier=PGG_MULTIPLIER,
        initial_endowment=INITIAL_ENDOWMENT
    )
    
    # Initialize agents
    agents = {}
    for i in range(N_AGENTS):
        agent_id = f"agent_{i}"
        agents[agent_id] = TabularQLearningAgent(
            agent_id=agent_id,
            observation_space_size=env.get_observation_space_size(),
            action_space_size=env.get_action_space_size(),
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN
        )
    
    # Training statistics
    round_contributions = defaultdict(list)  # Track individual agent contributions per round
    round_agent_rewards = defaultdict(list)  # Track individual agent rewards per round
    round_cooperation_rates = []
    
    # Training loop
    round_counter = 0
    for episode in range(N_EPISODES):
        observations, infos = env.reset()
        cum_rewards = defaultdict(float)
        
        
        # Run episode
        while env.agents:  # Continue while there are active agents
            round_counter += 1
            
            # Get actions from all active agents
            actions = {}
            for agent_id in env.agents:
                if agent_id in agents:
                    current_state = observations[agent_id]
                    action = agents[agent_id].select_action(current_state)
                    actions[agent_id] = action
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent_id in env.agents:
                if agent_id in agents:
                    action = actions[agent_id]  
                    round_contributions[agent_id].append(action) # Store contribution for this agent this round
                    
                    reward = rewards.get(agent_id, 0)
                    next_obs = next_observations.get(agent_id)
                    done_agent = terminations.get(agent_id, False)
                    
                    round_agent_rewards[agent_id].append(reward) # Store reward for this agent this round
                    
                    # Learn from this step
                    current_obs = observations[agent_id]
                    agents[agent_id].learn(current_obs, action, reward, next_obs, done_agent)
                    cum_rewards[agent_id] += reward

            # Track overall cooperation rate - calculate directly from actions
            total_contribution = sum(actions.values())
            max_possible = len(actions)  # Since actions are 0 or 1
            cooperation_rate = total_contribution / max_possible if max_possible > 0 else 0
            round_cooperation_rates.append(cooperation_rate)
            
            observations = next_observations
            
            # Print progress every PRINT_EVERY rounds
            if round_counter % PRINT_EVERY == 0:
                avg_reward = np.mean([cum_rewards[agent_id] / round_counter for agent_id in agents.keys() if agent_id in cum_rewards])
                recent_coop = np.mean(round_cooperation_rates[-PRINT_EVERY:]) if len(round_cooperation_rates) >= PRINT_EVERY else np.mean(round_cooperation_rates)
                
                # Get epsilon from any agent (they should all be the same)
                epsilon_val = list(agents.values())[0].epsilon
                print(f"Round {round_counter:4d}: "
                      f"Avg Reward/Round: {avg_reward:7.2f}, "
                      f"Recent Cooperation Rate: {recent_coop:5.2%}, "
                      f"Epsilon: {epsilon_val:.3f}")
        
        # End episode for all agents
        for agent in agents.values():
            agent.end_episode()
    
    print("\nTraining completed!")
    
    # Create plots using plotting module
    config_params = {
        'MOVING_AVERAGE_WINDOW': MOVING_AVERAGE_WINDOW,
        'INITIAL_ENDOWMENT': INITIAL_ENDOWMENT,
        'N_AGENTS': N_AGENTS,
        'EPSILON': EPSILON,
        'EPSILON_DECAY': EPSILON_DECAY,
        'EPSILON_MIN': EPSILON_MIN
    }
    
    create_training_plots(
        round_contributions=round_contributions,
        round_agent_rewards=round_agent_rewards,
        round_cooperation_rates=round_cooperation_rates,
        round_counter=round_counter,
        agents=agents,
        config_params=config_params
    )
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Rounds: {round_counter}")
    print(f"Final Overall Cooperation Rate: {round_cooperation_rates[-1]:.2%}")
    
    # Agent statistics
    print("\nAgent Statistics:")
    for agent_id, agent in agents.items():
        contributions = round_contributions[agent_id]
        recent_contrib = np.mean(contributions[-min(100, len(contributions)):]) if contributions else 0
        stats = agent.get_stats()
        print(f"  {agent_id}: Recent Avg Contribution: {recent_contrib:.2f}, "
              f"Steps: {stats['step_count']}, Epsilon: {stats['epsilon']:.3f}")

if __name__ == "__main__":
    main()