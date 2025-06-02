import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from environment import TabularPublicGoodsGame
from agent import TabularQLearningAgent
from config import *

def main():
    """Main function to run the tabular PGG experiment."""
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
    episode_rewards = defaultdict(list)
    round_contributions = defaultdict(list)  # Track individual agent contributions per round
    round_cooperation_rates = []
    
    # Training loop
    round_counter = 0
    for episode in range(N_EPISODES):
        observations = env.reset()
        episode_reward_sum = defaultdict(float)
        
        # Reset agents for new episode
        for agent in agents.values():
            agent.last_state = None
            agent.last_action = None
        
        # Run episode
        done = False
        while not done:
            round_counter += 1
            
            # Get actions from all agents
            actions = {}
            for agent_id, agent in agents.items():
                action = agent.act_and_learn(observations[agent_id], training=True)
                actions[agent_id] = action
            
            # Step environment
            next_observations, rewards, terminations, truncations, info = env.step(actions)
            
            # Track individual agent contributions for this round
            if info:
                for i, agent_id in enumerate(agents.keys()):
                    agent_info = info[agent_id]
                    contribution = agent_info["actual_contribution"]
                    round_contributions[agent_id].append(contribution)
                
                # Track overall cooperation rate
                total_contribution = info[list(info.keys())[0]]["total_contribution"]
                max_possible = N_AGENTS * INITIAL_ENDOWMENT
                cooperation_rate = total_contribution / max_possible if max_possible > 0 else 0
                round_cooperation_rates.append(cooperation_rate)
            
            # Update agents with rewards
            for agent_id, agent in agents.items():
                reward = rewards.get(agent_id, 0.0)
                next_obs = next_observations.get(agent_id, observations[agent_id])
                done_agent = terminations.get(agent_id, False)
                
                # Learn from this step
                agent.act_and_learn(next_obs, reward, done_agent, training=True)
                episode_reward_sum[agent_id] += reward
            
            observations = next_observations
            done = all(terminations.values()) if terminations else False
            
            # Print progress every PRINT_EVERY rounds
            if round_counter % PRINT_EVERY == 0:
                avg_reward = np.mean([episode_reward_sum[agent_id] / round_counter for agent_id in agents.keys()])
                avg_epsilon = np.mean([agent.epsilon for agent in agents.values()])
                recent_coop = np.mean(round_cooperation_rates[-PRINT_EVERY:]) if len(round_cooperation_rates) >= PRINT_EVERY else np.mean(round_cooperation_rates)
                
                print(f"Round {round_counter:4d}: "
                      f"Avg Reward/Round: {avg_reward:7.2f}, "
                      f"Recent Cooperation Rate: {recent_coop:5.2%}, "
                      f"Epsilon: {avg_epsilon:.3f}")
        
        # End episode for all agents
        for agent in agents.values():
            agent.end_episode()
    
    print("\nTraining completed!")
    
    # Calculate moving averages for each agent
    def calculate_moving_average(data, window_size):
        """Calculate moving average with given window size."""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Individual agent contributions with moving average
    plt.subplot(2, 2, 1)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (agent_id, contributions) in enumerate(round_contributions.items()):
        color = colors[i % len(colors)]
        rounds = range(1, len(contributions) + 1)
        
        # Plot raw contributions (light)
        plt.plot(rounds, contributions, color=color, alpha=0.3, linewidth=0.5, label=f'{agent_id} (raw)')
        
        # Plot moving average (bold)
        if len(contributions) >= MOVING_AVERAGE_WINDOW:
            ma_contributions = calculate_moving_average(contributions, MOVING_AVERAGE_WINDOW)
            ma_rounds = range(MOVING_AVERAGE_WINDOW, len(contributions) + 1)
            plt.plot(ma_rounds, ma_contributions, color=color, linewidth=2, label=f'{agent_id} (MA{MOVING_AVERAGE_WINDOW})')
    
    plt.xlabel('Round')
    plt.ylabel('Contribution')
    plt.title('Individual Agent Contributions Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Overall cooperation rate
    plt.subplot(2, 2, 2)
    rounds = range(1, len(round_cooperation_rates) + 1)
    plt.plot(rounds, round_cooperation_rates, 'b-', alpha=0.3, linewidth=0.5, label='Raw cooperation rate')
    
    if len(round_cooperation_rates) >= MOVING_AVERAGE_WINDOW:
        ma_coop = calculate_moving_average(round_cooperation_rates, MOVING_AVERAGE_WINDOW)
        ma_rounds = range(MOVING_AVERAGE_WINDOW, len(round_cooperation_rates) + 1)
        plt.plot(ma_rounds, ma_coop, 'b-', linewidth=2, label=f'MA{MOVING_AVERAGE_WINDOW} cooperation rate')
    
    plt.xlabel('Round')
    plt.ylabel('Cooperation Rate')
    plt.title('Overall Cooperation Rate Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Average epsilon decay
    plt.subplot(2, 2, 3)
    epsilon_values = []
    for round_num in range(1, round_counter + 1):
        # Calculate what epsilon would be at this round
        epsilon = max(EPSILON_MIN, EPSILON * (EPSILON_DECAY ** round_num))
        epsilon_values.append(epsilon)
    
    plt.plot(range(1, len(epsilon_values) + 1), epsilon_values, 'r-', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon) Decay')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final contribution distribution
    plt.subplot(2, 2, 4)
    final_contributions = []
    agent_labels = []
    
    for agent_id, contributions in round_contributions.items():
        if contributions:
            # Take average of last 100 rounds or all if less than 100
            recent_contrib = np.mean(contributions[-min(100, len(contributions)):])
            final_contributions.append(recent_contrib)
            agent_labels.append(agent_id)
    
    plt.bar(agent_labels, final_contributions, color=colors[:len(agent_labels)])
    plt.xlabel('Agent')
    plt.ylabel('Average Contribution (Last 100 rounds)')
    plt.title('Final Agent Cooperation Levels')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
    print(f"Results saved to: results/training_results.png")
    
    # Show the plot
    plt.show()
    
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
              f"Total Reward: {stats['total_reward']:.2f}, Epsilon: {stats['epsilon']:.3f}")

if __name__ == "__main__":
    main()