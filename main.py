import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from environment import TabularPublicGoodsGame
from agent import TabularQLearningAgent
from config import *

# Helper function to calculate moving average
def _calculate_moving_average(data, window_size):
    """Calculate moving average. Returns convolved data or empty array if too short."""
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Helper function to format plot labels
def _format_label(fmt_str, base_label, series_type_label, window):
    if not fmt_str:
        return None
    combined_base = base_label
    if series_type_label:
        combined_base = f"{base_label} {series_type_label}"
    
    return fmt_str.format(
        base_label=base_label, 
        series_type_label=series_type_label,
        combined_base=combined_base,
        window=window
    ).strip()

# Helper function to plot a time series with its moving average and standard deviation band
def plot_timeseries_with_ma_std(ax, series_data, window, color, base_label,
                                series_type_label="", 
                                y_max_clip=None, is_rate=False,
                                raw_plot_linewidth=1, ma_plot_linewidth=2,
                                std_alpha=0.15,
                                show_std_band=True, # Added
                                ma_label_format="{combined_base} (MA{window})",
                                std_label_format="{combined_base} ±1σ", # This is for the LABEL of the band
                                raw_label_format="{combined_base} (Raw)"
                               ):
    series_data_np = np.asarray(series_data)
    x_coords_raw = np.arange(1, len(series_data_np) + 1)

    ma_values = _calculate_moving_average(series_data_np, window)

    if ma_values.size > 0: # Successfully calculated MA
        # Calculate rolling_std_values regardless of whether it's labeled
        rolling_std_values = []
        for i in range(window - 1, len(series_data_np)):
            window_slice = series_data_np[i - window + 1 : i + 1]
            rolling_std_values.append(np.std(window_slice))
        rolling_std_values = np.array(rolling_std_values)

        ma_x_coords = np.arange(window, len(series_data_np) + 1)
        
        current_min_len = min(len(ma_values), len(rolling_std_values))
        ma_values_plot = ma_values[:current_min_len]
        rolling_std_values_plot = rolling_std_values[:current_min_len]
        ma_x_coords_plot = ma_x_coords[:current_min_len]

        if show_std_band: # Control filling the band
            lower_band = ma_values_plot - rolling_std_values_plot
            upper_band = ma_values_plot + rolling_std_values_plot

            if is_rate:
                lower_band = np.maximum(0, lower_band)
                upper_band = np.minimum(1, upper_band)
            else:
                lower_band = np.maximum(0, lower_band)
                if y_max_clip is not None:
                    upper_band = np.minimum(y_max_clip, upper_band)
            
            # Get the label for the std band, will be None if std_label_format is None/empty
            actual_std_legend_label = _format_label(std_label_format, base_label, series_type_label, window)
            ax.fill_between(
                ma_x_coords_plot, lower_band, upper_band,
                alpha=std_alpha, color=color, label=actual_std_legend_label
            )
        
        actual_ma_legend_label = _format_label(ma_label_format, base_label, series_type_label, window)
        ax.plot(
            ma_x_coords_plot, ma_values_plot,
            color=color, linewidth=ma_plot_linewidth, label=actual_ma_legend_label
        )
    elif len(series_data_np) > 0: # Fallback to raw data
        actual_raw_legend_label = _format_label(raw_label_format, base_label, series_type_label, window)
        ax.plot(
            x_coords_raw, series_data_np,
            color=color, linewidth=raw_plot_linewidth, alpha=0.6, label=actual_raw_legend_label
        )

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
    episode_rewards = defaultdict(list)
    round_contributions = defaultdict(list)  # Track individual agent contributions per round
    round_agent_rewards = defaultdict(list)  # Track individual agent rewards per round
    round_cooperation_rates = []
    
    # Training loop
    round_counter = 0
    for episode in range(N_EPISODES):
        observations = env.reset()
        cum_rewards = defaultdict(float)
        
        # Reset agents for new episode
        for agent in agents.values():
            # agent.last_state = None # No longer needed
            # agent.last_action = None # No longer needed
            pass # Agent reset is handled internally if needed, or not at all for these attributes
        
        # Run episode
        done = False
        while not done:
            round_counter += 1
            
            # Get actions from all agents
            actions = {}
            for agent_id, agent in agents.items():
                current_state = observations[agent_id]
                action = agent.act(current_state)
                actions[agent_id] = action            
            # Step environment
            next_observations, rewards, terminations, truncations, info = env.step(actions)
            
            for agent_id, agent in agents.items():
                action = actions[agent_id]  
                round_contributions[agent_id].append(action) # Store contribution for this agent this round
                
                reward = rewards.get(agent_id)
                next_obs = next_observations.get(agent_id)
                done_agent = terminations.get(agent_id, False)
                
                round_agent_rewards[agent_id].append(reward) # Store reward for this agent this round
                
                # Learn from this step
                current_obs= observations[agent_id]
                agent.learn(current_obs, action, reward, next_obs, done_agent)
                cum_rewards[agent_id] += reward

            # Track overall cooperation rate
            total_contribution = info[list(info.keys())[0]]["total_contribution"]
            max_possible = N_AGENTS * INITIAL_ENDOWMENT
            cooperation_rate = total_contribution / max_possible
            round_cooperation_rates.append(cooperation_rate)
            
            observations = next_observations
            done = all(terminations.values()) if terminations else False
            
            # Print progress every PRINT_EVERY rounds
            if round_counter % PRINT_EVERY == 0:
                avg_reward = np.mean([cum_rewards[agent_id] / round_counter for agent_id in agents.keys()])
                recent_coop = np.mean(round_cooperation_rates[-PRINT_EVERY:]) if len(round_cooperation_rates) >= PRINT_EVERY else np.mean(round_cooperation_rates)
                
                print(f"Round {round_counter:4d}: "
                      f"Avg Reward/Round: {avg_reward:7.2f}, "
                      f"Recent Cooperation Rate: {recent_coop:5.2%}, "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        # End episode for all agents
        for agent in agents.values():
            agent.end_episode()
    
    print("\nTraining completed!")
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Define colors for agents
    if N_AGENTS <= 10:
        agent_colors = plt.colormaps.get_cmap('tab10').colors
    else:
        agent_colors = plt.colormaps.get_cmap('tab20').colors

    # Store handles for the legend
    legend_handles = []
    legend_labels = []

    # Plot 1: Individual agent contributions
    for i, (agent_id, contributions) in enumerate(round_contributions.items()):
        color = agent_colors[i % len(agent_colors)]
        plot_timeseries_with_ma_std(
            ax=ax1,
            series_data=contributions,
            window=MOVING_AVERAGE_WINDOW,
            color=color,
            base_label=agent_id, # Used for MA line label
            series_type_label=" Contribution", # Added for clarity if needed, but MA label format overrides
            y_max_clip=INITIAL_ENDOWMENT,
            is_rate=False,
            ma_plot_linewidth=2,
            show_std_band=True, # Show std band for contributions
            ma_label_format="{base_label}", # MA line label will be just agent_id
            std_label_format=None, # No separate legend entry for std band
            raw_label_format="{base_label} (Raw)"
        )
        # Add to legend only once per agent
        if agent_id not in legend_labels:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2))
            legend_labels.append(agent_id)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Contribution')
    ax1.set_title('Individual Agent Contributions')
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend will be added globally
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, INITIAL_ENDOWMENT)
    
    # Plot 2: Overall cooperation rate with std bands and Epsilon Decay
    plot_timeseries_with_ma_std(
        ax=ax2,
        series_data=round_cooperation_rates,
        window=MOVING_AVERAGE_WINDOW,
        color='gray',
        base_label="Cooperation Rate",
        series_type_label="",
        is_rate=True,
        ma_plot_linewidth=3,
        show_std_band=True,
        ma_label_format="{base_label} (MA{window})",
        std_label_format="{base_label} ±1σ", # Label for std band
        raw_label_format="{base_label}"
    )

    # Calculate and plot Epsilon Decay on the same axes
    epsilon_values = []
    for round_num in range(1, round_counter + 1):
        epsilon = max(EPSILON_MIN, EPSILON * (EPSILON_DECAY ** round_num))
        epsilon_values.append(epsilon)
    
    ax2.plot(range(1, len(epsilon_values) + 1), epsilon_values, 'red', linewidth=2, linestyle='--', label='Epsilon Decay')
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Rate / Value') # Adjusted Y-axis label
    ax2.set_title('Overall Cooperation Rate & Epsilon Decay')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Individual agent rewards with their own std bands
    for i, (agent_id, agent_rewards_history) in enumerate(round_agent_rewards.items()):
        color = agent_colors[i % len(agent_colors)]
        plot_timeseries_with_ma_std(
            ax=ax3,
            series_data=agent_rewards_history,
            window=MOVING_AVERAGE_WINDOW,
            color=color,
            base_label=agent_id, # Used for MA line label
            series_type_label=" Reward", # Added for clarity if needed
            is_rate=False,
            ma_plot_linewidth=2,
            show_std_band=False, # Now show std band for rewards
            ma_label_format="{base_label}", # MA line label will be just agent_id
            std_label_format=None, # No separate legend entry for std band
            raw_label_format="{combined_base} (Raw)"
        )

    ax3.set_xlabel('Round')
    ax3.set_ylabel('Reward')
    ax3.set_title('Individual Agent Rewards') # Title updated
    # ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend will be added globally
    ax3.grid(True, alpha=0.3)

    # Plot 4: Average reward across all agents with std bands
    ax4 = plt.subplot(2, 2, 4)

    max_reward_rounds = 0
    if round_agent_rewards:
        # Ensure that we only consider lists that are not empty
        agent_reward_lengths = [len(rewards_list) for rewards_list in round_agent_rewards.values() if rewards_list]
        if agent_reward_lengths: # Check if there are any non-empty reward lists
            max_reward_rounds = max(agent_reward_lengths)

    average_round_rewards_data = []
    # std_average_round_rewards_data = [] # No longer needed for this plot's band calculation

    if max_reward_rounds > 0: # Proceed only if there is data
        for round_idx in range(max_reward_rounds):
            current_round_agent_rewards_list = []
            for agent_id_key in agents.keys(): 
                if round_idx < len(round_agent_rewards[agent_id_key]):
                    current_round_agent_rewards_list.append(round_agent_rewards[agent_id_key][round_idx])
            
            if current_round_agent_rewards_list:
                average_round_rewards_data.append(np.mean(current_round_agent_rewards_list))
                # std_average_round_rewards_data.append(np.std(current_round_agent_rewards_list)) # This line is removed
            else:
                # Append NaN if no data for this round to maintain array length
                average_round_rewards_data.append(np.nan)
                # std_average_round_rewards_data.append(np.nan) # This line is removed
            
    average_round_rewards_data_np = np.array(average_round_rewards_data)
    # std_average_round_rewards_data_np = np.array(std_average_round_rewards_data) # This line is removed

    # Filter out NaNs that might have been introduced if some agents finish earlier or data is sparse
    valid_indices_avg_reward = ~np.isnan(average_round_rewards_data_np)
    # plot_rounds_avg_reward_x = np.arange(1, len(average_round_rewards_data_np) + 1)[valid_indices_avg_reward] # x-coords handled by helper
    plot_avg_rewards_y = average_round_rewards_data_np[valid_indices_avg_reward]
    # plot_std_avg_rewards_y = std_average_round_rewards_data_np[valid_indices_avg_reward] # This line is removed

    # The plot_timeseries_with_ma_std function will now handle MA and rolling std for plot_avg_rewards_y
    if len(plot_avg_rewards_y) > 0:
        plot_timeseries_with_ma_std(
            ax=ax4,
            series_data=plot_avg_rewards_y,
            window=MOVING_AVERAGE_WINDOW,
            color='purple',
            base_label="Avg Reward",
            series_type_label="", # Base label is descriptive enough
            y_max_clip=None, # Rewards don't have a natural upper clip like rates or endowments here
            is_rate=False,
            raw_plot_linewidth=2, # Matches previous raw plot style
            ma_plot_linewidth=3,  # Matches previous MA plot style
            std_alpha=0.2,        # Matches previous MA band alpha for consistency
            ma_label_format="{base_label} (MA{window})",
            std_label_format="{base_label} ±1σ", # Std dev of the average reward series itself
            raw_label_format="{base_label} (Raw)"
        )
    # The old custom plotting logic for ax4 (if/elif based on MOVING_AVERAGE_WINDOW and ma_std_for_band) is now replaced.

    ax4.set_xlabel('Round')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Average Reward Across All Agents')
    ax4.legend(loc='best') # Keep this legend for plot-specific items like "Avg Reward"
    ax4.grid(True, alpha=0.3)

    # Add a single figure-level legend for agent colors
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=min(len(legend_labels), 5), bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for fig legend
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
    print(f"Results saved to: results/training_results.png")
    
    # Show the plot
    # plt.show()
    
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
              f"Total Reward: {stats['total_reward']:.2f},Average Reward: {stats['average_reward']:.2f} Epsilon: {stats['epsilon']:.3f}")

if __name__ == "__main__":
    main()