import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from collections import defaultdict
import os
from agent import calculate_epsilon


def _calculate_moving_average(data, window_size):
    """Calculate moving average."""
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def _format_label(fmt_str, base_label, series_type_label, window):
    """Helper function to format plot labels."""
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


def plot_timeseries_with_ma_std(ax, series_data, window, color, base_label,
                                series_type_label="", 
                                y_max_clip=None, is_rate=False,
                                raw_plot_linewidth=1, ma_plot_linewidth=2,
                                std_alpha=0.15,
                                show_std_band=True,
                                ma_label_format="{combined_base} (MA{window})",
                                std_label_format="{combined_base} ±1σ",
                                raw_label_format="{combined_base} (Raw)"):
    """
    Helper function to plot a time series with its moving average and standard deviation band.
    """
    series_data_np = np.asarray(series_data)
    x_coords_raw = np.arange(1, len(series_data_np) + 1)

    ma_values = _calculate_moving_average(series_data_np, window)

    if ma_values.size > 0:  # Successfully calculated MA
        # Calculate rolling_std_values
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

        if show_std_band:  # Control filling the band
            lower_band = ma_values_plot - rolling_std_values_plot
            upper_band = ma_values_plot + rolling_std_values_plot

            if is_rate:
                lower_band = np.maximum(0, lower_band)
                upper_band = np.minimum(1, upper_band)
            else:
                lower_band = np.maximum(0, lower_band)
                if y_max_clip is not None:
                    upper_band = np.minimum(y_max_clip, upper_band)
            
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
    elif len(series_data_np) > 0:  # Fallback to raw data
        actual_raw_legend_label = _format_label(raw_label_format, base_label, series_type_label, window)
        ax.plot(
            x_coords_raw, series_data_np,
            color=color, linewidth=raw_plot_linewidth, alpha=0.6, label=actual_raw_legend_label
        )


def create_training_plots(round_contributions: Dict[str, List], 
                         round_agent_rewards: Dict[str, List],
                         round_cooperation_rates: List[float],
                         round_counter: int,
                         agents: Dict,
                         config_params: Dict,
                         save_path: str = 'results/training_results.png'):
    """
    Create training plots.
    
    Args:
        round_contributions: Dictionary of agent contributions per round
        round_agent_rewards: Dictionary of agent rewards per round
        round_cooperation_rates: List of cooperation rates per round
        round_counter: Total number of rounds
        agents: Dictionary of agents
        config_params: Configuration parameters (MOVING_AVERAGE_WINDOW, etc.)
        save_path: Path to save the plot
    """
    # Extract config parameters
    moving_average_window = config_params['MOVING_AVERAGE_WINDOW']
    initial_endowment = config_params['INITIAL_ENDOWMENT']
    n_agents = config_params['N_AGENTS']
    epsilon = config_params['EPSILON']
    epsilon_decay = config_params['EPSILON_DECAY']
    epsilon_min = config_params['EPSILON_MIN']
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Define colors for agents
    if n_agents <= 10:
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
            window=moving_average_window,
            color=color,
            base_label=agent_id,
            series_type_label=" Contribution",
            y_max_clip=initial_endowment,
            is_rate=False,
            ma_plot_linewidth=2,
            show_std_band=True,
            ma_label_format="{base_label}",
            std_label_format=None,
            raw_label_format="{base_label} (Raw)"
        )
        # Add to legend only once per agent
        if agent_id not in legend_labels:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2))
            legend_labels.append(agent_id)
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Contribution')
    ax1.set_title('Individual Agent Contributions')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, initial_endowment)
    
    # Plot 2: Overall cooperation rate with Epsilon Decay
    plot_timeseries_with_ma_std(
        ax=ax2,
        series_data=round_cooperation_rates,
        window=moving_average_window,
        color='gray',
        base_label="Cooperation Rate",
        series_type_label="",
        is_rate=True,
        ma_plot_linewidth=3,
        show_std_band=True,
        ma_label_format="{base_label} (MA{window})",
        std_label_format="{base_label} ±1σ",
        raw_label_format="{base_label}"
    )

    # Calculate and plot Epsilon Decay using centralized function
    epsilon_values = []
    epsilon_val=epsilon
    for round_num in range(1, round_counter + 1):
        epsilon_val = calculate_epsilon(epsilon_val, epsilon_decay, epsilon_min)
        epsilon_values.append(epsilon_val)
    
    ax2.plot(range(1, len(epsilon_values) + 1), epsilon_values, 'red', 
             linewidth=2, linestyle='--', label='Epsilon Decay')
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Rate / Value')
    ax2.set_title('Overall Cooperation Rate & Epsilon Decay')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Individual agent rewards
    for i, (agent_id, agent_rewards_history) in enumerate(round_agent_rewards.items()):
        color = agent_colors[i % len(agent_colors)]
        plot_timeseries_with_ma_std(
            ax=ax3,
            series_data=agent_rewards_history,
            window=moving_average_window,
            color=color,
            base_label=agent_id,
            series_type_label=" Reward",
            is_rate=False,
            ma_plot_linewidth=2,
            show_std_band=False,
            ma_label_format="{base_label}",
            std_label_format=None,
            raw_label_format="{combined_base} (Raw)"
        )

    ax3.set_xlabel('Round')
    ax3.set_ylabel('Reward')
    ax3.set_title('Individual Agent Rewards')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Average reward across all agents
    max_reward_rounds = 0
    if round_agent_rewards:
        agent_reward_lengths = [len(rewards_list) for rewards_list in round_agent_rewards.values() if rewards_list]
        if agent_reward_lengths:
            max_reward_rounds = max(agent_reward_lengths)

    average_round_rewards_data = []

    if max_reward_rounds > 0:
        for round_idx in range(max_reward_rounds):
            current_round_agent_rewards_list = []
            for agent_id_key in agents.keys(): 
                if round_idx < len(round_agent_rewards[agent_id_key]):
                    current_round_agent_rewards_list.append(round_agent_rewards[agent_id_key][round_idx])
            
            if current_round_agent_rewards_list:
                average_round_rewards_data.append(np.mean(current_round_agent_rewards_list))
            else:
                average_round_rewards_data.append(np.nan)
            
    average_round_rewards_data_np = np.array(average_round_rewards_data)

    # Filter out NaNs
    valid_indices_avg_reward = ~np.isnan(average_round_rewards_data_np)
    plot_avg_rewards_y = average_round_rewards_data_np[valid_indices_avg_reward]

    if len(plot_avg_rewards_y) > 0:
        plot_timeseries_with_ma_std(
            ax=ax4,
            series_data=plot_avg_rewards_y,
            window=moving_average_window,
            color='purple',
            base_label="Avg Reward",
            series_type_label="",
            y_max_clip=None,
            is_rate=False,
            raw_plot_linewidth=2,
            ma_plot_linewidth=3,
            std_alpha=0.2,
            ma_label_format="{base_label} (MA{window})",
            std_label_format="{base_label} ±1σ",
            raw_label_format="{base_label} (Raw)"
        )

    ax4.set_xlabel('Round')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Average Reward Across All Agents')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # Add a single figure-level legend for agent colors
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', 
                  ncol=min(len(legend_labels), 5), bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to: {save_path}")
    
    # Show the plot
    plt.show()
