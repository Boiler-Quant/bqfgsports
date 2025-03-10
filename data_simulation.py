import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns

# Set the path to your CSV file
CSV_PATH = "/Users/jamieborst/Documents/Purdue Senior Year/BQFG/arbitrage_results_with_ev_and_kelly.csv"

def validate_data(data):
    """Validate the CSV data and print basic statistics"""
    print(f"Loaded {len(data)} betting opportunities")
    
    # Count arbitrage opportunities
    arb_opps = data[data['Bet on Over'].notna() & data['Bet on Under'].notna()]
    print(f"Found {len(arb_opps)} arbitrage opportunities")
    
    # Convert percentage columns to numeric before comparison
    data['Over Edge (%)'] = pd.to_numeric(data['Over Edge (%)'], errors='coerce')
    data['Under Edge (%)'] = pd.to_numeric(data['Under Edge (%)'], errors='coerce')
    
    # Count positive EV opportunities
    pos_ev_over = data[(data['Over Edge (%)'].notna()) & (data['Over Edge (%)'] > 0)]
    pos_ev_under = data[(data['Under Edge (%)'].notna()) & (data['Under Edge (%)'] > 0)]
    print(f"Found {len(pos_ev_over)} positive EV over opportunities")
    print(f"Found {len(pos_ev_under)} positive EV under opportunities")

def odds_to_implied_probability(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def simulate_outcome(probability):
    """Simulate a binary outcome based on probability"""
    return random.random() < probability

def calculate_profit(bet, odds, outcome):
    """Calculate profit from a bet given the odds and outcome"""
    if outcome:
        # Convert to decimal odds for calculation
        decimal_odds = odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1
        return bet * (decimal_odds - 1)
    else:
        return -bet

def process_bankroll_histories(bankroll_histories):
    """Calculate statistics from bankroll histories"""
    if not bankroll_histories:
        return [], {'p10': [], 'p25': [], 'p50': [], 'p75': [], 'p90': []}
    
    # Find the maximum length of any bankroll history
    max_length = max(len(history) for history in bankroll_histories)
    
    # Pad shorter histories with their last value
    padded_histories = []
    for history in bankroll_histories:
        padded = history.copy()
        if len(padded) < max_length:
            padded.extend([padded[-1]] * (max_length - len(padded)))
        padded_histories.append(padded)
    
    # Calculate the average at each step
    avg_history = []
    for i in range(max_length):
        avg_value = sum(history[i] for history in padded_histories) / len(padded_histories)
        avg_history.append(avg_value)
    
    # Calculate percentiles at each step
    percentiles = {'p10': [], 'p25': [], 'p50': [], 'p75': [], 'p90': []}
    
    for i in range(max_length):
        values_at_step = [history[i] for history in padded_histories]
        percentiles['p10'].append(np.percentile(values_at_step, 10))
        percentiles['p25'].append(np.percentile(values_at_step, 25))
        percentiles['p50'].append(np.percentile(values_at_step, 50))
        percentiles['p75'].append(np.percentile(values_at_step, 75))
        percentiles['p90'].append(np.percentile(values_at_step, 90))
    
    return avg_history, percentiles

def calculate_statistics(run_profits):
    """Calculate statistics from run profits"""
    sorted_profits = sorted(run_profits)
    avg_profit = sum(run_profits) / len(run_profits)
    profitable_runs_pct = (sum(1 for p in run_profits if p > 0) / len(run_profits)) * 100
    std_dev = np.std(run_profits)
    
    # Avoid division by zero
    risk_adjusted_return = float('inf') if std_dev == 0 else avg_profit / std_dev
    
    percentiles = {
        'min': sorted_profits[0],
        'p25': np.percentile(sorted_profits, 25),
        'median': np.percentile(sorted_profits, 50),
        'p75': np.percentile(sorted_profits, 75),
        'max': sorted_profits[-1]
    }
    
    return avg_profit, profitable_runs_pct, std_dev, risk_adjusted_return, percentiles

def print_strategy_results(strategy_name, total_runs, total_bets, run_profits):
    """Print formatted simulation results"""
    avg_bets_per_run = total_bets / total_runs
    avg_profit, profitable_runs_pct, std_dev, risk_adjusted, percentiles = calculate_statistics(run_profits)
    
    print(f"\n{strategy_name.upper()} STRATEGY:")
    print(f"Average bets per simulation: {avg_bets_per_run:.2f}")
    print(f"Average profit percentage: {avg_profit:.2f}%")
    print(f"Percentage of profitable runs: {profitable_runs_pct:.2f}%")
    print(f"Standard deviation: {std_dev:.2f}%")
    print("Profit percentiles:")
    print(f"  Min: {percentiles['min']:.2f}%")
    print(f"  25th: {percentiles['p25']:.2f}%")
    print(f"  Median: {percentiles['median']:.2f}%")
    print(f"  75th: {percentiles['p75']:.2f}%")
    print(f"  Max: {percentiles['max']:.2f}%")
    
    return avg_profit, std_dev, risk_adjusted

def simulate_arbitrage(data, num_simulations, track_bankroll=True):
    """Simulate arbitrage betting strategy"""
    total_bets = 0
    run_profits = []
    bankroll_histories = []
    
    for _ in range(num_simulations):
        run_profit = 0
        run_bets = 0
        
        # Initialize starting bankroll at 100%
        bankroll = 100.0
        bankroll_history = [bankroll]
        
        for _, bet in data.iterrows():
            # Check if this is an arbitrage opportunity
            if pd.notna(bet['Bet on Over']) and pd.notna(bet['Bet on Under']):
                run_bets += 1
                
                # For arbitrage, the profit should be guaranteed regardless of outcome
                # Convert to numeric in case they're strings
                over_profit = pd.to_numeric(bet['% Profit if Over Hits'], errors='coerce')
                under_profit = pd.to_numeric(bet['% Profit if Under Hits'], errors='coerce')
                
                # Skip if conversion resulted in NaN
                if pd.isna(over_profit) or pd.isna(under_profit):
                    continue
                
                # Determine fair probability from fair odds
                fair_odds_over = pd.to_numeric(bet['Fair Odds Over'], errors='coerce')
                if pd.isna(fair_odds_over):
                    continue
                
                fair_prob_over = odds_to_implied_probability(fair_odds_over)
                
                # Simulate the outcome based on fair probability
                over_outcome = simulate_outcome(fair_prob_over)
                
                # Determine profit for this bet
                bet_profit = over_profit if over_outcome else under_profit
                
                # Update run profit and bankroll
                run_profit += bet_profit
                bankroll += bet_profit
                
                # Track bankroll evolution if requested
                if track_bankroll:
                    bankroll_history.append(bankroll)
        
        total_bets += run_bets
        run_profits.append(run_profit)
        
        if track_bankroll:
            bankroll_histories.append(bankroll_history)
    
    return total_bets, run_profits, bankroll_histories

def simulate_positive_ev(data, num_simulations, kelly_fraction=1.0, track_bankroll=True):
    """Simulate positive EV betting strategy using Kelly criterion with bankruptcy check
    
    Parameters:
    data (DataFrame): Betting data
    num_simulations (int): Number of simulation runs
    kelly_fraction (float): Fraction of the Kelly bet to use (default=1.0 for full Kelly)
    track_bankroll (bool): Whether to track bankroll history
    
    Returns:
    tuple: (total_bets, run_profits, bankroll_histories)
    """
    total_bets = 0
    run_profits = []
    bankroll_histories = []
    
    for _ in range(num_simulations):
        run_profit = 0
        run_bets = 0
        
        # Initialize starting bankroll at 100%
        bankroll = 100.0
        bankroll_history = [bankroll]
        
        for _, bet in data.iterrows():
            # Stop betting if bankrupt (bankroll <= 0 or reaching -100% loss)
            if bankroll <= 0:
                break
            
            # Check if this is a positive EV opportunity (either over or under)
            has_positive_ev = False
            
            # Ensure numeric values for comparison
            over_edge = pd.to_numeric(bet['Over Edge (%)'], errors='coerce')
            under_edge = pd.to_numeric(bet['Under Edge (%)'], errors='coerce')
            
            # Check for positive Over EV
            if pd.notna(over_edge) and over_edge > 0:
                has_positive_ev = True
                run_bets += 1
                
                kelly_bet = pd.to_numeric(bet['Kelly Over Bet (% of bankroll)'], errors='coerce')
                over_odds = pd.to_numeric(bet['Best Over Odds'], errors='coerce')
                fair_odds_over = pd.to_numeric(bet['Fair Odds Over'], errors='coerce')
                
                # Skip if any conversion resulted in NaN
                if pd.isna(kelly_bet) or pd.isna(over_odds) or pd.isna(fair_odds_over):
                    continue
                
                # Apply Kelly fraction
                kelly_bet = kelly_bet * kelly_fraction
                
                fair_prob_over = odds_to_implied_probability(fair_odds_over)
                over_outcome = simulate_outcome(fair_prob_over)
                
                # Calculate profit for this bet (adjusted for current bankroll)
                bet_size_pct = kelly_bet
                bet_profit = calculate_profit(bet_size_pct, over_odds, over_outcome)
                
                # Update run profit and bankroll
                run_profit += bet_profit
                bankroll += bet_profit
                
                # Check for bankruptcy after bet
                if bankroll <= 0:
                    # Set bankroll to 0 to represent total loss
                    bankroll = 0
                    run_profit = -100
                    
                    # Track bankroll evolution if requested
                    if track_bankroll:
                        bankroll_history.append(bankroll)
                    break
                
                # Track bankroll evolution if requested
                if track_bankroll:
                    bankroll_history.append(bankroll)
            
            # Check for positive Under EV
            if pd.notna(under_edge) and under_edge > 0:
                # Only count as a new bet if we haven't already counted the Over
                if not has_positive_ev:
                    run_bets += 1
                    has_positive_ev = True
                
                kelly_bet = pd.to_numeric(bet['Kelly Under Bet (% of bankroll)'], errors='coerce')
                under_odds = pd.to_numeric(bet['Best Under Odds'], errors='coerce')
                fair_odds_under = pd.to_numeric(bet['Fair Odds Under'], errors='coerce')
                
                # Skip if any conversion resulted in NaN
                if pd.isna(kelly_bet) or pd.isna(under_odds) or pd.isna(fair_odds_under):
                    continue
                
                # Apply Kelly fraction
                kelly_bet = kelly_bet * kelly_fraction
                
                fair_prob_under = odds_to_implied_probability(fair_odds_under)
                under_outcome = simulate_outcome(fair_prob_under)
                
                # Calculate profit for this bet (adjusted for current bankroll)
                bet_size_pct = kelly_bet
                bet_profit = calculate_profit(bet_size_pct, under_odds, under_outcome)
                
                # Update run profit and bankroll
                run_profit += bet_profit
                bankroll += bet_profit
                
                # Check for bankruptcy after bet
                if bankroll <= 0:
                    # Set bankroll to 0 to represent total loss
                    bankroll = 0
                    run_profit = -100
                    
                    # Track bankroll evolution if requested
                    if track_bankroll:
                        bankroll_history.append(bankroll)
                    break
                
                # Track bankroll evolution if requested
                if track_bankroll:
                    bankroll_history.append(bankroll)
        
        total_bets += run_bets
        run_profits.append(run_profit)
        
        if track_bankroll:
            bankroll_histories.append(bankroll_history)
    
    return total_bets, run_profits, bankroll_histories

def plot_kelly_comparison_results(arb_results, ev_results_full, ev_results_half, ev_results_quarter, output_path='/Users/jamieborst/Documents/Purdue Senior Year/BQFG/betting_strategy_comparison.png'):
    """Create visualizations comparing all strategies including different Kelly fractions
    
    Parameters:
    arb_results, ev_results_full, ev_results_half, ev_results_quarter: Simulation results
    output_path (str): Path where the visualization should be saved
    """
    arb_total_bets, arb_run_profits, arb_bankroll_histories = arb_results
    
    ev_full_total_bets, ev_full_run_profits, ev_full_bankroll_histories = ev_results_full
    ev_half_total_bets, ev_half_run_profits, ev_half_bankroll_histories = ev_results_half
    ev_quarter_total_bets, ev_quarter_run_profits, ev_quarter_bankroll_histories = ev_results_quarter
    
    # Calculate statistics
    arb_avg_profit, _, arb_std_dev, arb_risk_adjusted, _ = calculate_statistics(arb_run_profits)
    
    ev_full_avg_profit, _, ev_full_std_dev, ev_full_risk_adjusted, _ = calculate_statistics(ev_full_run_profits)
    ev_half_avg_profit, _, ev_half_std_dev, ev_half_risk_adjusted, _ = calculate_statistics(ev_half_run_profits)
    ev_quarter_avg_profit, _, ev_quarter_std_dev, ev_quarter_risk_adjusted, _ = calculate_statistics(ev_quarter_run_profits)
    
    # Process bankroll histories
    arb_avg_history, arb_percentiles = process_bankroll_histories(arb_bankroll_histories)
    
    ev_full_avg_history, ev_full_percentiles = process_bankroll_histories(ev_full_bankroll_histories)
    ev_half_avg_history, ev_half_percentiles = process_bankroll_histories(ev_half_bankroll_histories)
    ev_quarter_avg_history, ev_quarter_percentiles = process_bankroll_histories(ev_quarter_bankroll_histories)
    
    plt.figure(figsize=(20, 16))
    
    # Plot 1: Profit distribution
    plt.subplot(2, 3, 1)
    sns.histplot(arb_run_profits, kde=True, label='Arbitrage', alpha=0.4, color='blue')
    sns.histplot(ev_full_run_profits, kde=True, label='Full Kelly', alpha=0.4, color='red')
    sns.histplot(ev_half_run_profits, kde=True, label='Half Kelly', alpha=0.4, color='green')
    sns.histplot(ev_quarter_run_profits, kde=True, label='Quarter Kelly', alpha=0.4, color='purple')
    plt.title('Profit Distribution')
    plt.xlabel('Profit Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Box plot comparison
    plt.subplot(2, 3, 2)
    data = [arb_run_profits, ev_full_run_profits, ev_half_run_profits, ev_quarter_run_profits]
    plt.boxplot(data, tick_labels=['Arbitrage', 'Full Kelly', 'Half Kelly', 'Quarter Kelly'])
    plt.title('Profit Comparison')
    plt.ylabel('Profit Percentage')
    
    # Plot 3: Risk-adjusted return
    plt.subplot(2, 3, 3)
    labels = ['Arbitrage', 'Full Kelly', 'Half Kelly', 'Quarter Kelly']
    returns = [arb_avg_profit, ev_full_avg_profit, ev_half_avg_profit, ev_quarter_avg_profit]
    stds = [arb_std_dev, ev_full_std_dev, ev_half_std_dev, ev_quarter_std_dev]
    risk_adjusted = [arb_risk_adjusted, ev_full_risk_adjusted, ev_half_risk_adjusted, ev_quarter_risk_adjusted]
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.bar(x - width, returns, width, label='Avg Return %')
    plt.bar(x, stds, width, label='Std Dev %')
    plt.bar(x + width, risk_adjusted, width, label='Risk-Adjusted Return')
    plt.title('Return and Risk Comparison')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    
    # Plot 4: Average Bankroll Evolution
    plt.subplot(2, 3, 4)
    x_arb = np.arange(len(arb_avg_history))
    plt.plot(x_arb, arb_avg_history, label='Arbitrage', color='blue')
    plt.plot(np.arange(len(ev_full_avg_history)), ev_full_avg_history, label='Full Kelly', color='red')
    plt.plot(np.arange(len(ev_half_avg_history)), ev_half_avg_history, label='Half Kelly', color='green')
    plt.plot(np.arange(len(ev_quarter_avg_history)), ev_quarter_avg_history, label='Quarter Kelly', color='purple')
    plt.title('Average Bankroll Evolution')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll (%)')
    plt.legend()
    
    # Plot 5: Final Bankroll Distribution
    plt.subplot(2, 3, 5)
    # Get the final bankroll values for each strategy
    arb_final_bankrolls = [history[-1] for history in arb_bankroll_histories]
    ev_full_final_bankrolls = [history[-1] for history in ev_full_bankroll_histories]
    ev_half_final_bankrolls = [history[-1] for history in ev_half_bankroll_histories]
    ev_quarter_final_bankrolls = [history[-1] for history in ev_quarter_bankroll_histories]
    
    sns.kdeplot(arb_final_bankrolls, label='Arbitrage', color='blue', warn_singular=False)
    sns.kdeplot(ev_full_final_bankrolls, label='Full Kelly', color='red')
    sns.kdeplot(ev_half_final_bankrolls, label='Half Kelly', color='green')
    sns.kdeplot(ev_quarter_final_bankrolls, label='Quarter Kelly', color='purple')
    plt.title('Final Bankroll Distribution')
    plt.xlabel('Final Bankroll (%)')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 6: Bankruptcy Rate
    plt.subplot(2, 3, 6)
    bankruptcy_rates = [
        sum(1 for b in arb_final_bankrolls if b <= 0) / len(arb_final_bankrolls) * 100,
        sum(1 for b in ev_full_final_bankrolls if b <= 0) / len(ev_full_final_bankrolls) * 100,
        sum(1 for b in ev_half_final_bankrolls if b <= 0) / len(ev_half_final_bankrolls) * 100,
        sum(1 for b in ev_quarter_final_bankrolls if b <= 0) / len(ev_quarter_final_bankrolls) * 100
    ]
    
    plt.bar(labels, bankruptcy_rates)
    plt.title('Bankruptcy Rate')
    plt.ylabel('Percentage of Runs (%)')
    plt.ylim(0, max(bankruptcy_rates) * 1.2 if max(bankruptcy_rates) > 0 else 5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved as '{output_path}'")

def export_results_to_csv(strategy_names, stats, output_path):
    """Export simulation statistics to a CSV file
    
    Parameters:
    strategy_names (list): List of strategy names
    stats (list): List of tuples containing (avg_profit, std_dev, risk_adjusted)
    output_path (str): Path to save the CSV file
    """
    # Create a DataFrame with all statistics
    results_df = pd.DataFrame({
        'Strategy': strategy_names,
        'Average Profit (%)': [stat[0] for stat in stats],
        'Standard Deviation (%)': [stat[1] for stat in stats],
        'Risk-Adjusted Return': [stat[2] for stat in stats]
    })
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def collect_detailed_stats(strategy_name, num_runs, total_bets, run_profits):
    """Collect detailed statistics for CSV output without printing"""
    avg_bets_per_run = total_bets / num_runs
    avg_profit, profitable_runs_pct, std_dev, risk_adjusted, percentiles = calculate_statistics(run_profits)
    
    # Return as dictionary for easier CSV conversion
    return {
        'Strategy': strategy_name,
        'Average Bets Per Run': avg_bets_per_run,
        'Average Profit (%)': avg_profit,
        'Profitable Runs (%)': profitable_runs_pct,
        'Standard Deviation (%)': std_dev,
        'Risk-Adjusted Return': risk_adjusted,
        'Min Profit (%)': percentiles['min'],
        'P25 Profit (%)': percentiles['p25'],
        'Median Profit (%)': percentiles['median'],
        'P75 Profit (%)': percentiles['p75'],
        'Max Profit (%)': percentiles['max']
    }

def main():
    # Define output paths
    import os.path
    # Get user's documents folder - works cross-platform
    if os.name == 'nt':  # Windows
        documents_folder = os.path.join(os.path.expanduser('~'), 'Documents')
    else:  # macOS/Linux
        documents_folder = os.path.join(os.path.expanduser('~'), 'Documents')
        
    # Create output paths
    results_csv_path = os.path.join(documents_folder, '/Users/jamieborst/Documents/Purdue Senior Year/BQFG/betting_strategy_results.csv')
    detailed_results_csv_path = os.path.join(documents_folder, '/Users/jamieborst/Documents/Purdue Senior Year/BQFG/betting_strategy_detailed_results.csv')
    kelly_plot_path = os.path.join(documents_folder, '/Users/jamieborst/Documents/Purdue Senior Year/BQFG/betting_strategy_comparison.png')
    
    # Load the data
    data = pd.read_csv(CSV_PATH)
    
    # Convert numeric columns to proper type at the start
    numeric_columns = [
        'Over Edge (%)', 'Under Edge (%)', 
        'Fair Odds Over', 'Fair Odds Under', 
        'Best Over Odds', 'Best Under Odds',
        'Kelly Over Bet (% of bankroll)', 'Kelly Under Bet (% of bankroll)',
        '% Profit if Over Hits', '% Profit if Under Hits'
    ]
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    validate_data(data)
    
    # Number of simulations to run
    num_simulations = 10000
    print(f"Running {num_simulations} simulations for each strategy...")
    
    # Run simulations
    print("Simulating Arbitrage strategy...")
    arb_results = simulate_arbitrage(data, num_simulations)
    
    print("Simulating Full Kelly (100%) strategy...")
    ev_full_results = simulate_positive_ev(data, num_simulations, kelly_fraction=1.0)
    
    print("Simulating Half Kelly (50%) strategy...")
    ev_half_results = simulate_positive_ev(data, num_simulations, kelly_fraction=0.5)
    
    print("Simulating Quarter Kelly (25%) strategy...")
    ev_quarter_results = simulate_positive_ev(data, num_simulations, kelly_fraction=0.25)
    
    # Print results to console for reference
    print(f"\n=== SIMULATION RESULTS ({num_simulations} runs) ===")
    arb_stats = print_strategy_results("Arbitrage", num_simulations, arb_results[0], arb_results[1])
    ev_full_stats = print_strategy_results("Full Kelly (100%)", num_simulations, ev_full_results[0], ev_full_results[1])
    ev_half_stats = print_strategy_results("Half Kelly (50%)", num_simulations, ev_half_results[0], ev_half_results[1])
    ev_quarter_stats = print_strategy_results("Quarter Kelly (25%)", num_simulations, ev_quarter_results[0], ev_quarter_results[1])
    
    print("\nCOMPARISON:")
    print(f"Risk-adjusted return (Profit/StdDev) - Arbitrage: {arb_stats[2]:.4f}")
    print(f"Risk-adjusted return (Profit/StdDev) - Full Kelly: {ev_full_stats[2]:.4f}")
    print(f"Risk-adjusted return (Profit/StdDev) - Half Kelly: {ev_half_stats[2]:.4f}")
    print(f"Risk-adjusted return (Profit/StdDev) - Quarter Kelly: {ev_quarter_stats[2]:.4f}")
    
    # Generate visualizations for all strategies and save to Documents folder
    plot_kelly_comparison_results(arb_results, ev_full_results, ev_half_results, ev_quarter_results, 
                                 output_path=kelly_plot_path)
    
    # Export summary results to CSV
    strategy_names = ['Arbitrage', 'Full Kelly (100%)', 'Half Kelly (50%)', 'Quarter Kelly (25%)']
    stats = [arb_stats, ev_full_stats, ev_half_stats, ev_quarter_stats]
    export_results_to_csv(strategy_names, stats, results_csv_path)
    
    # Export detailed results to CSV
    detailed_stats = []
    detailed_stats.append(collect_detailed_stats("Arbitrage", num_simulations, arb_results[0], arb_results[1]))
    detailed_stats.append(collect_detailed_stats("Full Kelly (100%)", num_simulations, ev_full_results[0], ev_full_results[1]))
    detailed_stats.append(collect_detailed_stats("Half Kelly (50%)", num_simulations, ev_half_results[0], ev_half_results[1]))
    detailed_stats.append(collect_detailed_stats("Quarter Kelly (25%)", num_simulations, ev_quarter_results[0], ev_quarter_results[1]))
    
    # Convert to DataFrame and save
    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df.to_csv(detailed_results_csv_path, index=False)
    print(f"Detailed results saved to {detailed_results_csv_path}")

if __name__ == "__main__":
    main()
