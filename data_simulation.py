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

def calculate_bankruptcy_rate(run_profits):
    """Calculate the percentage of simulations that ended in bankruptcy"""
    bankruptcies = sum(1 for profit in run_profits if profit <= -99)  # Consider -99 or less as bankruptcy
    return (bankruptcies / len(run_profits)) * 100

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

def print_kelly_comparison_results(full_results, half_results, quarter_results, num_simulations):
    """Print comparison of different Kelly fraction strategies"""
    # Unpack results
    full_total_bets, full_run_profits, _ = full_results
    half_total_bets, half_run_profits, _ = half_results
    quarter_total_bets, quarter_run_profits, _ = quarter_results
    
    # Calculate statistics
    full_avg_profit, full_profitable_pct, full_std_dev, full_risk_adjusted, full_percentiles = calculate_statistics(full_run_profits)
    half_avg_profit, half_profitable_pct, half_std_dev, half_risk_adjusted, half_percentiles = calculate_statistics(half_run_profits)
    quarter_avg_profit, quarter_profitable_pct, quarter_std_dev, quarter_risk_adjusted, quarter_percentiles = calculate_statistics(quarter_run_profits)
    
    # Calculate bankruptcy rates
    full_bankruptcy_rate = calculate_bankruptcy_rate(full_run_profits)
    half_bankruptcy_rate = calculate_bankruptcy_rate(half_run_profits)
    quarter_bankruptcy_rate = calculate_bankruptcy_rate(quarter_run_profits)
    
    print("\n=== KELLY FRACTION COMPARISON ===")
    
    # Format data for tabular display
    strategies = ["Full Kelly", "Half Kelly", "Quarter Kelly"]
    avg_bets = [full_total_bets / num_simulations, half_total_bets / num_simulations, quarter_total_bets / num_simulations]
    avg_profits = [full_avg_profit, half_avg_profit, quarter_avg_profit]
    profitable_pcts = [full_profitable_pct, half_profitable_pct, quarter_profitable_pct]
    std_devs = [full_std_dev, half_std_dev, quarter_std_dev]
    risk_adjusteds = [full_risk_adjusted, half_risk_adjusted, quarter_risk_adjusted]
    bankruptcy_rates = [full_bankruptcy_rate, half_bankruptcy_rate, quarter_bankruptcy_rate]
    min_profits = [full_percentiles['min'], half_percentiles['min'], quarter_percentiles['min']]
    median_profits = [full_percentiles['median'], half_percentiles['median'], quarter_percentiles['median']]
    max_profits = [full_percentiles['max'], half_percentiles['max'], quarter_percentiles['max']]
    
    # Print table headers
    print(f"{'Strategy':<15} {'Avg Bets':<10} {'Avg Profit':<12} {'Prof. %':<10} {'StdDev':<10} {'Risk/Reward':<12} {'Bankrupt %':<12} {'Min':<8} {'Median':<8} {'Max':<8}")
    print("-" * 105)
    
    # Print table rows
    for i in range(len(strategies)):
        print(f"{strategies[i]:<15} {avg_bets[i]:<10.2f} {avg_profits[i]:<12.2f}% {profitable_pcts[i]:<10.2f}% {std_devs[i]:<10.2f}% {risk_adjusteds[i]:<12.4f} {bankruptcy_rates[i]:<12.2f}% {min_profits[i]:<8.2f}% {median_profits[i]:<8.2f}% {max_profits[i]:<8.2f}%")

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

def simulate_positive_ev(data, num_simulations, track_bankroll=True):
    """Simulate positive EV betting strategy using Kelly criterion with bankruptcy check"""
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

def simulate_positive_ev_fractional_kelly(data, num_simulations, kelly_fraction=1.0, track_bankroll=True):
    """
    Simulate positive EV betting strategy using fractional Kelly criterion
    
    Parameters:
    data (DataFrame): The betting data
    num_simulations (int): Number of simulation runs
    kelly_fraction (float): Fraction of Kelly to bet (1.0 = full Kelly, 0.5 = half Kelly, etc.)
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
                
                # Apply the Kelly fraction to the bet size
                kelly_bet = pd.to_numeric(bet['Kelly Over Bet (% of bankroll)'], errors='coerce')
                if pd.notna(kelly_bet):
                    kelly_bet = kelly_bet * kelly_fraction
                
                over_odds = pd.to_numeric(bet['Best Over Odds'], errors='coerce')
                fair_odds_over = pd.to_numeric(bet['Fair Odds Over'], errors='coerce')
                
                # Skip if any conversion resulted in NaN
                if pd.isna(kelly_bet) or pd.isna(over_odds) or pd.isna(fair_odds_over):
                    continue
                
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
                
                # Apply the Kelly fraction to the bet size
                kelly_bet = pd.to_numeric(bet['Kelly Under Bet (% of bankroll)'], errors='coerce')
                if pd.notna(kelly_bet):
                    kelly_bet = kelly_bet * kelly_fraction
                
                under_odds = pd.to_numeric(bet['Best Under Odds'], errors='coerce')
                fair_odds_under = pd.to_numeric(bet['Fair Odds Under'], errors='coerce')
                
                # Skip if any conversion resulted in NaN
                if pd.isna(kelly_bet) or pd.isna(under_odds) or pd.isna(fair_odds_under):
                    continue
                
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

def plot_results(arb_results, ev_results):
    """Create visualizations comparing the two strategies"""
    arb_total_bets, arb_run_profits, arb_bankroll_histories = arb_results
    ev_total_bets, ev_run_profits, ev_bankroll_histories = ev_results
    
    # Calculate statistics
    arb_avg_profit, _, arb_std_dev, arb_risk_adjusted, _ = calculate_statistics(arb_run_profits)
    ev_avg_profit, _, ev_std_dev, ev_risk_adjusted, _ = calculate_statistics(ev_run_profits)
    
    # Process bankroll histories
    arb_avg_history, _ = process_bankroll_histories(arb_bankroll_histories)
    ev_avg_history, _ = process_bankroll_histories(ev_bankroll_histories)
    
    plt.figure(figsize=(12, 12))
    
    # Plot 1: Profit distribution
    plt.subplot(2, 2, 1)
    sns.histplot(arb_run_profits, kde=True, label='Arbitrage', alpha=0.6)
    sns.histplot(ev_run_profits, kde=True, label='Positive EV', alpha=0.6)
    plt.title('Profit Distribution')
    plt.xlabel('Profit Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Box plot comparison
    plt.subplot(2, 2, 2)
    data = [arb_run_profits, ev_run_profits]
    plt.boxplot(data, labels=['Arbitrage', 'Positive EV'])
    plt.title('Profit Comparison')
    plt.ylabel('Profit Percentage')
    
    # Plot 3: Risk-adjusted return
    plt.subplot(2, 2, 3)
    labels = ['Arbitrage', 'Positive EV']
    returns = [arb_avg_profit, ev_avg_profit]
    stds = [arb_std_dev, ev_std_dev]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, returns, width, label='Avg Return %')
    plt.bar(x + width/2, stds, width, label='Std Dev %')
    plt.title('Return and Risk Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # Plot 4: Average Bankroll Evolution
    plt.subplot(2, 2, 4)
    x_arb = np.arange(len(arb_avg_history))
    x_ev = np.arange(len(ev_avg_history))
    
    plt.plot(x_arb, arb_avg_history, label='Arbitrage')
    plt.plot(x_ev, ev_avg_history, label='Positive EV')
    plt.title('Average Bankroll Evolution')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('betting_strategy_comparison.png')
    print("Visualization saved as 'betting_strategy_comparison.png'")

def plot_kelly_comparison(full_kelly_results, half_kelly_results, quarter_kelly_results):
    """Create visualizations comparing different Kelly fractions"""
    # Unpack results
    full_total_bets, full_run_profits, full_bankroll_histories = full_kelly_results
    half_total_bets, half_run_profits, half_bankroll_histories = half_kelly_results
    quarter_total_bets, quarter_run_profits, quarter_bankroll_histories = quarter_kelly_results
    
    # Calculate statistics
    full_avg_profit, _, full_std_dev, full_risk_adjusted, full_percentiles = calculate_statistics(full_run_profits)
    half_avg_profit, _, half_std_dev, half_risk_adjusted, half_percentiles = calculate_statistics(half_run_profits)
    quarter_avg_profit, _, quarter_std_dev, quarter_risk_adjusted, quarter_percentiles = calculate_statistics(quarter_run_profits)
    
    # Process bankroll histories
    full_avg_history, full_percentile_histories = process_bankroll_histories(full_bankroll_histories)
    half_avg_history, half_percentile_histories = process_bankroll_histories(half_bankroll_histories)
    quarter_avg_history, quarter_percentile_histories = process_bankroll_histories(quarter_bankroll_histories)
    
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Profit distribution
    plt.subplot(2, 2, 1)
    sns.histplot(full_run_profits, kde=True, label='Full Kelly', alpha=0.6, color='blue')
    sns.histplot(half_run_profits, kde=True, label='Half Kelly', alpha=0.6, color='green')
    sns.histplot(quarter_run_profits, kde=True, label='Quarter Kelly', alpha=0.6, color='red')
    plt.title('Profit Distribution by Kelly Fraction')
    plt.xlabel('Profit Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Box plot comparison
    plt.subplot(2, 2, 2)
    data = [full_run_profits, half_run_profits, quarter_run_profits]
    plt.boxplot(data, labels=['Full Kelly', 'Half Kelly', 'Quarter Kelly'])
    plt.title('Profit Comparison by Kelly Fraction')
    plt.ylabel('Profit Percentage')
    
    # Plot 3: Risk-adjusted return
    plt.subplot(2, 2, 3)
    labels = ['Full Kelly', 'Half Kelly', 'Quarter Kelly']
    returns = [full_avg_profit, half_avg_profit, quarter_avg_profit]
    stds = [full_std_dev, half_std_dev, quarter_std_dev]
    risk_adjusted = [full_risk_adjusted, half_risk_adjusted, quarter_risk_adjusted]
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.bar(x - width, returns, width, label='Avg Return %')
    plt.bar(x, stds, width, label='Std Dev %')
    plt.bar(x + width, risk_adjusted, width, label='Return/Risk')
    plt.title('Return and Risk Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # Plot 4: Average Bankroll Evolution
    plt.subplot(2, 2, 4)
    
    # Add shaded regions for percentiles
    x_full = np.arange(len(full_avg_history))
    if len(x_full) > 0:
        plt.fill_between(x_full, full_percentile_histories['p25'], full_percentile_histories['p75'], 
                         alpha=0.2, color='blue', label='Full Kelly IQR')
    
    x_half = np.arange(len(half_avg_history))
    if len(x_half) > 0:
        plt.fill_between(x_half, half_percentile_histories['p25'], half_percentile_histories['p75'], 
                        alpha=0.2, color='green', label='Half Kelly IQR')
    
    x_quarter = np.arange(len(quarter_avg_history))
    if len(x_quarter) > 0:
        plt.fill_between(x_quarter, quarter_percentile_histories['p25'], quarter_percentile_histories['p75'], 
                        alpha=0.2, color='red', label='Quarter Kelly IQR')
    
    # Add median lines
    if len(x_full) > 0:
        plt.plot(x_full, full_avg_history, label='Full Kelly Avg', color='blue', linewidth=2)
    if len(x_half) > 0:
        plt.plot(x_half, half_avg_history, label='Half Kelly Avg', color='green', linewidth=2)
    if len(x_quarter) > 0:
        plt.plot(x_quarter, quarter_avg_history, label='Quarter Kelly Avg', color='red', linewidth=2)
    
    plt.title('Average Bankroll Evolution by Kelly Fraction')
    plt.xlabel('Number of Bets')
    plt.ylabel('Bankroll (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kelly_fraction_comparison.png')
    print("Kelly fraction comparison visualization saved as 'kelly_fraction_comparison.png'")

def main():
    # Check if file exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return
    
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
    
    # Run simulations for original strategies
    arb_results = simulate_arbitrage(data, num_simulations)
    ev_results = simulate_positive_ev(data, num_simulations)  # This is full Kelly
    
    # Run simulations for fractional Kelly strategies
    half_kelly_results = simulate_positive_ev_fractional_kelly(data, num_simulations, kelly_fraction=0.5)
    quarter_kelly_results = simulate_positive_ev_fractional_kelly(data, num_simulations, kelly_fraction=0.25)
    
    # Print results for original strategies
    print(f"\n=== ORIGINAL STRATEGY RESULTS ({num_simulations} runs) ===")
    arb_stats = print_strategy_results("Arbitrage", num_simulations, arb_results[0], arb_results[1])
    ev_stats = print_strategy_results("Positive EV (Full Kelly)", num_simulations, ev_results[0], ev_results[1])
    
    print("\nCOMPARISON:")
    print(f"Risk-adjusted return (Profit/StdDev) - Arbitrage: {arb_stats[2]:.4f}")
    print(f"Risk-adjusted return (Profit/StdDev) - Positive EV (Full Kelly): {ev_stats[2]:.4f}")
    
    # Print results for Kelly fraction comparison
    print_kelly_comparison_results(ev_results, half_kelly_results, quarter_kelly_results, num_simulations)
    
    # Generate visualizations
    plot_results(arb_results, ev_results)
    plot_kelly_comparison(ev_results, half_kelly_results, quarter_kelly_results)

if __name__ == "__main__":
    main()