import pandas as pd

# Load the CSV file
file_path = 'combined_data_with_best_odds_and_fair_odds.csv'
df = pd.read_csv(file_path)

# Function to calculate arbitrage opportunity
def calculate_arbitrage(over_odds, under_odds, over_line, under_line):
    if over_line > under_line:  # Check if the Over Line is smaller than the Under Line
        return None, None    
    over_implied_prob = 100 / (over_odds + 100) if over_odds > 0 else -over_odds / (-over_odds + 100)
    under_implied_prob = 100 / (under_odds + 100) if under_odds > 0 else -under_odds / (-under_odds + 100)
    
    if over_implied_prob + under_implied_prob < 1:
        total_investment = 1 / (over_implied_prob + under_implied_prob)
        bet_on_over = (over_implied_prob * total_investment)
        bet_on_under = (under_implied_prob * total_investment)

        return bet_on_over, bet_on_under
    else:
        return None, None


def calculate_ev_and_kelly(best_odds, consensus_odds):
    def implied_probability(american_odds):
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return -american_odds / (-american_odds + 100)

    # Convert American odds to payout multiplier
    def payout_multiplier(american_odds):
        if american_odds > 0:
            return american_odds / 100
        else:
            return 100 / -american_odds

    # Calculate probabilities of winning and losing
    p_win = implied_probability(consensus_odds)
    p_lose = 1 - p_win

    # Calculate payout from best odds
    payout = payout_multiplier(best_odds)

    edge = (p_win * payout) - (1 - p_win)

    # Kelly Fraction
    kelly_fraction = edge / payout if edge > 0 else 0
    
    return kelly_fraction, edge

df['Over Odds'] = df['best_odds_over']
df['Under Odds'] = df['best_odds_under']
df['Over Line'] = df['best_odds_over_line']
df['Under Line'] = df['best_odds_under_line']
df['Over Consensus Odds'] = df['fair_odds_over']
df['Under Consensus Odds'] = df['fair_odds_under']
df['Player Prop'] = df['player_name'] + " - " + df['market']

# List to hold the results
results = []

# Calculate arbitrage opportunities, EV, and Kelly Criterion
for index, row in df.iterrows():
    # Initialize variables for arbitrage and profit metrics
    bet_on_over = None
    bet_on_under = None
    profit_over_hits = ''
    profit_under_hits = ''
    profit_middle_hits = ''
    kelly_over = 0
    edge_over = 0
    kelly_under = 0
    edge_under = 0

    # Perform arbitrage and calculate profit metrics only if Over Line <= Under Line
    if row['Over Line'] <= row['Under Line']:
        bet_on_over, bet_on_under = calculate_arbitrage(
            row['Over Odds'], 
            row['Under Odds'], 
            row['Over Line'], 
            row['Under Line']
        )
    
    # Only calculate profit metrics if there's a valid arbitrage opportunity
    if bet_on_over is not None and bet_on_under is not None:
        # Calculate the implied probabilities for over and under directly
        over_implied_prob = 100 / (row['Over Odds'] + 100) if row['Over Odds'] > 0 else -row['Over Odds'] / (-row['Over Odds'] + 100)
        under_implied_prob = 100 / (row['Under Odds'] + 100) if row['Under Odds'] > 0 else -row['Under Odds'] / (-row['Under Odds'] + 100)
        total_implied_prob = over_implied_prob + under_implied_prob
        
        # Only valid if the total implied probability is less than 1 (i.e. an arbitrage exists)
        if total_implied_prob < 1:
            profit_value = (1 - total_implied_prob) * 100  # Revised profit calculation
            profit_over_hits = round(profit_value, 2)
            profit_under_hits = round(profit_value, 2)
            
            # If lines differ, double the profit for a middle hit scenario
            if row['best_odds_over_line'] != row['best_odds_under_line']:
                profit_middle_hits = round(profit_value * 2, 2)
            else:
                profit_middle_hits = ''


    # Calculate EV and Kelly only if line conditions are met
    # For Over: best odds over line <= fair odds over line
    if row['best_odds_over_line'] <= row['fair_line']:
        kelly_over, edge_over = calculate_ev_and_kelly(row['Over Odds'], row['Over Consensus Odds'])
    
    # For Under: best odds under line >= fair odds under line
    if row['best_odds_under_line'] >= row['fair_line']:
        kelly_under, edge_under = calculate_ev_and_kelly(row['Under Odds'], row['Under Consensus Odds'])

    # Append the results to the list
    results.append({
        'Player Prop': row['Player Prop'],
        'Game Start': row['game_start'],
        'Over Bet Sportsbook': row['best_odds_over_sportsbook'],
        'Best Over Odds': row['best_odds_over'],
        'Best Odds Over Line': row['best_odds_over_line'],
        'Under Bet Sportsbook': row['best_odds_under_sportsbook'], 
        'Best Under Odds': row['best_odds_under'],
        'Best Odds Under Line': row['best_odds_under_line'],
        'Fair Odds Over': row['fair_odds_over'],
        'Fair Odds Over Line': row['fair_line'],
        'Fair Odds Under': row['fair_odds_under'],
        'Fair Odds Under Line': row['fair_line'],
        'Bet on Over': round(bet_on_over * 100, 2) if bet_on_over is not None else '',
        'Bet on Under': round(bet_on_under * 100, 2) if bet_on_under is not None else '',
        '% Profit if Over Hits': profit_over_hits,
        '% Profit if Under Hits': profit_under_hits,
        '% Profit if Middle Hits': profit_middle_hits,
        'Over Edge (%)': round(edge_over * 100, 2) if edge_over > 0 else '',
        'Under Edge (%)': round(edge_under * 100, 2) if edge_under > 0 else '',        
        'Kelly Over Bet (% of bankroll)': round(kelly_over * 100, 2) if kelly_over else '',
        'Kelly Under Bet (% of bankroll)': round(kelly_under * 100, 2) if kelly_under else ''
    })

# Create a DataFrame from the results and save it to a CSV file
output_df = pd.DataFrame(results)
# Identify columns to check after 'Under Bet Sportsbook'
columns_to_check = [
    'Bet on Over', 'Bet on Under', '% Profit if Over Hits', 
    '% Profit if Under Hits', '% Profit if Middle Hits', 
    'Over Edge (%)', 'Under Edge (%)',
    'Kelly Over Bet (% of bankroll)', 'Kelly Under Bet (% of bankroll)'
]

# Filter out rows where all these columns are empty
output_df = output_df[~output_df[columns_to_check].isnull().all(axis=1)]
output_df = output_df[~output_df[columns_to_check].eq('').all(axis=1)]

# Save the filtered DataFrame to a CSV file
output_df.to_csv('arbitrage_results_with_ev_and_kelly.csv', index=False)

print("Filtered rows with empty data have been removed and saved to 'arbitrage_results_with_ev_and_kelly.csv'.")
